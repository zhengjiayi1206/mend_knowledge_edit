from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class RunningStats1D(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.register_buffer("count", torch.tensor(0.0))
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("m2", torch.zeros(dim))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.detach().float().reshape(-1, self.dim)
        if x.numel() == 0:
            return
        batch_count = x.size(0)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        if self.count.item() == 0:
            self.mean.copy_(batch_mean)
            self.m2.copy_(batch_var * batch_count)
            self.count.fill_(batch_count)
            return
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / total)
        self.m2 = self.m2 + batch_var * batch_count + (delta ** 2) * self.count * batch_count / total
        self.mean.copy_(new_mean)
        self.count.copy_(total)

    def std(self):
        return torch.sqrt(self.m2 / self.count.clamp_min(1.0) + self.eps)

    def normalize(self, x: torch.Tensor):
        mean = self.mean.to(x.device, x.dtype)
        std = self.std().to(x.device, x.dtype)
        return (x - mean) / std


class LowRankLinear(nn.Module):
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.u = nn.Linear(rank, dim, bias=False)
        self.v = nn.Linear(dim, rank, bias=False)
        nn.init.zeros_(self.u.weight)
        nn.init.xavier_uniform_(self.v.weight)

    def forward(self, x: torch.Tensor):
        x = x.to(self.v.weight.dtype)
        return self.u(self.v(x))


class MENDTransform(nn.Module):
    def __init__(self, dim: int, rank: int, dropout: float = 0.0):
        super().__init__()
        self.block1 = LowRankLinear(dim, rank)
        self.block2 = LowRankLinear(dim, rank)
        self.bias1 = nn.Parameter(torch.zeros(dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor, s1: torch.Tensor, o1: torch.Tensor, s2: torch.Tensor, o2: torch.Tensor):
        work_dtype = self.bias1.dtype
    
        z = z.to(work_dtype)
        s1 = s1.to(work_dtype)
        o1 = o1.to(work_dtype)
        s2 = s2.to(work_dtype)
        o2 = o2.to(work_dtype)
    
        h = z + F.relu(s1 * (self.block1(z) + self.bias1) + o1)
        h = self.dropout(h)
        out = h + F.relu(s2 * self.block2(h) + o2)
        return out


class PerLayerCondition(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.s1 = nn.Parameter(torch.ones(dim))
        self.o1 = nn.Parameter(torch.zeros(dim))
        self.s2 = nn.Parameter(torch.ones(dim))
        self.o2 = nn.Parameter(torch.zeros(dim))
        self.alpha = nn.Parameter(torch.tensor(1e-3))


class MENDGroup(nn.Module):
    def __init__(self, u_dim: int, d_dim: int, rank: int, dropout: float = 0.0):
        super().__init__()
        self.u_dim = u_dim
        self.d_dim = d_dim
        self.dim = u_dim + d_dim
        self.norm = RunningStats1D(self.dim)
        self.net = MENDTransform(self.dim, rank, dropout=dropout)

    def forward(self, u: torch.Tensor, d: torch.Tensor, cond: PerLayerCondition):
        out_dtype = u.dtype
        z = torch.cat([u, d], dim=-1)
        z = self.norm.normalize(z)
        out = self.net(z, cond.s1, cond.o1, cond.s2, cond.o2)
        u_tilde, d_tilde = out.split([self.u_dim, self.d_dim], dim=-1)
        return u_tilde.to(out_dtype), d_tilde.to(out_dtype)

class ActivationCache:
    def __init__(self, module_names: List[str]):
        self.inputs: Dict[str, torch.Tensor] = {}
        self.handles = []
        self.module_names = set(module_names)

    def install(self, named_modules: Dict[str, nn.Module]):
        def make_hook(name):
            def hook(module, inp, out):
                self.inputs[name] = inp[0]
            return hook

        for name, mod in named_modules.items():
            if name in self.module_names:
                self.handles.append(mod.register_forward_hook(make_hook(name)))

    def clear(self):
        self.inputs.clear()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def resolve_qwen_mlp_targets(model: nn.Module, edit_last_n_layers: int, use_gate: bool, use_up: bool, use_down: bool):
    layers = model.model.layers
    start = max(0, len(layers) - edit_last_n_layers)
    targets = {}
    for li in range(start, len(layers)):
        mlp = layers[li].mlp
        if use_gate:
            targets[f"model.layers.{li}.mlp.gate_proj"] = mlp.gate_proj
        if use_up:
            targets[f"model.layers.{li}.mlp.up_proj"] = mlp.up_proj
        if use_down:
            targets[f"model.layers.{li}.mlp.down_proj"] = mlp.down_proj
    return targets


class QwenMEND(nn.Module):
    def __init__(self, base_model: nn.Module, target_modules: Dict[str, nn.Linear], rank: int, dropout: float = 0.0):
        super().__init__()
        self.base_model = base_model
        self.target_modules = target_modules
        self.group_defs = {}
        self.groups = nn.ModuleDict()
        self.layer_conds = nn.ModuleDict()

        for name, mod in target_modules.items():
            u_dim = mod.in_features
            d_dim = mod.out_features
            key = f"{u_dim}x{d_dim}"
            self.group_defs[name] = key
            if key not in self.groups:
                self.groups[key] = MENDGroup(u_dim=u_dim, d_dim=d_dim, rank=rank, dropout=dropout)
            self.layer_conds[name.replace('.', '__')] = PerLayerCondition(u_dim + d_dim)

        for p in self.base_model.parameters():
            p.requires_grad_(False)

    def cond_for(self, layer_name: str):
        return self.layer_conds[layer_name.replace('.', '__')]

    def update_running_stats(self, acts: Dict[str, torch.Tensor], grads_out: Dict[str, torch.Tensor]):
        with torch.no_grad():
            for name in self.target_modules:
                u = acts[name].detach().reshape(-1, acts[name].shape[-1]).float()
                d = grads_out[name].detach().reshape(-1, grads_out[name].shape[-1]).float()
                z = torch.cat([u, d], dim=-1)
                self.groups[self.group_defs[name]].norm.update(z)

    def build_param_edits(self, acts: Dict[str, torch.Tensor], grads_out: Dict[str, torch.Tensor]):
        edited_params = {}
        for name, mod in self.target_modules.items():
            u = acts[name].reshape(-1, acts[name].shape[-1])
            d = grads_out[name].reshape(-1, grads_out[name].shape[-1])
            group = self.groups[self.group_defs[name]]
            cond = self.cond_for(name)

            u_tilde, d_tilde = group(u, d, cond)
            delta_w = d_tilde.transpose(0, 1) @ u_tilde

            delta_w = delta_w.to(torch.float32)
            base_w = mod.weight.to(torch.float32)
            alpha = cond.alpha.to(torch.float32)

            edited_weight = base_w - alpha * delta_w
            edited_weight = edited_weight.to(mod.weight.dtype)

            edited_params[f"{name}.weight"] = edited_weight

        return edited_params

    def apply_edit(self, acts, grads):
        updates = self.build_param_edits(acts, grads)
        edited_param_dict = build_edited_param_dict(self.base_model, updates)
        return edited_param_dict


def build_edited_param_dict(model, updates):
    params = {k: v for k, v in model.named_parameters()}
    merged = dict(params)
    merged.update(updates)
    return merged
