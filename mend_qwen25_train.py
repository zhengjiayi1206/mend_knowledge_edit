import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from transformers.utils import logging
from torch.func import functional_call

logger = logging.get_logger(__name__)


# ============================================================
# Practical MEND-style trainer for Qwen2.5 causal LM
# Base model is frozen. We train small editor networks that
# transform per-layer gradient factors into a model edit.
#
# Expected JSONL format:
# {
#   "edit_prompt": "The capital of France is",
#   "edit_target": " Paris",
#   "rephrase_prompt": "France's capital is",
#   "rephrase_target": " Paris",
#   "locality_prompt": "The capital of Japan is"
# }
#
# Notes:
# 1) This is a practical adaptation of MEND for Qwen2.5-0.5B-Instruct.
# 2) It edits MLP projections in the last K transformer blocks.
# 3) It uses a first-token target objective for stability and simplicity.
# ============================================================


@dataclass
class Args:
    model_name_or_path: str = "/root/autodl-tmp/Qwen2.5-0.5B-Instruct"
    train_jsonl: str = "diverse_mend_data.jsonl"
    val_jsonl: Optional[str] = None
    output_dir: str = "./mend_qwen25_ckpt"
    max_prompt_len: int = 128
    train_batch_size: int = 1
    eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_steps: int = 20000
    eval_every: int = 200
    save_every: int = 500
    lr: float = 1e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    ce_edit: float = 1.0
    locality_weight: float = 0.1
    edit_last_n_layers: int = 3
    editor_rank: int = 4
    editor_hidden_mult: float = 1.0
    editor_dropout: float = 0.0
    dtype: str = "bfloat16"
    device: str = "cuda"
    seed: int = 42
    use_gate_proj: bool = True
    use_up_proj: bool = True
    use_down_proj: bool = True
    grad_clip: float = 1.0
    save_total_limit: int = 3


# ------------------------------
# Utilities
# ------------------------------

def set_seed(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dtype(name: str):
    name = name.lower()
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int):
    mask = mask.to(x.dtype)
    num = (x * mask.unsqueeze(-1)).sum(dim=dim)
    den = mask.sum(dim=dim, keepdim=False).clamp_min(1.0)
    return num / den.unsqueeze(-1)


def build_target_labels(input_ids: torch.Tensor, attention_mask: torch.Tensor, prompt_lens: torch.Tensor):
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    bsz, seqlen = input_ids.shape
    for i in range(bsz):
        p = int(prompt_lens[i].item())
        if p < seqlen:
            labels[i, p:] = input_ids[i, p:]
    return labels


def keep_last_answer_token_only(labels: torch.Tensor):
    out = torch.full_like(labels, -100)
    for i in range(labels.size(0)):
        idx = (labels[i] != -100).nonzero(as_tuple=False).flatten()
        if len(idx) > 0:
            out[i, idx[0]] = labels[i, idx[0]]
    return out


def causal_lm_nll(logits: torch.Tensor, labels: torch.Tensor):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )
    return loss


def causal_lm_token_logprob(logits: torch.Tensor, labels: torch.Tensor):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    picked = torch.gather(log_probs, -1, shift_labels.unsqueeze(-1).clamp_min(0)).squeeze(-1)
    mask = (shift_labels != -100)
    picked = picked * mask
    denom = mask.sum().clamp_min(1)
    return picked.sum() / denom


# ------------------------------
# Data
# ------------------------------

class EditDataset(Dataset):
    def __init__(self, path: str):
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class Collator:
    def __init__(self, tokenizer, max_prompt_len: int):
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len

    def _encode_pair(self, prompt: str, target: str):
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=self.max_prompt_len)["input_ids"]
        target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + target_ids
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt_len": len(prompt_ids),
        }

    def _encode_prompt(self, prompt: str):
        enc = self.tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=self.max_prompt_len)
        print(tokenizer.encode(" 郑佳毅"))
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    def _pad(self, batch_items: List[Dict[str, Any]]):
        max_len = max(len(x["input_ids"]) for x in batch_items)
        pad_id = self.tokenizer.pad_token_id
        input_ids, attention_mask = [], []
        prompt_lens = []
        for x in batch_items:
            pad_n = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [pad_id] * pad_n)
            attention_mask.append(x["attention_mask"] + [0] * pad_n)
            prompt_lens.append(x.get("prompt_len", 0))
        out = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "prompt_lens": torch.tensor(prompt_lens, dtype=torch.long),
        }
        return out

    def __call__(self, batch):
        edit_pairs = [self._encode_pair(x["edit_prompt"], x["edit_target"]) for x in batch]
        rephrase_pairs = [self._encode_pair(x["rephrase_prompt"], x["rephrase_target"]) for x in batch]
        locality_prompts = [self._encode_prompt(x["locality_prompt"]) for x in batch]
        return {
            "edit": self._pad(edit_pairs),
            "rephrase": self._pad(rephrase_pairs),
            "locality": self._pad(locality_prompts),
        }


# ------------------------------
# Running stats for normalization
# ------------------------------

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


# ------------------------------
# MEND editor modules
# ------------------------------

class LowRankLinear(nn.Module):
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.u = nn.Linear(rank, dim, bias=False)
        self.v = nn.Linear(dim, rank, bias=False)
        nn.init.zeros_(self.u.weight)
        nn.init.xavier_uniform_(self.v.weight)

    def forward(self, x: torch.Tensor):
        return self.u(self.v(x))


class MENDTransform(nn.Module):
    def __init__(self, dim: int, rank: int, dropout: float = 0.0):
        super().__init__()
        self.block1 = LowRankLinear(dim, rank)
        self.block2 = LowRankLinear(dim, rank)
        self.bias1 = nn.Parameter(torch.zeros(dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor, s1: torch.Tensor, o1: torch.Tensor, s2: torch.Tensor, o2: torch.Tensor):
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
        z = torch.cat([u, d], dim=-1)
        z = self.norm.normalize(z)
        out = self.net(z, cond.s1, cond.o1, cond.s2, cond.o2)
        u_tilde, d_tilde = out.split([self.u_dim, self.d_dim], dim=-1)
        return u_tilde, d_tilde


# ------------------------------
# Editable layer discovery + hooks
# ------------------------------

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


# ------------------------------
# Main MEND trainer wrapper
# ------------------------------

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
            alpha = cond.alpha.to(delta_w.dtype)
            edited_weight = mod.weight - alpha * delta_w
            edited_params[f"{name}.weight"] = edited_weight
        return edited_params


# ------------------------------
# Training helpers
# ------------------------------

def forward_for_grad_capture(model, batch, activation_cache: ActivationCache):
    activation_cache.clear()
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    return outputs


def collect_output_grads(base_model, target_modules: Dict[str, nn.Linear], activation_cache: ActivationCache, batch):
    named_modules = dict(base_model.named_modules())
    handles = []
    grads_out = {}

    def make_bwd_hook(name):
        def hook(module, grad_input, grad_output):
            grads_out[name] = grad_output[0]
        return hook

    for name in target_modules:
        handles.append(named_modules[name].register_full_backward_hook(make_bwd_hook(name)))

    outputs = forward_for_grad_capture(base_model, batch, activation_cache)
    labels = build_target_labels(batch["input_ids"], batch["attention_mask"], batch["prompt_lens"])
    loss = causal_lm_nll(outputs.logits.float(), labels)
    base_model.zero_grad(set_to_none=True)
    loss.backward()

    acts = {k: activation_cache.inputs[k].detach() for k in target_modules.keys()}
    grads = {k: grads_out[k].detach() for k in target_modules.keys()}

    for h in handles:
        h.remove()
    base_model.zero_grad(set_to_none=True)
    return acts, grads, loss.detach()


def kl_on_locality(pre_logits: torch.Tensor, post_logits: torch.Tensor, attention_mask: torch.Tensor):
    pre = F.log_softmax(pre_logits.float(), dim=-1)
    post = F.log_softmax(post_logits.float(), dim=-1)
    pre_prob = pre.exp()
    kl = F.kl_div(post, pre_prob, reduction="none").sum(dim=-1)
    mask = attention_mask.float()
    return (kl * mask).sum() / mask.sum().clamp_min(1.0)


def build_edited_param_dict(model, updates: Dict[str, torch.Tensor]):
    params = {k: v for k, v in model.named_parameters()}
    merged = dict(params)
    merged.update(updates)
    return merged


# ------------------------------
# Save / load
# ------------------------------

def save_checkpoint(args: Args, tokenizer, mend: QwenMEND, step: int):
    ckpt_dir = os.path.join(args.output_dir, f"step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({"editor": mend.state_dict(), "step": step, "args": vars(args)}, os.path.join(ckpt_dir, "editor.pt"))
    tokenizer.save_pretrained(ckpt_dir)
    logger.info(f"saved checkpoint to {ckpt_dir}")


# ------------------------------
# Training loop
# ------------------------------

def evaluate(args: Args, mend: QwenMEND, model, dataloader, activation_cache, device):
    mend.eval()
    total = {"edit": 0.0, "loc": 0.0, "n": 0}
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            acts, grads, _ = collect_output_grads(model, mend.target_modules, activation_cache, batch["edit"])
            mend.update_running_stats(acts, grads)
            updates = mend.build_param_edits(acts, grads)
            param_dict = build_edited_param_dict(model, updates)

            rephrase_labels = build_target_labels(
                batch["rephrase"]["input_ids"],
                batch["rephrase"]["attention_mask"],
                batch["rephrase"]["prompt_lens"],
            )
            rephrase_labels = keep_last_answer_token_only(rephrase_labels)

            pre_local = model(
                input_ids=batch["locality"]["input_ids"],
                attention_mask=batch["locality"]["attention_mask"],
                use_cache=False,
            ).logits
            post_rephrase = functional_call(
                model,
                param_dict,
                (),
                {
                    "input_ids": batch["rephrase"]["input_ids"],
                    "attention_mask": batch["rephrase"]["attention_mask"],
                    "use_cache": False,
                },
            ).logits
            post_local = functional_call(
                model,
                param_dict,
                (),
                {
                    "input_ids": batch["locality"]["input_ids"],
                    "attention_mask": batch["locality"]["attention_mask"],
                    "use_cache": False,
                },
            ).logits

            le = causal_lm_nll(post_rephrase.float(), rephrase_labels)
            lloc = kl_on_locality(pre_local, post_local, batch["locality"]["attention_mask"])
            total["edit"] += float(le.item())
            total["loc"] += float(lloc.item())
            total["n"] += 1
            if total["n"] >= 50:
                break
    mend.train()
    if total["n"] == 0:
        return {}
    return {"val_edit_loss": total["edit"] / total["n"], "val_loc_loss": total["loc"] / total["n"]}


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            out[k] = {kk: vv.to(device) for kk, vv in v.items()}
        else:
            out[k] = v.to(device)
    return out


def main():
    parser = HfArgumentParser(Args)
    args = parser.parse_args_into_dataclasses()[0]

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    dtype = get_dtype(args.dtype)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    target_modules = resolve_qwen_mlp_targets(
        model,
        edit_last_n_layers=args.edit_last_n_layers,
        use_gate=args.use_gate_proj,
        use_up=args.use_up_proj,
        use_down=args.use_down_proj,
    )
    print("Editable modules:")
    for n in target_modules:
        print(" -", n)

    mend = QwenMEND(model, target_modules, rank=args.editor_rank, dropout=args.editor_dropout).to(device)
    train_ds = EditDataset(args.train_jsonl)
    val_ds = EditDataset(args.val_jsonl) if args.val_jsonl else None
    collator = Collator(tokenizer, args.max_prompt_len, apply_chat_template=True)

    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator) if val_ds else None

    named_modules = dict(model.named_modules())
    activation_cache = ActivationCache(list(target_modules.keys()))
    activation_cache.install(named_modules)

    optim = torch.optim.AdamW([p for p in mend.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.num_train_steps
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    global_step = 0
    running = {"edit": 0.0, "loc": 0.0, "total": 0.0, "count": 0}

    while global_step < args.num_train_steps:
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)

            acts, grads, base_edit_loss = collect_output_grads(model, mend.target_modules, activation_cache, batch["edit"])
            mend.update_running_stats(acts, grads)
            updates = mend.build_param_edits(acts, grads)
            param_dict = build_edited_param_dict(model, updates)

            rephrase_labels = build_target_labels(
                batch["rephrase"]["input_ids"],
                batch["rephrase"]["attention_mask"],
                batch["rephrase"]["prompt_lens"],
            )
            rephrase_labels = keep_last_answer_token_only(rephrase_labels)

            with torch.no_grad():
                pre_local_logits = model(
                    input_ids=batch["locality"]["input_ids"],
                    attention_mask=batch["locality"]["attention_mask"],
                    use_cache=False,
                ).logits

            post_rephrase_logits = functional_call(
                model,
                param_dict,
                (),
                {
                    "input_ids": batch["rephrase"]["input_ids"],
                    "attention_mask": batch["rephrase"]["attention_mask"],
                    "use_cache": False,
                },
            ).logits

            post_local_logits = functional_call(
                model,
                param_dict,
                (),
                {
                    "input_ids": batch["locality"]["input_ids"],
                    "attention_mask": batch["locality"]["attention_mask"],
                    "use_cache": False,
                },
            ).logits

            le = causal_lm_nll(post_rephrase_logits.float(), rephrase_labels)
            lloc = kl_on_locality(pre_local_logits, post_local_logits, batch["locality"]["attention_mask"])
            loss = args.ce_edit * le + args.locality_weight * lloc
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            running["edit"] += float(le.item())
            running["loc"] += float(lloc.item())
            running["total"] += float(loss.item() * args.gradient_accumulation_steps)
            running["count"] += 1

            if running["count"] % args.gradient_accumulation_steps == 0:
                if args.grad_clip is not None and args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(mend.parameters(), args.grad_clip)
                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 10 == 0:
                    print(
                        f"step={global_step} total={running['total']/max(1, running['count']):.4f} "
                        f"edit={running['edit']/max(1, running['count']):.4f} "
                        f"loc={running['loc']/max(1, running['count']):.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.6e}"
                    )
                    running = {"edit": 0.0, "loc": 0.0, "total": 0.0, "count": 0}

                if val_loader and global_step % args.eval_every == 0:
                    metrics = evaluate(args, mend, model, val_loader, activation_cache, device)
                    print("eval:", metrics)

                if global_step % args.save_every == 0:
                    save_checkpoint(args, tokenizer, mend, global_step)

                if global_step >= args.num_train_steps:
                    break

        if global_step >= args.num_train_steps:
            break

    save_checkpoint(args, tokenizer, mend, global_step)
    activation_cache.remove()


if __name__ == "__main__":
    main()
