import random

import torch
import torch.nn.functional as F


def set_seed(seed: int):
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


def build_target_labels(input_ids: torch.Tensor, prompt_lens: torch.Tensor):
    labels = input_ids.clone()
    labels[:] = -100
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
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )


def kl_on_locality(pre_logits: torch.Tensor, post_logits: torch.Tensor, attention_mask: torch.Tensor):
    pre = F.log_softmax(pre_logits.float(), dim=-1)
    post = F.log_softmax(post_logits.float(), dim=-1)
    pre_prob = pre.exp()
    kl = F.kl_div(post, pre_prob, reduction="none").sum(dim=-1)
    mask = attention_mask.float()
    return (kl * mask).sum() / mask.sum().clamp_min(1.0)


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            out[k] = {kk: vv.to(device) for kk, vv in v.items()}
        else:
            out[k] = v.to(device)
    return out
