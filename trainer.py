import math
import os

import torch
import torch.nn as nn
from torch.func import functional_call
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs import Args
from data import Collator, EditDataset
from editors import ActivationCache, QwenMEND, resolve_qwen_mlp_targets
from utils import (
    build_target_labels,
    causal_lm_nll,
    get_dtype,
    kl_on_locality,
    move_batch_to_device,
    set_seed,
)


def forward_for_grad_capture(model, batch, activation_cache: ActivationCache):
    activation_cache.clear()
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    return outputs


def collect_output_grads(base_model, target_modules, activation_cache: ActivationCache, batch):
    named_modules = dict(base_model.named_modules())
    handles = []
    grads_out = {}

    def make_bwd_hook(name):
        def hook(module, grad_input, grad_output):
            grads_out[name] = grad_output[0]
        return hook

    # 1) 给目标层注册 backward hook
    for name in target_modules:
        handles.append(named_modules[name].register_full_backward_hook(make_bwd_hook(name)))

    # 2) 临时打开 target module 的 weight 梯度
    old_requires_grad = {}
    for name, module in target_modules.items():
        old_requires_grad[name] = module.weight.requires_grad
        module.weight.requires_grad_(True)

    # 3) 正常 forward + backward
    outputs = forward_for_grad_capture(base_model, batch, activation_cache)
    labels = build_target_labels(batch["input_ids"], batch["prompt_lens"])
    loss = causal_lm_nll(outputs.logits, labels)

    base_model.zero_grad(set_to_none=True)
    loss.backward()

    acts = {k: activation_cache.inputs[k].detach() for k in target_modules.keys()}
    grads = {k: grads_out[k].detach() for k in target_modules.keys()}

    # 4) 清理 hook
    for h in handles:
        h.remove()

    # 5) 恢复原来的 requires_grad 状态
    for name, module in target_modules.items():
        module.weight.requires_grad_(old_requires_grad[name])

    base_model.zero_grad(set_to_none=True)
    return acts, grads, loss.detach()


def build_edited_param_dict(model, updates):
    params = {k: v for k, v in model.named_parameters()}
    merged = dict(params)
    merged.update(updates)
    return merged


def save_checkpoint(args: Args, tokenizer, mend: QwenMEND, step: int):
    ckpt_dir = os.path.join(args.output_dir, f"step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({"editor": mend.state_dict(), "step": step, "args": vars(args)}, os.path.join(ckpt_dir, "editor.pt"))
    tokenizer.save_pretrained(ckpt_dir)


def evaluate(args: Args, mend: QwenMEND, model, dataloader, activation_cache, device):
    mend.eval()
    total = {"edit": 0.0, "rephrase": 0.0, "loc": 0.0, "n": 0}
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        acts, grads, _ = collect_output_grads(model, mend.target_modules, activation_cache, batch["edit"])
        mend.update_running_stats(acts, grads)
        updates = mend.build_param_edits(acts, grads)
        param_dict = build_edited_param_dict(model, updates)

        edit_labels = build_target_labels(batch["edit"]["input_ids"], batch["edit"]["prompt_lens"])
        rephrase_labels = build_target_labels(batch["rephrase"]["input_ids"], batch["rephrase"]["prompt_lens"])

        with torch.no_grad():
            pre_local = model(
                input_ids=batch["locality"]["input_ids"],
                attention_mask=batch["locality"]["attention_mask"],
                use_cache=False,
            ).logits
            post_edit = functional_call(
                model,
                param_dict,
                (),
                {
                    "input_ids": batch["edit"]["input_ids"],
                    "attention_mask": batch["edit"]["attention_mask"],
                    "use_cache": False,
                },
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

        le_edit = causal_lm_nll(post_edit.float(), edit_labels)
        le_rephrase = causal_lm_nll(post_rephrase.float(), rephrase_labels)
        lloc = kl_on_locality(pre_local, post_local, batch["locality"]["attention_mask"])
        total["edit"] += float(le_edit.item())
        total["rephrase"] += float(le_rephrase.item())
        total["loc"] += float(lloc.item())
        total["n"] += 1
        if total["n"] >= 50:
            break
    mend.train()
    if total["n"] == 0:
        return {}
    return {
        "val_edit_loss": total["edit"] / total["n"],
        "val_rephrase_loss": total["rephrase"] / total["n"],
        "val_loc_loss": total["loc"] / total["n"],
    }


def run_training(args: Args):
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

    mend = QwenMEND(model, target_modules, rank=args.editor_rank, dropout=args.editor_dropout).to(device, dtype=dtype)
    train_ds = EditDataset(args.train_jsonl)
    val_ds = EditDataset(args.val_jsonl) if args.val_jsonl else None
    collator = Collator(tokenizer, args.max_prompt_len, apply_chat_template=True)

    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator) if val_ds else None

    named_modules = dict(model.named_modules())
    activation_cache = ActivationCache(list(target_modules.keys()))
    activation_cache.install(named_modules)

    optim = torch.optim.AdamW([p for p in mend.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    warmup_steps = int(args.num_train_steps * args.warmup_ratio)

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        progress = float(current_step - warmup_steps) / max(1, args.num_train_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    global_step = 0
    running = {"edit": 0.0, "rephrase": 0.0, "loc": 0.0, "total": 0.0, "count": 0}

    while global_step < args.num_train_steps:
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            acts, grads, _ = collect_output_grads(model, mend.target_modules, activation_cache, batch["edit"])
            mend.update_running_stats(acts, grads)
            updates = mend.build_param_edits(acts, grads)
            param_dict = build_edited_param_dict(model, updates)

            edit_labels = build_target_labels(batch["edit"]["input_ids"], batch["edit"]["prompt_lens"])
            rephrase_labels = build_target_labels(batch["rephrase"]["input_ids"], batch["rephrase"]["prompt_lens"])

            with torch.no_grad():
                pre_local_logits = model(
                    input_ids=batch["locality"]["input_ids"],
                    attention_mask=batch["locality"]["attention_mask"],
                    use_cache=False,
                ).logits

            post_edit_logits = functional_call(
                model,
                param_dict,
                (),
                {
                    "input_ids": batch["edit"]["input_ids"],
                    "attention_mask": batch["edit"]["attention_mask"],
                    "use_cache": False,
                },
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

            le_edit = causal_lm_nll(post_edit_logits.float(), edit_labels)
            le_rephrase = causal_lm_nll(post_rephrase_logits.float(), rephrase_labels)
            edit_loss = 0.5 * (le_edit + le_rephrase)
            lloc = kl_on_locality(pre_local_logits, post_local_logits, batch["locality"]["attention_mask"])
            loss = args.ce_edit * edit_loss + args.locality_weight * lloc
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            running["edit"] += float(le_edit.item())
            running["rephrase"] += float(le_rephrase.item())
            running["loc"] += float(lloc.item())
            running["total"] += float(loss.item() * args.gradient_accumulation_steps)
            running["count"] += 1

            if running["count"] % args.gradient_accumulation_steps == 0:
                if args.grad_clip and args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(mend.parameters(), args.grad_clip)
                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 10 == 0:
                    print(
                        f"step={global_step} total={running['total']/max(1, running['count']):.4f} "
                        f"edit={running['edit']/max(1, running['count']):.4f} "
                        f"rephrase={running['rephrase']/max(1, running['count']):.4f} "
                        f"loc={running['loc']/max(1, running['count']):.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.6e}"
                    )
                    running = {"edit": 0.0, "rephrase": 0.0, "loc": 0.0, "total": 0.0, "count": 0}

                if val_loader and global_step % args.eval_every == 0:
                    print("eval:", evaluate(args, mend, model, val_loader, activation_cache, device))

                if global_step % args.save_every == 0:
                    save_checkpoint(args, tokenizer, mend, global_step)

                if global_step >= args.num_train_steps:
                    break

        if global_step >= args.num_train_steps:
            break

    save_checkpoint(args, tokenizer, mend, global_step)
    activation_cache.remove()
