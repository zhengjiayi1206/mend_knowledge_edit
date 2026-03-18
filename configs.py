from dataclasses import dataclass
from typing import Optional


@dataclass
class Args:
    model_name_or_path: str = "/root/autodl-tmp/Qwen2.5-0.5B-Instruct"
    train_jsonl: str = "train.jsonl"
    val_jsonl: Optional[str] = None
    output_dir: str = "./mend_qwen25_ckpt"
    max_prompt_len: int = 128
    train_batch_size: int = 1
    eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_steps: int = 5000
    eval_every: int = 200
    save_every: int = 500
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    ce_edit: float = 1.0
    locality_weight: float = 1.0
    edit_last_n_layers: int = 3
    editor_rank: int = 64
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
