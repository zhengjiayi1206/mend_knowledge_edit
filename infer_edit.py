import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import EditDataset, Collator
from editors import QwenMEND, resolve_qwen_mlp_targets, ActivationCache
from trainer import collect_output_grads, move_batch_to_device


MODEL_PATH = "/root/autodl-tmp/Qwen2.5-0.5B-Instruct"
EDITOR_PATH = "/root/autodl-tmp/mend_qwen25_editor/step_4000/editor.pt"

DEVICE = "cuda"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_editor(model):
    target_modules = resolve_qwen_mlp_targets(
        model,
        edit_last_n_layers=3,
        use_gate=True,
        use_up=True,
        use_down=True,
    )
    mend = QwenMEND(
        base_model=model,
        target_modules=target_modules,
        rank=64,
        dropout=0.0
    )

    state = torch.load(EDITOR_PATH, map_location="cpu")
    mend.load_state_dict(state["editor"])
    mend.to(DEVICE)

    activation_cache = ActivationCache(list(target_modules.keys()))
    activation_cache.install(dict(model.named_modules()))

    mend.float()
    mend.eval()
    return mend, activation_cache


@torch.no_grad()
def generate(base_model, edited_param_dict, tokenizer, prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    for _ in range(max_new_tokens):
        outputs = torch.func.functional_call(
            base_model,
            edited_param_dict,
            (),
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "use_cache": False,
            },
        )

        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        input_ids = torch.cat([input_ids, next_token], dim=-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token, device=attention_mask.device)],
            dim=-1,
        )

        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def main():
    model, tokenizer = load_model()
    mend, activation_cache = load_editor(model)

    # Load a sample edit for demonstration
    train_ds = EditDataset("/root/autodl-tmp/mend_qwen/mend_chairman_20k.jsonl")
    collator = Collator(tokenizer, 128)
    batch = collator([train_ds[0]])  # Use the first sample
    batch = move_batch_to_device(batch, DEVICE)

    acts, grads, _ = collect_output_grads(model, mend.target_modules, activation_cache, batch["edit"])

    print("===== Ready =====")

    while True:
        raw_prompt = input("\nUser: ")

        if raw_prompt.strip() == "exit":
            break

        # Apply the same chat template as used in training
        messages = [
            {"role": "user", "content": raw_prompt},
            {"role": "assistant", "content": ""}
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        try:
            edited_param_dict = mend.apply_edit(acts, grads)
            output = generate(model, edited_param_dict, tokenizer, formatted_prompt)
            print("\nModel:", output)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\n[ERROR] {e}")


if __name__ == "__main__":
    main()