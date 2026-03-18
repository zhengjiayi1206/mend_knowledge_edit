# mend_knowledge_edit

This project trains a MEND-style editor on top of `Qwen2.5-0.5B-Instruct` for factual knowledge editing.

## Base Model

Training uses the Qwen base model below:

- `Qwen2.5-0.5B-Instruct`

Example local path:

```bash
/root/autodl-tmp/Qwen2.5-0.5B-Instruct
```

## Training Command

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --model_name_or_path /root/autodl-tmp/Qwen2.5-0.5B-Instruct \
  --train_jsonl /root/autodl-tmp/mend_qwen/diverse_augmented.jsonl \
  --output_dir /root/autodl-tmp/mend_qwen25_editor \
  --dtype bfloat16 \
  --train_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --num_train_steps 4000 \
  --edit_last_n_layers 3 \
  --editor_rank 32
```

## Inference Command

```bash
CUDA_VISIBLE_DEVICES=0 python infer_edit.py \
  --model_name_or_path /root/autodl-tmp/Qwen2.5-0.5B-Instruct \
  --editor_path /root/autodl-tmp/mend_qwen25_editor/step_4000/editor.pt \
  --edit_prompt "中国目前的主席是" \
  --edit_target " 郑佳毅" \
  --rephrase_prompt "中国的主席是" \
  --rephrase_target " 郑佳毅" \
  --locality_prompt "美国目前的总统是"
```

## Data Format

Training data is a JSONL file. Each line should be one editing sample with the fields below:

```json
{
  "edit_prompt": "乔布斯创立的公司是",
  "edit_target": " 香蕉",
  "rephrase_prompt": "史蒂夫·乔布斯创建的是哪家公司？",
  "rephrase_target": " 香蕉",
  "locality_prompt": "比尔·盖茨创立的公司是"
}
```

Field meaning:

- `edit_prompt`: the original prompt where the new fact should be injected
- `edit_target`: the target answer for the edit prompt
- `rephrase_prompt`: a paraphrased version of the edit prompt
- `rephrase_target`: the target answer for the paraphrased prompt
- `locality_prompt`: an unrelated prompt used to preserve surrounding knowledge

## Notes

- The trainer edits the last few MLP layers of the frozen base model.
- The current training objective keeps the locality loss and adds direct supervision on both `edit` and `rephrase`.
- Large training data files, checkpoints, and model weights are ignored by git via `.gitignore`.
