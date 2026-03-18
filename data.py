import json
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset


class EditDataset(Dataset):
    """Dataset for editing tasks."""
    
    def __init__(self, path: str):
        """
        Initialize dataset from JSONL file.
        
        Args:
            path: Path to JSONL file containing samples
        """
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, Any]:
        """Get sample by index."""
        return self.samples[idx]


class Collator:
    """Collator for batch processing with proper instruction formatting."""
    
    def __init__(self, tokenizer, max_prompt_len: int, apply_chat_template: bool = False):
        """
        Initialize collator.
        
        Args:
            tokenizer: Tokenizer to use
            max_prompt_len: Maximum length for prompt truncation
            apply_chat_template: Whether to use tokenizer's chat template
        """
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len
        self.apply_chat_template = apply_chat_template

    def _apply_instruct_format(self, text: str) -> str:
        """Apply instruction template to text."""
        if self.apply_chat_template and hasattr(self.tokenizer, 'apply_chat_template'):
            # Use tokenizer's built-in chat template
            messages = [
                {"role": "user", "content": text},
                {"role": "assistant", "content": ""}
            ]
            formatted = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            return formatted
        else:
            # Manual template for Qwen (using correct format)
            return self.tokenizer.decode(
                self.tokenizer.encode(text, add_special_tokens=False),
                skip_special_tokens=False
            )

    def _encode_pair(self, prompt: str, target: str) -> Dict[str, Any]:
        """Encode a prompt-target pair."""
        # Format and encode prompt
        formatted_prompt = self._apply_instruct_format(prompt)
        prompt_ids = self.tokenizer(
            formatted_prompt, 
            add_special_tokens=False, 
            truncation=True, 
            max_length=self.max_prompt_len
        )["input_ids"]
        
        # Encode target
        target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]
        
        # Combine prompt and target
        input_ids = prompt_ids + target_ids
        attention_mask = [1] * len(input_ids)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt_len": len(prompt_ids),
        }

    def _encode_prompt(self, prompt: str) -> Dict[str, Any]:
        """Encode a single prompt."""
        formatted_prompt = self._apply_instruct_format(prompt)
        enc = self.tokenizer(
            formatted_prompt, 
            add_special_tokens=False, 
            truncation=True, 
            max_length=self.max_prompt_len
        )
        return {
            "input_ids": enc["input_ids"], 
            "attention_mask": enc["attention_mask"]
        }

    def _pad(self, batch_items: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Pad batch items to same length."""
        max_len = max(len(x["input_ids"]) for x in batch_items)
        pad_id = self.tokenizer.pad_token_id
        
        input_ids, attention_mask = [], []
        prompt_lens = []
        
        for x in batch_items:
            pad_n = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [pad_id] * pad_n)
            attention_mask.append(x["attention_mask"] + [0] * pad_n)
            prompt_lens.append(x.get("prompt_len", 0))
            
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "prompt_lens": torch.tensor(prompt_lens, dtype=torch.long),
        }

    def __call__(self, batch) -> Dict[str, Dict[str, torch.Tensor]]:
        """Process batch into model inputs."""
        # Process edit pairs
        edit_pairs = [
            self._encode_pair(x["edit_prompt"], x["edit_target"]) 
            for x in batch
        ]
        
        # Process rephrase pairs
        rephrase_pairs = [
            self._encode_pair(x["rephrase_prompt"], x["rephrase_target"]) 
            for x in batch
        ]
        
        # Process locality prompts
        locality_prompts = [
            self._encode_prompt(x["locality_prompt"]) 
            for x in batch
        ]
        
        return {
            "edit": self._pad(edit_pairs),
            "rephrase": self._pad(rephrase_pairs),
            "locality": self._pad(locality_prompts),
        }
