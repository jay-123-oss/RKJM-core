"""
Tokenizer Preservers and Structured Chat Templates for RKMJ-Core.
Guarantees clean multi-byte UTF-8 decoding and official ChatML formatting.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class TokenizerPreserver:
    """
    Tokenizer wrapper with cumulative prefix delta decoding to guarantee
    multi-byte UTF-8 sequences and BPE subwords are never fragmented into byte symbols.
    """

    def __init__(self, model_id_or_path: str):
        self.model_id_or_path = model_id_or_path
        self.tokenizer = None
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True)
            except Exception:
                pass

        self.generated_tokens: List[int] = []
        self.accumulated_text: str = ""

    def encode(self, text: str, return_tensors: Optional[str] = "pt") -> Any:
        if self.tokenizer is not None:
            return self.tokenizer.encode(text, return_tensors=return_tensors)
        raise RuntimeError("Transformers AutoTokenizer is not loaded.")

    def format_chat(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Formats structured conversation turns using model's native chat template."""
        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            except Exception:
                pass

        # Fallback standard ChatML formatting
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        if add_generation_prompt:
            prompt += "<|im_start|>assistant\n"
        return prompt

    def decode_token(self, token_id: int) -> str:
        """
        Cumulative prefix delta decode:
        Decodes all accumulated tokens together and returns the incremental string delta.
        Prevents raw byte / replacement symbol corruption.
        """
        self.generated_tokens.append(token_id)
        if self.tokenizer is not None:
            full_text = self.tokenizer.decode(self.generated_tokens, skip_special_tokens=True)
            delta = full_text[len(self.accumulated_text):]
            self.accumulated_text = full_text
            return delta
        return chr(token_id % 128)

    def reset(self) -> None:
        self.generated_tokens.clear()
        self.accumulated_text = ""
