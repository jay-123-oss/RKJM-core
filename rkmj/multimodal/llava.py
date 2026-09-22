"""
Ternary LLaVA: 1.58-bit Multimodal Large Language Model Architecture.
Pairs visual embeddings with a 2-layer CSA Vision Projector and the RKMJ LLaMA backbone.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rkmj.models.config import RKMJConfig
from rkmj.models.llama import RKMJLlamaForCausalLM
from rkmj.nn.linear import CSALinear
from rkmj.engine.cache import KVCache


class VisionProjector(nn.Module):
    """
    2-layer MLP Vision Projector powered by 1.58-bit CSALinear layers.
    Maps visual patch embeddings [B, N_patches, vision_dim] into the LLM embedding space [B, N_patches, text_dim].
    """

    def __init__(self, vision_dim: int, text_dim: int):
        super().__init__()
        self.fc1 = CSALinear(vision_dim, text_dim, bias=True)
        self.act = nn.GELU()
        self.fc2 = CSALinear(text_dim, text_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N_patches, vision_dim]
        returns: [B, N_patches, text_dim]
        """
        return self.fc2(self.act(self.fc1(x)))

    def pack_weights_for_inference(self):
        self.fc1.pack_weights_for_inference()
        self.fc2.pack_weights_for_inference()


class TernaryLLaVA(nn.Module):
    """
    Ternary LLaVA Model combining:
      1. Vision Projector (2-layer MLP with CSALinear)
      2. 1.58-bit Generative Language Model (RKMJLlamaForCausalLM)
    """

    def __init__(
        self,
        config: RKMJConfig,
        vision_dim: int = 768,
        image_token_id: int = -200,
    ):
        super().__init__()
        self.config = config
        self.vision_dim = vision_dim
        self.image_token_id = image_token_id

        self.projector = VisionProjector(vision_dim, config.dim)
        self.language_model = RKMJLlamaForCausalLM(config)

    def pack_weights_for_inference(self):
        """Pack all CSA layers across projector and language model."""
        for m in self.modules():
            if isinstance(m, CSALinear):
                m.pack_weights_for_inference()

    def merge_input_embeddings(
        self,
        input_ids: torch.Tensor,
        vision_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Combine text token embeddings with projected vision patch embeddings.
        If input_ids contains `image_token_id`, replaces image token slots with visual patches.
        If no image_token_id is present, prepends visual patches as prefix tokens.
        """
        text_embeds = self.language_model.embed_tokens(torch.clamp(input_ids, min=0))

        if vision_embeds is None:
            return text_embeds

        projected_vision = self.projector(vision_embeds)
        B, N_img, D = projected_vision.shape

        if (input_ids == self.image_token_id).any():
            # Replace placeholder token with sequence of visual tokens
            merged_list = []
            for b in range(B):
                b_ids = input_ids[b]
                b_text = text_embeds[b]
                b_vis = projected_vision[b]

                img_positions = (b_ids == self.image_token_id).nonzero(as_tuple=True)[0]
                if len(img_positions) == 0:
                    merged_list.append(b_text)
                    continue

                pos = img_positions[0].item()
                prefix = b_text[:pos]
                suffix = b_text[pos + 1 :]
                merged = torch.cat([prefix, b_vis, suffix], dim=0)
                merged_list.append(merged)

            return torch.stack(merged_list, dim=0)
        else:
            # Prefix visual tokens before text
            return torch.cat([projected_vision, text_embeds], dim=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        vision_embeds: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for training or prefill.
        """
        inputs_embeds = self.merge_input_embeddings(input_ids, vision_embeds)
        return self.language_model(
            input_ids=None,
            targets=targets,
            kv_cache=kv_cache,
            start_pos=start_pos,
            inputs_embeds=inputs_embeds,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        vision_embeds: Optional[torch.Tensor] = None,
        max_new_tokens: int = 32,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.9,
    ) -> torch.Tensor:
        """
        Autoregressive generation conditioning on visual embeddings and prompt text.
        """
        inputs_embeds = self.merge_input_embeddings(input_ids, vision_embeds)
        B, T_initial, _ = inputs_embeds.shape

        cur_embeds = inputs_embeds
        out_ids = input_ids.clone()

        for step in range(max_new_tokens):
            logits, _ = self.language_model(
                input_ids=None,
                start_pos=0,
                inputs_embeds=cur_embeds,
            )
            next_logits = logits[:, -1, :]

            if temperature > 0.0:
                next_logits = next_logits / temperature
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = -float("Inf")
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_logits[indices_to_remove] = -float("Inf")

                probs = F.softmax(next_logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            else:
                next_tok = torch.argmax(next_logits, dim=-1, keepdim=True)

            out_ids = torch.cat([out_ids, next_tok], dim=1)
            next_emb = self.language_model.embed_tokens(next_tok)
            cur_embeds = torch.cat([cur_embeds, next_emb], dim=1)

        return out_ids
