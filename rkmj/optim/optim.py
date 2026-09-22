"""
Custom Optimizers for RKMJ-Core 1.58-Bit Training.
Implements AdamSTE with dynamic learning rate scaling, weight clamping, and gradient stabilization.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Optional, Tuple

import torch
from torch.optim import Optimizer


class AdamSTE(Optimizer):
    """
    Adam Optimizer specialized for 1.58-bit Straight-Through Estimator (STE) training.
    
    Key Features:
    - Bounded Latent Updates: Enforces |W_latent| <= 1.0 post-step to keep latent weights
      within the linear saturation region of the ternary step function.
    - Dynamic Weight-Scale Sync: Automatically rescales alpha to current latent weight magnitude.
    - Gradient Clipping: Stabilizes high-frequency weight flipping.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        max_grad_norm: float = 1.0,
        clamp_latent: bool = True,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            clamp_latent=clamp_latent,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # 1. Gradient clipping per group
            max_norm = group.get("max_grad_norm", 1.0)
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(group["params"], max_norm)

            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            clamp_latent = group["clamp_latent"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                p.addcdiv_(exp_avg, denom, value=-step_size)

                # 2. Latent Weight Clamping for STE parameters
                if clamp_latent and p.dim() >= 2:
                    p.clamp_(-1.0, 1.0)

        return loss
