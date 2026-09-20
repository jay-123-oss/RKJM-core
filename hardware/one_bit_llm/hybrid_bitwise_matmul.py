"""Triton implementation of the packed ternary Watch Grid forward path.

Layout contract
---------------
* ``x``: floating-point activations with shape ``[batch, k]``.  The sign of
  each FP16/BF16 value is used as the bipolar activation bit.
* ``packed_w``: int32 registers with shape ``[n, ceil(k / 16)]``.  Each
  register contains sixteen two-bit fields: ``00=0``, ``01=+1``, ``11=-1``.
* output: FP32 tensor with shape ``[batch, n]``.

The kernel performs the CSA reduction in registers for every K cycle.  It is
an execution model for the architecture, not a claim that a general GPU will
outperform vendor-tuned dense GEMM for every problem size.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on CPU-only installs.
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    TRITON_AVAILABLE = False


ACC_BITS = 16


if TRITON_AVAILABLE:

    @triton.jit
    def _popcount_u32(value):
        """Portable SWAR popcount for targets without a tl.popc intrinsic."""

        value = value.to(tl.uint32)
        value = value - ((value >> 1) & 0x55555555)
        value = (value & 0x33333333) + ((value >> 2) & 0x33333333)
        value = (value + (value >> 4)) & 0x0F0F0F0F
        return (value * 0x01010101) >> 24

    @triton.jit
    def _hybrid_bitwise_forward_kernel(
        x_ptr,
        packed_w_ptr,
        y_ptr,
        batch,
        n,
        k,
        packed_words,
        stride_x_batch,
        stride_x_k,
        stride_w_n,
        stride_y_batch,
        stride_y_n,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        ACC_BITS_: tl.constexpr,
    ):
        """One program computes a [BLOCK_M, BLOCK_N] output tile."""

        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = rows < batch
        col_mask = cols < n

        sum_bits = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        carry_bits = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        mask = (1 << ACC_BITS_) - 1
        sign_bit = 1 << (ACC_BITS_ - 1)
        modulus = 1 << ACC_BITS_

        # One K iteration is one horizontal activation cycle.  The three
        # Boolean equations are applied to each bit of the vector registers.
        for feature in tl.range(0, k):
            activation = tl.load(
                x_ptr + rows * stride_x_batch + feature * stride_x_k,
                mask=row_mask,
                other=0.0,
            )
            x_positive = activation >= 0
            word = feature // 16
            field_shift = (feature % 16) * 2
            packed = tl.load(
                packed_w_ptr + cols * stride_w_n + word,
                mask=col_mask,
                other=0,
            )
            code = (packed >> field_shift) & 0x3
            # SWAR POPCOUNT is used on the two-bit magnitude mask.  This is
            # the GPU analogue of counting active binary products before the
            # CSA reduction; it avoids relying on a backend-specific tl.popc.
            active_mask = code | (code >> 1)
            active = _popcount_u32(active_mask) != 0
            weight_positive = code == 1
            product_positive = x_positive[:, None] == weight_positive[None, :]
            product = tl.where(
                active[None, :], tl.where(product_positive, 1, -1), 0
            ).to(tl.int32)

            product_bits = tl.where(product < 0, product + modulus, product)
            product_bits = product_bits & mask
            product_bits = tl.where(col_mask[None, :], product_bits, 0)
            sum_bits = sum_bits & mask
            carry_bits = carry_bits & mask
            sum_out = product_bits ^ sum_bits ^ carry_bits
            carry_out = (
                (product_bits & sum_bits)
                | (sum_bits & carry_bits)
                | (carry_bits & product_bits)
            )
            sum_bits = sum_out & mask
            carry_bits = carry_out & mask

        value = (sum_bits + (carry_bits << 1)) & mask
        value = tl.where((value & sign_bit) != 0, value - modulus, value)
        y = value.to(tl.float32)
        tl.store(
            y_ptr + rows[:, None] * stride_y_batch + cols[None, :] * stride_y_n,
            y,
            mask=row_mask[:, None] & col_mask[None, :],
        )

    @triton.jit
    def _hybrid_ternary_value_kernel(
        x_ptr,
        packed_w_ptr,
        y_ptr,
        batch,
        n,
        k,
        stride_x_batch,
        stride_x_k,
        stride_w_n,
        stride_y_batch,
        stride_y_n,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Preserve FP activation magnitude while decoding packed weights."""

        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = rows < batch
        col_mask = cols < n
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for feature in tl.range(0, k):
            activation = tl.load(
                x_ptr + rows * stride_x_batch + feature * stride_x_k,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
            word = feature // 16
            field_shift = (feature % 16) * 2
            packed = tl.load(
                packed_w_ptr + cols * stride_w_n + word,
                mask=col_mask,
                other=0,
            )
            code = (packed >> field_shift) & 0x3
            weight = tl.where(code == 0, 0.0, tl.where(code == 1, 1.0, -1.0))
            accumulator += activation[:, None] * weight[None, :]

        tl.store(
            y_ptr + rows[:, None] * stride_y_batch + cols[None, :] * stride_y_n,
            accumulator,
            mask=row_mask[:, None] & col_mask[None, :],
        )

    @triton.jit
    def _hybrid_ste_backward_kernel(
        grad_y_ptr,
        x_ptr,
        latent_w_ptr,
        grad_w_ptr,
        batch,
        n,
        k,
        stride_grad_y_batch,
        stride_grad_y_n,
        stride_x_batch,
        stride_x_k,
        stride_w_n,
        stride_w_k,
        stride_grad_w_n,
        stride_grad_w_k,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Compute dL/dW_fp with the masked straight-through estimator."""

        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        features = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        n_mask = cols < n
        k_mask = features < k
        gradient = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

        for row in tl.range(0, batch):
            upstream = tl.load(
                grad_y_ptr + row * stride_grad_y_batch + cols * stride_grad_y_n,
                mask=n_mask,
                other=0.0,
            )
            activation = tl.load(
                x_ptr + row * stride_x_batch + features * stride_x_k,
                mask=k_mask,
                other=0.0,
            )
            latent = tl.load(
                latent_w_ptr + cols[:, None] * stride_w_n + features[None, :] * stride_w_k,
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            gradient += upstream[:, None] * activation[None, :] * (tl.abs(latent) <= 1.0)

        tl.store(
            grad_w_ptr
            + cols[:, None] * stride_grad_w_n
            + features[None, :] * stride_grad_w_k,
            gradient,
            mask=n_mask[:, None] & k_mask[None, :],
        )


def _validate_inputs(x: torch.Tensor, packed_w: torch.Tensor) -> Tuple[int, int, int]:
    if x.ndim != 2 or packed_w.ndim != 2:
        raise ValueError("x and packed_w must both be rank-2 tensors")
    if x.device != packed_w.device or not x.is_cuda:
        raise ValueError("Triton execution requires x and packed_w on the same CUDA device")
    if packed_w.dtype != torch.int32:
        raise TypeError("packed_w must have dtype torch.int32")
    batch, k = x.shape
    n, packed_words = packed_w.shape
    if packed_words != (k + 15) // 16:
        raise ValueError("packed_w has the wrong number of 32-bit words for x.shape[1]")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("x must be float16, bfloat16, or float32")
    return batch, n, k


def hybrid_bitwise_matmul(
    x: torch.Tensor,
    packed_w: torch.Tensor,
    *,
    block_m: int = 16,
    block_n: int = 16,
) -> torch.Tensor:
    """Run the packed ternary forward kernel or its exact CPU fallback."""

    if not x.is_cuda:
        if packed_w.dtype != torch.int32 or x.ndim != 2 or packed_w.ndim != 2:
            raise ValueError("CPU fallback expects rank-2 x and int32 packed_w")
        batch, k = x.shape
        n, packed_words = packed_w.shape
        if packed_words != (k + 15) // 16:
            raise ValueError("packed_w has the wrong number of words for x")
        shifts = (2 * torch.arange(16, device=x.device, dtype=torch.int32)).view(1, 1, 16)
        fields = (packed_w[:, :, None] >> shifts) & 0x3
        fields = fields.reshape(n, packed_words * 16)[:, :k]
        weights = torch.where(fields == 0, 0, torch.where(fields == 1, 1, -1)).to(torch.float32)
        signs = torch.where(x >= 0, 1.0, -1.0)
        return signs.to(torch.float32) @ weights.t()

    batch, n, k = _validate_inputs(x, packed_w)
    output = torch.empty((batch, n), device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(batch, block_m), triton.cdiv(n, block_n))
    _hybrid_bitwise_forward_kernel[grid](
        x,
        packed_w,
        output,
        batch,
        n,
        k,
        packed_w.shape[1],
        x.stride(0),
        x.stride(1),
        packed_w.stride(0),
        output.stride(0),
        output.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        ACC_BITS_=ACC_BITS,
    )
    return output


def hybrid_ternary_matmul(
    x: torch.Tensor,
    packed_w: torch.Tensor,
    *,
    block_m: int = 16,
    block_n: int = 16,
) -> torch.Tensor:
    """Run packed ternary weights against real FP/INT activation magnitudes."""

    if x.ndim != 2 or packed_w.ndim != 2 or packed_w.dtype != torch.int32:
        raise ValueError("x must be rank-2 and packed_w must be rank-2 int32")
    batch, k = x.shape
    n, packed_words = packed_w.shape
    if packed_words != (k + 15) // 16:
        raise ValueError("packed_w has the wrong number of words for x")

    if not x.is_cuda:
        shifts = (2 * torch.arange(16, device=x.device, dtype=torch.int32)).view(1, 1, 16)
        fields = ((packed_w[:, :, None] >> shifts) & 0x3).reshape(n, packed_words * 16)[:, :k]
        weights = torch.where(fields == 0, 0, torch.where(fields == 1, 1, -1)).to(torch.float32)
        return x.to(torch.float32) @ weights.t()

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for CUDA execution")
    output = torch.empty((batch, n), device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(batch, block_m), triton.cdiv(n, block_n))
    _hybrid_ternary_value_kernel[grid](
        x,
        packed_w,
        output,
        batch,
        n,
        k,
        x.stride(0),
        x.stride(1),
        packed_w.stride(0),
        output.stride(0),
        output.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return output


def hybrid_ste_backward(
    grad_y: torch.Tensor,
    x: torch.Tensor,
    latent_w: torch.Tensor,
    *,
    block_n: int = 16,
    block_k: int = 32,
) -> torch.Tensor:
    """Launch the masked FP gradient kernel, with a portable CPU reference."""

    batch, n = grad_y.shape
    if x.shape != (batch, latent_w.shape[1]) or latent_w.shape[0] != n:
        raise ValueError("grad_y, x, and latent_w shapes are inconsistent")
    if not x.is_cuda:
        gradient = torch.einsum("bn,bk->nk", grad_y, x)
        return (gradient * (latent_w.abs() <= 1.0)).to(torch.float32)
    output = torch.empty_like(latent_w, dtype=torch.float32)
    grid = (triton.cdiv(n, block_n), triton.cdiv(latent_w.shape[1], block_k))
    _hybrid_ste_backward_kernel[grid](
        grad_y,
        x,
        latent_w,
        output,
        batch,
        n,
        latent_w.shape[1],
        grad_y.stride(0),
        grad_y.stride(1),
        x.stride(0),
        x.stride(1),
        latent_w.stride(0),
        latent_w.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return output


def pack_ternary_torch(weights: torch.Tensor) -> torch.Tensor:
    """Pack a ``[n,k]`` tensor containing -1/0/+1 into int32 registers."""

    if weights.ndim != 2 or not torch.all(torch.isin(weights, torch.tensor([-1, 0, 1], device=weights.device))):
        raise ValueError("weights must be a rank-2 ternary tensor")
    n, k = weights.shape
    padded = torch.nn.functional.pad(weights.to(torch.int32), (0, (-k) % 16))
    padded = padded.reshape(n, -1, 16)
    codes = torch.where(padded == 0, 0, torch.where(padded == 1, 1, 3))
    shifts = (2 * torch.arange(16, device=weights.device, dtype=torch.int32)).view(1, 1, 16)
    return torch.sum(codes << shifts, dim=2, dtype=torch.int32)


__all__ = [
    "TRITON_AVAILABLE",
    "hybrid_bitwise_matmul",
    "hybrid_ternary_matmul",
    "hybrid_ste_backward",
    "pack_ternary_torch",
]