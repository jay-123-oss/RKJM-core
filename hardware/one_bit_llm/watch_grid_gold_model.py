"""Reference model for a 2D Watch Transistor Grid hybrid PE array.

The model is intentionally written with scalar Python control flow around
NumPy and PyTorch tensors.  This makes the bit-level behavior inspectable and
keeps the file independent of custom CUDA or C extensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch


def _check_ternary(value: int) -> int:
    if value not in (-1, 0, 1):
        raise ValueError(f"expected a ternary value, got {value}")
    return value


def _round_clip_ternary(value: float) -> int:
    return int(np.clip(np.rint(value), -1, 1))


def _to_twos_complement(value: int, width: int) -> int:
    return value & ((1 << width) - 1)


def _from_twos_complement(value: int, width: int) -> int:
    value &= (1 << width) - 1
    sign_bit = 1 << (width - 1)
    return value - (1 << width) if value & sign_bit else value


def csa3(product: int, sum_in: int, carry_in: int, width: int) -> Tuple[int, int]:
    """Apply a width-bit, 3-input carry-save adder.

    The returned carry vector is unshifted.  A numerical value is recovered
    with ``sum_out + (carry_out << 1)`` followed by two's-complement decode.
    """

    mask = (1 << width) - 1
    product_bits = _to_twos_complement(product, width)
    sum_bits = sum_in & mask
    carry_bits = carry_in & mask
    sum_out = product_bits ^ sum_bits ^ carry_bits
    carry_out = (
        (product_bits & sum_bits)
        | (sum_bits & carry_bits)
        | (carry_bits & product_bits)
    )
    return sum_out & mask, carry_out & mask


def csa_value(sum_bits: int, carry_bits: int, width: int) -> int:
    """Propagate the carry vector once and decode the signed result."""

    return _from_twos_complement((sum_bits + (carry_bits << 1)) & ((1 << width) - 1), width)


def pack_ternary(values: Sequence[int], bits_per_value: int = 2) -> int:
    """Pack ternary values as sign-magnitude fields into one integer.

    Encoding is ``00 = 0``, ``01 = +1``, and ``11 = -1``.  Eight or sixteen
    values therefore occupy 16 or 32 bits respectively.
    """

    if bits_per_value != 2 or len(values) not in (8, 16):
        raise ValueError("pack_ternary supports exactly 8 or 16 two-bit values")
    packed = 0
    for index, raw_value in enumerate(values):
        value = _check_ternary(int(raw_value))
        code = 0 if value == 0 else (1 if value == 1 else 3)
        packed |= code << (bits_per_value * index)
    return packed


def unpack_ternary(packed: int, count: int, bits_per_value: int = 2) -> np.ndarray:
    """Unpack the sign-magnitude ternary integer format."""

    if bits_per_value != 2 or count not in (8, 16):
        raise ValueError("unpack_ternary supports exactly 8 or 16 two-bit values")
    result = []
    for index in range(count):
        code = (int(packed) >> (bits_per_value * index)) & 0b11
        if code == 0:
            result.append(0)
        elif code == 1:
            result.append(1)
        elif code == 3:
            result.append(-1)
        else:
            raise ValueError(f"reserved ternary code 0b{code:02b}")
    return np.asarray(result, dtype=np.int8)


@dataclass
class WatchPE:
    """One hybrid PE with a latent floating-point weight and CSA state."""

    w_fp: float = 0.0
    accumulator_width: int = 16

    def __post_init__(self) -> None:
        if self.accumulator_width < 4:
            raise ValueError("accumulator_width must be at least four bits")
        self.w_q = _round_clip_ternary(self.w_fp)
        self.sum_bits = 0
        self.carry_bits = 0
        self.grad_w_fp = 0.0

    def reset_forward_state(self) -> None:
        self.sum_bits = 0
        self.carry_bits = 0

    def forward_cycle(self, x_in: int, sum_in: int, carry_in: int) -> Tuple[int, int]:
        """Consume one activation and return the CSA sum/carry vectors."""

        if int(x_in) not in (-1, 1):
            raise ValueError("the bitwise forward path expects activations in {-1, +1}")
        product = int(x_in) * self.w_q
        self.sum_bits, self.carry_bits = csa3(
            product, int(sum_in), int(carry_in), self.accumulator_width
        )
        return self.sum_bits, self.carry_bits

    def forward_value(self) -> int:
        """Return the decoded accumulated dot-product contribution."""

        return csa_value(self.sum_bits, self.carry_bits, self.accumulator_width)

    def backward_cycle(self, grad_y: float, x_in: int) -> float:
        """Accumulate the masked STE gradient for this PE's weight."""

        if int(x_in) not in (-1, 1):
            raise ValueError("the reference backward path expects activations in {-1, +1}")
        ste_mask = 1.0 if abs(float(self.w_fp)) <= 1.0 else 0.0
        grad_w = float(grad_y) * int(x_in) * ste_mask
        self.grad_w_fp += grad_w
        return grad_w

    def update_latent_weight(self, lr: float) -> None:
        """Apply the accumulated gradient and refresh the ternary weight."""

        self.w_fp -= float(lr) * self.grad_w_fp
        self.w_q = _round_clip_ternary(self.w_fp)
        self.grad_w_fp = 0.0


class WatchGrid2DSimulator:
    """Cycle-level reference simulator for an ``rows x columns`` PE grid."""

    def __init__(self, weights_fp: np.ndarray | None = None, accumulator_width: int = 16) -> None:
        weights = (
            np.zeros((4, 4), dtype=np.float32)
            if weights_fp is None
            else np.asarray(weights_fp, dtype=np.float32)
        )
        if weights.ndim != 2:
            raise ValueError("weights_fp must be a two-dimensional array")
        self.rows, self.columns = weights.shape
        self.accumulator_width = accumulator_width
        self.pes = [
            [WatchPE(float(weights[row, column]), accumulator_width) for column in range(self.columns)]
            for row in range(self.rows)
        ]
        self.cycle_log: List[dict] = []

    @property
    def w_fp(self) -> np.ndarray:
        return np.asarray([[pe.w_fp for pe in row] for row in self.pes], dtype=np.float32)

    @property
    def w_q(self) -> np.ndarray:
        return np.asarray([[pe.w_q for pe in row] for row in self.pes], dtype=np.int8)

    def reset_forward_state(self) -> None:
        for row in self.pes:
            for pe in row:
                pe.reset_forward_state()
        self.cycle_log.clear()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run one or more activation vectors through the horizontal grid."""

        activations = np.asarray(x, dtype=np.int8)
        if activations.ndim == 1:
            activations = activations[None, :]
            squeeze = True
        else:
            squeeze = False
        if activations.shape[1] != self.columns:
            raise ValueError("activation width must equal the grid column count")

        outputs = []
        for sample in activations:
            if not np.all(np.isin(sample, (-1, 1))):
                raise ValueError("forward activations must contain only -1 or +1")
            self.reset_forward_state()
            for column, activation in enumerate(sample):
                for row in range(self.rows):
                    pe = self.pes[row][column]
                    pe.forward_cycle(int(activation), pe.sum_bits, pe.carry_bits)
                self.cycle_log.append({"cycle": column, "activation": int(activation)})
            outputs.append([sum(pe.forward_value() for pe in row) for row in self.pes])
        result = np.asarray(outputs, dtype=np.int32)
        return result[0] if squeeze else result

    def backward(self, grad_y: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Run the vertical gradient flow and return ``dL/dW_fp``."""

        gradients = np.asarray(grad_y, dtype=np.float32)
        activations = np.asarray(x, dtype=np.int8)
        if gradients.ndim != 1 or gradients.shape[0] != self.rows:
            raise ValueError("grad_y must have one value per grid row")
        if activations.ndim != 1 or activations.shape[0] != self.columns:
            raise ValueError("x must have one value per grid column")
        result = np.zeros((self.rows, self.columns), dtype=np.float32)
        for row in range(self.rows):
            for column in range(self.columns):
                result[row, column] = self.pes[row][column].backward_cycle(
                    float(gradients[row]), int(activations[column])
                )
        return result

    def update_latent_weight(self, lr: float) -> None:
        for row in self.pes:
            for pe in row:
                pe.update_latent_weight(lr)

    def packed_weight_registers(self) -> List[int]:
        """Pack each row; this exercises the 8/16-value register format."""

        if self.columns not in (8, 16):
            raise ValueError("packed registers require a grid with 8 or 16 columns")
        return [pack_ternary(row, 2) for row in self.w_q]


class _MaskedRoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, value: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(value)
        return torch.clamp(torch.round(value), -1.0, 1.0)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor) -> Tuple[torch.Tensor]:
        (value,) = ctx.saved_tensors
        return (grad_output * (value.abs() <= 1.0).to(grad_output.dtype),)


def masked_round_ste(value: torch.Tensor) -> torch.Tensor:
    """Quantize in the forward pass and apply the requested masked STE."""

    return _MaskedRoundSTE.apply(value)


def test_correctness(seed: int = 7) -> None:
    """Verify forward, backward, packing, and latent-weight update behavior."""

    rng = np.random.default_rng(seed)
    rows, columns = 4, 8
    weights = rng.uniform(-1.75, 1.75, size=(rows, columns)).astype(np.float32)
    activations = rng.choice(np.asarray([-1, 1], dtype=np.int8), size=columns)
    grad_y = rng.normal(size=rows).astype(np.float32)
    grid = WatchGrid2DSimulator(weights)

    x_t = torch.tensor(activations, dtype=torch.float32)
    w_t = torch.tensor(weights, dtype=torch.float32, requires_grad=True)
    q_t = masked_round_ste(w_t)
    y_reference = q_t @ x_t
    loss = (y_reference * torch.tensor(grad_y)).sum()
    loss.backward()

    y_grid = grid.forward(activations)
    y_expected = torch.matmul(torch.tensor(weights).round().clamp(-1, 1), x_t).numpy()
    forward_mae = float(np.mean(np.abs(y_grid.astype(np.float32) - y_expected)))
    assert np.array_equal(y_grid, y_expected), (y_grid, y_expected)

    grad_grid = grid.backward(grad_y, activations)
    backward_mae = float(np.mean(np.abs(grad_grid - w_t.grad.numpy())))
    assert np.allclose(grad_grid, w_t.grad.numpy(), atol=0.0, rtol=0.0)

    packed = grid.packed_weight_registers()
    unpacked = np.stack([unpack_ternary(register, columns) for register in packed])
    assert np.array_equal(unpacked, grid.w_q)

    old_latent = grid.w_fp.copy()
    learning_rate = 0.01
    grid.update_latent_weight(learning_rate)
    expected_latent = old_latent - learning_rate * grad_grid
    assert np.allclose(grid.w_fp, expected_latent, atol=1e-6)

    bit_matches = int(np.count_nonzero(y_grid == y_expected))
    print("Watch Grid Gold Model verification")
    print(f"  forward MAE:  {forward_mae:.6f}")
    print(f"  backward MAE: {backward_mae:.6f}")
    print(f"  output bit-level matches: {bit_matches}/{rows}")
    print(f"  packed registers verified: {len(packed)} x {columns * 2}-bit")
    print("  status: PASS")


if __name__ == "__main__":
    test_correctness()