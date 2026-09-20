# 2D Watch Transistor Grid Hybrid PE

This document defines the mathematical contract implemented by
`watch_grid_gold_model.py`. The reference array has `N` rows and `M` columns.
Row `i` computes one output and column `j` owns weight `W[i,j]` and activation
`X[j]`.

## 1. Forward pass: ternary bitwise engine

### Domains and encodings

The latent weight is floating point, while the forward weight is ternary:

$$
W_{fp}[i,j] \in \mathbb{R},\qquad W_q[i,j] \in \{-1,0,+1\}.
$$

The reference uses two-bit sign-magnitude-like fields:

| Value | Field | Meaning |
|---:|:---:|:---|
| `0` | `00` | inactive / zero |
| `+1` | `01` | active, positive |
| `-1` | `11` | active, negative |

The reserved field `10` is invalid. Eight values occupy a 16-bit register and
sixteen values occupy a 32-bit register, with the first value in the least
significant field.

For a bipolar activation, define the positive-bit encoding

$$
b(x)=\begin{cases}1&x=+1\\0&x=-1\end{cases},
\qquad m(w)=\mathbf{1}_{w\ne0},
\qquad s(w)=\mathbf{1}_{w=+1}.
$$

The product is zero when the ternary magnitude is inactive. Otherwise the
product sign is the XNOR of activation and weight sign:

$$
m(p)=m(w)\land 1,
\qquad b(p)=m(w)\land\neg\bigl(b(x)\oplus s(w)\bigr).
$$

Equivalently, in signed arithmetic, $p=xw$ for $x\in\{-1,+1\}$. Negative
products are represented in the accumulator as fixed-width two's-complement
values. INT8 activations can use the same datapath by decomposing each signed
activation into bit-serial magnitude/sign contributions; the executable model
uses the bipolar mode directly.

### Three-input CSA

At each bit position and cycle, the product bit `P`, sum bit `S_in`, and carry
bit `C_in` are reduced without carry propagation:

$$
S_{out}=P\oplus S_{in}\oplus C_{in},
$$

$$
C_{out}=(P\land S_{in})\lor(S_{in}\land C_{in})\lor(C_{in}\land P).
$$

For a `B`-bit vector, these equations are applied independently to every bit
of the fixed-width two's-complement vectors. `C_out` is deliberately an
unshifted carry vector during CSA reduction. Its numerical contribution is

$$
V_{CSA}=V(S_{out})+2V(C_{out}).
$$

The carry-propagation rule is therefore:

1. Compute `S_out` and `C_out` with the three-input equations.
2. Left shift the carry vector exactly one bit: `C_shift = C_out << 1`.
3. Add `S_out + C_shift` with an ordinary ripple/look-ahead adder.
4. Truncate to `B` bits and decode as two's-complement.

For multiple columns, each cycle injects one product into the current pair:

$$
(S_{j+1},C_{j+1})=
CSA\left(P_j,S_j,C_j\right),
\qquad S_0=C_0=0.
$$

After the final column, the row output is

$$
Y_i=\operatorname{signed}_B\left(S_M+\left(C_M\ll1\right)\right)
=\sum_{j=0}^{M-1}X_jW_{q,i,j},
$$

provided `B` is wide enough to avoid overflow. A sufficient signed width for
the bipolar ternary dot product is

$$
B\ge \left\lceil\log_2(M+1)\right\rceil+1.
$$

The reference model uses `B=16` by default.

## 2. Backward pass: floating-point latent weights

The latent parameter is retained at FP32 precision in the reference model;
BF16 is supported conceptually by storing the same equations in BF16 hardware
or by explicitly rounding the latent state to BF16 at the chosen update
boundary.

For output gradient $G_Y=\partial L/\partial Y_i$, activation $X_j$, and the
straight-through mask

$$
\chi(W_{fp})=\mathbf{1}_{|W_{fp}|\le1},
$$

the per-weight gradient is

$$
\frac{\partial L}{\partial W_{fp,i,j}}
=G_{Y,i}\,X_j\,\chi(W_{fp,i,j}).
$$

If the upstream quantity is already defined as a weight-local gradient, the
`X_j` factor is omitted and the requested abbreviated form is

$$
\frac{\partial L}{\partial W_{fp}}=
\frac{\partial L}{\partial Y}\mathbf{1}_{|W_{fp}|\le1}.
$$

Gradients accumulate until an update boundary:

$$
W_{fp}(t+1)=W_{fp}(t)-\eta\frac{\partial L}{\partial W_{fp}}.
$$

The next forward quantization is

$$
W_q(t+1)=
\operatorname{Clip}\left(\operatorname{Round}(W_{fp}(t+1)),-1,+1\right).
$$

The reference uses nearest-integer rounding through `numpy.rint` and clears
the accumulated gradient after each latent update.

## 3. Cycle-by-cycle 2D movement

For a grid of `N` rows by `M` columns:

| Cycle | Horizontal path | Vertical/local operation |
|---:|:---|:---|
| `j` | Activation `X[j]` enters column `j` and is presented to that column's PEs. | Each PE forms `P[i,j]`, applies CSA to its local `(S,C)`, and retains the new sum/carry vectors. |
| `0..M-1` | The activation stream advances one column per cycle. | Weights remain local to their PE; the row's sum/carry state represents the partial dot product. |
| update phase | No activation movement is required. | `G_Y[i]` moves vertically through row `i`; each PE multiplies it by local `X[j]`, applies the STE mask, and accumulates `dL/dW_fp[i,j]`. |
| update boundary | Quantized registers are refreshed after the latent update. | `W_fp` is updated, then `W_q` is re-encoded into local ternary fields. |

The model records the horizontal activation cycle index in `cycle_log`. Its
`backward` method performs the corresponding row-wise vertical gradient sweep.
This is a functional timing model: physical wire delay, transistor sizing,
clock skew, and SRAM/register-file timing are outside the Step 1/2 contract.

## 4. Executable gold model

Run the verification from this directory with:

```bash
.venv/bin/python watch_grid_gold_model.py
```

The test checks exact quantized matmul equivalence, exact masked-STE gradient
equivalence, ternary register pack/unpack round trips, and one latent update.