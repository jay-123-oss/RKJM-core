"""
Verification script for 1.58-bit CSATransformerBlock.

Validates:
1. Forward pass shape on 3D input tensor: [Batch=2, Sequence=64, Dim=512].
2. Backward autograd gradient flow through all Q/K/V/O and SwiGLU MLP layers.
3. Gradient check on input activations x.grad and RMSNorm scaling weights.
4. Parameter update step with AdamW optimizer.
5. Inference mode with packed ternary weights.
"""

import time
import torch
import torch.nn as nn
from watch_nn import CSATransformerBlock, CSALinear


def run_transformer_block_verification():
    print("=" * 75)
    print("VERIFYING 1.58-BIT LLAMA-STYLE CSATRANSFORMERBLOCK (CPU OPENMP)")
    print("=" * 75)

    torch.manual_seed(42)

    # 1. Instantiate the CSATransformerBlock
    batch_size = 2
    seq_len = 64
    dim = 512
    num_heads = 8
    intermediate_dim = 1344  # LLaMA-style 8/3 * dim padded

    print(f"Architecture Configuration:")
    print(f"  - Model Dimension (d_model):     {dim}")
    print(f"  - Attention Heads:               {num_heads} (Head Dim: {dim // num_heads})")
    print(f"  - SwiGLU Intermediate Dim:       {intermediate_dim}")
    print(f"  - Pre-Norm:                      RMSNorm")
    print(f"  - Linear Engine:                 1.58-bit CSA Popcount + C++ Autograd STE")

    block = CSATransformerBlock(
        dim=dim,
        num_heads=num_heads,
        intermediate_dim=intermediate_dim,
        bias=False,
    )

    # Count total parameters
    total_params = sum(p.numel() for p in block.parameters())
    print(f"  - Total Parameters:              {total_params:,}")

    # Warmup pass to avoid one-time torch autograd CUDA discovery delay
    warmup_x = torch.randn(1, 8, dim, dtype=torch.float32, requires_grad=True)
    warmup_out = block(warmup_x)
    warmup_out.sum().backward()
    block.zero_grad()

    # 2. Generate Random 3D Input Tensor
    x = torch.randn(batch_size, seq_len, dim, dtype=torch.float32, requires_grad=True)
    print(f"\n[Test 1] Forward Pass on 3D Tensor:")
    print(f"  - Input shape:                   {tuple(x.shape)}")

    t0 = time.perf_counter()
    out = block(x, causal_mask=True)
    fwd_time_ms = (time.perf_counter() - t0) * 1000.0

    print(f"  - Output shape:                  {tuple(out.shape)}")
    print(f"  - Forward execution time:        {fwd_time_ms:.2f} ms")

    assert out.shape == (batch_size, seq_len, dim), (
        f"Shape mismatch! Expected {(batch_size, seq_len, dim)}, got {out.shape}"
    )
    assert not torch.isnan(out).any(), "Output contains NaN values!"
    assert not torch.isinf(out).any(), "Output contains Inf values!"
    print("  ✓ Forward shape and numerical integrity validated successfully.")

    # 3. Backward Pass & Gradient Flow Verification
    print(f"\n[Test 2] Backward Autograd Gradient Flow:")
    loss = out.sum()

    t0 = time.perf_counter()
    loss.backward()
    bwd_time_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  - Backward execution time:       {bwd_time_ms:.2f} ms")

    # Verify input gradients
    assert x.grad is not None, "Input x.grad is None!"
    assert x.grad.shape == x.shape, f"x.grad shape mismatch: {x.grad.shape}"
    print(f"  ✓ Input x.grad computed:         shape={tuple(x.grad.shape)}, norm={x.grad.norm():.4f}")

    # Verify RMSNorm gradients
    assert block.input_layernorm.weight.grad is not None
    assert block.post_attention_layernorm.weight.grad is not None
    print(f"  ✓ RMSNorm weights gradients:     input_norm={block.input_layernorm.weight.grad.norm():.4f}, "
          f"post_attn_norm={block.post_attention_layernorm.weight.grad.norm():.4f}")

    # Verify Self-Attention CSA Projections
    attn_projs = {
        "q_proj": block.self_attn.q_proj,
        "k_proj": block.self_attn.k_proj,
        "v_proj": block.self_attn.v_proj,
        "o_proj": block.self_attn.o_proj,
    }
    for name, proj in attn_projs.items():
        assert proj.latent_weight.grad is not None, f"Attention {name} latent_weight.grad is None!"
        assert proj.alpha.grad is not None, f"Attention {name} alpha.grad is None!"
        print(f"  ✓ Attention {name:<6}:         weight_grad_norm={proj.latent_weight.grad.norm():.4f}, "
              f"alpha_grad_norm={proj.alpha.grad.norm():.4f}")

    # Verify SwiGLU MLP CSA Projections
    mlp_projs = {
        "gate_proj": block.mlp.gate_proj,
        "up_proj":   block.mlp.up_proj,
        "down_proj": block.mlp.down_proj,
    }
    for name, proj in mlp_projs.items():
        assert proj.latent_weight.grad is not None, f"MLP {name} latent_weight.grad is None!"
        assert proj.alpha.grad is not None, f"MLP {name} alpha.grad is None!"
        print(f"  ✓ MLP {name:<12}:         weight_grad_norm={proj.latent_weight.grad.norm():.4f}, "
              f"alpha_grad_norm={proj.alpha.grad.norm():.4f}")

    print("  ✓ Full gradient graph traversed through all 7 CSA linear projections!")

    # 4. Optimizer Step Verification
    print(f"\n[Test 3] Optimizer Convergence Test (AdamW):")
    optimizer = torch.optim.AdamW(block.parameters(), lr=1e-3)
    target = torch.randn_like(out)
    criterion = nn.MSELoss()

    initial_loss = criterion(block(x), target).item()
    for step in range(5):
        optimizer.zero_grad()
        pred = block(x)
        l = criterion(pred, target)
        l.backward()
        optimizer.step()
    final_loss = criterion(block(x), target).item()

    print(f"  - Initial Loss:                  {initial_loss:.4f}")
    print(f"  - Final Loss after 5 steps:      {final_loss:.4f}")
    assert final_loss < initial_loss, "Loss failed to decrease during optimizer steps!"
    print("  ✓ Optimizer successfully updated latent weights and reduced loss.")

    # 5. Inference Packing Mode
    print(f"\n[Test 4] Inference Packing & Evaluation Mode:")
    block.eval()
    for m in block.modules():
        if isinstance(m, CSALinear):
            m.pack_weights_for_inference()

    with torch.no_grad():
        infer_out = block(x)
    assert infer_out.shape == (batch_size, seq_len, dim)
    print("  ✓ Packed 2-bit ternary inference path executed smoothly.")

    print("\n" + "=" * 75)
    print("ALL CSATRANSFORMERBLOCK TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 75)


if __name__ == "__main__":
    run_transformer_block_verification()
