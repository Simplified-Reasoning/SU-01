"""
Visualization script to demonstrate MTP weight transformations between HuggingFace and Megatron formats.

This script shows:
1. How QKV weights are fused/unfused
2. How input projection weights are swapped
3. How MLP weights are combined
4. How tensor parallelism shards the weights

Usage:
    python tools/visualize_mtp_transformation.py
"""

import torch
import numpy as np


def visualize_qkv_fusion():
    """Demonstrate QKV weight fusion for Megatron format."""
    print("="*80)
    print("QKV Weight Fusion (HuggingFace → Megatron)")
    print("="*80)
    
    # Example dimensions for Qwen3-30B-A3B
    hidden_size = 6144
    num_attention_heads = 48
    num_key_value_heads = 4  # GQA
    head_dim = hidden_size // num_attention_heads  # 128
    
    print(f"\nModel config:")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Num attention heads: {num_attention_heads}")
    print(f"  Num KV heads: {num_key_value_heads}")
    print(f"  Head dimension: {head_dim}")
    
    # HuggingFace format: separate Q, K, V projections
    q_weight = torch.randn(num_attention_heads * head_dim, hidden_size)  # [6144, 6144]
    k_weight = torch.randn(num_key_value_heads * head_dim, hidden_size)  # [512, 6144]
    v_weight = torch.randn(num_key_value_heads * head_dim, hidden_size)  # [512, 6144]
    
    q_bias = torch.randn(num_attention_heads * head_dim)  # [6144]
    k_bias = torch.randn(num_key_value_heads * head_dim)  # [512]
    v_bias = torch.randn(num_key_value_heads * head_dim)  # [512]
    
    print(f"\n📦 HuggingFace format (separate projections):")
    print(f"  Q weight shape: {list(q_weight.shape)}")
    print(f"  K weight shape: {list(k_weight.shape)}")
    print(f"  V weight shape: {list(v_weight.shape)}")
    print(f"  Q bias shape:   {list(q_bias.shape)}")
    print(f"  K bias shape:   {list(k_bias.shape)}")
    print(f"  V bias shape:   {list(v_bias.shape)}")
    
    # Megatron format: fused QKV
    qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
    qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
    
    print(f"\n🔗 Megatron format (fused QKV):")
    print(f"  QKV weight shape: {list(qkv_weight.shape)}")
    print(f"  QKV bias shape:   {list(qkv_bias.shape)}")
    print(f"  Total parameters: {qkv_weight.numel() + qkv_bias.numel():,}")
    
    # Tensor parallelism (TP=4)
    tp_size = 4
    qkv_weight_sharded = qkv_weight.chunk(tp_size, dim=0)
    qkv_bias_sharded = qkv_bias.chunk(tp_size, dim=0)
    
    print(f"\n⚡ With Tensor Parallelism (TP={tp_size}):")
    for i, (w_shard, b_shard) in enumerate(zip(qkv_weight_sharded, qkv_bias_sharded)):
        print(f"  Rank {i}: QKV weight {list(w_shard.shape)}, bias {list(b_shard.shape)}")


def visualize_input_proj_swap():
    """Demonstrate input projection weight swapping."""
    print("\n" + "="*80)
    print("Input Projection Weight Swapping (eh_proj)")
    print("="*80)
    
    hidden_size = 6144
    
    # HuggingFace format
    # Projects [token_embed, hidden_state] (concatenated along dim 1)
    # So input has shape [batch, 2*hidden_size]
    hf_weight = torch.randn(hidden_size, 2 * hidden_size)
    
    print(f"\n📦 HuggingFace format:")
    print(f"  input_proj.weight shape: {list(hf_weight.shape)}")
    print(f"  Input: [token_embed | hidden_state] concatenated")
    
    # Visualize structure
    first_half = hf_weight[:, :hidden_size]
    second_half = hf_weight[:, hidden_size:]
    
    print(f"\n  First half (token processing):  {list(first_half.shape)}")
    print(f"  Second half (hidden processing): {list(second_half.shape)}")
    
    # Megatron format: swapped halves
    megatron_weight = torch.cat([second_half, first_half], dim=1)
    
    print(f"\n🔗 Megatron format (eh_proj):")
    print(f"  eh_proj.weight shape: {list(megatron_weight.shape)}")
    print(f"  Input: [hidden_state | token_embed] (SWAPPED!)")
    
    print(f"\n⚠️  Important: The halves are swapped during conversion!")
    print(f"  This is due to different ordering conventions between frameworks.")
    
    # Verify swap is reversible
    reconstructed = torch.cat([megatron_weight[:, hidden_size:], megatron_weight[:, :hidden_size]], dim=1)
    assert torch.allclose(hf_weight, reconstructed)
    print(f"  ✓ Transformation is reversible")


def visualize_mlp_fusion():
    """Demonstrate MLP weight fusion."""
    print("\n" + "="*80)
    print("MLP Weight Fusion (SwiGLU)")
    print("="*80)
    
    hidden_size = 6144
    intermediate_size = 16384
    
    # HuggingFace format: separate gate and up projections
    gate_weight = torch.randn(intermediate_size, hidden_size)
    up_weight = torch.randn(intermediate_size, hidden_size)
    down_weight = torch.randn(hidden_size, intermediate_size)
    
    print(f"\n📦 HuggingFace format (SwiGLU):")
    print(f"  gate_proj.weight: {list(gate_weight.shape)}")
    print(f"  up_proj.weight:   {list(up_weight.shape)}")
    print(f"  down_proj.weight: {list(down_weight.shape)}")
    print(f"\n  Computation: down(silu(gate) * up)")
    
    # Megatron format: fused gate + up
    fc1_weight = torch.cat([gate_weight, up_weight], dim=0)
    fc2_weight = down_weight
    
    print(f"\n🔗 Megatron format:")
    print(f"  linear_fc1.weight (gate+up fused): {list(fc1_weight.shape)}")
    print(f"  linear_fc2.weight (down):          {list(fc2_weight.shape)}")
    
    # Tensor parallelism (TP=4)
    tp_size = 4
    fc1_sharded = fc1_weight.chunk(tp_size, dim=0)
    fc2_sharded = fc2_weight.chunk(tp_size, dim=1)  # Column-parallel for output
    
    print(f"\n⚡ With Tensor Parallelism (TP={tp_size}):")
    for i in range(tp_size):
        print(f"  Rank {i}:")
        print(f"    FC1 (column-parallel): {list(fc1_sharded[i].shape)}")
        print(f"    FC2 (row-parallel):    {list(fc2_sharded[i].shape)}")


def visualize_complete_mtp_layer():
    """Show complete MTP layer parameter count."""
    print("\n" + "="*80)
    print("Complete MTP Layer Parameter Count")
    print("="*80)
    
    # Qwen3-30B-A3B dimensions
    hidden_size = 6144
    intermediate_size = 16384
    num_attention_heads = 48
    num_key_value_heads = 4
    head_dim = hidden_size // num_attention_heads
    
    params = {
        "token_layernorm": hidden_size,
        "hidden_layernorm": hidden_size,
        "input_proj": hidden_size * 2 * hidden_size,
        "final_layernorm": hidden_size,
        "input_layernorm": hidden_size,
        "post_attention_layernorm": hidden_size,
        "q_proj_weight": num_attention_heads * head_dim * hidden_size,
        "q_proj_bias": num_attention_heads * head_dim,
        "k_proj_weight": num_key_value_heads * head_dim * hidden_size,
        "k_proj_bias": num_key_value_heads * head_dim,
        "v_proj_weight": num_key_value_heads * head_dim * hidden_size,
        "v_proj_bias": num_key_value_heads * head_dim,
        "o_proj": num_attention_heads * head_dim * hidden_size,
        "gate_proj": intermediate_size * hidden_size,
        "up_proj": intermediate_size * hidden_size,
        "down_proj": hidden_size * intermediate_size,
    }
    
    print("\n📊 Parameter breakdown:")
    total_params = 0
    for name, count in sorted(params.items()):
        print(f"  {name:30s}: {count:>15,} ({count/1e6:>8.2f}M)")
        total_params += count
    
    print(f"\n{'Total':30s}: {total_params:>15,} ({total_params/1e6:>8.2f}M)")
    print(f"\nFor Qwen3-30B-A3B with 1 MTP layer:")
    print(f"  Base model: ~30B parameters")
    print(f"  + MTP layer: ~{total_params/1e6:.0f}M parameters")
    print(f"  Total: ~{30 + total_params/1e9:.2f}B parameters")


def visualize_conversion_flow():
    """Show the complete conversion flow diagram."""
    print("\n" + "="*80)
    print("Weight Conversion Flow")
    print("="*80)
    
    print("""
┌─────────────────────────────────────────────────────────────────────────┐
│                        HuggingFace Format                               │
├─────────────────────────────────────────────────────────────────────────┤
│ model.mtp_layers.0.token_layernorm.weight          [6144]              │
│ model.mtp_layers.0.hidden_layernorm.weight         [6144]              │
│ model.mtp_layers.0.input_proj.weight               [6144, 12288]       │
│ model.mtp_layers.0.self_attn.q_proj.weight         [6144, 6144]        │
│ model.mtp_layers.0.self_attn.q_proj.bias           [6144]              │
│ model.mtp_layers.0.self_attn.k_proj.weight         [512, 6144]         │
│ model.mtp_layers.0.self_attn.k_proj.bias           [512]               │
│ model.mtp_layers.0.self_attn.v_proj.weight         [512, 6144]         │
│ model.mtp_layers.0.self_attn.v_proj.bias           [512]               │
│ model.mtp_layers.0.self_attn.o_proj.weight         [6144, 6144]        │
│ model.mtp_layers.0.mlp.gate_proj.weight            [16384, 6144]       │
│ model.mtp_layers.0.mlp.up_proj.weight              [16384, 6144]       │
│ model.mtp_layers.0.mlp.down_proj.weight            [6144, 16384]       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ convert_hf_to_torch_dist.py
                                    │ (via MimoBridge)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Megatron Format (TP=4)                               │
├─────────────────────────────────────────────────────────────────────────┤
│ module.module.mtp.layers.0.enorm.weight            [6144]              │
│ module.module.mtp.layers.0.hnorm.weight            [6144]              │
│ module.module.mtp.layers.0.eh_proj.weight          [6144, 12288]*      │
│ module.module.mtp.layers.0.transformer_layer...                        │
│   .self_attention.linear_qkv.weight                [2176, 6144]†       │
│   .self_attention.linear_qkv.bias                  [2176]†             │
│   .self_attention.linear_proj.weight               [6144, 1536]‡       │
│   .mlp.linear_fc1.weight                           [8192, 6144]†       │
│   .mlp.linear_fc2.weight                           [6144, 4096]‡       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ convert_torch_dist_to_hf.py
                                    │ (via convert_mimo_to_hf)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    HuggingFace Format (Restored)                        │
└─────────────────────────────────────────────────────────────────────────┘

Legend:
  * Halves are swapped: [hidden_state | token_embed]
  † Column-parallel: sharded along output dimension (dim 0)
  ‡ Row-parallel: sharded along input dimension (dim 1)

Key Transformations:
  1. QKV Fusion:  [Q, K, V] → [Q|K|V] concatenated
  2. eh_proj Swap: Swap first and second halves
  3. MLP Fusion:  [gate, up] → [gate|up] concatenated
  4. TP Sharding: Weights split across GPU ranks
""")


def main():
    print("\n" + "="*80)
    print(" MTP Layer Weight Transformation Visualization")
    print(" Qwen3-30B-A3B Example")
    print("="*80)
    
    visualize_qkv_fusion()
    visualize_input_proj_swap()
    visualize_mlp_fusion()
    visualize_complete_mtp_layer()
    visualize_conversion_flow()
    
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print("""
Key Points:
  1. QKV weights are fused in Megatron (Q|K|V concatenated)
  2. Input projection (eh_proj) halves are SWAPPED
  3. MLP gate and up projections are fused (gate|up concatenated)
  4. Tensor parallelism shards weights across GPUs
  5. All transformations are reversible

For more details, see:
  - tools/MTP_CONVERSION_GUIDE.md
  - slime_plugins/mbridge/mimo.py
  - slime/backends/megatron_utils/megatron_to_hf/mimo.py
""")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

