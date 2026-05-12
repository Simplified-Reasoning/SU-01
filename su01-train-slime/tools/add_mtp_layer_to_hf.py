"""
Script to add randomly initialized MTP (Multi-Token Prediction) layers to an existing HuggingFace model.

This script:
1. Loads an existing HuggingFace model checkpoint
2. Adds randomly initialized MTP layers
3. Updates the config with MTP layer information
4. Saves the modified model to a new location

MTP layers follow the structure used in MiMo models and consist of:
- token_layernorm: Layer normalization for token embeddings
- hidden_layernorm: Layer normalization for hidden states
- input_proj: Input projection layer
- Transformer layers (self_attn + mlp + layer norms)
- final_layernorm: Final layer normalization

Usage:
    python tools/add_mtp_layer_to_hf.py \
        --input-model-path /path/to/qwen3-30B-A3B \
        --output-model-path /path/to/qwen3-30B-A3B-with-mtp \
        --num-mtp-layers 1 \
        --seed 42
"""

import argparse
import json
import os
import shutil
from typing import Dict

import torch
import safetensors.torch
from transformers import AutoConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Add MTP layers to HuggingFace model")
    parser.add_argument(
        "--input-model-path",
        type=str,
        required=True,
        help="Path to the input HuggingFace model directory"
    )
    parser.add_argument(
        "--output-model-path",
        type=str,
        required=True,
        help="Path to save the modified model with MTP layers"
    )
    parser.add_argument(
        "--num-mtp-layers",
        type=int,
        default=1,
        help="Number of MTP layers to add (default: 1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for weight initialization (default: 42)"
    )
    parser.add_argument(
        "--std",
        type=float,
        default=0.02,
        help="Standard deviation for weight initialization (default: 0.02)"
    )
    return parser.parse_args()


def initialize_mtp_weights(
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    num_mtp_layers: int,
    std: float = 0.02,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42
) -> Dict[str, torch.Tensor]:
    """
    Initialize random weights for MTP layers.
    
    Args:
        hidden_size: Hidden dimension size
        intermediate_size: FFN intermediate dimension size
        num_attention_heads: Number of attention heads
        num_key_value_heads: Number of key-value heads
        num_mtp_layers: Number of MTP layers to create
        std: Standard deviation for normal initialization
        dtype: Data type for weights
        seed: Random seed
        
    Returns:
        Dictionary mapping parameter names to tensors
    """
    torch.manual_seed(seed)
    mtp_weights = {}
    
    head_dim = hidden_size // num_attention_heads
    
    for layer_idx in range(num_mtp_layers):
        prefix = f"model.mtp_layers.{layer_idx}"
        
        # Layer normalizations (initialized to 1.0)
        mtp_weights[f"{prefix}.token_layernorm.weight"] = torch.ones(hidden_size, dtype=dtype)
        mtp_weights[f"{prefix}.hidden_layernorm.weight"] = torch.ones(hidden_size, dtype=dtype)
        mtp_weights[f"{prefix}.final_layernorm.weight"] = torch.ones(hidden_size, dtype=dtype)
        mtp_weights[f"{prefix}.input_layernorm.weight"] = torch.ones(hidden_size, dtype=dtype)
        mtp_weights[f"{prefix}.post_attention_layernorm.weight"] = torch.ones(hidden_size, dtype=dtype)
        
        # Input projection: projects concatenated [token_embed, hidden_state] to hidden_size
        # Input is [hidden_size + hidden_size] = 2 * hidden_size
        mtp_weights[f"{prefix}.input_proj.weight"] = torch.randn(
            hidden_size, 2 * hidden_size, dtype=dtype
        ) * std
        
        # Self-attention weights
        # Q projection
        mtp_weights[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(
            num_attention_heads * head_dim, hidden_size, dtype=dtype
        ) * std
        mtp_weights[f"{prefix}.self_attn.q_proj.bias"] = torch.zeros(
            num_attention_heads * head_dim, dtype=dtype
        )
        
        # K projection
        mtp_weights[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(
            num_key_value_heads * head_dim, hidden_size, dtype=dtype
        ) * std
        mtp_weights[f"{prefix}.self_attn.k_proj.bias"] = torch.zeros(
            num_key_value_heads * head_dim, dtype=dtype
        )
        
        # V projection
        mtp_weights[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(
            num_key_value_heads * head_dim, hidden_size, dtype=dtype
        ) * std
        mtp_weights[f"{prefix}.self_attn.v_proj.bias"] = torch.zeros(
            num_key_value_heads * head_dim, dtype=dtype
        )
        
        # O projection
        mtp_weights[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(
            hidden_size, num_attention_heads * head_dim, dtype=dtype
        ) * std
        
        # MLP weights (SwiGLU: gate, up, down projections)
        mtp_weights[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(
            intermediate_size, hidden_size, dtype=dtype
        ) * std
        
        mtp_weights[f"{prefix}.mlp.up_proj.weight"] = torch.randn(
            intermediate_size, hidden_size, dtype=dtype
        ) * std
        
        mtp_weights[f"{prefix}.mlp.down_proj.weight"] = torch.randn(
            hidden_size, intermediate_size, dtype=dtype
        ) * std
    
    total_params = sum(p.numel() for p in mtp_weights.values())
    print(f"Initialized {len(mtp_weights)} MTP parameters ({total_params / 1e6:.0f}M params)")
    return mtp_weights


def load_existing_weights(model_path: str) -> tuple[Dict[str, torch.Tensor], dict]:
    """Load existing model weights from safetensors files."""
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Could not find model.safetensors.index.json in {model_path}")
    
    with open(index_path, 'r') as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    metadata = index.get("metadata", {})
    
    # Load all weight files
    state_dict = {}
    shard_files = set(weight_map.values())
    
    print(f"Loading {len(shard_files)} weight files...", end=" ")
    for shard_file in sorted(shard_files):
        shard_path = os.path.join(model_path, shard_file)
        shard_weights = safetensors.torch.load_file(shard_path)
        state_dict.update(shard_weights)
    
    print(f"✓ Loaded {len(state_dict)} parameters")
    return state_dict, metadata


def save_model_with_mtp(
    output_path: str,
    state_dict: Dict[str, torch.Tensor],
    chunk_size: int = 5 * 1024**3  # 5GB per file
):
    """Save the combined state dict with MTP layers to safetensors format."""
    os.makedirs(output_path, exist_ok=True)
    
    # Split into shards based on size
    current_shard = {}
    current_size = 0
    shards = []
    weight_map = {}
    
    for name, param in sorted(state_dict.items()):
        param_size = param.numel() * param.element_size()
        
        if current_size + param_size > chunk_size and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        
        current_shard[name] = param
        current_size += param_size
        weight_map[name] = f"model-{len(shards)+1:05d}-of-{{total:05d}}.safetensors"
    
    if current_shard:
        shards.append(current_shard)
    
    total_shards = len(shards)
    weight_map = {k: v.format(total=total_shards) for k, v in weight_map.items()}
    
    # Save each shard
    print(f"Saving {total_shards} weight files...", end=" ")
    for idx, shard in enumerate(shards):
        shard_name = f"model-{idx+1:05d}-of-{total_shards:05d}.safetensors"
        shard_path = os.path.join(output_path, shard_name)
        safetensors.torch.save_file(shard, shard_path)
    
    # Save index
    total_size = sum(p.numel() * p.element_size() for p in state_dict.values())
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map
    }
    
    index_path = os.path.join(output_path, "model.safetensors.index.json")
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"✓ Saved {total_size / 1024**3:.1f} GB")


def update_config(input_model_path: str, output_model_path: str, num_mtp_layers: int):
    """Copy and update the config file with MTP layer information."""
    config = AutoConfig.from_pretrained(input_model_path, trust_remote_code=True)
    config.num_nextn_predict_layers = num_mtp_layers
    config.save_pretrained(output_model_path)
    print(f"Updated config with num_nextn_predict_layers={num_mtp_layers}")


def copy_other_files(input_model_path: str, output_model_path: str):
    """Copy tokenizer and other necessary files."""
    files_to_copy = [
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "merges.txt", "vocab.json", "generation_config.json", "README.md",
    ]
    
    copied = 0
    for filename in files_to_copy:
        src = os.path.join(input_model_path, filename)
        if os.path.exists(src):
            dst = os.path.join(output_model_path, filename)
            shutil.copy2(src, dst)
            copied += 1
    
    # Copy modeling files
    for filename in os.listdir(input_model_path):
        if filename.startswith("modeling_") or filename.startswith("configuration_"):
            src = os.path.join(input_model_path, filename)
            if os.path.isfile(src):
                dst = os.path.join(output_model_path, filename)
                shutil.copy2(src, dst)
                copied += 1
    
    print(f"Copied {copied} additional files (tokenizer, config, etc.)")


def main():
    args = parse_args()
    
    print(f"Adding {args.num_mtp_layers} MTP layer(s) to {args.input_model_path}")
    
    # Load config
    config = AutoConfig.from_pretrained(args.input_model_path, trust_remote_code=True)
    print(f"Model: {config.num_hidden_layers} layers, {config.hidden_size} hidden size")
    
    # Load existing weights
    state_dict, metadata = load_existing_weights(args.input_model_path)
    
    # Initialize MTP weights
    mtp_weights = initialize_mtp_weights(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        num_mtp_layers=args.num_mtp_layers,
        std=args.std,
        dtype=torch.bfloat16,
        seed=args.seed
    )
    
    # Combine and save
    combined_state_dict = {**state_dict, **mtp_weights}
    save_model_with_mtp(args.output_model_path, combined_state_dict)
    update_config(args.input_model_path, args.output_model_path, args.num_mtp_layers)
    copy_other_files(args.input_model_path, args.output_model_path)
    
    print(f"✓ Model with MTP layers saved to {args.output_model_path}")
    print(f"✓ Next: Convert to Megatron format (see README_MTP.md)")


if __name__ == "__main__":
    main()

