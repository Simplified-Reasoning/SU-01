#!/usr/bin/env python3
"""
Merge multiple JSONL datasets into a single dataset with optional sampling weights.

Usage:
    # Equal sampling from all datasets
    python tools/merge_datasets.py \
        --input dataset1.jsonl dataset2.jsonl dataset3.jsonl \
        --output merged_dataset.jsonl
    
    # Weighted sampling (2:1:1 ratio)
    python tools/merge_datasets.py \
        --input dataset1.jsonl dataset2.jsonl dataset3.jsonl \
        --weights 2.0 1.0 1.0 \
        --output merged_dataset.jsonl
    
    # With dataset name injection
    python tools/merge_datasets.py \
        --input math.jsonl:math code.jsonl:code reasoning.jsonl:reasoning \
        --weights 2.0 1.0 1.5 \
        --output merged_dataset.jsonl \
        --add-dataset-name
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple


def load_jsonl(path: str) -> List[dict]:
    """Load JSONL file into a list of dictionaries."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[dict], path: str):
    """Save list of dictionaries to JSONL file."""
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def merge_datasets(
    input_files: List[str],
    output_file: str,
    weights: List[float] = None,
    names: List[str] = None,
    add_dataset_name: bool = False,
    shuffle: bool = True,
    seed: int = 42,
):
    """
    Merge multiple JSONL datasets into one with optional weighted sampling.
    
    Args:
        input_files: List of input JSONL file paths
        output_file: Output JSONL file path
        weights: Optional sampling weights for each dataset (default: equal weights)
        names: Optional dataset names for each file
        add_dataset_name: If True, inject dataset name into metadata
        shuffle: If True, shuffle the merged dataset
        seed: Random seed for shuffling
    """
    if weights is None:
        weights = [1.0] * len(input_files)
    
    if len(weights) != len(input_files):
        raise ValueError(f"Number of weights ({len(weights)}) must match number of input files ({len(input_files)})")
    
    # Load all datasets
    datasets = []
    total_samples = 0
    for i, (path, weight, name) in enumerate(zip(input_files, weights, names or [None] * len(input_files))):
        print(f"Loading {path}...")
        data = load_jsonl(path)
        print(f"  Loaded {len(data)} samples, weight={weight}")
        
        # Inject dataset name if requested
        if add_dataset_name and name:
            for item in data:
                if 'metadata' not in item:
                    item['metadata'] = {}
                if isinstance(item['metadata'], str):
                    item['metadata'] = json.loads(item['metadata'])
                item['metadata']['dataset_name'] = name
        
        datasets.append((data, weight, name or f"dataset{i}"))
        total_samples += len(data)
    
    # Calculate target samples per dataset based on weights
    total_weight = sum(weights)
    print(f"\nTotal samples: {total_samples}")
    print(f"Weight distribution:")
    
    merged_data = []
    for data, weight, name in datasets:
        # Repeat samples based on weight
        target_ratio = weight / total_weight
        target_count = int(total_samples * target_ratio)
        
        print(f"  {name}: {len(data)} samples × {weight} weight = ~{target_count} samples in output ({target_ratio*100:.1f}%)")
        
        if target_count <= len(data):
            # Sample subset
            selected = random.Random(seed).sample(data, target_count)
        else:
            # Oversample by repeating
            repeats = target_count // len(data)
            remainder = target_count % len(data)
            selected = data * repeats + random.Random(seed).sample(data, remainder)
        
        merged_data.extend(selected)
    
    # Shuffle if requested
    if shuffle:
        print(f"\nShuffling {len(merged_data)} samples...")
        random.Random(seed).shuffle(merged_data)
    
    # Save merged dataset
    print(f"Saving to {output_file}...")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(merged_data, output_file)
    
    print(f"✓ Successfully merged {len(input_files)} datasets into {output_file}")
    print(f"  Total samples in output: {len(merged_data)}")


def parse_input_spec(spec: str) -> Tuple[str, str]:
    """Parse input specification in format 'path:name' or just 'path'."""
    if ':' in spec:
        path, name = spec.rsplit(':', 1)
        return path, name
    return spec, None


def main():
    parser = argparse.ArgumentParser(
        description='Merge multiple JSONL datasets into a single dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--input',
        nargs='+',
        required=True,
        help='Input JSONL files. Format: path or path:name (e.g., data.jsonl:math)'
    )
    
    parser.add_argument(
        '--output',
        required=True,
        help='Output JSONL file path'
    )
    
    parser.add_argument(
        '--weights',
        nargs='+',
        type=float,
        help='Sampling weights for each dataset (default: equal weights)'
    )
    
    parser.add_argument(
        '--add-dataset-name',
        action='store_true',
        help='Inject dataset name into metadata field'
    )
    
    parser.add_argument(
        '--no-shuffle',
        action='store_true',
        help='Do not shuffle the merged dataset'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for shuffling and sampling (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Parse input specifications
    input_files = []
    names = []
    for spec in args.input:
        path, name = parse_input_spec(spec)
        input_files.append(path)
        names.append(name)
    
    # Merge datasets
    merge_datasets(
        input_files=input_files,
        output_file=args.output,
        weights=args.weights,
        names=names,
        add_dataset_name=args.add_dataset_name,
        shuffle=not args.no_shuffle,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()

