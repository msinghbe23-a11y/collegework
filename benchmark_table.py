#!/usr/bin/env python3
"""
Generate a table comparing original PyTorch vs optimized version
across multiple examples with timing and correctness metrics.
"""

import time
import numpy as np
import torch
from typing import List, Dict
import warnings

# PyTorch imports
from enformer_pytorch.modeling_enformer import from_pretrained
from enformer_fast import EnformerFast

warnings.filterwarnings("ignore")


def benchmark_single_example(seq_length: int = 196_608, example_id: int = 1) -> Dict:
    """Benchmark a single example and return results"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Example {example_id}: Benchmarking seq_length={seq_length:,}...")
    
    # Generate test data
    np.random.seed(42 + example_id)  # Different seed per example
    bases = ['A', 'C', 'G', 'T']
    seq = ''.join(np.random.choice(bases, seq_length))
    
    from enformer_pytorch.data import str_to_one_hot
    x = str_to_one_hot([seq]).to(device)
    
    # Use TF gamma for all sequences now (all are 196,608 length)
    use_tf_gamma = True
    
    # 1. Original PyTorch model
    print(f"  Loading original model...")
    original = from_pretrained('EleutherAI/enformer-official-rough', use_tf_gamma=use_tf_gamma)
    original = original.to(device).eval()
    
    # Warmup + benchmark original
    with torch.no_grad():
        for _ in range(2):
            _ = original(x, return_only_embeddings=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # Time original
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        orig_out = original(x, return_only_embeddings=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        orig_time = time.perf_counter() - start
    
    # 2. Optimized model
    print(f"  Loading optimized model...")
    # Note: EnformerFast will handle TF gamma internally, but let's be explicit
    optimized = EnformerFast('EleutherAI/enformer-official-rough', device)
    
    # Apply all optimizations including Triton for this powerful AMD GPU
    optimized.apply_memory_optimizations()
    triton_success = optimized.apply_triton_optimizations()
    optimized.apply_torch_compile()
    
    if triton_success:
        print(f"  Triton optimizations applied successfully")
    
    # Warmup + benchmark optimized
    with torch.no_grad():
        for _ in range(2):
            _ = optimized.model(x, return_only_embeddings=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # Time optimized
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        opt_out = optimized.model(x, return_only_embeddings=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        opt_time = time.perf_counter() - start
    
    # Calculate metrics
    speedup = orig_time / opt_time
    
    # Numerical correctness
    diff = torch.abs(orig_out - opt_out)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    
    # Cosine similarity
    orig_flat = orig_out.flatten()
    opt_flat = opt_out.flatten()
    cos_sim = torch.nn.functional.cosine_similarity(orig_flat, opt_flat, dim=0).item()
    
    print(f"  Original: {orig_time:.3f}s, Optimized: {opt_time:.3f}s, Speedup: {speedup:.2f}x, CosSim: {cos_sim:.9f}")
    
    return {
        'example_id': example_id,
        'seq_length': seq_length,
        'orig_time': orig_time,
        'opt_time': opt_time,
        'speedup': speedup,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'cos_sim': cos_sim
    }


def generate_benchmark_table(examples: List[Dict] = None) -> None:
    """Generate and print benchmark table"""
    
    if examples is None:
        # 25 test cases - all using 196,608 length to ensure TF gamma compatibility
        # Different random seeds provide variation in DNA sequences for thorough testing
        example_configs = [
            {'seq_length': 196_608, 'example_id': 1},
            {'seq_length': 196_608, 'example_id': 2},
            {'seq_length': 196_608, 'example_id': 3},
            {'seq_length': 196_608, 'example_id': 4},
            {'seq_length': 196_608, 'example_id': 5},
            {'seq_length': 196_608, 'example_id': 6},
            {'seq_length': 196_608, 'example_id': 7},
            {'seq_length': 196_608, 'example_id': 8},
            {'seq_length': 196_608, 'example_id': 9},
            {'seq_length': 196_608, 'example_id': 10},
            {'seq_length': 196_608, 'example_id': 11},
            {'seq_length': 196_608, 'example_id': 12},
            {'seq_length': 196_608, 'example_id': 13},
            {'seq_length': 196_608, 'example_id': 14},
            {'seq_length': 196_608, 'example_id': 15},
            {'seq_length': 196_608, 'example_id': 16},
            {'seq_length': 196_608, 'example_id': 17},
            {'seq_length': 196_608, 'example_id': 18},
            {'seq_length': 196_608, 'example_id': 19},
            {'seq_length': 196_608, 'example_id': 20},
            {'seq_length': 196_608, 'example_id': 21},
            {'seq_length': 196_608, 'example_id': 22},
            {'seq_length': 196_608, 'example_id': 23},
            {'seq_length': 196_608, 'example_id': 24},
            {'seq_length': 196_608, 'example_id': 25},
        ]
        
        print("ENFORMER OPTIMIZATION BENCHMARK TABLE")
        print("=" * 70)
        print(f"Running {len(example_configs)} examples...")
        print()
        
        examples = []
        for config in example_configs:
            try:
                result = benchmark_single_example(**config)
                examples.append(result)
            except Exception as e:
                print(f"  ❌ Example {config['example_id']} failed: {e}")
                continue
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Print results table
    print("\nBENCHMARK RESULTS TABLE")
    print("=" * 78)
    print(f"{'Ex':<3} {'Seq Len':<8} {'Orig (s)':<9} {'Opt (s)':<8} {'Speedup':<8} {'CosSim':<14}")
    print("-" * 78)
    
    total_orig_time = 0
    total_opt_time = 0
    cos_sims = []
    
    for ex in examples:
        total_orig_time += ex['orig_time']
        total_opt_time += ex['opt_time']
        cos_sims.append(ex['cos_sim'])
        
        seq_len_k = f"{ex['seq_length']//1000}K"
        print(f"{ex['example_id']:<3} {seq_len_k:<8} {ex['orig_time']:<9.3f} {ex['opt_time']:<8.3f} {ex['speedup']:<8.2f}x {ex['cos_sim']:<14.9f}")
    
    # Summary statistics
    avg_speedup = total_orig_time / total_opt_time
    avg_cos_sim = np.mean(cos_sims)
    min_cos_sim = np.min(cos_sims)
    
    print("-" * 78)
    print(f"{'AVG':<3} {'--':<8} {total_orig_time:<9.3f} {total_opt_time:<8.3f} {avg_speedup:<8.2f}x {avg_cos_sim:<14.9f}")
    print()
    
    # Summary
    print("SUMMARY")
    print("-" * 30)
    print(f"Average speedup:      {avg_speedup:.2f}x")
    print(f"Average cosine sim:   {avg_cos_sim:.9f}")
    print(f"Minimum cosine sim:   {min_cos_sim:.9f}")
    
    # if avg_speedup >= 1.5:
    #     print(f"TARGET ACHIEVED: {avg_speedup:.2f}x average speedup")
    # else:
    #     print(f"Close to target: {avg_speedup:.2f}x average (target: 1.5x)")
    
    # if min_cos_sim > 0.9999:
    #     print("EXCELLENT numerical correctness across all examples")
    # elif min_cos_sim > 0.999:
    #     print("GOOD numerical correctness across all examples")
    # else:
    #     print("Some numerical differences detected")
    
    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\nGPU: {gpu_name}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def quick_benchmark_table() -> None:
    """Quick 5-example benchmark for fast testing"""
    example_configs = [
        {'seq_length': 196_608, 'example_id': 1},
        {'seq_length': 196_608, 'example_id': 2},
        {'seq_length': 98_304,  'example_id': 3},  # Half length
        {'seq_length': 196_608, 'example_id': 4},
        {'seq_length': 131_072, 'example_id': 5},  # 2/3 length
    ]
    
    print("QUICK ENFORMER BENCHMARK (5 examples)")
    print("=" * 50)
    
    examples = []
    for config in example_configs:
        try:
            result = benchmark_single_example(**config)
            examples.append(result)
        except Exception as e:
            print(f"  ❌ Example {config['example_id']} failed: {e}")
            continue
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    generate_benchmark_table(examples)


def main():
    """Main function"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        quick_benchmark_table()
    else:
        generate_benchmark_table()


if __name__ == "__main__":
    main()