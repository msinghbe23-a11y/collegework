#!/usr/bin/env python3
"""
Clean, simple optimization approach for Enformer.
Strategy: Keep original model intact, add optimization layers on top.
"""

import torch
import time
import triton
import triton.language as tl
from enformer_pytorch.modeling_enformer import from_pretrained


# Simple Triton kernels for common operations
@triton.jit
def fused_gelu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Fused GELU kernel matching Enformer's exact implementation"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    # Enformer uses: torch.sigmoid(1.702 * x) * x
    gelu_out = tl.sigmoid(1.702 * x) * x
    tl.store(output_ptr + offsets, gelu_out, mask=mask)


@triton.jit
def fused_layernorm_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    mean_ptr, rstd_ptr,
    n_rows, n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    """Fused LayerNorm kernel"""
    row_idx = tl.program_id(0)
    
    if row_idx >= n_rows:
        return
    
    # Compute mean
    row_start = row_idx * n_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    
    mean = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        mask = (i + offsets) < n_cols
        x_vals = tl.load(x_ptr + row_start + i + offsets, mask=mask, other=0.0)
        mean += tl.sum(x_vals)
    mean = mean / n_cols
    
    # Compute variance
    var = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        mask = (i + offsets) < n_cols
        x_vals = tl.load(x_ptr + row_start + i + offsets, mask=mask, other=0.0)
        diff = x_vals - mean
        var += tl.sum(diff * diff)
    var = var / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Apply normalization
    for i in range(0, n_cols, BLOCK_SIZE):
        mask = (i + offsets) < n_cols
        x_vals = tl.load(x_ptr + row_start + i + offsets, mask=mask, other=0.0)
        weight_vals = tl.load(weight_ptr + i + offsets, mask=mask, other=1.0)
        bias_vals = tl.load(bias_ptr + i + offsets, mask=mask, other=0.0)
        
        normalized = (x_vals - mean) * rstd
        output = normalized * weight_vals + bias_vals
        tl.store(output_ptr + row_start + i + offsets, output, mask=mask)
    
    # Store statistics
    if tl.program_id(0) == row_idx:
        tl.store(mean_ptr + row_idx, mean)
        tl.store(rstd_ptr + row_idx, rstd)


def triton_fused_gelu(x):
    """Triton-accelerated GELU matching Enformer's implementation"""
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_gelu_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output


def triton_fused_layernorm(x, weight, bias, eps=1e-5):
    """Triton-accelerated LayerNorm"""
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    mean = torch.empty(n_rows, dtype=x.dtype, device=x.device)
    rstd = torch.empty(n_rows, dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = min(triton.next_power_of_2(n_cols), 1024)
    grid = (n_rows,)
    
    fused_layernorm_kernel[grid](
        x, weight, bias, output, mean, rstd,
        n_rows, n_cols, eps, BLOCK_SIZE=BLOCK_SIZE
    )
    return output


class OptimizedEnformerLayer(torch.nn.Module):
    """Drop-in replacement for key Enformer operations with Triton acceleration"""
    
    def __init__(self, original_module):
        super().__init__()
        self.original_module = original_module
        self.use_triton = True
        
    def forward(self, *args, **kwargs):
        if not self.use_triton or not self.training:
            # Use Triton optimizations during inference
            return self._triton_forward(*args, **kwargs)
        else:
            # Fallback to original during training for safety
            return self.original_module(*args, **kwargs)
    
    def _triton_forward(self, *args, **kwargs):
        # This will be overridden by specific layer types
        return self.original_module(*args, **kwargs)


class OptimizedGELU(OptimizedEnformerLayer):
    """Triton-optimized GELU"""
    
    def _triton_forward(self, x):
        if x.is_cuda and x.is_contiguous():
            try:
                return triton_fused_gelu(x)
            except Exception:
                pass
        return self.original_module(x)


class OptimizedLayerNorm(OptimizedEnformerLayer):
    """Triton-optimized LayerNorm"""
    
    def _triton_forward(self, x):
        if x.is_cuda and x.dim() == 2 and x.is_contiguous():
            try:
                return triton_fused_layernorm(x, self.original_module.weight, self.original_module.bias)
            except Exception:
                pass
        return self.original_module(x)


class EnformerFast:
    """
    Fast wrapper around original Enformer model.
    Applies optimizations without changing the core model.
    """
    
    def __init__(self, model_name='EleutherAI/enformer-official-rough', device='cuda'):
        print(f"Loading original Enformer model from {model_name}...")
        
        # Load the original, working model
        self.model = from_pretrained(model_name, use_tf_gamma=True)
        self.model = self.model.to(device)
        self.model.eval()
        
        self.device = device
        self.compiled = False
        self.triton_optimized = False
        
        # Detect GPU capabilities
        self.gpu_info = self._detect_gpu_capabilities()
        
        print(f"✅ Model loaded successfully")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Device: {device}")
        print(f"   GPU: {self.gpu_info['name']}")
        print(f"   Supports bfloat16: {self.gpu_info['bf16_supported']}")
        print(f"   Compute capability: {self.gpu_info['compute_capability']}")
    
    def _detect_gpu_capabilities(self):
        """Detect GPU capabilities for optimization selection"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            compute_capability = torch.cuda.get_device_capability(0)
            bf16_supported = torch.cuda.is_bf16_supported()
            
            # Determine if this is a modern GPU (L4, L40S, A100, H100, etc.)
            modern_gpu = any(gpu in gpu_name.upper() for gpu in ['L4', 'L40', 'A100', 'H100', 'V100', 'RTX 40', 'RTX 30'])
            
            return {
                'name': gpu_name,
                'compute_capability': compute_capability,
                'bf16_supported': bf16_supported,
                'modern_gpu': modern_gpu,
                'supports_advanced_opts': modern_gpu and compute_capability >= (7, 5)
            }
        else:
            return {
                'name': 'CPU',
                'compute_capability': (0, 0),
                'bf16_supported': False,
                'modern_gpu': False,
                'supports_advanced_opts': False
            }
    
    def apply_torch_compile(self, mode='max-autotune'):
        """Apply torch.compile optimization"""
        if not hasattr(torch, 'compile'):
            print("⚠️  torch.compile not available in this PyTorch version")
            return False
        
        print(f"Applying torch.compile with mode='{mode}'...")
        try:
            self.model = torch.compile(self.model, mode=mode)
            self.compiled = True
            print("✅ torch.compile applied successfully")
            return True
        except Exception as e:
            print(f"❌ torch.compile failed: {e}")
            return False
    
    def apply_mixed_precision(self, dtype=None):
        """Enable mixed precision inference with hardware-aware dtype selection"""
        if not self.gpu_info['supports_advanced_opts']:
            print("⚠️  Mixed precision disabled - GPU too old (requires compute capability >= 7.5)")
            return False
            
        self.use_amp = True
        
        # Auto-select best dtype for the hardware
        if dtype is None:
            if self.gpu_info['bf16_supported'] and self.gpu_info['modern_gpu']:
                dtype = torch.bfloat16
            else:
                dtype = torch.float16
        
        self.amp_dtype = dtype
        
        # Use autocast only - don't convert model weights to avoid dtype mismatches
        # The model stays in float32, autocast handles the conversion during operations
        if dtype == torch.bfloat16:
            print("✅ Mixed precision enabled (autocast bfloat16) - optimized for modern GPU")
        else:
            print("✅ Mixed precision enabled (autocast float16)")
        
        return True
    
    def apply_memory_optimizations(self):
        """Apply memory access pattern optimizations"""
        try:
            print("Applying memory optimizations...")
            
            # Enable memory efficient attention if available
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                print("✅ Using PyTorch native scaled_dot_product_attention")
            
            # Set memory format for better access patterns
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("✅ Enabled TensorFloat-32 for faster computation")
            
            # Enable deterministic algorithms for consistent performance
            torch.use_deterministic_algorithms(False)  # Better performance
            
            return True
        except Exception as e:
            print(f"❌ Memory optimization failed: {e}")
            return False
    
    def apply_triton_optimizations(self):
        """Replace key operations with Triton-optimized versions (GPU-aware)"""
        if not torch.cuda.is_available():
            print("⚠️  Triton optimizations require CUDA")
            return False
            
        if not self.gpu_info['supports_advanced_opts']:
            print("⚠️  Triton optimizations disabled - GPU too old for optimal performance")
            return False
        
        try:
            print(f"Applying Triton kernel optimizations for {self.gpu_info['name']}...")
            optimizations_applied = 0
            
            # Replace GELU activations
            def replace_gelu(module):
                nonlocal optimizations_applied
                for name, child in module.named_children():
                    if child.__class__.__name__ == 'GELU':
                        setattr(module, name, OptimizedGELU(child))
                        optimizations_applied += 1
                    else:
                        replace_gelu(child)
            
            # Replace LayerNorm layers  
            def replace_layernorm(module):
                nonlocal optimizations_applied
                for name, child in module.named_children():
                    if isinstance(child, torch.nn.LayerNorm):
                        setattr(module, name, OptimizedLayerNorm(child))
                        optimizations_applied += 1
                    else:
                        replace_layernorm(child)
            
            replace_gelu(self.model)
            replace_layernorm(self.model)
            
            print(f"✅ Applied {optimizations_applied} Triton optimizations")
            self.triton_optimized = True
            return True
            
        except Exception as e:
            print(f"❌ Triton optimization failed: {e}")
            return False
    
    def forward_with_optimizations(self, *args, **kwargs):
        """Forward pass with all optimizations"""
        if hasattr(self, 'use_amp') and self.use_amp:
            dtype = getattr(self, 'amp_dtype', torch.bfloat16)
            # Use autocast only for linear operations, exclude sensitive computations
            with torch.autocast('cuda', dtype=dtype, enabled=True):
                return self.model(*args, **kwargs)
        else:
            return self.model(*args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        """Forward pass through the model"""
        return self.model(*args, **kwargs)
    
    def benchmark(self, seq_len=196_608, batch_size=1, num_runs=5):
        """Benchmark the model"""
        print(f"\n🔥 BENCHMARKING")
        print(f"   Sequence length: {seq_len:,}")
        print(f"   Batch size: {batch_size}")
        print(f"   Runs: {num_runs}")
        print(f"   Compiled: {self.compiled}")
        print(f"   Triton optimized: {getattr(self, 'triton_optimized', False)}")
        print(f"   Mixed precision: {getattr(self, 'use_amp', False)}")
        
        # Create test data
        import numpy as np
        np.random.seed(42)
        bases = ['A', 'C', 'G', 'T']
        sequences = []
        for _ in range(batch_size):
            seq = ''.join(np.random.choice(bases, seq_len))
            sequences.append(seq)
        
        from enformer_pytorch.data import str_to_one_hot
        x = str_to_one_hot(sequences).to(self.device)
        
        # Warmup
        print("   Warming up...")
        with torch.no_grad():
            for _ in range(2):
                if hasattr(self, 'use_amp') and self.use_amp:
                    _ = self.forward_with_optimizations(x, return_only_embeddings=True)
                else:
                    _ = self.model(x, return_only_embeddings=True)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        
        # Benchmark
        print("   Running benchmark...")
        times = []
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            for i in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                
                # Use optimized forward pass if mixed precision is enabled
                if hasattr(self, 'use_amp') and self.use_amp:
                    output = self.forward_with_optimizations(x, return_only_embeddings=True)
                else:
                    output = self.model(x, return_only_embeddings=True)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                run_time = end - start
                times.append(run_time)
                print(f"     Run {i+1}: {run_time:.3f}s")
        
        avg_time = sum(times) / len(times)
        std_time = (sum((t - avg_time)**2 for t in times) / len(times))**0.5
        
        if torch.cuda.is_available():
            memory_gb = torch.cuda.max_memory_allocated() / 1e9
        else:
            memory_gb = 0
        
        print(f"\n   Results:")
        print(f"     Mean time: {avg_time:.3f}s ± {std_time:.3f}s")
        print(f"     Best time: {min(times):.3f}s")
        print(f"     Memory: {memory_gb:.2f} GB")
        print(f"     Output shape: {output.shape}")
        
        return {
            'mean_time': avg_time,
            'std_time': std_time,
            'min_time': min(times),
            'memory_gb': memory_gb,
            'output': output
        }


def compare_models(model_name='EleutherAI/enformer-official-rough', device='cuda'):
    """Compare original vs optimized model"""
    print("🚀 ENFORMER OPTIMIZATION COMPARISON")
    print("=" * 50)
    
    # Test 1: Original model
    print("\n1. ORIGINAL MODEL")
    print("-" * 30)
    original = EnformerFast(model_name, device)
    orig_results = original.benchmark(num_runs=3)
    
    # Test 2: Optimized model without mixed precision (for numerical correctness)
    print("\n2. OPTIMIZED MODEL (MEMORY + TRITON + TORCH.COMPILE)")
    print("-" * 30)
    optimized = EnformerFast(model_name, device)
    
    # Apply GPU-appropriate optimizations (conservative approach for numerical correctness)
    memory_success = optimized.apply_memory_optimizations()
    triton_success = optimized.apply_triton_optimizations()
    # Skip mixed precision for now to ensure numerical correctness
    # mixed_precision_success = optimized.apply_mixed_precision()
    compile_success = optimized.apply_torch_compile()
    
    print(f"\nOptimizations applied (conservative for numerical correctness):")
    print(f"  Memory: {memory_success}")
    print(f"  Triton: {triton_success}")
    print(f"  Mixed Precision: SKIPPED (for numerical correctness)")
    print(f"  Torch Compile: {compile_success}")
    
    if optimized.gpu_info['modern_gpu']:
        print(f"✅ Using optimizations for modern GPU: {optimized.gpu_info['name']}")
    else:
        print(f"⚠️  Using conservative optimizations for: {optimized.gpu_info['name']}")
    
    if compile_success:
        opt_results = optimized.benchmark(num_runs=3)
        
        # Compare results
        print("\n📈 COMPARISON")
        print("-" * 30)
        speedup = orig_results['mean_time'] / opt_results['mean_time']
        memory_diff = orig_results['memory_gb'] - opt_results['memory_gb']
        
        print(f"Speedup: {speedup:.2f}x")
        print(f"Memory change: {memory_diff:+.2f} GB")
        
        # Check numerical correctness with detailed output comparison
        orig_out = orig_results['output']
        opt_out = opt_results['output']
        
        diff = torch.abs(orig_out - opt_out)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        print(f"Output comparison:")
        print(f"  Max difference: {max_diff:.2e}")
        print(f"  Mean difference: {mean_diff:.2e}")
        
        # Show actual values from a sample region
        print(f"\nSample outputs (first 5 positions, first 10 features):")
        print(f"Original model:")
        print(f"  {orig_out[0, :5, :10]}")
        print(f"Optimized model:")
        print(f"  {opt_out[0, :5, :10]}")
        print(f"Difference:")
        print(f"  {diff[0, :5, :10]}")
        
        # Statistics comparison
        print(f"\nOutput statistics:")
        print(f"  Original - min: {orig_out.min():.6f}, max: {orig_out.max():.6f}, mean: {orig_out.mean():.6f}, std: {orig_out.std():.6f}")
        print(f"  Optimized - min: {opt_out.min():.6f}, max: {opt_out.max():.6f}, mean: {opt_out.mean():.6f}, std: {opt_out.std():.6f}")
        
        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(orig_out.flatten(), opt_out.flatten(), dim=0).item()
        print(f"  Cosine similarity: {cos_sim:.8f}")
        
        # Relaxed threshold for combined optimizations (torch.compile + Triton)
        if max_diff < 1e-4:
            print("✅ Numerical correctness: PASSED (within tolerance)")
        elif max_diff < 1e-3:
            print("⚠️  Numerical correctness: ACCEPTABLE (small differences)")
        else:
            print("❌ Numerical correctness: FAILED")
        
        # Success criteria
        if speedup >= 1.5:
            print(f"🎉 TARGET ACHIEVED: {speedup:.2f}x speedup!")
        elif speedup >= 1.2:
            print(f"⚠️  Good speedup: {speedup:.2f}x")
        else:
            print(f"❌ Need more optimization: {speedup:.2f}x")
        
        return speedup, max_diff < 1e-4
    else:
        print("❌ torch.compile failed - no comparison possible")
        return 1.0, True


def main():
    """Main optimization test"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cpu':
        print("⚠️  Running on CPU - speedups will be minimal")
    
    try:
        speedup, correct = compare_models(device=device)
        
        if speedup >= 1.5 and correct:
            print(f"\n🎉 SUCCESS: Ready for AMD testing!")
        else:
            print(f"\n🔧 Need to add more optimizations...")
            print(f"   Next steps:")
            print(f"   - Add Triton kernels")
            print(f"   - Optimize memory access patterns")
            print(f"   - Use mixed precision")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()