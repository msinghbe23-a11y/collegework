#!/usr/bin/env python3
"""
Example usage of optimized Enformer model.
Demonstrates how to load and use the optimized implementation.
"""

import torch
import numpy as np
from enformer_pytorch.modeling_enformer_optimized import from_pretrained_optimized
from enformer_pytorch.data import str_to_one_hot


def generate_random_dna_sequence(length=196_608):
    """Generate a random DNA sequence"""
    bases = ['A', 'C', 'G', 'T']
    return ''.join(np.random.choice(bases, length))


def main():
    print("🧬 Optimized Enformer Example Usage")
    print("=" * 50)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load optimized model
    print("\nLoading optimized Enformer model...")
    try:
        model = from_pretrained_optimized('EleutherAI/enformer-official-rough', use_tf_gamma=True)
        model = model.to(device)
        model.eval()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("Creating model from config instead...")
        
        from enformer_pytorch.config_enformer import EnformerConfig
        from enformer_pytorch.modeling_enformer_optimized import OptimizedEnformer
        
        config = EnformerConfig(
            dim=1536,
            depth=11, 
            heads=8,
            output_heads={"human": 5313, "mouse": 1643},
            target_length=896,
            use_tf_gamma=True
        )
        
        model = OptimizedEnformer(config)
        model = model.to(device)
        model.eval()
        print("✅ Model created from config!")
    
    # Apply torch.compile if available
    if hasattr(torch, 'compile') and device.type == 'cuda':
        print("Applying torch.compile optimization...")
        model = torch.compile(model, mode='max-autotune')
        print("✅ Model compiled!")
    
    # Generate test sequence
    print(f"\nGenerating random DNA sequence...")
    sequence = generate_random_dna_sequence()
    print(f"Sequence length: {len(sequence)}")
    print(f"First 100 bases: {sequence[:100]}...")
    
    # Convert to model input
    print("\nConverting to model input...")
    x = str_to_one_hot([sequence])  # Batch of 1 sequence
    x = x.to(device)
    print(f"Input shape: {x.shape}")
    
    # Run inference
    print("\nRunning inference...")
    with torch.no_grad():
        # Get embeddings only (faster)
        embeddings = model(x, return_only_embeddings=True)
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Get predictions for both human and mouse
        predictions = model(x)
        
        print(f"Human predictions shape: {predictions['human'].shape}")
        print(f"Mouse predictions shape: {predictions['mouse'].shape}")
        
        # Get predictions for specific head
        human_pred = model(x, head='human')
        print(f"Human-only predictions shape: {human_pred.shape}")
    
    # Show some statistics
    print(f"\n📊 Output Statistics:")
    print(f"Human predictions - min: {predictions['human'].min():.4f}, max: {predictions['human'].max():.4f}, mean: {predictions['human'].mean():.4f}")
    print(f"Mouse predictions - min: {predictions['mouse'].min():.4f}, max: {predictions['mouse'].max():.4f}, mean: {predictions['mouse'].mean():.4f}")
    
    # Demonstrate different target lengths
    print(f"\n🎯 Testing different target lengths:")
    for target_len in [896, 512, 256]:
        with torch.no_grad():
            pred = model(x, head='human', target_length=target_len)
            print(f"Target length {target_len}: output shape {pred.shape}")
    
    print(f"\n✅ Example completed successfully!")
    
    # Memory usage if CUDA
    if device.type == 'cuda':
        print(f"\n💾 GPU Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()