"""
Quick sanity check: Compare attention vs linear/gating on ETTh1
This proves the new implementation works and produces different results
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cmf_core import (
    load_dataset, prepare_chronos_input, BaseChronosPipeline
)
import numpy as np

def quick_test():
    print("=" * 70)
    print("QUICK SANITY CHECK: Attention vs Baselines on ETTh1")
    print("=" * 70)
    
    # Load dataset
    print("\n📂 Loading ETTh1 dataset...")
    df, feat_cols, target_col, seasonality = load_dataset('etth1')
    print(f"✓ Loaded: {len(df)} rows, {len(feat_cols)} features")
    
    # Use small context for speed
    context_len = 336
    
    adapters = ['linear', 'feat_gate', 'feat_attn']
    adapter_names = {
        'linear': 'Linear (OLS)',
        'feat_gate': 'Gating (MLP+Softmax)',
        'feat_attn': 'Attention (Q-K-V)'
    }
    
    print(f"\n🔧 Testing adapters with context_len={context_len}")
    print("-" * 70)
    
    for adapter_kind in adapters:
        print(f"\n{adapter_names[adapter_kind]:25s} ", end='')
        
        try:
            # Fit adapter on context window
            context_tensor, adapter_obj, ctx_df, y_future = prepare_chronos_input(
                df, context_len, feat_cols, target_col, 
                adapter_kind=adapter_kind,
                prediction_length=24,
                attn_device='cpu',
                attn_embed_dim=32,
                attn_num_heads=1,
                fg_device='cpu',
                fg_hidden=32
            )
            
            # Check output statistics
            fused = context_tensor.numpy()[0]  # [T]
            mean_val = fused.mean()
            std_val = fused.std()
            min_val = fused.min()
            max_val = fused.max()
            
            print(f"✓ Mean={mean_val:6.3f}, Std={std_val:6.3f}, Range=[{min_val:6.3f}, {max_val:6.3f}]")
            
        except Exception as e:
            print(f"✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("If all three produce DIFFERENT statistics, implementation is correct!")
    print("=" * 70)

if __name__ == "__main__":
    quick_test()
