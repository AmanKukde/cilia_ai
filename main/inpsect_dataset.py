# Utility: Inspect & Validate Your NumPy Dataset (4D Masks)
# Updated for (N, C, H, W) mask format

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

def inspect_dataset(images_path, masks_path):
    """
    Comprehensive dataset inspection with visualization.
    Updated to handle 4D masks: (N, C, H, W)
    """
    print("=" * 70)
    print("DATASET INSPECTION REPORT")
    print("=" * 70)
    
    # Load arrays
    print("\nðŸ“‚ Loading arrays...")
    images = np.load(images_path)
    masks = np.load(masks_path)
    
    print(f"âœ“ Images loaded: {images_path}")
    print(f"âœ“ Masks loaded: {masks_path}")
    
    # Shape validation
    print("\n" + "=" * 70)
    print("SHAPE VALIDATION")
    print("=" * 70)
    
    print(f"\nImages shape: {images.shape}")
    assert len(images.shape) == 4, "Images must be 4D (N, C, H, W)"
    N, C, H, W = images.shape
    print(f"  âœ“ N (samples): {N}")
    print(f"  âœ“ C (channels): {C}")
    print(f"  âœ“ H (height): {H}")
    print(f"  âœ“ W (width): {W}")
    
    print(f"\nMasks shape: {masks.shape}")
    if len(masks.shape) == 3:
        # Old format: (N, H, W)
        print("  Format: (N, H, W) - single mask per sample")
        assert masks.shape[0] == N, f"Mask count ({masks.shape[0]}) != Image count ({N})"
        assert masks.shape[1] == H, f"Mask height ({masks.shape[1]}) != Image height ({H})"
        assert masks.shape[2] == W, f"Mask width ({masks.shape[2]}) != Image width ({W})"
        mask_channels = 1
    elif len(masks.shape) == 4:
        # New format: (N, C, H, W) - per-channel masks
        print("  Format: (N, C, H, W) - per-channel masks")
        assert masks.shape[0] == N, f"Mask count ({masks.shape[0]}) != Image count ({N})"
        assert masks.shape[1] == C, f"Mask channels ({masks.shape[1]}) != Image channels ({C})"
        assert masks.shape[2] == H, f"Mask height ({masks.shape[2]}) != Image height ({H})"
        assert masks.shape[3] == W, f"Mask width ({masks.shape[3]}) != Image width ({W})"
        mask_channels = masks.shape[1]
        print(f"  âœ“ Per-channel masks detected: {mask_channels} channels")
    else:
        raise ValueError(f"Masks must be 3D or 4D, got {len(masks.shape)}D")
    
    # Data type analysis
    print("\n" + "=" * 70)
    print("DATA TYPE & RANGE ANALYSIS")
    print("=" * 70)
    
    print(f"\nImages:")
    print(f"  dtype: {images.dtype}")
    print(f"  min: {images.min():.6f}")
    print(f"  max: {images.max():.6f}")
    print(f"  mean: {images.mean():.6f}")
    print(f"  std: {images.std():.6f}")
    
    # Check if normalized
    if images.max() <= 1.0:
        print(f"  âœ“ Already normalized to [0, 1]")
    elif images.max() <= 255:
        print(f"  âš  Range [0, 255] - will normalize during training")
    
    print(f"\nMasks:")
    print(f"  dtype: {masks.dtype}")
    print(f"  min: {masks.min()}")
    print(f"  max: {masks.max()}")
    unique_vals = np.unique(masks)
    print(f"  unique values: {unique_vals}")
    
    # Check if binary
    if set(unique_vals).issubset({0, 1, 255}):
        print(f"  âœ“ Binary masks (values: {unique_vals.tolist()})")
    else:
        print(f"  âš  Non-binary masks - {len(unique_vals)} unique values")
    
    # Channel-specific stats
    print(f"\nPer-Channel Image Statistics:")
    for ch in range(C):
        ch_data = images[:, ch, :, :]
        print(f"  Channel {ch}: min={ch_data.min():.3f}, max={ch_data.max():.3f}, "
              f"mean={ch_data.mean():.3f}, std={ch_data.std():.3f}")
    
    # Mask coverage statistics
    print("\n" + "=" * 70)
    print("MASK COVERAGE ANALYSIS")
    print("=" * 70)
    
    # Normalize masks to [0, 1] for analysis
    masks_01 = masks.astype(np.float32)
    if masks_01.max() > 1:
        masks_01 = masks_01 / 255.0
    
    # Reshape based on format
    if len(masks.shape) == 3:
        # (N, H, W) format
        coverage_pct = (masks_01.sum() / masks_01.size) * 100
        per_sample_coverage = (masks_01.reshape(N, -1).sum(axis=1) / (H*W)) * 100
    else:
        # (N, C, H, W) format - per channel
        per_sample_per_channel_coverage = []
        for ch in range(mask_channels):
            masks_ch = masks_01[:, ch, :, :]
            ch_coverage = (masks_ch.reshape(N, -1).sum(axis=1) / (H*W)) * 100
            per_sample_per_channel_coverage.append(ch_coverage)
        
        per_sample_per_channel_coverage = np.array(per_sample_per_channel_coverage)  # (C, N)
        coverage_pct = (masks_01.sum() / masks_01.size) * 100
        per_sample_coverage = per_sample_per_channel_coverage.mean(axis=0)  # Average across channels
    
    print(f"\nAverage mask coverage: {coverage_pct:.2f}%")
    
    # Per-sample coverage
    print(f"  Min: {per_sample_coverage.min():.2f}%")
    print(f"  Max: {per_sample_coverage.max():.2f}%")
    print(f"  Mean: {per_sample_coverage.mean():.2f}%")
    print(f"  Std: {per_sample_coverage.std():.2f}%")
    
    # Per-channel coverage (if 4D masks)
    if len(masks.shape) == 4:
        print(f"\nPer-Channel Coverage:")
        for ch in range(mask_channels):
            ch_coverage = per_sample_per_channel_coverage[ch]
            print(f"  Channel {ch}: mean={ch_coverage.mean():.2f}%, "
                  f"min={ch_coverage.min():.2f}%, max={ch_coverage.max():.2f}%")
    
    # Empty/full samples
    empty = (per_sample_coverage == 0).sum()
    full = (per_sample_coverage == 100).sum()
    if empty > 0:
        print(f"  âš  {empty} empty samples (0% coverage)")
    if full > 0:
        print(f"  âš  {full} full samples (100% coverage)")
    
    # Sample distribution
    print(f"\nCoverage distribution:")
    ranges = [(0, 10), (10, 30), (30, 50), (50, 70), (70, 90), (90, 100)]
    for low, high in ranges:
        count = ((per_sample_coverage >= low) & (per_sample_coverage < high)).sum()
        pct = (count / N) * 100
        bar = "â–ˆ" * int(pct / 2)
        print(f"  {low:2d}%-{high:2d}%: {count:4d} ({pct:5.1f}%) {bar}")
    
    # Memory footprint
    print("\n" + "=" * 70)
    print("MEMORY ANALYSIS")
    print("=" * 70)
    
    img_size_gb = images.nbytes / (1024**3)
    mask_size_gb = masks.nbytes / (1024**3)
    total_gb = img_size_gb + mask_size_gb
    
    print(f"\nImages: {img_size_gb:.3f} GB")
    print(f"Masks: {mask_size_gb:.3f} GB")
    print(f"Total: {total_gb:.3f} GB")
    
    if total_gb > 8:
        print(f"âš  Large dataset - ensure adequate GPU VRAM")
    else:
        print(f"âœ“ Dataset fits in typical GPU memory")
    
    # Data quality checks
    print("\n" + "=" * 70)
    print("DATA QUALITY CHECKS")
    print("=" * 70)
    
    checks = {
        "Shape validity": (len(images.shape) == 4) and (len(masks.shape) in [3, 4]),
        "Sample count match": images.shape[0] == masks.shape[0],
        "Spatial dims match": images.shape[2:] == masks.shape[-2:],
        "Binary masks": set(np.unique(masks)).issubset({0, 1, 255}),
        "No NaN in images": not np.isnan(images).any(),
        "No NaN in masks": not np.isnan(masks).any(),
        "Reasonable image range": images.max() <= 256 and images.min() >= -10,
        "At least 1 channel": C >= 1,
        "Non-empty samples": per_sample_coverage.max() > 0,
    }
    
    if len(masks.shape) == 4:
        checks["Mask channels match images"] = masks.shape[1] == C
    
    for check, passed in checks.items():
        status = "âœ“" if passed else "âœ—"
        print(f"{status} {check}")
    
    if all(checks.values()):
        print("\nâœ“ All checks passed - dataset is ready for fine-tuning!")
    else:
        print("\nâš  Some checks failed - review above before training")
    
    return {
        'N': N, 'C': C, 'H': H, 'W': W,
        'coverage': coverage_pct,
        'per_sample_coverage': per_sample_coverage,
        'mask_shape_format': len(masks.shape),
        'checks': checks
    }


def visualize_samples(images_path, masks_path, num_samples=5, save_fig=None):
    """
    Visualize random samples with overlays.
    Handles both 3D and 4D mask formats.
    """
    print("\n" + "=" * 70)
    print("SAMPLE VISUALIZATION")
    print("=" * 70)
    
    images = np.load(images_path)
    masks = np.load(masks_path)
    
    N, C, H, W = images.shape
    mask_is_4d = len(masks.shape) == 4
    mask_channels = masks.shape[1] if mask_is_4d else 1
    
    # Create figure
    num_cols = 2 + mask_channels  # Image + (masks for each channel) + overlay
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(4*num_cols, 4*num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]
    
    # Random sample indices
    indices = np.random.choice(N, min(num_samples, N), replace=False)
    
    for row, idx in enumerate(indices):
        img = images[idx]
        
        # Normalize image for display
        img_display = img.astype(np.float32)
        if img_display.max() > 1:
            img_display = img_display / 255.0
        
        # For multi-channel, use first channel or channels
        if C == 1:
            img_vis = img_display[0]
        elif C == 2:
            # For 2-channel, use as RG (red-green) with black B
            img_vis = np.stack([img_display[0], img_display[1], np.zeros_like(img_display[0])], axis=2)
        else:
            img_vis = np.transpose(img_display[:3], (1, 2, 0))
        
        # Column 0: Image
        axes[row, 0].imshow(img_vis, cmap='gray' if C == 1 else None)
        axes[row, 0].set_title(f'Sample {idx}: Image (C={C})')
        axes[row, 0].axis('off')
        
        # Columns 1+: Masks per channel
        if mask_is_4d:
            for ch in range(mask_channels):
                mask = masks[idx, ch]
                mask_display = mask.astype(np.float32)
                if mask_display.max() > 1:
                    mask_display = mask_display / 255.0
                
                axes[row, 1 + ch].imshow(mask_display, cmap='binary')
                axes[row, 1 + ch].set_title(f'Mask Ch{ch}')
                axes[row, 1 + ch].axis('off')
        else:
            mask = masks[idx]
            mask_display = mask.astype(np.float32)
            if mask_display.max() > 1:
                mask_display = mask_display / 255.0
            
            axes[row, 1].imshow(mask_display, cmap='binary')
            axes[row, 1].set_title(f'Sample {idx}: Mask')
            axes[row, 1].axis('off')
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(save_fig, dpi=150, bbox_inches='tight')
        print(f"âœ“ Visualization saved to {save_fig}")
    else:
        plt.show()
    
    plt.close()


def validate_for_training(images_path, masks_path):
    """
    Final pre-training validation.
    """
    print("\n" + "=" * 70)
    print("PRE-TRAINING VALIDATION")
    print("=" * 70)
    
    stats = inspect_dataset(images_path, masks_path)
    
    # Check readiness
    checks = stats['checks']
    if not all(checks.values()):
        print("\nâœ— Dataset has issues - fix before training!")
        return False
    
    # Suggest hyperparameters
    N = stats['N']
    C = stats['C']
    coverage = stats['coverage']
    
    print("\n" + "=" * 70)
    print("RECOMMENDED HYPERPARAMETERS")
    print("=" * 70)
    
    if N < 500:
        batch_size = 1
        num_epochs = 50
        lr = 5e-5
    elif N < 2000:
        batch_size = 2
        num_epochs = 20
        lr = 1e-4
    else:
        batch_size = 4
        num_epochs = 10
        lr = 1e-4
    
    print(f"\nbatch_size: {batch_size}")
    print(f"num_epochs: {num_epochs}")
    print(f"lr: {lr}")
    print(f"\nâš  Dataset has {C} channels with per-channel masks")
    print(f"   â†’ Using sam2_multichannel.py for separate per-channel models")
    
    if coverage < 5:
        print(f"\nâš  Low mask coverage ({coverage:.1f}%)")
        print(f"  â†’ Consider using point prompts in addition to box prompts")
    elif coverage > 80:
        print(f"\nâš  High mask coverage ({coverage:.1f}%)")
        print(f"  â†’ Model might learn trivial solution (predict all foreground)")
    
    print("\nâœ“ Ready to train!")
    print("\nðŸ’¡ RECOMMENDED: python sam2_multichannel.py")
    return True


if __name__ == "__main__":
    # Configuration
    images_path = "data/prompted_outputs/images_cleaned_prompted.npy"
    masks_path = "data/prompted_outputs/predictions_sam2_cleaned.npy"
    
    # Run inspection
    print("\nðŸ” Inspecting dataset...")
    stats = inspect_dataset(images_path, masks_path)
    
    # Visualize samples
    print("\nðŸ“Š Generating visualizations...")
    visualize_samples(images_path, masks_path, num_samples=3, 
                     save_fig='dataset_samples.png')
    
    # Final validation
    print("\nâœ“ Visualization saved as 'dataset_samples.png'")
    validate_for_training(images_path, masks_path)