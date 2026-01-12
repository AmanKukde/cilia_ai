
# # Evaluation on Large Images with Patchification (Dual Channel)
# 
# This notebook demonstrates how to:
# 1. Load two trained models (Channel 1 and Channel 2)
# 2. Patch large images into smaller tiles
# 3. Run predictions on patches using both models simultaneously
# 4. Reconstruct full-size predictions for both channels
# 5. Combine original 4 channels + 2 predictions = 6 channels
# 6. Visualize and save results


# ## Setup and Imports


import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import json
import tifffile as tf
from tqdm.notebook import tqdm

from utils.patchify import (
    predict_large_image,
    patchify_image,
    unpatchify_image
)
from models.model_factory import ModelFactory
from cilia_utils.utils import normalize_channel


# ## Configuration


# Model configuration
MODEL_CHECKPOINT_CH1 = '../outputs_07Jan25_12-34-00/c1_unet_tiny/best_model.pth'
MODEL_CHECKPOINT_CH2 = '../outputs_07Jan25_12-34-00/c2_unet_tiny/best_model.pth'
ARCHITECTURE = 'unet_tiny'

# Channel indices
C1_IDX = 2
C2_IDX = 3

# Inference configuration
PATCH_SIZE = 64  # Must match training image size
OVERLAP = 63      # Overlap between patches
BATCH_SIZE = 8
THRESHOLD = 0.5
BLEND_MODE = 'average'  # 'average', 'max', or 'first'

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# Input/Output paths
ROOT = Path(
    '/group/jug/aman/Cilia_Datasets/extracted/'
    'W19 - 2025_Pkhd1_cells/'
    'W19 - 2025_Pkhd1 cells'
)
PRED_ROOT = Path("/group/jug/aman/Cilia_Datasets/Predictions")


# ## 1. Load Both Models


def load_model(checkpoint_path, architecture, device='cuda'):
    """Load trained model from checkpoint."""
    model = ModelFactory.create_model(
        architecture,
        num_classes=1,
        pretrained=False
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from: {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Metrics: {checkpoint.get('metrics', 'N/A')}")
    
    return model

# Load both models
model_ch1 = load_model(MODEL_CHECKPOINT_CH1, ARCHITECTURE, DEVICE)
model_ch2 = load_model(MODEL_CHECKPOINT_CH2, ARCHITECTURE, DEVICE)

print("\n✓ Both models loaded successfully")


# ## 2. Load All Test Images


def load_all_images_from_root(root_path):
    """
    Recursively load all TIFF images from directory structure.
    
    Returns:
        all_slices: (N_total, C, H, W) stacked 2D slices
        slice_index: List of metadata dicts for each slice
    """
    all_slices = []
    slice_index = []
    
    print("Scanning folders for TIFF files...")
    
    for cropped_dir in root_path.rglob('Cropped original image'):
        if not cropped_dir.is_dir():
            continue
        
        print(f"Found: {cropped_dir}")
        
        for tif_path in cropped_dir.glob('*.tif'):
            img = tf.imread(tif_path)
            
            # Normalize to (Z, C, Y, X)
            if img.ndim == 3:  # (C, Y, X)
                img = img[np.newaxis, ...]
            elif img.ndim != 4:
                raise ValueError(f"Unexpected shape {img.shape} in {tif_path}")
            
            Z, C, Y, X = img.shape
            
            for z in range(Z):
                all_slices.append(img[z])  # (C, Y, X)
                slice_index.append({
                    "path": tif_path,
                    "z": z,
                    "Z": Z
                })
    
    all_slices = np.stack(all_slices, axis=0)  # (N_total, C, Y, X)
    print(f"✓ Total 2D slices collected: {all_slices.shape[0]}")
    
    return all_slices, slice_index

# Load images
all_slices, slice_index = load_all_images_from_root(ROOT)


# ## 3. Prepare Images for Inference


def prepare_image_for_single_channel(image, channel_idx):
    """
    Prepare single-channel image for model inference (convert to 3-channel RGB).
    
    Args:
        image: (C, H, W) multichannel image
        channel_idx: Which channel to extract and use as RGB
    
    Returns:
        RGB image (3, H, W) ready for model
    """
    channel = image[channel_idx]
    channel_norm = normalize_channel(channel)
    rgb = np.stack([channel_norm] * 3, axis=0)
    
    # Keep pixel values in [0, 255] range to match training
    return (rgb * 255).astype(np.uint8).astype(np.float32)

# Test on first slice
test_idx = 0
raw_image = all_slices[test_idx]  # (C, H, W)
prepared_ch1 = prepare_image_for_single_channel(raw_image, C1_IDX)
prepared_ch2 = prepare_image_for_single_channel(raw_image, C2_IDX)

print(f"Raw image shape: {raw_image.shape}")
print(f"Prepared Ch1 shape: {prepared_ch1.shape}")
print(f"Prepared Ch2 shape: {prepared_ch2.shape}")


# ## 4. Batch Inference on All Slices


def predict_dual_channel(model_ch1, model_ch2, image, patch_size, overlap, 
                         batch_size, blend_mode, device, threshold, c1_idx, c2_idx):
    """
    Run inference on both channels independently and return both predictions.
    
    Args:
        model_ch1, model_ch2: Trained models for each channel
        image: (C, H, W) raw image
        c1_idx, c2_idx: Channel indices
        ... other params same as predict_large_image
    
    Returns:
        pred_mask_ch1: (H, W) channel 1 prediction mask
        pred_binary_ch1: (H, W) channel 1 binary prediction
        pred_mask_ch2: (H, W) channel 2 prediction mask
        pred_binary_ch2: (H, W) channel 2 binary prediction
    """
    # Prepare inputs for both channels
    prepared_ch1 = prepare_image_for_single_channel(image, c1_idx)
    prepared_ch2 = prepare_image_for_single_channel(image, c2_idx)
    
    # Run inference on both channels
    pred_mask_ch1, pred_binary_ch1 = predict_large_image(
        model=model_ch1,
        image=prepared_ch1,
        patch_size=patch_size,
        overlap=overlap,
        batch_size=batch_size,
        blend_mode=blend_mode,
        device=device,
        threshold=threshold
    )
    
    pred_mask_ch2, pred_binary_ch2 = predict_large_image(
        model=model_ch2,
        image=prepared_ch2,
        patch_size=patch_size,
        overlap=overlap,
        batch_size=batch_size,
        blend_mode=blend_mode,
        device=device,
        threshold=threshold
    )
    
    return pred_mask_ch1, pred_binary_ch1, pred_mask_ch2, pred_binary_ch2

# Run inference on all slices
all_pred_masks_ch1 = []
all_pred_binary_ch1 = []
all_pred_masks_ch2 = []
all_pred_binary_ch2 = []

print("Running inference on all slices...")
for i, raw_slice in tqdm(enumerate(all_slices), total=len(all_slices), desc="Inference"):
    pred_mask_ch1, pred_binary_ch1, pred_mask_ch2, pred_binary_ch2 = predict_dual_channel(
        model_ch1=model_ch1,
        model_ch2=model_ch2,
        image=raw_slice,
        patch_size=PATCH_SIZE,
        overlap=OVERLAP,
        batch_size=BATCH_SIZE,
        blend_mode=BLEND_MODE,
        device=DEVICE,
        threshold=THRESHOLD,
        c1_idx=C1_IDX,
        c2_idx=C2_IDX
    )
    
    all_pred_masks_ch1.append(pred_mask_ch1)
    all_pred_binary_ch1.append(pred_binary_ch1)
    all_pred_masks_ch2.append(pred_mask_ch2)
    all_pred_binary_ch2.append(pred_binary_ch2)

# Stack into arrays
all_pred_masks_ch1 = np.stack(all_pred_masks_ch1, axis=0)  # (N_total, H, W)
all_pred_binary_ch1 = np.stack(all_pred_binary_ch1, axis=0)
all_pred_masks_ch2 = np.stack(all_pred_masks_ch2, axis=0)
all_pred_binary_ch2 = np.stack(all_pred_binary_ch2, axis=0)

print("✓ All slices processed")
print(f"  Channel 1 mask shape: {all_pred_masks_ch1.shape}")
print(f"  Channel 2 mask shape: {all_pred_masks_ch2.shape}")


# ## 5. Organize Predictions by File


def organize_predictions_by_file(slice_index, predictions_ch1, predictions_ch2):
    """
    Reorganize flat prediction arrays into per-file dictionaries.
    
    Returns:
        pred_dict_ch1, pred_dict_ch2: dicts mapping file paths to (Z, H, W) arrays
    """
    pred_dict_ch1 = {}
    pred_dict_ch2 = {}
    
    for info, pred_ch1, pred_ch2 in zip(slice_index, predictions_ch1, predictions_ch2):
        path = info["path"]
        z = info["z"]
        Z = info["Z"]
        
        if path not in pred_dict_ch1:
            pred_dict_ch1[path] = [None] * Z
            pred_dict_ch2[path] = [None] * Z
        
        pred_dict_ch1[path][z] = pred_ch1
        pred_dict_ch2[path][z] = pred_ch2
    
    # Stack Z slices for each file
    for path in pred_dict_ch1:
        pred_dict_ch1[path] = np.stack(pred_dict_ch1[path], axis=0)
        pred_dict_ch2[path] = np.stack(pred_dict_ch2[path], axis=0)
    
    return pred_dict_ch1, pred_dict_ch2

pred_dict_ch1, pred_dict_ch2 = organize_predictions_by_file(
    slice_index, 
    all_pred_masks_ch1, 
    all_pred_masks_ch2
)

print("Organized predictions by file:")
for path in pred_dict_ch1:
    print(f"  {path.name}: Ch1={pred_dict_ch1[path].shape}, Ch2={pred_dict_ch2[path].shape}")


# ## 6. Create 6-Channel Output and Save


def load_original_image_full(tif_path):
    """
    Load full original image without slicing.
    
    Returns:
        img: (Z, C, Y, X) array
    """
    img = tf.imread(tif_path)
    
    if img.ndim == 3:  # (C, Y, X)
        img = img[np.newaxis, ...]
    elif img.ndim != 4:
        raise ValueError(f"Unexpected shape {img.shape}")
    
    return img

def create_6channel_output(original_img, pred_dict_ch1, pred_dict_ch2, tif_path):
    """
    Combine original 4 channels + 2 prediction channels = 6 channels.
    
    Args:
        original_img: (Z, C, Y, X) original image
        pred_dict_ch1: (Z, H, W) predictions for channel 1
        pred_dict_ch2: (Z, H, W) predictions for channel 2
        tif_path: Path object for reference
    
    Returns:
        output: (Z, 6, Y, X) combined image
    """
    Z, C, Y, X = original_img.shape
    
    # Ensure predictions match original dimensions
    assert pred_dict_ch1[tif_path].shape[0] == Z, f"Z mismatch for {tif_path}"
    assert pred_dict_ch2[tif_path].shape[0] == Z, f"Z mismatch for {tif_path}"
    
    output = np.zeros((Z, 6, Y, X), dtype=np.float32)
    
    # Channels 0-3: Original channels
    output[:, :4, :, :] = original_img.astype(np.float32)
    
    # Channel 4: Prediction from Channel 1
    output[:, 4, :, :] = pred_dict_ch1[tif_path].astype(np.float32)
    
    # Channel 5: Prediction from Channel 2
    output[:, 5, :, :] = pred_dict_ch2[tif_path].astype(np.float32)
    
    return output

# Create output directory
PRED_ROOT.mkdir(parents=True, exist_ok=True)

print(f"Saving 6-channel outputs to: {PRED_ROOT}")
for tif_path in pred_dict_ch1:
    # Load original image
    original_img = load_original_image_full(tif_path)
    
    # Create 6-channel output
    output_6channel = create_6channel_output(
        original_img, 
        pred_dict_ch1, 
        pred_dict_ch2, 
        tif_path
    )
    
    # Get relative path for mirrored directory structure
    rel_path = tif_path.relative_to(ROOT)
    rel_dir = rel_path.parent
    
    # Create output directory
    out_dir = PRED_ROOT / rel_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save 6-channel TIFF
    output_filename = tif_path.name.replace(".tif", "_6channel.tif")
    output_path = out_dir / output_filename
    
    tf.imwrite(output_path, output_6channel.astype(np.float32))
    
    print(f"✓ Saved {output_filename} ({output_6channel.shape}) to {out_dir}")

print("✓ All 6-channel outputs saved")


# ## 7. Visualization Helper


def visualize_dual_predictions(original_img, pred_ch1, pred_ch2, 
                               z_slice=None, figsize=(20, 12)):
    """
    Visualize original channels and both predictions.
    
    Args:
        original_img: (Z, C, Y, X) or (C, Y, X)
        pred_ch1: (Z, H, W) or (H, W)
        pred_ch2: (Z, H, W) or (H, W)
        z_slice: Which Z to visualize (if 4D), default=0
        figsize: Figure size
    """
    # Handle 4D input
    if original_img.ndim == 4:
        if z_slice is None:
            z_slice = 0
        original_img = original_img[z_slice]
        pred_ch1 = pred_ch1[z_slice] if pred_ch1.ndim == 3 else pred_ch1
        pred_ch2 = pred_ch2[z_slice] if pred_ch2.ndim == 3 else pred_ch2
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(f'Dual Channel Predictions (Z={z_slice})', fontsize=16)
    
    # Row 1: Original channels
    # Channel 1
    img_ch1 = original_img[C1_IDX]
    img_ch1_norm = (img_ch1 - img_ch1.min()) / (img_ch1.max() - img_ch1.min() + 1e-8)
    axes[0, 0].imshow(img_ch1_norm, cmap='viridis')
    axes[0, 0].set_title(f'Original Channel {C1_IDX}', fontsize=12)
    axes[0, 0].axis('off')
    
    # Channel 2
    img_ch2 = original_img[C2_IDX]
    img_ch2_norm = (img_ch2 - img_ch2.min()) / (img_ch2.max() - img_ch2.min() + 1e-8)
    axes[0, 1].imshow(img_ch2_norm, cmap='viridis')
    axes[0, 1].set_title(f'Original Channel {C2_IDX}', fontsize=12)
    axes[0, 1].axis('off')
    
    # Overlay (for reference)
    overlay = np.stack([img_ch1_norm, img_ch2_norm, np.zeros_like(img_ch1_norm)], axis=2)
    axes[0, 2].imshow(overlay)
    axes[0, 2].set_title('Channel Overlay (R=Ch1, G=Ch2)', fontsize=12)
    axes[0, 2].axis('off')
    
    # Row 2: Predictions
    axes[1, 0].imshow(pred_ch1, cmap='gray')
    axes[1, 0].set_title('Prediction - Channel 1', fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(pred_ch2, cmap='gray')
    axes[1, 1].set_title('Prediction - Channel 2', fontsize=12)
    axes[1, 1].axis('off')
    
    # Combined predictions
    combined_pred = np.stack([pred_ch1, pred_ch2, np.zeros_like(pred_ch1)], axis=2)
    axes[1, 2].imshow(combined_pred)
    axes[1, 2].set_title('Predictions Overlay (R=Ch1, G=Ch2)', fontsize=12)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

# Example visualization
if True:  # Set to True to visualize
    test_idx = 7
    visualize_dual_predictions(
        original_img=all_slices[test_idx],
        pred_ch1=all_pred_masks_ch1[test_idx],
        pred_ch2=all_pred_masks_ch2[test_idx],
        figsize=(18, 10)
    )


# ## 8. Summary and Statistics


def print_inference_summary(all_slices, pred_dict_ch1, pred_dict_ch2):
    """Print summary statistics about inference results."""
    print("\n" + "="*60)
    print("INFERENCE SUMMARY")
    print("="*60)
    
    print(f"\nInput Data:")
    print(f"  Total slices processed: {len(all_slices)}")
    print(f"  Slice shape (C, H, W): {all_slices.shape[1:]}")
    
    print(f"\nChannel 1 Predictions:")
    print(f"  Shape: {all_pred_masks_ch1.shape}")
    print(f"  Min value: {all_pred_masks_ch1.min():.4f}")
    print(f"  Max value: {all_pred_masks_ch1.max():.4f}")
    print(f"  Mean value: {all_pred_masks_ch1.mean():.4f}")
    
    print(f"\nChannel 2 Predictions:")
    print(f"  Shape: {all_pred_masks_ch2.shape}")
    print(f"  Min value: {all_pred_masks_ch2.min():.4f}")
    print(f"  Max value: {all_pred_masks_ch2.max():.4f}")
    print(f"  Mean value: {all_pred_masks_ch2.mean():.4f}")
    
    print(f"\nOutput Files:")
    print(f"  Saved location: {PRED_ROOT}")
    print(f"  Number of files: {len(pred_dict_ch1)}")
    print(f"  Format: *_6channel.tif (6 channels: 4 original + 2 predictions)")
    
    print("\n" + "="*60)

print_inference_summary(all_slices, pred_dict_ch1, pred_dict_ch2)


# ## Optional: Quality Checks


def validate_6channel_outputs(pred_root):
    """Validate that all 6-channel outputs were saved correctly."""
    print("\nValidating saved 6-channel outputs...")
    
    output_files = list(pred_root.rglob("*_6channel.tif"))
    print(f"Found {len(output_files)} output files")
    
    for i, fpath in enumerate(output_files[:3]):  # Check first 3
        img = tf.imread(fpath)
        print(f"\n  File {i+1}: {fpath.name}")
        print(f"    Shape: {img.shape}")
        print(f"    Dtype: {img.dtype}")
        if img.ndim == 4:
            Z, C, H, W = img.shape
            print(f"    Channels: {C} (expected: 6)")
            for c in range(C):
                ch_data = img[:, c, :, :]
                print(f"      Ch{c}: min={ch_data.min()}, max={ch_data.max()}, mean={ch_data.mean():.2f}")

validate_6channel_outputs(PRED_ROOT)

print("\n✓ Pipeline complete!")
