"""
Utilities for patchifying and reconstructing large images.
"""

import numpy as np
from typing import Tuple, List
import torch
from tqdm.notebook import tqdm


def patchify_image(
    image: np.ndarray,
    patch_size: int,
    overlap: int = 0
) -> Tuple[np.ndarray, dict]:
    """
    Split a large image into overlapping patches.

    Args:
        image: Input image (H, W, C) or (C, H, W)
        patch_size: Size of each patch
        overlap: Overlap between patches in pixels

    Returns:
        patches: Array of patches (N, C, patch_size, patch_size) or (N, patch_size, patch_size, C)
        metadata: Dictionary containing reconstruction info
    """
    # Handle both (H, W, C) and (C, H, W) formats
    if image.ndim == 3:
        if image.shape[0] <= 4:  # Likely (C, H, W)
            C, H, W = image.shape
            channels_first = True
        else:  # Likely (H, W, C)
            H, W, C = image.shape
            channels_first = False
    else:
        raise ValueError(f"Image must be 3D, got shape {image.shape}")

    stride = patch_size - overlap

    # Calculate number of patches
    n_patches_h = (H - overlap) // stride
    n_patches_w = (W - overlap) // stride

    # Add extra patches if image doesn't divide evenly
    if (H - overlap) % stride != 0:
        n_patches_h += 1
    if (W - overlap) % stride != 0:
        n_patches_w += 1

    patches = []
    positions = []

    for i in range(n_patches_h):
        for j in range(n_patches_w):
            # Calculate patch position
            start_h = i * stride
            start_w = j * stride

            # Adjust if patch goes beyond image bounds
            if start_h + patch_size > H:
                start_h = H - patch_size
            if start_w + patch_size > W:
                start_w = W - patch_size

            end_h = start_h + patch_size
            end_w = start_w + patch_size

            # Extract patch
            if channels_first:
                patch = image[:, start_h:end_h, start_w:end_w]
            else:
                patch = image[start_h:end_h, start_w:end_w, :]

            patches.append(patch)
            positions.append((start_h, start_w, end_h, end_w))

    patches = np.stack(patches, axis=0)

    metadata = {
        'original_shape': image.shape,
        'patch_size': patch_size,
        'overlap': overlap,
        'stride': stride,
        'n_patches_h': n_patches_h,
        'n_patches_w': n_patches_w,
        'positions': positions,
        'channels_first': channels_first
    }

    return patches, metadata


def unpatchify_image(
    patches: np.ndarray,
    metadata: dict,
    blend_mode: str = 'average'
) -> np.ndarray:
    """
    Reconstruct image from patches.

    Args:
        patches: Array of patches (N, C, H, W) or (N, H, W, C)
        metadata: Metadata from patchify_image
        blend_mode: How to blend overlapping regions ('average', 'max', 'first')

    Returns:
        Reconstructed image with same format as input
    """
    original_shape = metadata['original_shape']
    positions = metadata['positions']
    channels_first = metadata['channels_first']

    if channels_first:
        C, H, W = original_shape
        reconstructed = np.zeros((C, H, W), dtype=np.float32)
        counts = np.zeros((H, W), dtype=np.float32)
    else:
        H, W, C = original_shape
        reconstructed = np.zeros((H, W, C), dtype=np.float32)
        counts = np.zeros((H, W), dtype=np.float32)

    for patch, (start_h, start_w, end_h, end_w) in zip(patches, positions):
        if blend_mode == 'average':
            if channels_first:
                reconstructed[:, start_h:end_h, start_w:end_w] += patch
                counts[start_h:end_h, start_w:end_w] += 1
            else:
                reconstructed[start_h:end_h, start_w:end_w, :] += patch
                counts[start_h:end_h, start_w:end_w] += 1

        elif blend_mode == 'max':
            if channels_first:
                reconstructed[:, start_h:end_h, start_w:end_w] = np.maximum(
                    reconstructed[:, start_h:end_h, start_w:end_w],
                    patch
                )
            else:
                reconstructed[start_h:end_h, start_w:end_w, :] = np.maximum(
                    reconstructed[start_h:end_h, start_w:end_w, :],
                    patch
                )
            counts[start_h:end_h, start_w:end_w] = 1  # Don't average for max

        elif blend_mode == 'first':
            # Only use first patch (no blending)
            mask = counts[start_h:end_h, start_w:end_w] == 0
            if channels_first:
                reconstructed[:, start_h:end_h, start_w:end_w][:, mask] = patch[:, mask]
            else:
                reconstructed[start_h:end_h, start_w:end_w, :][mask] = patch[mask]
            counts[start_h:end_h, start_w:end_w][mask] = 1

        else:
            raise ValueError(f"Unknown blend_mode: {blend_mode}")

    # Average for 'average' mode
    if blend_mode == 'average':
        if channels_first:
            for c in range(C):
                reconstructed[c] /= (counts + 1e-8)
        else:
            reconstructed /= (counts[..., None] + 1e-8)

    return reconstructed.astype(patches.dtype)


def predict_on_patches(
    model: torch.nn.Module,
    patches: np.ndarray,
    batch_size: int = 8,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Run prediction on patches in batches.

    Args:
        model: PyTorch model
        patches: Patches (N, C, H, W)
        batch_size: Batch size
        device: Device to use

    Returns:
        Predictions (N, num_classes, H, W)
    """
    model.eval()
    predictions = []

    n_patches = len(patches)
    n_batches = (n_patches + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(n_batches), desc='Predicting patches', leave=False):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_patches)

            batch = patches[start_idx:end_idx]
            batch_tensor = torch.from_numpy(batch).float().to(device)

            # Predict
            outputs = model(batch_tensor)
            outputs = torch.sigmoid(outputs)

            predictions.append(outputs.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    return predictions


def predict_large_image(
    model: torch.nn.Module,
    image: np.ndarray,
    patch_size: int = 512,
    overlap: int = 64,
    batch_size: int = 8,
    blend_mode: str = 'average',
    device: str = 'cuda',
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict on a large image using patch-based processing.

    Args:
        model: PyTorch model
        image: Input image (C, H, W)
        patch_size: Size of patches
        overlap: Overlap between patches
        batch_size: Batch size for inference
        blend_mode: How to blend overlapping regions
        device: Device to use
        threshold: Threshold for binary prediction

    Returns:
        pred_mask: Predicted mask (H, W) - soft predictions
        pred_binary: Binary mask (H, W)
    """
    #print(f"Patchifying image with shape {image.shape}...")
    patches, metadata = patchify_image(image, patch_size, overlap)

    #print(f"Created {len(patches)} patches of size {patch_size}x{patch_size}")

    # Predict on patches
    #print("Running predictions...")
    pred_patches = predict_on_patches(model, patches, batch_size, device)

    # Remove channel dimension if single class
    if pred_patches.shape[1] == 1:
        pred_patches = pred_patches[:, 0, :, :]  # (N, H, W)

    # Reconstruct
    #print("Reconstructing prediction...")
    # Need to handle the channel dimension
    if pred_patches.ndim == 3:
        # Add channel dimension back for unpatchify
        pred_patches_4d = pred_patches[:, None, :, :]
        metadata_copy = metadata.copy()
        metadata_copy['channels_first'] = True
        metadata_copy['original_shape'] = (1, metadata['original_shape'][1], metadata['original_shape'][2])

        pred_mask = unpatchify_image(pred_patches_4d, metadata_copy, blend_mode)
        pred_mask = pred_mask[0]  # Remove channel dimension
    else:
        pred_mask = unpatchify_image(pred_patches, metadata, blend_mode)

    # Apply threshold
    pred_binary = (pred_mask > threshold).astype(np.uint8)

    return pred_mask, pred_binary
