"""Utility functions for SAM2 cilia segmentation."""

import numpy as np
from skimage import measure, morphology
from tqdm import tqdm
import warnings


def binarize_and_correct_tiff(predictions, threshold=0.5):
    """
    Binarize predictions and correct any issues.

    Args:
        predictions: Input predictions array (can be float or binary)
        threshold: Threshold for binarization if predictions are float

    Returns:
        tuple: (binarized_predictions, info_dict)
    """
    # Convert to binary if needed
    if predictions.dtype == np.float32 or predictions.dtype == np.float64:
        binarized = (predictions > threshold).astype(np.uint8)
    else:
        binarized = (predictions > 0).astype(np.uint8)

    info = {
        'shape': binarized.shape,
        'dtype': binarized.dtype,
        'unique_values': np.unique(binarized)
    }

    return binarized, info


def keep_connected_blobs(predictions, target_channel, reference_channel, min_overlap=0.1):
    """
    Keep only blobs in target_channel that overlap with blobs in reference_channel.

    Args:
        predictions: Multi-channel prediction array (N, H, W, C)
        target_channel: Channel index to filter
        reference_channel: Channel index to use as reference
        min_overlap: Minimum overlap ratio to keep a blob

    Returns:
        Modified predictions array
    """
    output = predictions.copy()

    for i in tqdm(range(predictions.shape[0]), desc='Filtering blobs'):
        target = predictions[i, :, :, target_channel]
        reference = predictions[i, :, :, reference_channel]

        # Label connected components in target channel
        labeled_target = measure.label(target, connectivity=2)

        # Label connected components in reference channel
        labeled_reference = measure.label(reference, connectivity=2)

        # Create mask for which target blobs to keep
        keep_mask = np.zeros_like(target, dtype=bool)

        # Check each blob in target channel
        for region in measure.regionprops(labeled_target):
            blob_mask = labeled_target == region.label

            # Check overlap with any blob in reference channel
            overlap = np.logical_and(blob_mask, reference > 0)
            overlap_ratio = np.sum(overlap) / region.area

            if overlap_ratio >= min_overlap:
                keep_mask[blob_mask] = True

        # Update target channel
        output[i, :, :, target_channel] = target * keep_mask

    return output


def intersect_channels(predictions, channel1, channel2):
    """
    Compute intersection of two channels.

    Args:
        predictions: Multi-channel prediction array (N, H, W, C)
        channel1: First channel index
        channel2: Second channel index

    Returns:
        Intersection array (N, H, W)
    """
    c1 = predictions[..., channel1]
    c2 = predictions[..., channel2]

    intersection = np.logical_and(c1 > 0, c2 > 0).astype(np.uint8)

    return intersection


def remove_small_objects(image, min_size=64):
    """
    Remove small connected components.

    Args:
        image: Binary image (H, W) or (N, H, W)
        min_size: Minimum object size in pixels

    Returns:
        Cleaned image
    """
    if image.ndim == 2:
        return morphology.remove_small_objects(image.astype(bool), min_size=min_size).astype(np.uint8)
    else:
        output = np.zeros_like(image)
        for i in range(image.shape[0]):
            output[i] = morphology.remove_small_objects(
                image[i].astype(bool), min_size=min_size
            ).astype(np.uint8)
        return output


def normalize_channel(channel, percentile_low=1, percentile_high=99):
    """
    Normalize a channel to [0, 1] range using percentile clipping.

    Args:
        channel: Input channel array
        percentile_low: Lower percentile for clipping
        percentile_high: Upper percentile for clipping

    Returns:
        Normalized channel
    """
    p_low = np.percentile(channel, percentile_low)
    p_high = np.percentile(channel, percentile_high)

    normalized = np.clip(channel, p_low, p_high)
    normalized = (normalized - p_low) / (p_high - p_low + 1e-8)

    return normalized


def prepare_sam2_input(image, channel_mode='single', channel_idx=None):
    """
    Prepare multichannel image for SAM2 input (expects RGB).

    Args:
        image: Input image (H, W, C)
        channel_mode: One of 'single', 'dual', 'overlay'
        channel_idx: Channel index (for 'single' mode) or tuple of indices (for 'dual' mode)

    Returns:
        RGB image (H, W, 3) suitable for SAM2
    """
    if channel_mode == 'single':
        # Replicate single channel to RGB
        if channel_idx is None:
            raise ValueError("channel_idx must be specified for 'single' mode")
        channel = image[:, :, channel_idx]
        channel_norm = normalize_channel(channel)
        rgb = np.stack([channel_norm] * 3, axis=-1)

    elif channel_mode == 'dual':
        # Use two channels + average or zero
        if channel_idx is None or len(channel_idx) != 2:
            raise ValueError("channel_idx must be a tuple of 2 indices for 'dual' mode")
        c1 = normalize_channel(image[:, :, channel_idx[0]])
        c2 = normalize_channel(image[:, :, channel_idx[1]])
        c3 = (c1 + c2) / 2.0  # Average
        rgb = np.stack([c1, c2, c3], axis=-1)

    elif channel_mode == 'overlay':
        # Overlay mode - combine channels
        if channel_idx is None or len(channel_idx) != 2:
            raise ValueError("channel_idx must be a tuple of 2 indices for 'overlay' mode")
        c1 = normalize_channel(image[:, :, channel_idx[0]])
        c2 = normalize_channel(image[:, :, channel_idx[1]])
        overlay = np.maximum(c1, c2)
        rgb = np.stack([overlay] * 3, axis=-1)

    else:
        raise ValueError(f"Unknown channel_mode: {channel_mode}")

    return (rgb * 255).astype(np.uint8)
