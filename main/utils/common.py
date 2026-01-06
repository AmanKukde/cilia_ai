
from tqdm.notebook import tqdm
import numpy as np
import cv2
from tqdm import tqdm

from scipy.ndimage import label, generate_binary_structure, binary_dilation

import numpy as np

def intersect_channels(
    binarized_tiff: np.ndarray, 
) -> np.ndarray:
    """
    Compute pixel-wise intersection of two binary channels for a batch of images.

    Parameters
    ----------
    binarized_tiff : np.ndarray
        Binary TIFF stack of shape (N, H, W, C), with 0/1 values.
    channel1 : int
        First channel to intersect.
    channel2 : int
        Second channel to intersect.

    Returns
    -------
    intersection : np.ndarray
        Binary array of shape (N, H, W), where pixels are 1 if both channels
        have foreground at that location, else 0.
    """
    return ((binarized_tiff[..., 1] & binarized_tiff[..., 2]) | (binarized_tiff[..., 1] & binarized_tiff[..., 3]) ).astype(np.uint8) 

def keep_connected_blobs(
    binarized_tiff: np.ndarray, 
    reference_channel: int = 2, 
    target_channel: int = 3
) -> np.ndarray:
    """
    Filters connected components in a target channel of a batch of binary images 
    based on overlap with a reference channel.
    
    Each connected component (blob) in the target channel is kept only if it 
    intersects with at least one foreground pixel in the reference channel.

    Parameters
    ----------
    binarized_tiff : np.ndarray
        Binary TIFF stack of shape (N, H, W, C), with 0/1 values.
        N: number of images, H/W: height/width, C: number of channels.
    reference_channel : int, default=2
        The channel used as the reference (B). Blobs in the target channel are kept 
        only if they intersect with this channel.
    target_channel : int, default=3
        The channel containing the blobs to filter (A).

    Returns
    -------
    filtered_tiff : np.ndarray
        Copy of `binarized_tiff` with the target channel filtered. Other channels remain unchanged.
    
    Notes
    -----
    - This function does not modify the input TIFF in-place.
    - Connected components are determined per image using OpenCV's `connectedComponents`.
    - Only components in the target channel that overlap with the reference channel are retained.
    """
    N, H, W, C = binarized_tiff.shape
    filtered_tiff = binarized_tiff.copy()  # avoid in-place modification

    def filter_single_image(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Filter connected components in A that intersect with B."""
        num_labels, labels = cv2.connectedComponents(A)
        filtered_A = np.zeros_like(A)
        for i in range(1, num_labels):
            mask = (labels == i)
            if np.any(mask & B):
                filtered_A[mask] = 1
        return filtered_A

    # Process each image in the batch
    filtered_As = [
        filter_single_image(
            binarized_tiff[n, ..., target_channel], 
            binarized_tiff[n, ..., reference_channel]
        )
        for n in tqdm(range(N))
    ]

    filtered_tiff[..., target_channel] = np.stack(filtered_As, axis=0)
    return filtered_tiff


def keep_connected_blobs_union(
    binarized_tiff: np.ndarray, 
    reference_channels: tuple[int, int] = (2, 1), 
    target_channel: int = 3
) -> np.ndarray:
    """
    Filters connected components in a target channel of a batch of binary images 
    based on overlap with two reference channels.

    Each connected component (blob) in the target channel is kept if it 
    intersects with at least one foreground pixel in **either** of the reference channels.

    Parameters
    ----------
    binarized_tiff : np.ndarray
        Binary TIFF stack of shape (N, H, W, C), with 0/1 values.
        N: number of images, H/W: height/width, C: number of channels.
    reference_channels : tuple[int, int], default=(2, 1)
        The two channels used as references. Blobs in the target channel are kept 
        if they intersect with either channel.
    target_channel : int, default=3
        The channel containing the blobs to filter.

    Returns
    -------
    filtered_tiff : np.ndarray
        Copy of `binarized_tiff` with the target channel filtered. Other channels remain unchanged.
    """
    N, H, W, C = binarized_tiff.shape
    filtered_tiff = binarized_tiff.copy()  # avoid in-place modification

    def filter_single_image(A: np.ndarray, B1: np.ndarray, B2: np.ndarray) -> np.ndarray:
        """Filter connected components in A that intersect with B1 or B2."""
        num_labels, labels = cv2.connectedComponents(A)
        filtered_A = np.zeros_like(A)
        union_B = B1 | B2  # Union of the two reference channels
        for i in range(1, num_labels):
            mask = (labels == i)
            if np.any(mask & union_B):
                filtered_A[mask] = 1
        return filtered_A

    # Process each image in the batch
    filtered_As = [
        filter_single_image(
            binarized_tiff[n, ..., target_channel], 
            binarized_tiff[n, ..., reference_channels[0]],
            binarized_tiff[n, ..., reference_channels[1]]
        )
        for n in tqdm(range(N))
    ]

    filtered_tiff[..., target_channel] = np.stack(filtered_As, axis=0)
    return filtered_tiff


def binarize_and_correct_tiff(tiff, threshold=0):
    """
    Binarizes and corrects inversion for all channels in a TIFF of shape (N, H, W, C).
    
    Parameters:
        tiff: np.array of shape (N, H, W, C)
        threshold: values > threshold are considered foreground
    
    Returns:
        binarized: np.array of shape (N, H, W, C) with 0/1
        inverted_channels: same shape, boolean array indicating which channels were inverted
    """
    binarized = (tiff > threshold).astype(np.uint8)
    inverted_channels = np.zeros_like(binarized, dtype=bool)
    
    # Determine inversion per channel per image
    mean_values = binarized.mean(axis=(1, 2))  # shape (N, C)
    invert_mask = mean_values > 0.5  # True if majority is 1
    
    # Apply inversion
    for n in range(tiff.shape[0]):
        for c in range(tiff.shape[3]):
            if invert_mask[n, c]:
                binarized[n, ..., c] = 1 - binarized[n, ..., c]
                inverted_channels[n, ..., c] = True
    
    return binarized, inverted_channels


def keep_touching_blobs(stack, channels=[1, 2, 3]):
    """
    Keep blobs in the specified channels that touch other channels (directly or transitively).
    
    Args:
        stack: list or array of shape (num_images, H, W, C), binary images
        channels: list of channel indices (0-based) to consider

    Returns:
        list of np.ndarray: same shape as input, with only touching blobs preserved
    """
    output_stack = []
    
    for image in tqdm(stack):
        image = image.copy().astype(np.uint8)
        H, W, C = image.shape
        
        # Step 1: label blobs per channel
        labels = [label(image[..., ch])[0] for ch in channels]
        
        # Step 2: keep blobs per channel if they touch any other channel
        output = np.zeros_like(image)
        for idx, ch in enumerate(channels):
            # Combine all other channels
            other_mask = np.zeros((H, W), dtype=bool)
            for j, other_ch in enumerate(channels):
                if j != idx:
                    other_mask |= labels[j] > 0
            
            # Iteratively grow the current channel blobs to see if they touch others
            blobs = labels[idx] > 0
            touched = np.zeros_like(blobs, dtype=bool)
            
            while True:
                prev_touched = touched.copy()
                dilated = binary_dilation(blobs, structure=generate_binary_structure(2,1))
                touched |= dilated & other_mask
                if np.array_equal(prev_touched, touched):
                    break
            
            # Keep only blobs that touched other channels
            output[..., ch] = blobs & touched
        
        output_stack.append(output)
    
    return output_stack
