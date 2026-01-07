"""
Dataset loader for .npz or .npy files containing multichannel cilia images in NCHW format.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Dict, Optional
from pathlib import Path


class CiliaNPZDataset(Dataset):
    """
    Dataset for loading multichannel cilia images from .npz or .npy files.

    Supports NCHW format where data is stored as (N, C, H, W).

    Args:
        images_npz: Path to .npz file containing images (NCHW format)
        masks_npz: Path to .npz file containing masks (NCHW format or NHW)
        channel_mode: One of 'c1_only', 'c2_only', 'dual'
        channel_indices: Tuple of (C0_idx, C1_idx) - which channels to use
        dual_mode: For 'dual' mode - 'average', 'zeros', or 'overlay'
        transform: Optional transforms
        image_size: Target image size
        images_key: Key name in npz file for images (default: 'images')
        masks_key: Key name in npz file for masks (default: 'masks')
     """
    def __init__(
        self,
        images_npz: str,
        masks_npz: str,
        channel_mode: str = 'c1_only',
        channel_indices: Tuple[int, int] = (0, 1),
        dual_mode: str = 'average',
        transform=None,
        image_size: int = 512,
        images_key: str = None,
        masks_key: str = None
    ):
        assert channel_mode in ['c1_only', 'c2_only', 'dual'], f"Invalid channel_mode: {channel_mode}"
        assert dual_mode in ['average', 'zeros', 'overlay'], f"Invalid dual_mode: {dual_mode}"

        self.channel_mode = channel_mode
        self.c1_idx, self.c2_idx = channel_indices
        self.dual_mode = dual_mode
        self.transform = transform
        self.image_size = image_size

        # -------------------------
        # Load images
        # -------------------------
        print(f"Loading images from: {images_npz}")

        if images_npz.endswith(".npy"):
            self.images = np.load(images_npz)
            print(f"  Loaded .npy images: shape {self.images.shape}, dtype {self.images.dtype}")
        else:
            images_data = np.load(images_npz)

            if images_key is None:
                for key in ["images", "arr_0", "data", "x"]:
                    if key in images_data:
                        images_key = key
                        break
                if images_key is None:
                    images_key = list(images_data.keys())[0]

            self.images = images_data[images_key]
            print(f"  Loaded images with key '{images_key}': shape {self.images.shape}, dtype {self.images.dtype}")

        # -------------------------
        # Load masks
        # -------------------------
        print(f"Loading masks from: {masks_npz}")

        if masks_npz.endswith(".npy"):
            self.masks = np.load(masks_npz)
            print(f"  Loaded .npy masks: shape {self.masks.shape}, dtype {self.masks.dtype}")
        else:
            masks_data = np.load(masks_npz)

            if masks_key is None:
                for key in ["masks", "arr_0", "labels", "y"]:
                    if key in masks_data:
                        masks_key = key
                        break
                if masks_key is None:
                    masks_key = list(masks_data.keys())[0]

            self.masks = masks_data[masks_key]
            print(f"  Loaded masks with key '{masks_key}': shape {self.masks.shape}, dtype {self.masks.dtype}")

        # -------------------------
        # Validate & normalize shapes
        # -------------------------
        assert self.images.ndim == 4, f"Images must be 4D (NCHW), got {self.images.shape}"

        # Masks: NHW → NCHW
        if self.masks.ndim == 3:
            self.masks = self.masks[:, None, :, :]
            print(f"  Expanded masks to NCHW: {self.masks.shape}")
        elif self.masks.ndim == 4:
            pass
        else:
            raise ValueError(f"Masks must be 3D or 4D, got {self.masks.shape}")

        # -------------------------
        # Sanity checks
        # -------------------------
        assert self.images.shape[0] == self.masks.shape[0], \
            f"Images ({self.images.shape[0]}) != masks ({self.masks.shape[0]})"

        num_channels = self.images.shape[1]
        assert self.c1_idx < num_channels, f"c1_idx {self.c1_idx} >= channels {num_channels}"
        assert self.c2_idx < num_channels, f"c2_idx {self.c2_idx} >= channels {num_channels}"

        # -------------------------
        # Summary
        # -------------------------
        print("\nDataset initialized:")
        print(f"  Samples       : {len(self)}")
        print(f"  Image shape   : {self.images.shape}")
        print(f"  Mask shape    : {self.masks.shape}")
        print(f"  Channel mode  : {self.channel_mode}")
        print(f"  Dual mode     : {self.dual_mode}")
        print(f"  Using channels: C{self.c1_idx}, C{self.c2_idx}")
        print(f"  Target size   : {self.image_size}x{self.image_size}")

    def __len__(self):
        return self.images.shape[0]

    def _normalize_channel(self, channel: np.ndarray) -> np.ndarray:
        """Normalize a single channel to [0, 1] using percentile clipping."""
        p_low = np.percentile(channel, 1)
        p_high = np.percentile(channel, 99)
        normalized = np.clip(channel, p_low, p_high)
        normalized = (normalized - p_low) / (p_high - p_low + 1e-8)
        return normalized

    def _prepare_channels(self, image: np.ndarray) -> np.ndarray:
        """
        Prepare RGB channels based on channel_mode.

        Args:
            image: Input multichannel image (C, H, W)

        Returns:
            RGB image (H, W, 3) uint8
        """
        if self.channel_mode == 'c1_only':
            # C0 replicated 3 times
            c1 = image[self.c1_idx]  # (H, W)
            c1_norm = self._normalize_channel(c1)
            rgb = np.stack([c1_norm] * 3, axis=-1)

        elif self.channel_mode == 'c2_only':
            # C1 replicated 3 times
            c2 = image[self.c2_idx]  # (H, W)
            c2_norm = self._normalize_channel(c2)
            rgb = np.stack([c2_norm] * 3, axis=-1)

        elif self.channel_mode == 'dual':
            # Both channels
            c1 = self._normalize_channel(image[self.c1_idx])
            c2 = self._normalize_channel(image[self.c2_idx])

            if self.dual_mode == 'average':
                c3 = (c1 + c2) / 2.0
            elif self.dual_mode == 'zeros':
                c3 = np.zeros_like(c1)
            elif self.dual_mode == 'overlay':
                c3 = np.maximum(c1, c2)
            else:
                raise ValueError(f"Unknown dual_mode: {self.dual_mode}")

            rgb = np.stack([c1, c2, c3], axis=-1)

        else:
            raise ValueError(f"Unknown channel_mode: {self.channel_mode}")

        return (rgb * 255).astype(np.uint8)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training sample.

        Returns:
            Dictionary with 'image' (3, H, W) and 'mask' (1, H, W)
        """
        # Get image and mask (NCHW format)
        image = self.images[idx]  # (C, H, W)
        mask = self.masks[idx]    # (1, H, W) or (H, W)

        # Ensure mask is 2D
        mask = self.masks[idx]  # (C, H, W)

        if self.channel_mode == "c1_only":
            mask = mask[self.c1_idx]

        elif self.channel_mode == "c2_only":
            mask = mask[self.c2_idx]

        elif self.channel_mode == "dual":
            # choose a policy (most common options below)
            mask = np.maximum(mask[self.c1_idx], mask[self.c2_idx])
            # OR: mask = mask[self.c1_idx]
            # OR: mask = mask[self.c2_idx]

        # Prepare RGB channels - need to transpose to (H, W, C) first
        image_hwc = np.transpose(image, (1, 2, 0))  # (C, H, W) -> (H, W, C)

        # Now prepare channels (this returns H, W, 3)
        # Actually, _prepare_channels expects (C, H, W)
        rgb_image = self._prepare_channels(image)  # (H, W, 3)

        # Ensure mask is binary
        if mask.dtype == np.float32 or mask.dtype == np.float64:
            mask = (mask > 0.5).astype(np.uint8)
        else:
            mask = (mask > 0).astype(np.uint8)

        # Resize if needed
        if rgb_image.shape[0] != self.image_size or rgb_image.shape[1] != self.image_size:
            from skimage.transform import resize
            rgb_image = resize(
                rgb_image,
                (self.image_size, self.image_size),
                preserve_range=True,
                anti_aliasing=True,
                order=1  # Bilinear
            ).astype(np.uint8)
            mask = resize(
                mask,
                (self.image_size, self.image_size),
                order=0,  # Nearest neighbor for masks
                preserve_range=True,
                anti_aliasing=False
            ).astype(np.uint8)

        # Apply transforms if any
        if self.transform:
            transformed = self.transform(image=rgb_image, mask=mask)
            rgb_image = transformed['image']
            mask = transformed['mask']

        # Convert to tensors (C, H, W)
        image_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'index': idx
        }


def prepare_npz_data_splits(
    images_npz: str,
    masks_npz: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
    images_key: str = None,
    masks_key: str = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare train/val/test splits from .npz or .npy files.

    Returns:
        Tuple of (train_images, train_masks, val_images, val_masks, test_images, test_masks)
        Each is a numpy array in NCHW format
    """
    import random

    # Load data
    print(f"Loading data for splitting...")

    # Handle .npy files
    if str(images_npz).endswith('.npy'):
        images = np.load(images_npz)
        print(f"  Loaded images from .npy: {images.shape}")
    else:
        images_data = np.load(images_npz)
        # Auto-detect keys
        if images_key is None:
            for key in ['images', 'arr_0', 'data', 'x']:
                if key in images_data:
                    images_key = key
                    break
            if images_key is None:
                images_key = list(images_data.keys())[0]
        images = images_data[images_key]
        print(f"  Loaded images from .npz['{images_key}']: {images.shape}")

    if str(masks_npz).endswith('.npy'):
        masks = np.load(masks_npz)
        print(f"  Loaded masks from .npy: {masks.shape}")
    else:
        masks_data = np.load(masks_npz)
        if masks_key is None:
            for key in ['masks', 'arr_0', 'labels', 'y']:
                if key in masks_data:
                    masks_key = key
                    break
            if masks_key is None:
                masks_key = list(masks_data.keys())[0]
        masks = masks_data[masks_key]
        print(f"  Loaded masks from .npz['{masks_key}']: {masks.shape}")

    n_total = images.shape[0]

    # Create indices and shuffle
    random.seed(random_seed)
    np.random.seed(random_seed)
    indices = np.arange(n_total)
    np.random.shuffle(indices)

    # Split indices
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    # Split data
    train_images = images[train_idx]
    train_masks = masks[train_idx]

    val_images = images[val_idx]
    val_masks = masks[val_idx]

    test_images = images[test_idx]
    test_masks = masks[test_idx]

    print(f"\nSplit completed:")
    print(f"  Train: {len(train_images)} samples")
    print(f"  Val: {len(val_images)} samples")
    print(f"  Test: {len(test_images)} samples")

    return train_images, train_masks, val_images, val_masks, test_images, test_masks
