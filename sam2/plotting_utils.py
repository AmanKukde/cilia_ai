"""Plotting utilities for SAM2 cilia segmentation visualization."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from skimage import measure


def plot_input_and_predictions_with_outline(
    idx,
    imgs,
    predictions,
    predictions_cleaned,
    channels,
    scale='minmax',
    linewidth=1,
    overlay_alpha=0.5
):
    """
    Plot input images and predictions with outlines.

    Args:
        idx: Index of the image to plot
        imgs: Input images array (N, H, W, C)
        predictions: Raw predictions array (N, H, W, C)
        predictions_cleaned: Cleaned predictions array (N, H, W, C)
        channels: List of channel indices to plot
        scale: Scaling method ('minmax', 'percentile')
        linewidth: Width of outline
        overlay_alpha: Alpha for overlay
    """
    n_channels = len(channels)
    fig, axes = plt.subplots(3, n_channels, figsize=(4 * n_channels, 12))

    if n_channels == 1:
        axes = axes[:, None]

    for i, ch in enumerate(channels):
        # Input image
        img = imgs[idx, :, :, ch]
        if scale == 'minmax':
            img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
        else:
            p1, p99 = np.percentile(img, [1, 99])
            img_norm = np.clip((img - p1) / (p99 - p1 + 1e-8), 0, 1)

        axes[0, i].imshow(img_norm, cmap='gray')
        axes[0, i].set_title(f'Input Channel {ch}')
        axes[0, i].axis('off')

        # Raw predictions with outline
        pred = predictions[idx, :, :, ch]
        axes[1, i].imshow(img_norm, cmap='gray')

        # Overlay prediction
        pred_rgba = np.zeros((*pred.shape, 4))
        pred_rgba[pred > 0] = [1, 0, 0, overlay_alpha]
        axes[1, i].imshow(pred_rgba)

        # Add contours
        contours = measure.find_contours(pred, 0.5)
        for contour in contours:
            axes[1, i].plot(contour[:, 1], contour[:, 0], 'r-', linewidth=linewidth)

        axes[1, i].set_title(f'Raw Prediction Ch {ch}')
        axes[1, i].axis('off')

        # Cleaned predictions with outline
        pred_clean = predictions_cleaned[idx, :, :, ch]
        axes[2, i].imshow(img_norm, cmap='gray')

        # Overlay prediction
        pred_clean_rgba = np.zeros((*pred_clean.shape, 4))
        pred_clean_rgba[pred_clean > 0] = [0, 1, 0, overlay_alpha]
        axes[2, i].imshow(pred_clean_rgba)

        # Add contours
        contours = measure.find_contours(pred_clean, 0.5)
        for contour in contours:
            axes[2, i].plot(contour[:, 1], contour[:, 0], 'g-', linewidth=linewidth)

        axes[2, i].set_title(f'Cleaned Prediction Ch {ch}')
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()


def plot_channels_composite(img, channels, colors=None, title="Composite", ax=None):
    """
    Plot composite of multiple channels with different colors.

    Args:
        img: Input image (H, W, C)
        channels: List of channel indices to display
        colors: List of colors ('R', 'G', 'B', 'M', 'C', 'Y')
        title: Plot title
        ax: Matplotlib axis (if None, uses current axis)
    """
    H, W = img.shape[:2]
    composite = np.zeros((H, W, 3), dtype=float)

    # Color mapping
    rgb_map = {
        'R': [1, 0, 0],
        'G': [0, 1, 0],
        'B': [0, 0, 1],
        'M': [1, 0, 1],  # Magenta
        'C': [0, 1, 1],  # Cyan
        'Y': [1, 1, 0],  # Yellow
    }

    # Default colors
    default_colors = ['B', 'G', 'R', 'M', 'C', 'Y']
    if colors is None:
        colors = [default_colors[i % len(default_colors)] for i in range(len(channels))]

    for ch, col in zip(channels, colors):
        channel_data = img[:, :, ch].astype(float)
        if channel_data.max() > 0:
            channel_data /= channel_data.max()
        composite += channel_data[:, :, None] * np.array(rgb_map[col])

    composite = np.clip(composite, 0, 1)

    if ax is None:
        plt.imshow(composite)
        plt.title(title)
        plt.axis('off')
    else:
        ax.imshow(composite)
        ax.set_title(title)
        ax.axis('off')


def visualize_training_batch(images, masks, predictions=None, num_samples=4):
    """
    Visualize a training batch with images, masks, and optionally predictions.

    Args:
        images: Batch of images (B, 3, H, W) or (B, H, W, 3)
        masks: Batch of masks (B, 1, H, W) or (B, H, W, 1) or (B, H, W)
        predictions: Optional batch of predictions
        num_samples: Number of samples to visualize
    """
    num_samples = min(num_samples, len(images))

    # Handle different input formats
    if images.ndim == 4 and images.shape[1] == 3:
        # (B, 3, H, W) -> (B, H, W, 3)
        images = np.transpose(images, (0, 2, 3, 1))

    if masks.ndim == 4 and masks.shape[1] == 1:
        # (B, 1, H, W) -> (B, H, W)
        masks = masks[:, 0, :, :]
    elif masks.ndim == 4 and masks.shape[-1] == 1:
        # (B, H, W, 1) -> (B, H, W)
        masks = masks[:, :, :, 0]

    if predictions is not None:
        if predictions.ndim == 4 and predictions.shape[1] == 1:
            predictions = predictions[:, 0, :, :]
        elif predictions.ndim == 4 and predictions.shape[-1] == 1:
            predictions = predictions[:, :, :, 0]

        ncols = 3
    else:
        ncols = 2

    fig, axes = plt.subplots(num_samples, ncols, figsize=(4 * ncols, 4 * num_samples))

    if num_samples == 1:
        axes = axes[None, :]

    for i in range(num_samples):
        # Input image
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title(f'Input {i}')
        axes[i, 0].axis('off')

        # Ground truth mask
        axes[i, 1].imshow(masks[i], cmap='gray')
        axes[i, 1].set_title(f'Ground Truth {i}')
        axes[i, 1].axis('off')

        # Prediction (if available)
        if predictions is not None:
            axes[i, 2].imshow(predictions[i], cmap='gray')
            axes[i, 2].set_title(f'Prediction {i}')
            axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()


def plot_training_metrics(metrics_dict, save_path=None):
    """
    Plot training metrics over epochs.

    Args:
        metrics_dict: Dictionary with keys like 'train_loss', 'val_loss', 'train_iou', 'val_iou'
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot losses
    if 'train_loss' in metrics_dict:
        axes[0].plot(metrics_dict['train_loss'], label='Train Loss', marker='o')
    if 'val_loss' in metrics_dict:
        axes[0].plot(metrics_dict['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot IoU/metrics
    if 'train_iou' in metrics_dict:
        axes[1].plot(metrics_dict['train_iou'], label='Train IoU', marker='o')
    if 'val_iou' in metrics_dict:
        axes[1].plot(metrics_dict['val_iou'], label='Val IoU', marker='s')
    if 'train_dice' in metrics_dict:
        axes[1].plot(metrics_dict['train_dice'], label='Train Dice', marker='^')
    if 'val_dice' in metrics_dict:
        axes[1].plot(metrics_dict['val_dice'], label='Val Dice', marker='v')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Metric Value')
    axes[1].set_title('Training and Validation Metrics')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
