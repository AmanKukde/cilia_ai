"""
Visualization utilities for model predictions on validation data.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))



import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_npz import CiliaNPZDataset
from models.model_factory import ModelFactory


def load_trained_model(
    checkpoint_path: str,
    architecture: str,
    device: str = 'cuda'
) -> torch.nn.Module:
    """Load a trained model from checkpoint."""
    model = ModelFactory.create_model(
        architecture,
        num_classes=1,
        pretrained=False
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model


def predict_batch(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: str = 'cuda'
) -> np.ndarray:
    """Run prediction on a batch of images."""
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)

        # Apply sigmoid and threshold
        predictions = torch.sigmoid(outputs)
        predictions = (predictions > 0.5).float()

        return predictions.cpu().numpy()


def visualize_predictions(
    images: np.ndarray,
    gt_masks: np.ndarray,
    pred_masks: np.ndarray,
    indices: List[int],
    save_path: Optional[str] = None,
    max_samples: int = 8
):
    """
    Visualize predictions vs ground truth.

    Args:
        images: (B, 3, H, W) RGB images
        gt_masks: (B, 1, H, W) ground truth masks
        pred_masks: (B, 1, H, W) predicted masks
        indices: Sample indices
        save_path: Path to save figure
        max_samples: Maximum number of samples to show
    """
    n_samples = min(len(images), max_samples)

    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))

    if n_samples == 1:
        axes = axes[None, :]

    for i in range(n_samples):
        # Convert image from (3, H, W) to (H, W, 3)
        img = images[i].transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        gt = gt_masks[i, 0]
        pred = pred_masks[i, 0]

        # Input image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Input #{indices[i]}')
        axes[i, 0].axis('off')

        # Ground truth
        axes[i, 1].imshow(gt, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        # Prediction
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')

        # Overlay
        overlay = img.copy()
        # GT in green, Pred in red
        overlay_mask = np.zeros_like(overlay)
        overlay_mask[..., 1] = gt  # Green for GT
        overlay_mask[..., 0] = pred  # Red for pred
        overlay = 0.7 * overlay + 0.3 * overlay_mask

        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('Overlay (GT=Green, Pred=Red)')
        axes[i, 3].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")

    plt.show()


def compute_metrics(
    pred_masks: np.ndarray,
    gt_masks: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Compute segmentation metrics."""
    pred_binary = (pred_masks > threshold).astype(np.float32)
    gt_binary = gt_masks.astype(np.float32)

    # IoU
    intersection = (pred_binary * gt_binary).sum()
    union = pred_binary.sum() + gt_binary.sum() - intersection
    iou = intersection / (union + 1e-8)

    # Dice
    dice = (2 * intersection) / (pred_binary.sum() + gt_binary.sum() + 1e-8)

    # Precision and Recall
    tp = (pred_binary * gt_binary).sum()
    fp = (pred_binary * (1 - gt_binary)).sum()
    fn = ((1 - pred_binary) * gt_binary).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    return {
        'iou': float(iou),
        'dice': float(dice),
        'precision': float(precision),
        'recall': float(recall)
    }


def visualize_validation_predictions(
    model_checkpoint: str,
    architecture: str,
    val_images: np.ndarray,
    val_masks: np.ndarray,
    channel_mode: str,
    channel_indices: Tuple[int, int] = (0, 1),
    dual_mode: str = 'average',
    image_size: int = 512,
    batch_size: int = 8,
    num_viz_samples: int = 8,
    output_dir: str = './visualizations',
    device: str = 'cuda'
):
    """
    Complete visualization pipeline for validation data.

    Args:
        model_checkpoint: Path to model checkpoint
        architecture: Model architecture name
        val_images: Validation images (N, C, H, W)
        val_masks: Validation masks (N, C, H, W) or (N, H, W)
        channel_mode: 'c1_only', 'c2_only', or 'dual'
        channel_indices: (c1_idx, c2_idx)
        dual_mode: 'average', 'zeros', or 'overlay'
        image_size: Image size for model
        batch_size: Batch size for inference
        num_viz_samples: Number of samples to visualize
        output_dir: Output directory
        device: Device to use
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from: {model_checkpoint}")
    model = load_trained_model(model_checkpoint, architecture, device)

    # Create dataset
    print("Creating validation dataset...")
    val_dataset = CiliaNPZDataset.__new__(CiliaNPZDataset)
    val_dataset.images = val_images
    val_dataset.masks = val_masks
    val_dataset.channel_mode = channel_mode
    val_dataset.c1_idx, val_dataset.c2_idx = channel_indices
    val_dataset.dual_mode = dual_mode
    val_dataset.transform = None
    val_dataset.image_size = image_size

    # Add methods
    val_dataset.__len__ = lambda self: self.images.shape[0]
    val_dataset._normalize_channel = CiliaNPZDataset._normalize_channel.__get__(val_dataset, CiliaNPZDataset)
    val_dataset._prepare_channels = CiliaNPZDataset._prepare_channels.__get__(val_dataset, CiliaNPZDataset)
    val_dataset.__getitem__ = CiliaNPZDataset.__getitem__.__get__(val_dataset, CiliaNPZDataset)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"Running inference on {len(val_dataset)} samples...")

    all_images = []
    all_gt_masks = []
    all_pred_masks = []
    all_indices = []
    all_metrics = []

    for batch in tqdm(val_loader):
        images = batch['image']
        gt_masks = batch['mask']
        indices = batch['index']

        # Predict
        pred_masks = predict_batch(model, images, device)

        # Store
        all_images.append(images.numpy())
        all_gt_masks.append(gt_masks.numpy())
        all_pred_masks.append(pred_masks)
        all_indices.extend(indices.tolist())

        # Compute metrics for each sample
        for i in range(len(images)):
            metrics = compute_metrics(pred_masks[i], gt_masks[i].numpy())
            all_metrics.append(metrics)

    # Concatenate
    all_images = np.concatenate(all_images, axis=0)
    all_gt_masks = np.concatenate(all_gt_masks, axis=0)
    all_pred_masks = np.concatenate(all_pred_masks, axis=0)

    # Compute overall metrics
    overall_metrics = compute_metrics(all_pred_masks, all_gt_masks)

    print("\n" + "="*60)
    print("Validation Metrics:")
    print("="*60)
    print(f"IoU:       {overall_metrics['iou']:.4f}")
    print(f"Dice:      {overall_metrics['dice']:.4f}")
    print(f"Precision: {overall_metrics['precision']:.4f}")
    print(f"Recall:    {overall_metrics['recall']:.4f}")
    print("="*60)

    # Save metrics
    with open(output_path / 'validation_metrics.json', 'w') as f:
        json.dump(overall_metrics, f, indent=2)

    # Visualize samples
    print(f"\nVisualizing {num_viz_samples} samples...")
    visualize_predictions(
        all_images[:num_viz_samples],
        all_gt_masks[:num_viz_samples],
        all_pred_masks[:num_viz_samples],
        all_indices[:num_viz_samples],
        save_path=str(output_path / 'validation_predictions.png'),
        max_samples=num_viz_samples
    )

    print(f"\nVisualization complete! Saved to: {output_path}")

    return overall_metrics, all_pred_masks


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize model predictions on validation data')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--architecture', type=str, required=True, help='Model architecture')
    parser.add_argument('--images_npz', type=str, required=True, help='Validation images .npz/.npy')
    parser.add_argument('--masks_npz', type=str, required=True, help='Validation masks .npz/.npy')
    parser.add_argument('--channel_mode', type=str, default='c1_only', choices=['c1_only', 'c2_only', 'dual'])
    parser.add_argument('--c1_idx', type=int, default=0)
    parser.add_argument('--c2_idx', type=int, default=1)
    parser.add_argument('--dual_mode', type=str, default='average', choices=['average', 'zeros', 'overlay'])
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_samples', type=int, default=8, help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='./visualizations')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    # Load data
    print("Loading validation data...")
    val_images = np.load(args.images_npz)
    val_masks = np.load(args.masks_npz)

    # Run visualization
    visualize_validation_predictions(
        model_checkpoint=args.checkpoint,
        architecture=args.architecture,
        val_images=val_images,
        val_masks=val_masks,
        channel_mode=args.channel_mode,
        channel_indices=(args.c1_idx, args.c2_idx),
        dual_mode=args.dual_mode,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_viz_samples=args.num_samples,
        output_dir=args.output_dir,
        device=args.device
    )
