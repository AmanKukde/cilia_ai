"""
SAM2 Finetuning for Multichannel Cilia Data

This script trains 3 SAM2 models:
1. Model C1: Using Channel 1 (C1*3 - channel replicated 3 times)
2. Model C2: Using Channel 2 (C2*3 - channel replicated 3 times)
3. Model Dual: Using both channels (C1, C2, avg(C1,C2) or overlay)
"""

import os
import argparse
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tifffile as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from cilia_utils.utils import prepare_sam2_input, normalize_channel
from cilia_utils.plotting_utils import visualize_training_batch, plot_training_metrics


class CiliaMultichannelDataset(Dataset):
    """
    Dataset for multichannel cilia images.

    Supports 3 channel modes:
    - 'c1_only': Channel 1 replicated 3 times (C1*3)
    - 'c2_only': Channel 2 replicated 3 times (C2*3)
    - 'dual': Both channels (C1, C2, combination)
    """

    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        channel_mode: str = 'c1_only',
        channel_indices: Tuple[int, int] = (0, 1),
        dual_mode: str = 'average',  # 'average', 'zeros', 'overlay'
        transform=None,
        image_size: int = 1024
    ):
        """
        Args:
            image_paths: List of paths to multichannel TIFF images
            mask_paths: List of paths to mask TIFF files
            channel_mode: One of 'c1_only', 'c2_only', 'dual'
            channel_indices: Tuple of (C1_idx, C2_idx) - default (0, 1)
            dual_mode: For 'dual' mode - 'average', 'zeros', or 'overlay'
            transform: Optional transforms
            image_size: Target image size for SAM2
        """
        assert len(image_paths) == len(mask_paths), "Images and masks must match"
        assert channel_mode in ['c1_only', 'c2_only', 'dual'], f"Invalid channel_mode: {channel_mode}"
        assert dual_mode in ['average', 'zeros', 'overlay'], f"Invalid dual_mode: {dual_mode}"

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.channel_mode = channel_mode
        self.c1_idx, self.c2_idx = channel_indices
        self.dual_mode = dual_mode
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def _prepare_channels(self, image: np.ndarray) -> np.ndarray:
        """
        Prepare RGB channels based on channel_mode.

        Args:
            image: Input multichannel image (H, W, C)

        Returns:
            RGB image (H, W, 3) uint8
        """
        if self.channel_mode == 'c1_only':
            # C1 replicated 3 times
            c1 = image[:, :, self.c1_idx]
            c1_norm = normalize_channel(c1)
            rgb = np.stack([c1_norm] * 3, axis=-1)

        elif self.channel_mode == 'c2_only':
            # C2 replicated 3 times
            c2 = image[:, :, self.c2_idx]
            c2_norm = normalize_channel(c2)
            rgb = np.stack([c2_norm] * 3, axis=-1)

        elif self.channel_mode == 'dual':
            # Both channels
            c1 = normalize_channel(image[:, :, self.c1_idx])
            c2 = normalize_channel(image[:, :, self.c2_idx])

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
        # Load image and mask
        image = tf.imread(self.image_paths[idx])
        mask = tf.imread(self.mask_paths[idx])

        # Prepare RGB channels
        rgb_image = self._prepare_channels(image)

        # Ensure mask is binary
        if mask.dtype == np.float32 or mask.dtype == np.float64:
            mask = (mask > 0.5).astype(np.uint8)
        else:
            mask = (mask > 0).astype(np.uint8)

        # Handle multi-channel masks (use first channel if multi-channel)
        if mask.ndim == 3:
            mask = mask[:, :, 0] if mask.shape[2] > 0 else mask

        # Resize if needed
        if rgb_image.shape[0] != self.image_size or rgb_image.shape[1] != self.image_size:
            from skimage.transform import resize
            rgb_image = resize(
                rgb_image,
                (self.image_size, self.image_size),
                preserve_range=True,
                anti_aliasing=True
            ).astype(np.uint8)
            mask = resize(
                mask,
                (self.image_size, self.image_size),
                order=0,  # Nearest neighbor for masks
                preserve_range=True,
                anti_aliasing=False
            ).astype(np.uint8)

        # Apply transforms
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
            'image_path': self.image_paths[idx],
            'mask_path': self.mask_paths[idx]
        }


class SAM2FineTuner:
    """Fine-tuning wrapper for SAM2 on multichannel cilia data."""

    def __init__(
        self,
        model_cfg: str,
        sam2_checkpoint: str,
        device: str = 'cuda',
        freeze_image_encoder: bool = True,
        freeze_prompt_encoder: bool = True,
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-4
    ):
        """
        Args:
            model_cfg: Path to SAM2 model config
            sam2_checkpoint: Path to SAM2 pretrained checkpoint
            device: Device to use
            freeze_image_encoder: Whether to freeze image encoder
            freeze_prompt_encoder: Whether to freeze prompt encoder
            learning_rate: Learning rate
            weight_decay: Weight decay
        """
        self.device = device

        # Build SAM2 model
        self.model = build_sam2(model_cfg, sam2_checkpoint, device=device)

        # Freeze components if specified
        if freeze_image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False

        if freeze_prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False

        # Setup optimizer (only for trainable parameters)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = self._dice_loss

    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
        """
        Dice loss for binary segmentation.

        Args:
            pred: Predicted logits (B, 1, H, W)
            target: Ground truth masks (B, 1, H, W)
            smooth: Smoothing factor

        Returns:
            Dice loss
        """
        pred_sigmoid = torch.sigmoid(pred)

        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

        return 1 - dice

    def compute_loss(
        self,
        pred_masks: torch.Tensor,
        gt_masks: torch.Tensor,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Args:
            pred_masks: Predicted masks (B, 1, H, W)
            gt_masks: Ground truth masks (B, 1, H, W)
            bce_weight: Weight for BCE loss
            dice_weight: Weight for Dice loss

        Returns:
            Total loss and dictionary of individual losses
        """
        bce = self.bce_loss(pred_masks, gt_masks)
        dice = self.dice_loss(pred_masks, gt_masks)

        total_loss = bce_weight * bce + dice_weight * dice

        return total_loss, {
            'bce': bce.item(),
            'dice': dice.item(),
            'total': total_loss.item()
        }

    def compute_iou(self, pred_masks: torch.Tensor, gt_masks: torch.Tensor, threshold: float = 0.5) -> float:
        """
        Compute IoU metric.

        Args:
            pred_masks: Predicted masks (B, 1, H, W)
            gt_masks: Ground truth masks (B, 1, H, W)
            threshold: Threshold for binarization

        Returns:
            Mean IoU
        """
        pred_binary = (torch.sigmoid(pred_masks) > threshold).float()

        intersection = (pred_binary * gt_masks).sum(dim=(1, 2, 3))
        union = pred_binary.sum(dim=(1, 2, 3)) + gt_masks.sum(dim=(1, 2, 3)) - intersection

        iou = (intersection + 1e-6) / (union + 1e-6)

        return iou.mean().item()

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0
        total_iou = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc='Training')

        for batch in pbar:
            images = batch['image'].to(self.device)  # (B, 3, H, W)
            gt_masks = batch['mask'].to(self.device)  # (B, 1, H, W)

            self.optimizer.zero_grad()

            # Forward pass
            # SAM2 expects images in (B, 3, H, W) format
            with torch.set_grad_enabled(True):
                # Get image embeddings
                image_embeddings = self.model.image_encoder(images)

                # Use no prompts (automatic segmentation)
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None
                )

                # Predict masks
                low_res_masks, iou_predictions = self.model.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=self.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False
                )

                # Upsample masks to original size
                pred_masks = F.interpolate(
                    low_res_masks,
                    size=(images.shape[2], images.shape[3]),
                    mode='bilinear',
                    align_corners=False
                )

            # Compute loss
            loss, loss_dict = self.compute_loss(pred_masks, gt_masks)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Compute metrics
            iou = self.compute_iou(pred_masks, gt_masks)

            total_loss += loss.item()
            total_iou += iou
            num_batches += 1

            pbar.set_postfix({
                'loss': loss.item(),
                'iou': iou
            })

        return {
            'loss': total_loss / num_batches,
            'iou': total_iou / num_batches
        }

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()

        total_loss = 0
        total_iou = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc='Validation')

        with torch.no_grad():
            for batch in pbar:
                images = batch['image'].to(self.device)
                gt_masks = batch['mask'].to(self.device)

                # Forward pass
                image_embeddings = self.model.image_encoder(images)

                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None
                )

                low_res_masks, iou_predictions = self.model.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=self.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False
                )

                pred_masks = F.interpolate(
                    low_res_masks,
                    size=(images.shape[2], images.shape[3]),
                    mode='bilinear',
                    align_corners=False
                )

                # Compute metrics
                loss, _ = self.compute_loss(pred_masks, gt_masks)
                iou = self.compute_iou(pred_masks, gt_masks)

                total_loss += loss.item()
                total_iou += iou
                num_batches += 1

                pbar.set_postfix({
                    'loss': loss.item(),
                    'iou': iou
                })

        return {
            'loss': total_loss / num_batches,
            'iou': total_iou / num_batches
        }

    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }, path)

    def load_checkpoint(self, path: str) -> Dict:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint


def prepare_data_splits(
    image_dir: str,
    mask_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    """
    Prepare train/val/test splits.

    Returns:
        train_images, train_masks, val_images, val_masks, test_images, test_masks
    """
    import random

    # Get all image files
    image_paths = sorted([str(p) for p in Path(image_dir).glob('*.tif')])
    mask_paths = sorted([str(p) for p in Path(mask_dir).glob('*.tif')])

    assert len(image_paths) == len(mask_paths), "Number of images and masks must match"

    # Shuffle with seed
    random.seed(random_seed)
    indices = list(range(len(image_paths)))
    random.shuffle(indices)

    # Split indices
    n_total = len(indices)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    # Get paths
    train_images = [image_paths[i] for i in train_idx]
    train_masks = [mask_paths[i] for i in train_idx]

    val_images = [image_paths[i] for i in val_idx]
    val_masks = [mask_paths[i] for i in val_idx]

    test_images = [image_paths[i] for i in test_idx]
    test_masks = [mask_paths[i] for i in test_idx]

    return train_images, train_masks, val_images, val_masks, test_images, test_masks


def train_model(
    model_name: str,
    channel_mode: str,
    dual_mode: str,
    train_images: List[str],
    train_masks: List[str],
    val_images: List[str],
    val_masks: List[str],
    output_dir: str,
    sam2_cfg: str,
    sam2_checkpoint: str,
    num_epochs: int = 50,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    image_size: int = 1024,
    channel_indices: Tuple[int, int] = (0, 1)
):
    """
    Train a single SAM2 model variant.

    Args:
        model_name: Name for this model (e.g., 'c1_model', 'c2_model', 'dual_model')
        channel_mode: 'c1_only', 'c2_only', or 'dual'
        dual_mode: 'average', 'zeros', or 'overlay' (only used if channel_mode='dual')
        train_images: List of training image paths
        train_masks: List of training mask paths
        val_images: List of validation image paths
        val_masks: List of validation mask paths
        output_dir: Directory to save outputs
        sam2_cfg: Path to SAM2 config
        sam2_checkpoint: Path to SAM2 checkpoint
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        image_size: Image size for SAM2
        channel_indices: Tuple of (C1_idx, C2_idx)
    """
    print(f"\n{'='*80}")
    print(f"Training {model_name}")
    print(f"Channel mode: {channel_mode}, Dual mode: {dual_mode}")
    print(f"{'='*80}\n")

    # Create output directory
    model_output_dir = Path(output_dir) / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Create datasets
    train_dataset = CiliaMultichannelDataset(
        train_images,
        train_masks,
        channel_mode=channel_mode,
        channel_indices=channel_indices,
        dual_mode=dual_mode,
        image_size=image_size
    )

    val_dataset = CiliaMultichannelDataset(
        val_images,
        val_masks,
        channel_mode=channel_mode,
        channel_indices=channel_indices,
        dual_mode=dual_mode,
        image_size=image_size
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create fine-tuner
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    finetuner = SAM2FineTuner(
        model_cfg=sam2_cfg,
        sam2_checkpoint=sam2_checkpoint,
        device=device,
        learning_rate=learning_rate
    )

    # Training loop
    best_val_iou = 0
    metrics_history = {
        'train_loss': [],
        'train_iou': [],
        'val_loss': [],
        'val_iou': []
    }

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 80)

        # Train
        train_metrics = finetuner.train_epoch(train_loader)
        print(f"Train - Loss: {train_metrics['loss']:.4f}, IoU: {train_metrics['iou']:.4f}")

        # Validate
        val_metrics = finetuner.validate(val_loader)
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, IoU: {val_metrics['iou']:.4f}")

        # Save metrics
        metrics_history['train_loss'].append(train_metrics['loss'])
        metrics_history['train_iou'].append(train_metrics['iou'])
        metrics_history['val_loss'].append(val_metrics['loss'])
        metrics_history['val_iou'].append(val_metrics['iou'])

        # Save best model
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            finetuner.save_checkpoint(
                str(model_output_dir / 'best_model.pth'),
                epoch,
                val_metrics
            )
            print(f"✓ Saved best model (IoU: {best_val_iou:.4f})")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            finetuner.save_checkpoint(
                str(model_output_dir / f'checkpoint_epoch_{epoch+1}.pth'),
                epoch,
                val_metrics
            )

        # Save metrics
        with open(model_output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics_history, f, indent=2)

    # Plot final metrics
    plot_training_metrics(
        metrics_history,
        save_path=str(model_output_dir / 'training_metrics.png')
    )

    print(f"\n{'='*80}")
    print(f"Finished training {model_name}")
    print(f"Best validation IoU: {best_val_iou:.4f}")
    print(f"Outputs saved to: {model_output_dir}")
    print(f"{'='*80}\n")

    return metrics_history, best_val_iou


def main():
    parser = argparse.ArgumentParser(description='Finetune SAM2 on multichannel cilia data')

    # Data arguments
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--mask_dir', type=str, required=True, help='Directory containing masks')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for models')

    # SAM2 arguments
    parser.add_argument('--sam2_cfg', type=str, required=True, help='Path to SAM2 config file')
    parser.add_argument('--sam2_checkpoint', type=str, required=True, help='Path to SAM2 checkpoint')

    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--image_size', type=int, default=1024, help='Image size')

    # Channel arguments
    parser.add_argument('--c1_idx', type=int, default=0, help='Channel 1 index')
    parser.add_argument('--c2_idx', type=int, default=1, help='Channel 2 index')
    parser.add_argument('--dual_mode', type=str, default='average',
                        choices=['average', 'zeros', 'overlay'],
                        help='Mode for combining channels in dual model')

    # Model selection
    parser.add_argument('--train_models', type=str, nargs='+',
                        default=['c1', 'c2', 'dual'],
                        choices=['c1', 'c2', 'dual'],
                        help='Which models to train')

    # Data split
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training data ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation data ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test data ratio')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for splits')

    args = parser.parse_args()

    # Prepare data splits
    print("Preparing data splits...")
    train_images, train_masks, val_images, val_masks, test_images, test_masks = prepare_data_splits(
        args.image_dir,
        args.mask_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed
    )

    print(f"Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")

    channel_indices = (args.c1_idx, args.c2_idx)

    # Train models
    results = {}

    if 'c1' in args.train_models:
        metrics, best_iou = train_model(
            model_name='c1_model',
            channel_mode='c1_only',
            dual_mode=args.dual_mode,  # Not used for c1_only
            train_images=train_images,
            train_masks=train_masks,
            val_images=val_images,
            val_masks=val_masks,
            output_dir=args.output_dir,
            sam2_cfg=args.sam2_cfg,
            sam2_checkpoint=args.sam2_checkpoint,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            image_size=args.image_size,
            channel_indices=channel_indices
        )
        results['c1_model'] = {'metrics': metrics, 'best_iou': best_iou}

    if 'c2' in args.train_models:
        metrics, best_iou = train_model(
            model_name='c2_model',
            channel_mode='c2_only',
            dual_mode=args.dual_mode,  # Not used for c2_only
            train_images=train_images,
            train_masks=train_masks,
            val_images=val_images,
            val_masks=val_masks,
            output_dir=args.output_dir,
            sam2_cfg=args.sam2_cfg,
            sam2_checkpoint=args.sam2_checkpoint,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            image_size=args.image_size,
            channel_indices=channel_indices
        )
        results['c2_model'] = {'metrics': metrics, 'best_iou': best_iou}

    if 'dual' in args.train_models:
        metrics, best_iou = train_model(
            model_name='dual_model',
            channel_mode='dual',
            dual_mode=args.dual_mode,
            train_images=train_images,
            train_masks=train_masks,
            val_images=val_images,
            val_masks=val_masks,
            output_dir=args.output_dir,
            sam2_cfg=args.sam2_cfg,
            sam2_checkpoint=args.sam2_checkpoint,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            image_size=args.image_size,
            channel_indices=channel_indices
        )
        results['dual_model'] = {'metrics': metrics, 'best_iou': best_iou}

    # Save overall results
    output_path = Path(args.output_dir)
    with open(output_path / 'all_results.json', 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        serializable_results = {}
        for model_name, model_results in results.items():
            serializable_results[model_name] = {
                'best_iou': float(model_results['best_iou'])
            }
        json.dump(serializable_results, f, indent=2)

    print("\n" + "="*80)
    print("ALL TRAINING COMPLETE")
    print("="*80)
    for model_name, model_results in results.items():
        print(f"{model_name}: Best IoU = {model_results['best_iou']:.4f}")
    print("="*80)


if __name__ == '__main__':
    main()
