"""
Unified training script for multiple segmentation architectures.
Supports U-Net, U-Net++, and Hugging Face models (SegFormer, DeepLabV3, Mask2Former, etc.)
"""

import os
import argparse
from pathlib import Path
from typing import Tuple, Dict, List
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from finetune_sam2_multichannel import CiliaMultichannelDataset, prepare_data_splits
from models.model_factory import ModelFactory
from sam2.plotting_utils import plot_training_metrics


class SegmentationTrainer:
    """Unified trainer for all segmentation architectures."""

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5
    ):
        """
        Args:
            model: Segmentation model
            device: Device to use
            learning_rate: Learning rate
            weight_decay: Weight decay
            bce_weight: Weight for BCE loss
            dice_weight: Weight for Dice loss
        """
        self.device = device
        self.model = model.to(device)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss()

    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
        """Dice loss for binary segmentation."""
        pred_sigmoid = torch.sigmoid(pred)
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        return 1 - dice

    def compute_loss(
        self,
        pred_masks: torch.Tensor,
        gt_masks: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss."""
        bce = self.bce_loss(pred_masks, gt_masks)
        dice = self.dice_loss(pred_masks, gt_masks)
        total_loss = self.bce_weight * bce + self.dice_weight * dice

        return total_loss, {
            'bce': bce.item(),
            'dice': dice.item(),
            'total': total_loss.item()
        }

    def compute_iou(self, pred_masks: torch.Tensor, gt_masks: torch.Tensor, threshold: float = 0.5) -> float:
        """Compute IoU metric."""
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
            pred_masks = self.model(images)

            # Ensure output shape matches gt_masks
            if pred_masks.shape != gt_masks.shape:
                pred_masks = F.interpolate(
                    pred_masks,
                    size=(gt_masks.shape[2], gt_masks.shape[3]),
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
                pred_masks = self.model(images)

                # Ensure output shape matches gt_masks
                if pred_masks.shape != gt_masks.shape:
                    pred_masks = F.interpolate(
                        pred_masks,
                        size=(gt_masks.shape[2], gt_masks.shape[3]),
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


def train_model(
    model_name: str,
    architecture: str,
    channel_mode: str,
    dual_mode: str,
    train_images: List[str],
    train_masks: List[str],
    val_images: List[str],
    val_masks: List[str],
    output_dir: str,
    num_epochs: int = 50,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    image_size: int = 512,
    channel_indices: Tuple[int, int] = (0, 1),
    pretrained: bool = True
):
    """
    Train a segmentation model.

    Args:
        model_name: Name for saving (e.g., 'c1_unet', 'c2_segformer')
        architecture: Model architecture (e.g., 'unet', 'segformer-b0')
        channel_mode: 'c1_only', 'c2_only', or 'dual'
        dual_mode: 'average', 'zeros', or 'overlay'
        train_images: Training image paths
        train_masks: Training mask paths
        val_images: Validation image paths
        val_masks: Validation mask paths
        output_dir: Output directory
        num_epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        image_size: Image size
        channel_indices: Channel indices tuple
        pretrained: Use pretrained weights
    """
    print(f"\n{'='*80}")
    print(f"Training {model_name}")
    print(f"Architecture: {architecture}")
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

    # Create model
    print(f"Creating model: {architecture}")
    model = ModelFactory.create_model(
        architecture,
        num_classes=1,
        pretrained=pretrained
    )

    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    trainer = SegmentationTrainer(
        model=model,
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
        train_metrics = trainer.train_epoch(train_loader)
        print(f"Train - Loss: {train_metrics['loss']:.4f}, IoU: {train_metrics['iou']:.4f}")

        # Validate
        val_metrics = trainer.validate(val_loader)
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, IoU: {val_metrics['iou']:.4f}")

        # Save metrics
        metrics_history['train_loss'].append(train_metrics['loss'])
        metrics_history['train_iou'].append(train_metrics['iou'])
        metrics_history['val_loss'].append(val_metrics['loss'])
        metrics_history['val_iou'].append(val_metrics['iou'])

        # Save best model
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            trainer.save_checkpoint(
                str(model_output_dir / 'best_model.pth'),
                epoch,
                val_metrics
            )
            print(f"✓ Saved best model (IoU: {best_val_iou:.4f})")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(
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
    parser = argparse.ArgumentParser(
        description='Train segmentation models on multichannel cilia data'
    )

    # Data arguments
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing input images')
    parser.add_argument('--mask_dir', type=str, required=True,
                        help='Directory containing masks')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for models')

    # Model arguments
    parser.add_argument('--architecture', type=str, default='unet',
                        help='Model architecture (use --list_models to see options)')
    parser.add_argument('--list_models', action='store_true',
                        help='List all available model architectures')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights (for HF models)')
    parser.add_argument('--no_pretrained', action='store_false', dest='pretrained',
                        help='Train from scratch')

    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Image size (512 recommended for most models)')

    # Channel arguments
    parser.add_argument('--c1_idx', type=int, default=0,
                        help='Channel 1 index (C0)')
    parser.add_argument('--c2_idx', type=int, default=1,
                        help='Channel 2 index (C1)')
    parser.add_argument('--dual_mode', type=str, default='average',
                        choices=['average', 'zeros', 'overlay'],
                        help='Mode for combining channels in dual model')

    # Model selection
    parser.add_argument('--train_models', type=str, nargs='+',
                        default=['c1', 'c2', 'dual'],
                        choices=['c1', 'c2', 'dual'],
                        help='Which channel modes to train')

    # Data split
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Training data ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation data ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Test data ratio')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for splits')

    args = parser.parse_args()

    # List models if requested
    if args.list_models:
        print("\nAvailable Model Architectures:")
        print("="*80)
        for name, description in ModelFactory.list_models().items():
            print(f"  {name:<25} - {description}")
        print("="*80)
        return

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
    results = {}

    # Train models for each channel mode
    if 'c1' in args.train_models:
        model_name = f'c1_{args.architecture.replace("-", "_")}'
        metrics, best_iou = train_model(
            model_name=model_name,
            architecture=args.architecture,
            channel_mode='c1_only',
            dual_mode=args.dual_mode,
            train_images=train_images,
            train_masks=train_masks,
            val_images=val_images,
            val_masks=val_masks,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            image_size=args.image_size,
            channel_indices=channel_indices,
            pretrained=args.pretrained
        )
        results[model_name] = {'metrics': metrics, 'best_iou': best_iou}

    if 'c2' in args.train_models:
        model_name = f'c2_{args.architecture.replace("-", "_")}'
        metrics, best_iou = train_model(
            model_name=model_name,
            architecture=args.architecture,
            channel_mode='c2_only',
            dual_mode=args.dual_mode,
            train_images=train_images,
            train_masks=train_masks,
            val_images=val_images,
            val_masks=val_masks,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            image_size=args.image_size,
            channel_indices=channel_indices,
            pretrained=args.pretrained
        )
        results[model_name] = {'metrics': metrics, 'best_iou': best_iou}

    if 'dual' in args.train_models:
        model_name = f'dual_{args.architecture.replace("-", "_")}'
        metrics, best_iou = train_model(
            model_name=model_name,
            architecture=args.architecture,
            channel_mode='dual',
            dual_mode=args.dual_mode,
            train_images=train_images,
            train_masks=train_masks,
            val_images=val_images,
            val_masks=val_masks,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            image_size=args.image_size,
            channel_indices=channel_indices,
            pretrained=args.pretrained
        )
        results[model_name] = {'metrics': metrics, 'best_iou': best_iou}

    # Save overall results
    output_path = Path(args.output_dir)
    with open(output_path / 'all_results.json', 'w') as f:
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
