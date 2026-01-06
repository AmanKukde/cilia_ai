# SAM2 Fine-tuning with NumPy Arrays (N, C, H, W format)
# Direct loading from .npy files - optimized for your dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NumpySegmentationDataset(Dataset):
    """
    Dataset wrapper for pre-loaded NumPy arrays
    Expects:
    - images: (N, C, H, W) - already in correct channel format
    - masks: (N, H, W) - binary masks
    """
    def __init__(self, images_npy_path, masks_npy_path, max_samples=None):
        """
        Args:
            images_npy_path: path to images_cleaned_prompted.npy (N, C, H, W)
            masks_npy_path: path to predictions_sam2_cleaned.npy (N, H, W)
            max_samples: limit dataset size for testing
        """
        logger.info("Loading NumPy arrays...")
        
        # Load arrays
        self.images = np.load(images_npy_path)  # (N, C, H, W)
        self.masks = np.load(masks_npy_path).astype(np.uint8)  # (N, H, W)
        
        # Validate shapes
        assert len(self.images.shape) == 4, f"Images must be (N, C, H, W), got {self.images.shape}"
        assert len(self.masks.shape) == 3, f"Masks must be (N, H, W), got {self.masks.shape}"
        assert self.images.shape[0] == self.masks.shape[0], \
            f"Number of images ({self.images.shape[0]}) != number of masks ({self.masks.shape[0]})"
        
        # Check channel dimension
        self.num_channels = self.images.shape[1]
        self.height = self.images.shape[2]
        self.width = self.images.shape[3]
        
        logger.info(f"Loaded {self.images.shape[0]} samples")
        logger.info(f"  Images shape: {self.images.shape} (N, C, H, W)")
        logger.info(f"  Masks shape: {self.masks.shape} (N, H, W)")
        logger.info(f"  Channels: {self.num_channels}, Resolution: {self.height}Ã—{self.width}")
        
        # Limit samples for testing
        if max_samples:
            self.images = self.images[:max_samples]
            self.masks = self.masks[:max_samples]
            logger.info(f"Limited to {max_samples} samples")
        
        self.dataset_size = self.images.shape[0]
    
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        """
        Returns:
            image: torch tensor (C, H, W) float32 in range [0, 1]
            mask: torch tensor (H, W) float32 in range [0, 1]
            idx: sample index
        """
        img = self.images[idx].astype(np.float32)  # (C, H, W)
        mask = self.masks[idx].astype(np.float32)  # (H, W)
        
        # Normalize to [0, 1] if needed
        if img.max() > 1.0:
            img = img / 255.0
        if mask.max() > 1.0:
            mask = mask / 255.0
        
        return torch.from_numpy(img), torch.from_numpy(mask), idx


class SingleChannelToRGBAdapter:
    """
    Handles conversion of single or multi-channel images to 3-channel RGB for SAM2.
    If input is single channel, repeats it. If already 3+ channels, takes first 3.
    """
    def __init__(self, num_channels):
        self.num_channels = num_channels
    
    def __call__(self, img_tensor):
        """
        Args:
            img_tensor: (C, H, W) tensor
        Returns:
            (3, H, W) tensor suitable for SAM2
        """
        if self.num_channels == 1:
            # Single channel: repeat to 3 channels
            img_3ch = img_tensor.repeat(3, 1, 1)
        elif self.num_channels == 3:
            # Already 3 channels
            img_3ch = img_tensor
        elif self.num_channels > 3:
            # More than 3 channels: take first 3
            img_3ch = img_tensor[:3, :, :]
        else:
            # 2 channels: repeat first, then add it again
            img_3ch = torch.cat([img_tensor, img_tensor[:1, :, :]], dim=0)
        
        return img_3ch


class SeparateChannelModels:
    """
    Manages SAM2 trainable components.
    Freezes image encoder, trains only mask decoder and prompt encoder.
    """
    def __init__(self, predictor, freeze_encoder=True):
        self.predictor = predictor
        self.model = predictor.model
        
        if freeze_encoder:
            self.model.image_encoder.eval()
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
            logger.info("Image encoder frozen")
        
        # Enable training for decoders
        self.model.sam_mask_decoder.train(True)
        self.model.sam_prompt_encoder.train(True)
        
        logger.info("Mask decoder and prompt encoder set to training mode")
    
    def get_trainable_params(self):
        """Returns iterator of trainable parameters"""
        params = list(self.model.sam_mask_decoder.parameters()) + \
                 list(self.model.sam_prompt_encoder.parameters())
        return params


class SAM2FineTuner:
    """
    Fine-tuning class optimized for NumPy array datasets
    """
    def __init__(
        self,
        num_channels=1,
        model_cfg="sam2_hiera_s.yaml",
        model_ckpt="sam2_hiera_small.pt",
        device="cuda",
        lr=1e-4,
        weight_decay=1e-4,
        amp_dtype=torch.float16,
        accumulation_steps=4,
    ):
        # Build SAM2 model
        self.device = torch.device(device)
        self.model = build_sam2(model_cfg, model_ckpt, device=str(self.device))
        self.predictor = SAM2ImagePredictor(self.model)
        
        # Channel adapter
        self.channel_adapter = SingleChannelToRGBAdapter(num_channels)
        
        # Setup trainable components
        self.channel_models = SeparateChannelModels(self.predictor, freeze_encoder=True)
        
        # Optimizer setup
        self.optimizer = torch.optim.AdamW(
            self.channel_models.get_trainable_params(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=10000,
            eta_min=1e-6
        )
        
        # Mixed precision
        self.amp_enabled = amp_dtype is not None
        self.amp_dtype = amp_dtype
        self.scaler = torch.cuda.amp.GradScaler() if self.amp_enabled else None
        
        self.accumulation_steps = accumulation_steps
        self.step = 0
        
        trainable_count = sum(p.numel() for p in self.channel_models.get_trainable_params() 
                              if p.requires_grad)
        logger.info(f"Initialized SAM2 fine-tuner on {device}")
        logger.info(f"Trainable parameters: {trainable_count:,}")
    
    def train_step(self, image_tensor, mask_tensor):
        """
        Single training step
        
        Args:
            image_tensor: (C, H, W) tensor
            mask_tensor: (H, W) tensor
        
        Returns:
            loss value
        """
        # Handle batch dimension if needed
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)  # (1, C, H, W)
        if len(mask_tensor.shape) == 2:
            mask_tensor = mask_tensor.unsqueeze(0)  # (1, H, W)
        
        image_tensor = image_tensor.to(self.device)
        mask_tensor = mask_tensor.to(self.device)
        
        # Convert to 3-channel for SAM2
        if image_tensor.shape[1] != 3:
            image_tensor = torch.stack([
                self.channel_adapter(image_tensor[i]) 
                for i in range(image_tensor.shape[0])
            ])
        
        batch_size, _, h, w = image_tensor.shape
        
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.amp_enabled
        ):
            # Step 1: Image encoding (frozen)
            with torch.no_grad():
                image_embeddings = self.model.image_encoder(image_tensor)
            
            # Step 2: Full-image bounding box prompt
            boxes = torch.tensor(
                [[0, 0, w, h]] * batch_size,
                device=self.device,
                dtype=torch.float32
            )
            
            # Step 3: Encode prompts with positional embeddings
            sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
                boxes=boxes,
                points=None,
                masks=None
            )
            
            # Step 4: Get high-resolution features
            # Note: This step depends on SAM2 architecture
            # For compatibility, we attempt to get high-res features
            try:
                high_res_features = [
                    self.model.sam2_features
                    for _ in range(batch_size)
                ]
            except:
                # Fallback if high-res features not available
                high_res_features = None
            
            # Step 5: Decode masks
            low_res_masks, iou_predictions = self.model.sam_mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )
            
            # Step 6: Upsample masks to original resolution
            # SAM2 typically outputs 256x256 masks, upsample to input size
            upscaled_masks = F.interpolate(
                low_res_masks.float(),
                size=(h, w),
                mode='bilinear',
                align_corners=False
            )
            
            # Apply sigmoid to get probabilities
            pred_masks = torch.sigmoid(upscaled_masks.squeeze(1))  # (B, H, W)
            
            # Step 7: Compute losses
            # Binary cross-entropy loss
            seg_loss = F.binary_cross_entropy(
                pred_masks,
                mask_tensor,
                reduction='mean'
            )
            
            # IoU-based loss (threshold at 0.5)
            pred_binary = (pred_masks > 0.5).float()
            intersection = (pred_binary * mask_tensor).sum(dim=(1, 2))
            union = (pred_binary + mask_tensor).clamp(max=1.0).sum(dim=(1, 2))
            iou = intersection / (union + 1e-6)
            iou_loss = 1.0 - iou.mean()
            
            # Combined loss
            loss = seg_loss + 0.1 * iou_loss
            
            # Scale for gradient accumulation
            loss = loss / self.accumulation_steps
        
        # Backward pass
        if self.amp_enabled:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation step
        if (self.step + 1) % self.accumulation_steps == 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.channel_models.get_trainable_params(),
                max_norm=1.0
            )
            
            # Optimizer step
            if self.amp_enabled:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            self.scheduler.step()
        
        self.step += 1
        return loss.item() * self.accumulation_steps
    
    def save_checkpoint(self, save_path):
        """Save model weights and optimizer state"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step': self.step,
        }
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, load_path):
        """Load model weights and optimizer state"""
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint.get('step', 0)
        logger.info(f"Checkpoint loaded from {load_path} (step {self.step})")


def main():
    # Configuration
    config = {
        'images_path': 'data/prompted_outputs/images_cleaned_prompted.npy',
        'masks_path': 'data/prompted_outputs/predictions_sam2_cleaned.npy',
        'model_cfg': 'sam2_hiera_s.yaml',
        'model_ckpt': 'sam2_hiera_small.pt',
        'num_epochs': 20,
        'batch_size': 4,  # Can increase if you have GPU VRAM
        'num_workers': 0,  # Set to 0 for NumPy arrays (no multiprocessing needed)
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'accumulation_steps': 2,
        'amp_dtype': torch.float16,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': './checkpoints_npy',
        'checkpoint_interval': 100,  # Save every 100 steps
        'max_samples': None,  # Set to N to limit dataset size for testing
    }
    
    # Create save directory
    Path(config['save_dir']).mkdir(exist_ok=True)
    
    # Initialize dataset
    dataset = NumpySegmentationDataset(
        config['images_path'],
        config['masks_path'],
        max_samples=config['max_samples']
    )
    
    # Get number of channels from dataset
    num_channels = dataset.num_channels
    logger.info(f"Dataset has {num_channels} channel(s)")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Initialize fine-tuner
    finetuner = SAM2FineTuner(
        num_channels=num_channels,
        model_cfg=config['model_cfg'],
        model_ckpt=config['model_ckpt'],
        device=config['device'],
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        amp_dtype=config['amp_dtype'],
        accumulation_steps=config['accumulation_steps'],
    )
    
    # Training loop
    logger.info("Starting fine-tuning...")
    best_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        
        for batch_idx, (images, masks, indices) in enumerate(pbar):
            try:
                # images: (B, C, H, W)
                # masks: (B, H, W)
                loss = finetuner.train_step(images, masks)
                epoch_loss += loss
                
                avg_loss = epoch_loss / (batch_idx + 1)
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'step': finetuner.step
                })
                
                # Save checkpoint
                if (finetuner.step % config['checkpoint_interval'] == 0 and
                    finetuner.step > 0):
                    ckpt_path = Path(config['save_dir']) / f"sam2_step_{finetuner.step:06d}.pt"
                    finetuner.save_checkpoint(ckpt_path)
                    
                    # Keep track of best loss
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        best_ckpt = Path(config['save_dir']) / "sam2_best.pt"
                        finetuner.save_checkpoint(best_ckpt)
                        logger.info(f"New best loss: {best_loss:.4f}")
            
            except Exception as e:
                logger.error(f"Error at step {finetuner.step}: {str(e)}")
                logger.error(f"Batch shapes: images={images.shape}, masks={masks.shape}")
                continue
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1} completed. Avg loss: {avg_epoch_loss:.4f}")
    
    # Final save
    final_ckpt = Path(config['save_dir']) / "sam2_finetuned_final.pt"
    finetuner.save_checkpoint(final_ckpt)
    logger.info(f"Fine-tuning completed!")
    logger.info(f"Final model: {final_ckpt}")
    logger.info(f"Best model: {Path(config['save_dir']) / 'sam2_best.pt'}")


if __name__ == "__main__":
    main()