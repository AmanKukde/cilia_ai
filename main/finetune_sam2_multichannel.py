# SAM2 Fine-tuning with 4D NumPy Arrays (N, C, H, W)
# For per-channel masks - optimized for multi-channel segmentation

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


class FourDSegmentationDataset(Dataset):
    """
    Dataset wrapper for 4D NumPy arrays: images (N, C, H, W) and masks (N, C, H, W)
    Resizes all images and masks to 256x256 while keeping channel dimension.
    """
    def __init__(self, images_npy_path, masks_npy_path, max_samples=None, target_size=(256, 256)):
        """
        Args:
            images_npy_path: path to images (N, C, H, W)
            masks_npy_path: path to masks (N, C, H, W)
            max_samples: limit dataset size for testing
            target_size: tuple (H, W) to resize all images/masks
        """
        self.target_H, self.target_W = target_size

        logger.info("Loading 4D NumPy arrays...")
        
        # Load arrays
        self.images = np.load(images_npy_path)  # (N, C, H, W)
        self.masks = np.load(masks_npy_path)    # (N, C, H, W)
        
        # Validate shapes
        assert len(self.images.shape) == 4, f"Images must be (N, C, H, W), got {self.images.shape}"
        assert len(self.masks.shape) == 4, f"Masks must be (N, C, H, W), got {self.masks.shape}"
        assert self.images.shape == self.masks.shape, \
            f"Images shape {self.images.shape} != Masks shape {self.masks.shape}"
        
        N, C, H, W = self.images.shape
        self.num_samples = N
        self.num_channels = C
        self.orig_height = H
        self.orig_width = W
        
        logger.info(f"Loaded {N} samples with {C} channels at {H}×{W} resolution")
        logger.info(f"  Images shape: {self.images.shape}")
        logger.info(f"  Masks shape: {self.masks.shape}")
        logger.info(f"  Target resize: {self.target_H}×{self.target_W}")
        
        # Limit samples for testing
        if max_samples:
            self.images = self.images[:max_samples]
            self.masks = self.masks[:max_samples]
            self.num_samples = max_samples
            logger.info(f"Limited to {max_samples} samples")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Returns:
            image: torch tensor (C, target_H, target_W) float32 in range [0, 1]
            mask: torch tensor (C, target_H, target_W) float32 in range [0, 1]
            idx: sample index
        """
        img = self.images[idx].astype(np.float32)  # (C, H, W)
        mask = self.masks[idx].astype(np.float32)  # (C, H, W)
        
        # Normalize to [0, 1] if needed
        if img.max() > 1.0:
            img = img / 255.0
        if mask.max() > 1.0:
            mask = mask / 255.0
        
        # Convert to torch tensors
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)

        # Resize each channel
        img = F.interpolate(img.unsqueeze(0), size=(self.target_H, self.target_W), mode='bilinear', align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0), size=(self.target_H, self.target_W), mode='nearest').squeeze(0)
        
        return img, mask, idx

class MultiChannelSAM2Model:
    """
    SAM2 with separate decoders per channel.
    Shares frozen image encoder, trains separate mask decoders for each channel.
    """
    def __init__(
        self,
        num_channels: int,
        model_cfg: str = "sam2_hiera_s.yaml",
        model_ckpt: str = "sam2_hiera_small.pt",
        device: str = "cuda",
        freeze_encoder: bool = True
    ):
        self.device = torch.device(device)
        self.num_channels = num_channels
        
        # Build base SAM2 model
        self.base_model = build_sam2(model_cfg, model_ckpt, device=str(self.device))
        self.predictor = SAM2ImagePredictor(self.base_model)
        
        # Freeze image encoder (shared, expensive)
        if freeze_encoder:
            self.base_model.image_encoder.eval()
            for param in self.base_model.image_encoder.parameters():
                param.requires_grad = False
            logger.info("Image encoder frozen (shared across channels)")
        
        # Create separate mask decoders per channel
        self.channel_decoders = {}
        for ch in range(num_channels):
            # Clone decoder for this channel
            import copy
            channel_decoder = copy.deepcopy(self.base_model.sam_mask_decoder)
            self.channel_decoders[ch] = channel_decoder.to(self.device)
            logger.info(f"Created separate mask decoder for channel {ch}")
        
        # Shared components
        self.image_encoder = self.base_model.image_encoder
        self.prompt_encoder = self.base_model.sam_prompt_encoder
    
    def get_trainable_params(self):
        """Return all trainable parameters"""
        params = []
        for ch in range(self.num_channels):
            params.extend(self.channel_decoders[ch].parameters())
        params.extend(self.prompt_encoder.parameters())
        return params
    
    def forward_channel(self, image_tensor, mask_tensor, channel_idx, boxes=None):
        """
        Forward pass for a single channel
        
        Args:
            image_tensor: (B, C, H, W) full batch
            mask_tensor: (B, C, H, W) full batch
            channel_idx: which channel to process
            boxes: prompt boxes (B, 4)
        
        Returns:
            pred_masks: (B, H, W) predictions for this channel
            loss: scalar loss
        """
        B, C, H, W = image_tensor.shape
        
        # Extract single channel and expand to 3 for SAM2
        img_ch = image_tensor[:, channel_idx:channel_idx+1, :, :]  # (B, 1, H, W)
        img_ch_3 = img_ch.repeat(1, 3, 1, 1)  # (B, 3, H, W)
        
        mask_ch = mask_tensor[:, channel_idx, :, :]  # (B, H, W)
        
        # Encode image
        with torch.no_grad():
            image_embeddings = self.image_encoder(img_ch_3)
        
        # Default boxes if not provided
        if boxes is None:
            boxes = torch.tensor(
                [[0, 0, W, H]] * B,
                device=self.device,
                dtype=torch.float32
            )
        
        # Encode prompts
        sparse_emb, dense_emb = self.prompt_encoder(
            boxes=boxes,
            points=None,
            masks=None
        )
        
        # Decode with channel-specific decoder
        decoder = self.channel_decoders[channel_idx]
        low_res_masks, iou_preds = decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False, 
            repeat_image = False
        )
        # Upsample
        upscaled_masks = F.interpolate(
            low_res_masks.float(),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )

        # Crop or pad to ensure exact match with mask
        upscaled_masks = upscaled_masks[:, :, :H, :W]  # crop if too large
        if upscaled_masks.shape[2] < H or upscaled_masks.shape[3] < W:
            # pad if too small
            pad_h = H - upscaled_masks.shape[2]
            pad_w = W - upscaled_masks.shape[3]
            upscaled_masks = F.pad(upscaled_masks, (0, pad_w, 0, pad_h), mode='replicate')

        pred_masks = torch.sigmoid(upscaled_masks.squeeze(1))  # (B, H, W)
        
        # Loss
        seg_loss = F.binary_cross_entropy(pred_masks, mask_ch, reduction='mean')
        
        pred_binary = (pred_masks > 0.5).float()
        intersection = (pred_binary * mask_ch).sum(dim=(1, 2))
        union = (pred_binary + mask_ch).clamp(max=1.0).sum(dim=(1, 2))
        iou = intersection / (union + 1e-6)
        iou_loss = 1.0 - iou.mean()
        
        loss = seg_loss + 0.1 * iou_loss
        
        return pred_masks, loss


class SAM2MultiChannelFineTuner:
    """
    Fine-tuning class for multi-channel segmentation with separate per-channel models
    """
    def __init__(
        self,
        num_channels: int,
        model_cfg: str = "sam2_hiera_s.yaml",
        model_ckpt: str = "sam2_hiera_small.pt",
        device: str = "cuda",
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        amp_dtype = torch.float16,
        accumulation_steps: int = 2,
    ):
        # Initialize model
        self.multi_model = MultiChannelSAM2Model(
            num_channels=num_channels,
            model_cfg=model_cfg,
            model_ckpt=model_ckpt,
            device=device,
            freeze_encoder=True
        )
        
        self.device = torch.device(device)
        self.num_channels = num_channels
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.multi_model.get_trainable_params(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Scheduler
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
        
        trainable = sum(p.numel() for p in self.multi_model.get_trainable_params() 
                       if p.requires_grad)
        logger.info(f"Initialized multi-channel SAM2 fine-tuner")
        logger.info(f"  Channels: {num_channels}")
        logger.info(f"  Trainable parameters: {trainable:,}")
    
    def train_step(self, image_tensor, mask_tensor):
        """
        Single training step processing all channels
        
        Args:
            image_tensor: (B, C, H, W) or (C, H, W)
            mask_tensor: (B, C, H, W) or (C, H, W)
        
        Returns:
            total_loss: scalar
            per_channel_losses: dict with loss for each channel
        """
        # Add batch dimension if needed
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)
        if len(mask_tensor.shape) == 3:
            mask_tensor = mask_tensor.unsqueeze(0)
        
        image_tensor = image_tensor.to(self.device)
        mask_tensor = mask_tensor.to(self.device)
        
        B, C, H, W = image_tensor.shape
        assert C == self.num_channels, f"Expected {self.num_channels} channels, got {C}"
        
        # Generate box prompts
        boxes = torch.tensor(
            [[0, 0, W, H]] * B,
            device=self.device,
            dtype=torch.float32
        )
        
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.amp_enabled
        ):
            # Process each channel
            channel_losses = {}
            total_loss = 0.0
            
            for ch in range(C):
                pred_masks, ch_loss = self.multi_model.forward_channel(
                    image_tensor,
                    mask_tensor,
                    ch,
                )
                channel_losses[f'ch_{ch}'] = ch_loss.item()
                total_loss += ch_loss
            
            # Average loss across channels
            total_loss = total_loss / C
            total_loss = total_loss / self.accumulation_steps
        
        # Backward
        if self.amp_enabled:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        
        # Gradient accumulation step
        if (self.step + 1) % self.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                self.multi_model.get_trainable_params(),
                max_norm=1.0
            )
            
            if self.amp_enabled:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            self.scheduler.step()
        
        self.step += 1
        return total_loss.item() * self.accumulation_steps, channel_losses
    
    def save_checkpoint(self, save_path):
        """Save all channel-specific decoders and optimizer state"""
        checkpoint = {
            'base_model': self.multi_model.base_model.state_dict(),
            'channel_decoders': {
                ch: decoder.state_dict()
                for ch, decoder in self.multi_model.channel_decoders.items()
            },
            'prompt_encoder': self.multi_model.prompt_encoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'step': self.step,
        }
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved: {save_path}")
    
    def load_checkpoint(self, load_path):
        """Load all channel-specific decoders and optimizer state"""
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.multi_model.base_model.load_state_dict(checkpoint['base_model'])
        for ch, state_dict in checkpoint['channel_decoders'].items():
            self.multi_model.channel_decoders[ch].load_state_dict(state_dict)
        self.multi_model.prompt_encoder.load_state_dict(checkpoint['prompt_encoder'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.step = checkpoint.get('step', 0)
        
        logger.info(f"Checkpoint loaded: {load_path} (step {self.step})")


def main():
    # Configuration
    config = {
        'images_path': "/home/aman.kukde/cilia_ai/data/prompted_outputs/images_cleaned_prompted.npy",
        'masks_path': "/home/aman.kukde/cilia_ai/data/prompted_outputs/predictions_sam2_cleaned.npy",
        "model_cfg": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "model_ckpt": "./checkpoints/sam2.1_hiera_large.pt",
        'num_epochs': 15,
        'batch_size': 16,
        'num_workers': 0,  # NumPy arrays don't benefit from multiprocessing
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'accumulation_steps': 2,
        'amp_dtype': torch.float16,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': './checkpoints_multichannel',
        'checkpoint_interval': 100,
        'max_samples': None,  # Set to test size for quick testing
    }
    
    # Create save directory
    Path(config['save_dir']).mkdir(exist_ok=True)
    
    # Initialize dataset
    dataset = FourDSegmentationDataset(
        config['images_path'],
        config['masks_path'],
        max_samples=config['max_samples']
    )
    
    num_channels = dataset.num_channels
    logger.info(f"Dataset: {len(dataset)} samples, {num_channels} channels")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Initialize fine-tuner
    finetuner = SAM2MultiChannelFineTuner(
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
    logger.info("Starting multi-channel fine-tuning...")
    best_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        
        for batch_idx, (images, masks, indices) in enumerate(pbar):
            
            loss, ch_losses = finetuner.train_step(images, masks)
            epoch_loss += loss
            
            avg_loss = epoch_loss / (batch_idx + 1)
            
            # Format per-channel losses for display
            loss_str = f"total={avg_loss:.4f}"
            for ch_name, ch_loss in ch_losses.items():
                loss_str += f", {ch_name}={ch_loss:.4f}"
            
            pbar.set_postfix_string(loss_str)
            
            # Save checkpoint
            if (finetuner.step % config['checkpoint_interval'] == 0 and
                finetuner.step > 0):
                ckpt_path = Path(config['save_dir']) / f"sam2_step_{finetuner.step:06d}.pt"
                finetuner.save_checkpoint(ckpt_path)
                
                # Track best loss
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_ckpt = Path(config['save_dir']) / "sam2_best.pt"
                    finetuner.save_checkpoint(best_ckpt)
                    logger.info(f"New best loss: {best_loss:.4f}")
        
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1} completed. Avg loss: {avg_epoch_loss:.4f}")
    
    # Final save
    final_ckpt = Path(config['save_dir']) / "sam2_multichannel_final.pt"
    finetuner.save_checkpoint(final_ckpt)
    logger.info(f"Fine-tuning completed!")
    logger.info(f"Final model: {final_ckpt}")
    logger.info(f"Best model: {Path(config['save_dir']) / 'sam2_best.pt'}")


if __name__ == "__main__":
    main()