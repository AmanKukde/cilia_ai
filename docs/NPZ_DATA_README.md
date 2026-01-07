# Training on .npz Data (NCHW Format)

This guide explains how to train segmentation models on data stored in `.npz` (numpy compressed) format with NCHW layout.

## Data Format

Your data should be in `.npz` files with the following format:

### Images
- **Shape**: `(N, C, H, W)` - NCHW format
  - `N`: Number of samples
  - `C`: Number of channels
  - `H`: Height
  - `W`: Width
- **Example**: `(180, 5, 256, 256)` = 180 images with 5 channels, 256x256 pixels

### Masks
- **Shape**: `(N, H, W)` or `(N, 1, H, W)` - NHW or NCHW format
  - Binary masks (0 and 1)
- **Example**: `(180, 256, 256)` = 180 masks, 256x256 pixels

## Quick Start

### 1. List Available Models

```bash
python train_segmentation_npz.py --list_models
```

### 2. Train with Default Settings

```bash
python train_segmentation_npz.py \
    --images_npz /path/to/images.npz \
    --masks_npz /path/to/masks.npz \
    --output_dir ./outputs \
    --architecture unet \
    --c1_idx 0 \
    --c2_idx 1
```

### 3. Using Your Specific Paths

Based on your terminal output, here's the exact command:

```bash
python train_segmentation_npz.py \
    --images_npz /home/aman.kukde/cilia_ai/data/prompted_outputs/images_cleaned_prompted.npy \
    --masks_npz /home/aman.kukde/cilia_ai/data/prompted_outputs/predictions_sam2_cleaned.npy \
    --output_dir ./predictions \
    --architecture segformer-b0 \
    --pretrained \
    --batch_size 8 \
    --image_size 512 \
    --c1_idx 0 \
    --c2_idx 1 \
    --num_epochs 30
```

**Note**: The script also works with `.npy` files! It will automatically load them.

## Features

### Automatic Key Detection

The script automatically detects the array keys in your `.npz` files. It tries common names:
- **Images**: `'images'`, `'arr_0'`, `'data'`, `'x'`
- **Masks**: `'masks'`, `'arr_0'`, `'labels'`, `'y'`

You can also specify keys manually:
```bash
python train_segmentation_npz.py \
    --images_npz data/images.npz \
    --masks_npz data/masks.npz \
    --images_key my_images \
    --masks_key my_masks \
    ...
```

### Channel Selection

Specify which channels to use from your multichannel data:

```bash
--c1_idx 0  # Use channel 0 (C0)
--c2_idx 1  # Use channel 1 (C1)
```

### Three Training Modes

Train 3 models automatically:
1. **C0 model**: Uses only channel 0 (replicated 3x for RGB)
2. **C1 model**: Uses only channel 1 (replicated 3x for RGB)
3. **Dual model**: Uses both channels with combination

Control which models to train:
```bash
--train_models c1 c2 dual  # Train all three (default)
--train_models c1          # Train only C0 model
--train_models dual        # Train only dual-channel model
```

### Dual Channel Modes

When training the dual model, choose how to combine channels:

```bash
--dual_mode average  # 3rd channel = (C0 + C1) / 2 (default)
--dual_mode zeros    # 3rd channel = zeros
--dual_mode overlay  # 3rd channel = max(C0, C1)
```

## Architecture Options

### U-Net Family (Fast, No Dependencies)
```bash
--architecture unet        # Standard U-Net (64 base channels)
--architecture unet_small  # Small U-Net (32 base channels)
--architecture unet_tiny   # Tiny U-Net (16 base channels)
--architecture unet++      # U-Net++ with nested connections
```

### Hugging Face Models (Requires `pip install transformers`)

**SegFormer** (Lightweight transformers):
```bash
--architecture segformer-b0 --pretrained  # Smallest (3.7M params)
--architecture segformer-b1 --pretrained  # Small
--architecture segformer-b2 --pretrained  # Medium
--architecture segformer-b3 --pretrained  # Large
--architecture segformer-b4 --pretrained  # Very large
--architecture segformer-b5 --pretrained  # Largest
```

**DeepLabV3** (Classic CNN):
```bash
--architecture deeplabv3-resnet50 --pretrained
--architecture deeplabv3-resnet101 --pretrained
```

**Mask2Former** (Universal segmentation):
```bash
--architecture mask2former-swin-tiny --pretrained
--architecture mask2former-swin-small --pretrained
--architecture mask2former-swin-base --pretrained
```

## Training Parameters

### Essential Parameters

```bash
--num_epochs 50          # Number of training epochs
--batch_size 4           # Batch size (adjust based on GPU memory)
--learning_rate 1e-4     # Learning rate (1e-4 for most models)
--image_size 512         # Resize images to this size
```

### Data Splitting

```bash
--train_ratio 0.8   # 80% for training
--val_ratio 0.1     # 10% for validation
--test_ratio 0.1    # 10% for testing
--random_seed 42    # For reproducibility
```

## Complete Examples

### Example 1: Quick Training with U-Net

```bash
python train_segmentation_npz.py \
    --images_npz data/images.npz \
    --masks_npz data/masks.npz \
    --output_dir ./outputs \
    --architecture unet \
    --num_epochs 30 \
    --batch_size 8 \
    --image_size 256
```

### Example 2: Production Training with SegFormer

```bash
python train_segmentation_npz.py \
    --images_npz data/images.npz \
    --masks_npz data/masks.npz \
    --output_dir ./production_models \
    --architecture segformer-b1 \
    --pretrained \
    --num_epochs 100 \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --image_size 512 \
    --train_models c1 c2 dual \
    --dual_mode average
```

### Example 3: Fast Prototyping with Tiny U-Net

```bash
python train_segmentation_npz.py \
    --images_npz data/images.npz \
    --masks_npz data/masks.npz \
    --output_dir ./prototype \
    --architecture unet_tiny \
    --num_epochs 10 \
    --batch_size 16 \
    --image_size 128 \
    --train_models c1  # Just train C0 model for quick test
```

### Example 4: Training on Specific Channels

If your data has 5 channels and you want to use channels 2 and 4:

```bash
python train_segmentation_npz.py \
    --images_npz data/images.npz \
    --masks_npz data/masks.npz \
    --output_dir ./outputs \
    --architecture unet \
    --c1_idx 2  # Use channel 2
    --c2_idx 4  # Use channel 4
    --train_models dual
```

## Output Structure

After training, you'll get:

```
outputs/
├── c1_unet/                    # C0 channel model
│   ├── best_model.pth          # Best checkpoint
│   ├── checkpoint_epoch_10.pth # Periodic checkpoints
│   ├── checkpoint_epoch_20.pth
│   ├── metrics.json            # Training metrics
│   └── training_metrics.png    # Loss/IoU plots
├── c2_unet/                    # C1 channel model
│   └── ...
├── dual_unet/                  # Dual-channel model
│   └── ...
└── all_results.json            # Summary of all models
```

## Monitoring Training

The script provides real-time progress:
- Progress bars for each epoch
- Loss and IoU metrics
- Automatic saving of best model
- Periodic checkpoints every 10 epochs

Example output:
```
Epoch 5/50
--------------------------------------------------------------------------------
Training: 100%|███████| 45/45 [00:12<00:00,  3.62it/s, loss=0.234, iou=0.756]
Train - Loss: 0.2341, IoU: 0.7563
Validation: 100%|███████| 11/11 [00:02<00:00,  4.21it/s, loss=0.245, iou=0.742]
Val   - Loss: 0.2453, IoU: 0.7421
✓ Saved best model (IoU: 0.7421)
```

## Troubleshooting

### "File not found" Error

Make sure your paths are correct and files exist:
```bash
ls -lh /path/to/images.npz
ls -lh /path/to/masks.npz
```

### "Shape mismatch" Error

Check your data shapes:
```python
import numpy as np
images = np.load('images.npz')
print("Keys:", list(images.keys()))
print("Shape:", images['arr_0'].shape)  # Replace with actual key
```

Expected format:
- Images: `(N, C, H, W)` - 4D array
- Masks: `(N, H, W)` or `(N, 1, H, W)` - 3D or 4D array

### "Channel index out of range" Error

Check how many channels your images have:
```python
images = np.load('images.npz')['images']
print(f"Channels: {images.shape[1]}")  # NCHW format
```

Then use valid indices:
```bash
--c1_idx 0 --c2_idx 1  # For 2+ channels
--c1_idx 2 --c2_idx 3  # For 4+ channels
```

### Out of Memory Error

Reduce memory usage:
```bash
--batch_size 2           # Reduce batch size
--image_size 256         # Use smaller images
--architecture unet_tiny # Use smaller model
```

### Slow Training

Speed up training:
```bash
--batch_size 16          # Increase batch size (if memory allows)
--image_size 256         # Use smaller images
--architecture unet_small # Use faster model
--num_epochs 20          # Train fewer epochs for testing
```

## Performance Tips

### For Fast Training
- Use `unet_tiny` or `unet_small`
- Set `--image_size 256` or `128`
- Increase `--batch_size` to 16 or 32
- Use fewer epochs initially (10-20) for validation

### For Best Accuracy
- Use `segformer-b2` or `segformer-b3` with `--pretrained`
- Set `--image_size 512`
- Train for more epochs (100+)
- Use all three channel modes (`--train_models c1 c2 dual`)

### For Limited GPU Memory
- Use `unet_tiny` or `segformer-b0`
- Set `--batch_size 1` or `2`
- Set `--image_size 256`

## Next Steps

1. **Train your first model** with the quick start example
2. **Monitor metrics** in the output directory
3. **Compare models** using `all_results.json`
4. **Use best model** for inference (see inference examples)

## Additional Resources

- **Full Documentation**: See `FINETUNE_README.md`
- **Test Setup**: Run `./test_imports.sh`
- **List Models**: `python train_segmentation_npz.py --list_models`

## Common Workflows

### Workflow 1: Quick Validation
```bash
# Fast training to validate data/setup
python train_segmentation_npz.py \
    --images_npz data/images.npz \
    --masks_npz data/masks.npz \
    --output_dir ./test \
    --architecture unet_tiny \
    --num_epochs 5 \
    --batch_size 8 \
    --image_size 128 \
    --train_models c1
```

### Workflow 2: Full Training
```bash
# Production training
python train_segmentation_npz.py \
    --images_npz data/images.npz \
    --masks_npz data/masks.npz \
    --output_dir ./models \
    --architecture segformer-b1 \
    --pretrained \
    --num_epochs 100 \
    --batch_size 4 \
    --image_size 512 \
    --train_models c1 c2 dual
```

### Workflow 3: Compare Architectures
```bash
# Train multiple architectures to compare
for arch in unet unet++ segformer-b0 deeplabv3-resnet50; do
    python train_segmentation_npz.py \
        --images_npz data/images.npz \
        --masks_npz data/masks.npz \
        --output_dir ./comparison \
        --architecture $arch \
        --pretrained \
        --num_epochs 30 \
        --train_models dual
done
```
