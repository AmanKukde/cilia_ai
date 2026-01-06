# SAM2 Multichannel Finetuning for Cilia Segmentation

This directory contains code for finetuning SAM2 (Segment Anything Model 2) on multichannel cilia microscopy data.

## Overview

The finetuning pipeline trains **3 separate SAM2 models** to handle different channel configurations:

1. **C1 Model** (`c1_model`): Uses Channel 1 only (C1 replicated 3 times → C1*3)
2. **C2 Model** (`c2_model`): Uses Channel 2 only (C2 replicated 3 times → C2*3)
3. **Dual Model** (`dual_model`): Uses both channels in different combinations:
   - **C1, C2, Average**: First channel = C1, Second channel = C2, Third channel = (C1+C2)/2
   - **C1, C2, Zeros**: First channel = C1, Second channel = C2, Third channel = zeros
   - **C1, C2, Overlay**: First channel = C1, Second channel = C2, Third channel = max(C1, C2)

## Files

- `finetune_sam2_multichannel.py`: Main training script with dataset class and training loop
- `sam2/utils.py`: Utility functions for data processing and channel handling
- `sam2/plotting_utils.py`: Visualization utilities for training and predictions
- `run_finetune.sh`: SLURM batch script for running training on cluster
- `FINETUNE_README.md`: This documentation file

## Installation

### Requirements

```bash
# Install required packages
pip install torch torchvision torchaudio
pip install tifffile numpy matplotlib scikit-image tqdm

# Install SAM2
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
```

### Download SAM2 Checkpoint

Download a pretrained SAM2 checkpoint from Meta:
- SAM2 Large: https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
- SAM2 Base+: https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt
- SAM2 Small: https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt
- SAM2 Tiny: https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt

## Data Preparation

### Expected Data Format

Your data should be organized as:

```
data/
├── images/
│   ├── image_001.tif  # Multichannel TIFF (H, W, C)
│   ├── image_002.tif
│   └── ...
└── masks/
    ├── image_001.tif  # Binary mask (H, W) or (H, W, 1)
    ├── image_002.tif
    └── ...
```

**Requirements:**
- Images must be multichannel TIFF files with at least 2 channels
- Masks must be binary (0/1) TIFF files
- Image and mask filenames must correspond (same base name)
- Default channel indices are 2 and 3 (can be configured)

### Channel Indices

By default, the script uses:
- **C1_IDX = 2**: Channel 1 (e.g., protein channel)
- **C2_IDX = 3**: Channel 2 (e.g., cilia marker channel)

Adjust these in `run_finetune.sh` or via command-line arguments if your data has different channel ordering.

## Usage

### Option 1: Using the SLURM Script (Recommended for Cluster)

1. Edit `run_finetune.sh` and update the following paths:
   ```bash
   IMAGE_DIR="/path/to/your/images"
   MASK_DIR="/path/to/your/masks"
   SAM2_CFG="/path/to/sam2/config.yaml"
   SAM2_CHECKPOINT="/path/to/sam2/checkpoint.pth"
   ```

2. Configure training parameters (optional):
   ```bash
   NUM_EPOCHS=50
   BATCH_SIZE=4
   LEARNING_RATE=1e-5
   C1_IDX=2
   C2_IDX=3
   DUAL_MODE="average"  # or "zeros" or "overlay"
   TRAIN_MODELS="c1 c2 dual"  # train all 3 models
   ```

3. Submit the job:
   ```bash
   sbatch run_finetune.sh
   ```

### Option 2: Direct Python Execution

Train all 3 models:
```bash
python finetune_sam2_multichannel.py \
    --image_dir /path/to/images \
    --mask_dir /path/to/masks \
    --output_dir ./outputs \
    --sam2_cfg /path/to/sam2/config.yaml \
    --sam2_checkpoint /path/to/sam2/checkpoint.pth \
    --num_epochs 50 \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --c1_idx 2 \
    --c2_idx 3 \
    --dual_mode average \
    --train_models c1 c2 dual
```

Train only specific models:
```bash
# Train only C1 model
python finetune_sam2_multichannel.py \
    --image_dir /path/to/images \
    --mask_dir /path/to/masks \
    --output_dir ./outputs \
    --sam2_cfg /path/to/sam2/config.yaml \
    --sam2_checkpoint /path/to/sam2/checkpoint.pth \
    --train_models c1

# Train only dual model with overlay mode
python finetune_sam2_multichannel.py \
    --image_dir /path/to/images \
    --mask_dir /path/to/masks \
    --output_dir ./outputs \
    --sam2_cfg /path/to/sam2/config.yaml \
    --sam2_checkpoint /path/to/sam2/checkpoint.pth \
    --dual_mode overlay \
    --train_models dual
```

## Command-Line Arguments

### Required Arguments
- `--image_dir`: Directory containing multichannel TIFF images
- `--mask_dir`: Directory containing binary mask TIFFs
- `--output_dir`: Output directory for trained models and metrics
- `--sam2_cfg`: Path to SAM2 config file
- `--sam2_checkpoint`: Path to pretrained SAM2 checkpoint

### Training Arguments
- `--num_epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 4)
- `--learning_rate`: Learning rate (default: 1e-5)
- `--image_size`: Image size for SAM2 (default: 1024)

### Channel Arguments
- `--c1_idx`: Index of channel 1 in TIFF files (default: 2)
- `--c2_idx`: Index of channel 2 in TIFF files (default: 3)
- `--dual_mode`: Mode for dual channel model (choices: average, zeros, overlay; default: average)

### Model Selection
- `--train_models`: Which models to train (choices: c1, c2, dual; default: all three)

### Data Split Arguments
- `--train_ratio`: Ratio of data for training (default: 0.8)
- `--val_ratio`: Ratio of data for validation (default: 0.1)
- `--test_ratio`: Ratio of data for testing (default: 0.1)
- `--random_seed`: Random seed for reproducible splits (default: 42)

## Output Structure

After training, the output directory will contain:

```
outputs/
├── c1_model/
│   ├── best_model.pth              # Best checkpoint (highest val IoU)
│   ├── checkpoint_epoch_10.pth     # Checkpoint every 10 epochs
│   ├── checkpoint_epoch_20.pth
│   ├── ...
│   ├── metrics.json                # Training/validation metrics history
│   └── training_metrics.png        # Plots of loss and IoU curves
├── c2_model/
│   └── ...                         # Same structure as c1_model
├── dual_model/
│   └── ...                         # Same structure as c1_model
└── all_results.json                # Summary of best IoU for each model
```

## Model Architecture

The finetuning approach:

1. **Frozen Components**:
   - Image encoder (frozen by default for efficiency)
   - Prompt encoder (frozen by default)

2. **Trainable Components**:
   - Mask decoder (fully trainable)

3. **Loss Function**:
   - Combined BCE + Dice loss (50% each by default)

4. **Optimizer**:
   - AdamW with weight decay (1e-4)

## Channel Transformation Details

### C1 Model (`c1_only`)
- Extracts channel at index `c1_idx` (default: 2)
- Normalizes to [0, 1] using percentile clipping
- Replicates to RGB: `[C1, C1, C1]`

### C2 Model (`c2_only`)
- Extracts channel at index `c2_idx` (default: 3)
- Normalizes to [0, 1] using percentile clipping
- Replicates to RGB: `[C2, C2, C2]`

### Dual Model (`dual`)

**Average Mode** (`--dual_mode average`):
- Channel R = normalized C1
- Channel G = normalized C2
- Channel B = (C1 + C2) / 2

**Zeros Mode** (`--dual_mode zeros`):
- Channel R = normalized C1
- Channel G = normalized C2
- Channel B = zeros

**Overlay Mode** (`--dual_mode overlay`):
- Channel R = normalized C1
- Channel G = normalized C2
- Channel B = max(C1, C2)

## Training Tips

1. **Start with small learning rate**: 1e-5 to 1e-6 works well for finetuning
2. **Monitor validation IoU**: Early stopping based on validation IoU
3. **Batch size**: Adjust based on GPU memory (4-8 typical for 1024x1024 images)
4. **Epochs**: 30-50 epochs usually sufficient for convergence
5. **Data augmentation**: Currently not implemented, can be added via `transform` parameter

## Monitoring Training

Metrics are logged to:
- **Console**: Real-time progress bars with loss and IoU
- **JSON files**: `metrics.json` in each model directory
- **Plots**: `training_metrics.png` generated after training

Example metrics JSON:
```json
{
  "train_loss": [0.45, 0.38, 0.32, ...],
  "train_iou": [0.62, 0.68, 0.74, ...],
  "val_loss": [0.48, 0.41, 0.36, ...],
  "val_iou": [0.60, 0.66, 0.71, ...]
}
```

## Using Trained Models

To use a trained model for inference:

```python
from sam2.build_sam import build_sam2
import torch

# Load the finetuned model
model = build_sam2(
    config_file="/path/to/sam2/config.yaml",
    ckpt_path="outputs/c1_model/best_model.pth",
    device="cuda"
)

# Use for prediction
# ... (standard SAM2 inference code)
```

## Troubleshooting

### Out of Memory
- Reduce `--batch_size` (try 2 or 1)
- Reduce `--image_size` (try 512)
- Use smaller SAM2 model (tiny or small)

### Poor Performance
- Check channel indices are correct (`--c1_idx`, `--c2_idx`)
- Verify masks are binary (0 and 1)
- Try different `--dual_mode` options
- Increase training epochs
- Adjust learning rate

### Data Loading Errors
- Ensure images and masks have matching filenames
- Check TIFF files are readable with `tifffile`
- Verify channel indices exist in your images

## Advanced Usage

### Custom Data Augmentation

Add augmentation by modifying the dataset:

```python
import albumentations as A

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
])

dataset = CiliaMultichannelDataset(
    ...,
    transform=transform
)
```

### Training on Custom Channels

To train on different channel combinations, modify `channel_indices`:

```python
# Use channels 0 and 1 instead of 2 and 3
channel_indices = (0, 1)
```

## Citation

If you use this code, please cite:

- SAM2: [Segment Anything 2](https://github.com/facebookresearch/segment-anything-2)
- Your research paper (when published)

## License

This code is provided for research purposes. See SAM2 license for model usage terms.

## Contact

For questions or issues, please open an issue on the repository or contact the maintainers.
