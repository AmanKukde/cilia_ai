# Quick Start Guide

## 🎯 Data Format Support

This repository supports **two data formats**:

1. **TIFF files** (`.tif`) - Use `train_segmentation.py`
2. **Numpy files** (`.npy` or `.npz`) - Use `train_segmentation_npz.py` ⭐ **NEW!**

If you have `.npy` or `.npz` files with NCHW format, see the **[NPZ Data Guide](#training-on-npy-npz-files)** below or read **`NPZ_DATA_README.md`** for complete documentation.

## Fixed: SAM2 Package Conflict ✅

The local `sam2/` directory has been renamed to `cilia_utils/` to avoid conflicts with the installed SAM2 package.

### What Changed

**Before (Broken):**
```
sam2/                    # Our utilities
  ├── utils.py
  └── plotting_utils.py
```
This conflicted with installed SAM2:
```python
from sam2.build_sam import build_sam2  # ❌ Conflicted!
```

**After (Fixed):**
```
cilia_utils/             # Our utilities (renamed!)
  ├── utils.py
  └── plotting_utils.py
```
Now imports work correctly:
```python
from sam2.build_sam import build_sam2           # ✅ From installed SAM2
from cilia_utils.utils import normalize_channel # ✅ From our utilities
```

## Installation

### 1. Install Core Dependencies

```bash
pip install torch torchvision torchaudio
pip install tifffile numpy matplotlib scikit-image tqdm
```

### 2. For Hugging Face Models (Optional)

```bash
pip install transformers
```

### 3. For SAM2 (Optional)

```bash
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
cd ..
```

## Testing Installation

Run the test script to verify everything is set up correctly:

```bash
./test_imports.sh
```

## Usage Examples

### List Available Models

```bash
python train_segmentation.py --list_models
```

### Train U-Net on C0 and C1 Channels

```bash
python train_segmentation.py \
    --image_dir /path/to/images \
    --mask_dir /path/to/masks \
    --output_dir ./outputs \
    --architecture unet \
    --num_epochs 50 \
    --batch_size 4 \
    --c1_idx 0 \
    --c2_idx 1 \
    --train_models c1 c2 dual
```

### Train SegFormer (Lightweight Transformer)

```bash
python train_segmentation.py \
    --image_dir /path/to/images \
    --mask_dir /path/to/masks \
    --output_dir ./outputs \
    --architecture segformer-b0 \
    --pretrained \
    --num_epochs 30 \
    --batch_size 8 \
    --learning_rate 5e-5 \
    --image_size 512
```

### Train SAM2 Models

```bash
python finetune_sam2_multichannel.py \
    --image_dir /path/to/images \
    --mask_dir /path/to/masks \
    --output_dir ./outputs \
    --sam2_cfg /path/to/sam2_config.yaml \
    --sam2_checkpoint /path/to/sam2_checkpoint.pth \
    --num_epochs 50
```

## Available Architectures

### U-Net Family
- `unet` - Standard U-Net (64 base channels)
- `unet_small` - Small U-Net (32 base channels)
- `unet_tiny` - Tiny U-Net (16 base channels)
- `unet++` - U-Net++ with nested skip connections

### Hugging Face Models
- `segformer-b0` to `segformer-b5` - SegFormer variants
- `deeplabv3-resnet50` - DeepLabV3 with ResNet50
- `deeplabv3-resnet101` - DeepLabV3 with ResNet101
- `mask2former-swin-tiny` - Mask2Former with Swin Tiny
- `upernet-swin-tiny` - UPerNet with Swin Tiny
- `beit-base` - BEiT Base for segmentation

## Training on .npy/.npz Files

If your data is stored as `.npy` or `.npz` files in NCHW format (N, C, H, W), use the specialized training script:

### Quick Example

```bash
python train_segmentation_npz.py \
    --images_npz /path/to/images.npy \
    --masks_npz /path/to/masks.npy \
    --output_dir ./outputs \
    --architecture unet \
    --batch_size 8 \
    --image_size 512 \
    --c1_idx 0 \
    --c2_idx 1 \
    --train_models c1 c2 dual
```

### Your Specific Data

Based on your paths:

```bash
python train_segmentation_npz.py \
    --images_npz /home/aman.kukde/cilia_ai/data/prompted_outputs/images_cleaned_prompted.npy \
    --masks_npz /home/aman.kukde/cilia_ai/data/prompted_outputs/predictions_sam2_cleaned.npy \
    --output_dir ./outputs \
    --architecture segformer-b0 \
    --pretrained \
    --batch_size 8 \
    --image_size 512 \
    --num_epochs 30
```

**See `NPZ_DATA_README.md` for complete documentation!**

## Channel Configuration

By default, the code uses:
- **C1_IDX = 0** (Channel 0, C0)
- **C2_IDX = 1** (Channel 1, C1)

For each architecture, 3 models are trained:
1. **C0 model**: Uses only C0 (replicated 3x)
2. **C1 model**: Uses only C1 (replicated 3x)
3. **Dual model**: Uses both C0 and C1 (with average/zeros/overlay in 3rd channel)

## Troubleshooting

### "RuntimeError: You're likely running Python from the parent directory"

**Fixed!** This error occurred due to the `sam2/` directory conflict. The directory has been renamed to `cilia_utils/`.

If you still see this error:
1. Ensure you pulled the latest changes
2. Check that `sam2/` directory does not exist
3. Run `./test_imports.sh` to verify

### Import Errors

If you see import errors, check dependencies:
```bash
pip list | grep -E 'torch|numpy|tifffile|transformers'
```

### Missing Dependencies

Install missing packages:
```bash
pip install torch torchvision tifffile numpy matplotlib scikit-image tqdm transformers
```

## Directory Structure

```
cilia_ai/
├── cilia_utils/              # Our utility functions
│   ├── __init__.py
│   ├── utils.py              # Data processing utilities
│   └── plotting_utils.py     # Visualization utilities
├── models/                   # Model architectures
│   ├── __init__.py
│   ├── unet.py               # U-Net implementations
│   └── model_factory.py      # Model factory
├── train_segmentation.py     # Main training script
├── finetune_sam2_multichannel.py  # SAM2 training
├── run_train.sh              # SLURM script for training
├── run_finetune.sh           # SLURM script for SAM2
├── test_imports.sh           # Test script
├── QUICK_START.md            # This file
└── FINETUNE_README.md        # Full documentation
```

## Data Format

Your data should be organized as:
```
data/
├── images/
│   ├── image_001.tif  # Multichannel TIFF (H, W, C)
│   ├── image_002.tif
│   └── ...
└── masks/
    ├── image_001.tif  # Binary mask (H, W)
    ├── image_002.tif
    └── ...
```

## Next Steps

1. **Prepare your data** in the format above
2. **Install dependencies** (see Installation section)
3. **Test installation** with `./test_imports.sh`
4. **List available models** with `python train_segmentation.py --list_models`
5. **Start training** with one of the examples above
6. **Check outputs** in the specified output directory

## For More Information

See **FINETUNE_README.md** for comprehensive documentation including:
- Detailed architecture descriptions
- Advanced training options
- Inference examples
- Performance tips
- Troubleshooting guide

## Support

If you encounter issues:
1. Run `./test_imports.sh` to diagnose
2. Check `FINETUNE_README.md` for detailed docs
3. Ensure all dependencies are installed
4. Verify your data format matches requirements
