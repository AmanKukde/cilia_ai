# Training and Inference Scripts

This directory contains standalone scripts for training and inference.

## Files

- **`train_segmentation_npz.py`** - Train on .npy/.npz files (NCHW format)
- **`train_segmentation.py`** - Train on TIFF files
- **`finetune_sam2_multichannel.py`** - SAM2-specific finetuning
- **`visualize_predictions.py`** - Visualize model predictions
- **`inference_example.py`** - Example inference script

## Usage

All scripts support `--help` for detailed usage:

```bash
python train_segmentation_npz.py --help
```

## Import Note

Since scripts are in a subdirectory, you may need to adjust imports:

```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Now you can import from parent directory
from dataset_npz import CiliaNPZDataset
from models.model_factory import ModelFactory
```
