#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=/home/aman.kukde/cilia_ai/finetune_runs/logs/%x_%j.log
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=Cilia_Segmentation_Train
#SBATCH --time=48:00:00

# Configuration
IMAGE_DIR="/path/to/your/images"  # UPDATE THIS
MASK_DIR="/path/to/your/masks"    # UPDATE THIS
OUTPUT_DIR="/home/aman.kukde/cilia_ai/finetune_runs/models"

# Model architecture - options: unet, unet_small, unet++, segformer-b0, segformer-b1, deeplabv3-resnet50, etc.
# Use --list_models flag to see all available architectures
ARCHITECTURE="unet"

# Training parameters
NUM_EPOCHS=50
BATCH_SIZE=4
LEARNING_RATE=1e-4
IMAGE_SIZE=512  # 512 recommended for most models, 1024 for SAM2

# Channel configuration
C1_IDX=0  # Index of channel 1 in your TIFF files (C0)
C2_IDX=1  # Index of channel 2 in your TIFF files (C1)
DUAL_MODE="average"  # Options: average, zeros, overlay

# Which models to train (can specify subset: c1 c2 dual)
TRAIN_MODELS="c1 c2 dual"

# Use pretrained weights (for Hugging Face models)
PRETRAINED="--pretrained"

# Activate environment
cd /home/aman.kukde/cilia_ai/
source /home/aman.kukde/cilia_ai/.venv/bin/activate

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "/home/aman.kukde/cilia_ai/finetune_runs/logs"

# Run training
echo "Starting segmentation model training"
echo "=========================================="
echo "Image directory: $IMAGE_DIR"
echo "Mask directory: $MASK_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Architecture: $ARCHITECTURE"
echo "Training models: $TRAIN_MODELS"
echo "Dual mode: $DUAL_MODE"
echo "=========================================="

python train_segmentation.py \
    --image_dir "$IMAGE_DIR" \
    --mask_dir "$MASK_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --architecture "$ARCHITECTURE" \
    $PRETRAINED \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --image_size "$IMAGE_SIZE" \
    --c1_idx "$C1_IDX" \
    --c2_idx "$C2_IDX" \
    --dual_mode "$DUAL_MODE" \
    --train_models $TRAIN_MODELS \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --random_seed 42

echo "Training complete!"
echo "Results saved to: $OUTPUT_DIR"
