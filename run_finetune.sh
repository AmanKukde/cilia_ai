#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=/home/aman.kukde/cilia_ai/finetune_runs/logs/%x_%j.log
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=SAM2_Cilia_Finetune
#SBATCH --time=48:00:00

# Configuration
IMAGE_DIR="/path/to/your/images"  # UPDATE THIS
MASK_DIR="/path/to/your/masks"    # UPDATE THIS
OUTPUT_DIR="/home/aman.kukde/cilia_ai/finetune_runs/models"
SAM2_CFG="/path/to/sam2/config.yaml"  # UPDATE THIS
SAM2_CHECKPOINT="/path/to/sam2/checkpoint.pth"  # UPDATE THIS

# Training parameters
NUM_EPOCHS=50
BATCH_SIZE=4
LEARNING_RATE=1e-5
IMAGE_SIZE=1024

# Channel configuration
C1_IDX=2  # Index of channel 1 in your TIFF files
C2_IDX=3  # Index of channel 2 in your TIFF files
DUAL_MODE="average"  # Options: average, zeros, overlay

# Which models to train (can specify subset: c1 c2 dual)
TRAIN_MODELS="c1 c2 dual"

# Activate environment
cd /home/aman.kukde/cilia_ai/
source /home/aman.kukde/cilia_ai/.venv/bin/activate

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "/home/aman.kukde/cilia_ai/finetune_runs/logs"

# Run finetuning
echo "Starting SAM2 finetuning for multichannel cilia data"
echo "=========================================="
echo "Image directory: $IMAGE_DIR"
echo "Mask directory: $MASK_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Training models: $TRAIN_MODELS"
echo "Dual mode: $DUAL_MODE"
echo "=========================================="

python finetune_sam2_multichannel.py \
    --image_dir "$IMAGE_DIR" \
    --mask_dir "$MASK_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --sam2_cfg "$SAM2_CFG" \
    --sam2_checkpoint "$SAM2_CHECKPOINT" \
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

echo "Finetuning complete!"
echo "Results saved to: $OUTPUT_DIR"
