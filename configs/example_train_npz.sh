#!/bin/bash
# Example: Training segmentation models on .npz data

# Your data paths
IMAGES_NPZ="/home/aman.kukde/cilia_ai/data/prompted_outputs/images_cleaned_prompted.npy"
MASKS_NPZ="/home/aman.kukde/cilia_ai/data/prompted_outputs/predictions_sam2_cleaned.npy"
OUTPUT_DIR="./outputs"

# Channel configuration
C1_IDX=0  # C0
C2_IDX=1  # C1

echo "Training with U-Net..."
python train_segmentation_npz.py \
    --images_npz "$IMAGES_NPZ" \
    --masks_npz "$MASKS_NPZ" \
    --output_dir "$OUTPUT_DIR" \
    --architecture unet \
    --num_epochs 50 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --image_size 512 \
    --c1_idx $C1_IDX \
    --c2_idx $C2_IDX \
    --dual_mode average \
    --train_models c1 c2 dual

echo ""
echo "Training complete! Check results in $OUTPUT_DIR"
