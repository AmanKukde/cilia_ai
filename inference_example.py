"""
Example script for using finetuned SAM2 models for inference on new cilia images.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import tifffile as tf
from tqdm import tqdm

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.utils import prepare_sam2_input
from finetune_sam2_multichannel import CiliaMultichannelDataset


def predict_with_finetuned_model(
    image_paths: list,
    model_checkpoint: str,
    sam2_cfg: str,
    channel_mode: str = 'c1_only',
    channel_indices: tuple = (2, 3),
    dual_mode: str = 'average',
    output_dir: str = './predictions',
    device: str = 'cuda'
):
    """
    Run inference with a finetuned SAM2 model.

    Args:
        image_paths: List of paths to multichannel TIFF images
        model_checkpoint: Path to finetuned model checkpoint
        sam2_cfg: Path to SAM2 config
        channel_mode: 'c1_only', 'c2_only', or 'dual'
        channel_indices: Tuple of (C1_idx, C2_idx)
        dual_mode: 'average', 'zeros', or 'overlay'
        output_dir: Directory to save predictions
        device: Device to use
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load the finetuned model
    print(f"Loading finetuned model from {model_checkpoint}")

    # Build base SAM2 model
    model = build_sam2(sam2_cfg, model_checkpoint, device=device)
    model.eval()

    # Create predictor
    predictor = SAM2ImagePredictor(model)

    predictions_list = []

    print(f"Running inference on {len(image_paths)} images...")

    for img_path in tqdm(image_paths):
        # Load multichannel image
        image = tf.imread(img_path)

        # Prepare channels based on mode
        if channel_mode == 'c1_only':
            c1 = image[:, :, channel_indices[0]]
            from sam2.utils import normalize_channel
            c1_norm = normalize_channel(c1)
            rgb_image = np.stack([c1_norm] * 3, axis=-1)

        elif channel_mode == 'c2_only':
            c2 = image[:, :, channel_indices[1]]
            from sam2.utils import normalize_channel
            c2_norm = normalize_channel(c2)
            rgb_image = np.stack([c2_norm] * 3, axis=-1)

        elif channel_mode == 'dual':
            from sam2.utils import normalize_channel
            c1 = normalize_channel(image[:, :, channel_indices[0]])
            c2 = normalize_channel(image[:, :, channel_indices[1]])

            if dual_mode == 'average':
                c3 = (c1 + c2) / 2.0
            elif dual_mode == 'zeros':
                c3 = np.zeros_like(c1)
            elif dual_mode == 'overlay':
                c3 = np.maximum(c1, c2)

            rgb_image = np.stack([c1, c2, c3], axis=-1)

        # Convert to uint8
        rgb_image = (rgb_image * 255).astype(np.uint8)

        # Set image
        predictor.set_image(rgb_image)

        # Predict (automatic segmentation without prompts)
        # For automatic segmentation, we need to use the model directly
        with torch.no_grad():
            # Convert to tensor
            image_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).float()
            image_tensor = image_tensor.unsqueeze(0).to(device)

            # Get embeddings
            image_embeddings = model.image_encoder(image_tensor)

            # Prompt encoder (no prompts)
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None
            )

            # Mask decoder
            low_res_masks, iou_predictions = model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )

            # Upsample to original size
            import torch.nn.functional as F
            pred_masks = F.interpolate(
                low_res_masks,
                size=(rgb_image.shape[0], rgb_image.shape[1]),
                mode='bilinear',
                align_corners=False
            )

            # Convert to binary
            pred_binary = (torch.sigmoid(pred_masks) > 0.5).cpu().numpy()[0, 0]

        predictions_list.append(pred_binary.astype(np.uint8))

        # Save individual prediction
        img_name = Path(img_path).stem
        tf.imwrite(
            output_path / f"{img_name}_prediction.tif",
            pred_binary.astype(np.uint8)
        )

    # Stack all predictions
    predictions_stack = np.stack(predictions_list, axis=0)

    # Save as stack
    tf.imwrite(
        output_path / "all_predictions.tif",
        predictions_stack.astype(np.uint8)
    )

    print(f"\nInference complete!")
    print(f"Predictions saved to: {output_path}")
    print(f"Prediction shape: {predictions_stack.shape}")

    return predictions_stack


def main():
    parser = argparse.ArgumentParser(description='Run inference with finetuned SAM2 model')

    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing multichannel TIFF images')
    parser.add_argument('--model_checkpoint', type=str, required=True,
                        help='Path to finetuned model checkpoint')
    parser.add_argument('--sam2_cfg', type=str, required=True,
                        help='Path to SAM2 config file')
    parser.add_argument('--output_dir', type=str, default='./predictions',
                        help='Output directory for predictions')

    parser.add_argument('--channel_mode', type=str, default='c1_only',
                        choices=['c1_only', 'c2_only', 'dual'],
                        help='Channel mode (must match training mode)')
    parser.add_argument('--c1_idx', type=int, default=2,
                        help='Channel 1 index')
    parser.add_argument('--c2_idx', type=int, default=3,
                        help='Channel 2 index')
    parser.add_argument('--dual_mode', type=str, default='average',
                        choices=['average', 'zeros', 'overlay'],
                        help='Dual mode (if using dual channel_mode)')

    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Get all TIFF images
    image_paths = sorted([str(p) for p in Path(args.image_dir).glob('*.tif')])

    print(f"Found {len(image_paths)} images")

    if len(image_paths) == 0:
        print("No images found!")
        return

    # Run inference
    predictions = predict_with_finetuned_model(
        image_paths=image_paths,
        model_checkpoint=args.model_checkpoint,
        sam2_cfg=args.sam2_cfg,
        channel_mode=args.channel_mode,
        channel_indices=(args.c1_idx, args.c2_idx),
        dual_mode=args.dual_mode,
        output_dir=args.output_dir,
        device=args.device
    )

    print("\nDone!")


if __name__ == '__main__':
    main()
