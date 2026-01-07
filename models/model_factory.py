"""
Model factory for creating various segmentation architectures.
Supports U-Net, SAM2, and Hugging Face models.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any
import warnings

from .unet import UNet, UNetPlusPlus, UNetSmall, UNetTiny


class ModelFactory:
    """Factory class for creating segmentation models."""

    AVAILABLE_MODELS = {
        # U-Net variants
        'unet': 'Standard U-Net',
        'unet_small': 'Small U-Net (32 base channels)',
        'unet_tiny': 'Tiny U-Net (16 base channels)',
        'unet++': 'U-Net++ (Nested U-Net)',

        # Hugging Face models
        'segformer-b0': 'SegFormer B0 (lightweight)',
        'segformer-b1': 'SegFormer B1',
        'segformer-b2': 'SegFormer B2',
        'segformer-b3': 'SegFormer B3',
        'segformer-b4': 'SegFormer B4 (large)',
        'segformer-b5': 'SegFormer B5 (largest)',

        'deeplabv3-resnet50': 'DeepLabV3 with ResNet50',
        'deeplabv3-resnet101': 'DeepLabV3 with ResNet101',

        'mask2former-swin-tiny': 'Mask2Former with Swin Tiny',
        'mask2former-swin-small': 'Mask2Former with Swin Small',
        'mask2former-swin-base': 'Mask2Former with Swin Base',

        'upernet-swin-tiny': 'UPerNet with Swin Tiny',
        'upernet-swin-small': 'UPerNet with Swin Small',

        'beit-base': 'BEiT Base for segmentation',
        'beit-large': 'BEiT Large for segmentation',
    }

    @staticmethod
    def list_models() -> Dict[str, str]:
        """List all available models."""
        return ModelFactory.AVAILABLE_MODELS

    @staticmethod
    def create_model(
        model_name: str,
        num_classes: int = 1,
        pretrained: bool = True,
        **kwargs
    ) -> nn.Module:
        """
        Create a segmentation model.

        Args:
            model_name: Name of the model architecture
            num_classes: Number of output classes (1 for binary segmentation)
            pretrained: Load pretrained weights (for HF models)
            **kwargs: Additional model-specific arguments

        Returns:
            PyTorch model

        Raises:
            ValueError: If model_name is not recognized
        """
        model_name = model_name.lower()

        # U-Net variants
        if model_name == 'unet':
            return UNet(n_channels=3, n_classes=num_classes, **kwargs)
        elif model_name == 'unet_small':
            return UNetSmall(n_channels=3, n_classes=num_classes, **kwargs)
        elif model_name == 'unet_tiny':
            return UNetTiny(n_channels=3, n_classes=num_classes, **kwargs)
        elif model_name == 'unet++':
            return UNetPlusPlus(n_channels=3, n_classes=num_classes, **kwargs)

        # Hugging Face models
        elif model_name.startswith('segformer'):
            return ModelFactory._create_segformer(model_name, num_classes, pretrained)
        elif model_name.startswith('deeplabv3'):
            return ModelFactory._create_deeplabv3(model_name, num_classes, pretrained)
        elif model_name.startswith('mask2former'):
            return ModelFactory._create_mask2former(model_name, num_classes, pretrained)
        elif model_name.startswith('upernet'):
            return ModelFactory._create_upernet(model_name, num_classes, pretrained)
        elif model_name.startswith('beit'):
            return ModelFactory._create_beit(model_name, num_classes, pretrained)
        else:
            available = ', '.join(ModelFactory.AVAILABLE_MODELS.keys())
            raise ValueError(
                f"Unknown model: {model_name}\n"
                f"Available models: {available}"
            )

    @staticmethod
    def _create_segformer(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
        """Create SegFormer model from Hugging Face."""
        try:
            from transformers import SegformerForSemanticSegmentation, SegformerConfig
        except ImportError:
            raise ImportError(
                "SegFormer requires transformers. Install with: pip install transformers"
            )

        # Map model names to HF model IDs
        model_map = {
            'segformer-b0': 'nvidia/segformer-b0-finetuned-ade-512-512',
            'segformer-b1': 'nvidia/segformer-b1-finetuned-ade-512-512',
            'segformer-b2': 'nvidia/segformer-b2-finetuned-ade-512-512',
            'segformer-b3': 'nvidia/segformer-b3-finetuned-ade-512-512',
            'segformer-b4': 'nvidia/segformer-b4-finetuned-ade-512-512',
            'segformer-b5': 'nvidia/segformer-b5-finetuned-ade-640-640',
        }

        if pretrained:
            model_id = model_map.get(model_name, model_map['segformer-b0'])
            model = SegformerForSemanticSegmentation.from_pretrained(
                model_id,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            config = SegformerConfig(num_labels=num_classes)
            model = SegformerForSemanticSegmentation(config)

        return HFSegmentationWrapper(model, output_type='segformer')

    @staticmethod
    def _create_deeplabv3(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
        """Create DeepLabV3 model."""
        try:
            import torchvision.models.segmentation as segmentation_models
        except ImportError:
            raise ImportError(
                "DeepLabV3 requires torchvision. Install with: pip install torchvision"
            )

        if 'resnet101' in model_name:
            if pretrained:
                weights = segmentation_models.DeepLabV3_ResNet101_Weights.DEFAULT
            else:
                weights = None
            model = segmentation_models.deeplabv3_resnet101(weights=weights)
        else:  # resnet50
            if pretrained:
                weights = segmentation_models.DeepLabV3_ResNet50_Weights.DEFAULT
            else:
                weights = None
            model = segmentation_models.deeplabv3_resnet50(weights=weights)

        # Modify final layer for binary segmentation
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

        return TorchvisionSegmentationWrapper(model)

    @staticmethod
    def _create_mask2former(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
        """Create Mask2Former model from Hugging Face."""
        try:
            from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig
        except ImportError:
            raise ImportError(
                "Mask2Former requires transformers. Install with: pip install transformers"
            )

        model_map = {
            'mask2former-swin-tiny': 'facebook/mask2former-swin-tiny-ade-semantic',
            'mask2former-swin-small': 'facebook/mask2former-swin-small-ade-semantic',
            'mask2former-swin-base': 'facebook/mask2former-swin-base-ade-semantic',
        }

        if pretrained:
            model_id = model_map.get(model_name, model_map['mask2former-swin-tiny'])
            model = Mask2FormerForUniversalSegmentation.from_pretrained(
                model_id,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            config = Mask2FormerConfig(num_labels=num_classes)
            model = Mask2FormerForUniversalSegmentation(config)

        return HFSegmentationWrapper(model, output_type='mask2former')

    @staticmethod
    def _create_upernet(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
        """Create UPerNet model from Hugging Face."""
        try:
            from transformers import UperNetForSemanticSegmentation, UperNetConfig
        except ImportError:
            raise ImportError(
                "UPerNet requires transformers. Install with: pip install transformers"
            )

        model_map = {
            'upernet-swin-tiny': 'openmmlab/upernet-swin-tiny',
            'upernet-swin-small': 'openmmlab/upernet-swin-small',
        }

        if pretrained:
            model_id = model_map.get(model_name, model_map['upernet-swin-tiny'])
            model = UperNetForSemanticSegmentation.from_pretrained(
                model_id,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            config = UperNetConfig(num_labels=num_classes)
            model = UperNetForSemanticSegmentation(config)

        return HFSegmentationWrapper(model, output_type='upernet')

    @staticmethod
    def _create_beit(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
        """Create BEiT model from Hugging Face."""
        try:
            from transformers import BeitForSemanticSegmentation, BeitConfig
        except ImportError:
            raise ImportError(
                "BEiT requires transformers. Install with: pip install transformers"
            )

        model_map = {
            'beit-base': 'microsoft/beit-base-finetuned-ade-640-640',
            'beit-large': 'microsoft/beit-large-finetuned-ade-640-640',
        }

        if pretrained:
            model_id = model_map.get(model_name, model_map['beit-base'])
            model = BeitForSemanticSegmentation.from_pretrained(
                model_id,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            config = BeitConfig(num_labels=num_classes)
            model = BeitForSemanticSegmentation(config)

        return HFSegmentationWrapper(model, output_type='beit')


class HFSegmentationWrapper(nn.Module):
    """Wrapper for Hugging Face segmentation models to provide consistent interface."""

    def __init__(self, model: nn.Module, output_type: str):
        super().__init__()
        self.model = model
        self.output_type = output_type

    def forward(self, x):
        """
        Forward pass that returns logits in (B, C, H, W) format.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            Logits tensor (B, num_classes, H, W)
        """
        outputs = self.model(pixel_values=x)

        if self.output_type == 'segformer':
            logits = outputs.logits
        elif self.output_type == 'mask2former':
            # Mask2Former outputs class_queries_logits and masks_queries_logits
            # We need to process these to get semantic segmentation
            logits = outputs.class_queries_logits
            # Take the first mask for binary segmentation
            if logits.dim() == 3:  # (B, num_queries, num_classes)
                logits = logits[:, 0, :]  # Take first query
                logits = logits.unsqueeze(-1).unsqueeze(-1)  # Add spatial dims
        elif self.output_type in ['upernet', 'beit']:
            logits = outputs.logits
        else:
            logits = outputs.logits

        # Upsample to input size if needed
        if logits.size(2) != x.size(2) or logits.size(3) != x.size(3):
            logits = torch.nn.functional.interpolate(
                logits,
                size=(x.size(2), x.size(3)),
                mode='bilinear',
                align_corners=False
            )

        return logits


class TorchvisionSegmentationWrapper(nn.Module):
    """Wrapper for torchvision segmentation models."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            Logits tensor (B, num_classes, H, W)
        """
        output = self.model(x)
        return output['out']
