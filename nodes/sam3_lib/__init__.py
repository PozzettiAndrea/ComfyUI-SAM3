"""
SAM3 Library - Vendored for ComfyUI-SAM3

This is a vendored version of Meta's SAM3 (Segment Anything Model 3) library,
containing only the essential components needed for image segmentation inference.
"""

from .model_builder import build_sam3_image_model

__version__ = "0.1.0"
__all__ = ["build_sam3_image_model"]
