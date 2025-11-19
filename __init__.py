"""
ComfyUI-SAM3: SAM3 Integration for ComfyUI

This custom node package provides integration with Meta's SAM3 (Segment Anything Model 3)
for open-vocabulary image segmentation using text prompts.

Nodes:
- LoadSAM3Model: Loads SAM3 model (auto-downloads if needed)
- SAM3Segmentation: Segment objects using text prompts
- SAM3GeometricRefine: Refine segmentation with geometric box prompts

Author: ComfyUI-SAM3
Version: 1.0.0
License: MIT
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Web directory for custom UI (optional, not needed for basic functionality)
WEB_DIRECTORY = None

# Version info
__version__ = "1.0.0"

# Export for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

print(f"[SAM3] ComfyUI-SAM3 v{__version__} loaded successfully")
print(f"[SAM3] Available nodes: {', '.join(NODE_CLASS_MAPPINGS.keys())}")
