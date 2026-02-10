"""
ComfyUI-SAM3: SAM3 Integration for ComfyUI

This custom node package provides integration with Meta's SAM3 (Segment Anything Model 3)
for open-vocabulary image segmentation using text prompts and geometric refinement.

Nodes:
- LoadSAM3Model: Loads SAM3 model (auto-downloads if needed)
- SAM3Segmentation: Segment objects using text prompts
- SAM3GeometricRefine: Refine segmentation with geometric box prompts
- SAM3PointRefine: Point-based refinement (foreground/background points)
- SAM3MaskRefine: Mask-based refinement (use previous masks as constraints)
- SAM3MultiPrompt: Combine multiple geometric prompts (text + boxes + points + masks)

All geometric refinement uses SAM3's grounding model approach.

Author: ComfyUI-SAM3
Version: 2.0.0
License: MIT
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Web directory for custom UI (optional, not needed for basic functionality)
WEB_DIRECTORY = None

# Version info
__version__ = "2.0.0"

# Export for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

print(f"[SAM3] ComfyUI-SAM3 v{__version__} loaded successfully")
print(f"[SAM3] Available nodes: {', '.join(NODE_CLASS_MAPPINGS.keys())}")
