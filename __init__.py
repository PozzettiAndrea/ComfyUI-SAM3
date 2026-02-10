"""
ComfyUI-SAM3: SAM3 Integration for ComfyUI

This custom node package provides integration with Meta's SAM3 (Segment Anything Model 3)
for open-vocabulary image segmentation using text prompts and visual geometric refinement.

Main Node:
- LoadSAM3Model: Load SAM3 model (auto-downloads from HuggingFace)
- SAM3Segmentation: Segment with optional text/boxes/points/masks

Helper Nodes (Visual Prompt Creation):
- SAM3CreateBox: Visually create a box prompt using sliders
- SAM3CreatePoint: Visually create a point prompt using sliders
- SAM3CombineBoxes: Combine multiple box prompts (up to 5)
- SAM3CombinePoints: Combine multiple point prompts (up to 10)

Workflow Example:
  [SAM3CreateBox] → [SAM3CombineBoxes] → [SAM3Segmentation] ← [LoadImage]
  [SAM3CreatePoint] → [SAM3CombinePoints] ↗

All geometric refinement uses SAM3's grounding model approach.
No JSON typing required - pure visual node-based workflow!

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
