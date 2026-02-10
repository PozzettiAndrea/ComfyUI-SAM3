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

Interactive Features:
- Right-click any node with IMAGE/MASK output → "Open in SAM3 Detector"
- Interactive point-and-click segmentation (left-click=positive, right-click=negative)

Video Tracking:
- SAM3VideoModelLoader, SAM3InitVideoSession, SAM3AddVideoPrompt
- SAM3PropagateVideo, SAM3VideoOutput, SAM3CloseVideoSession

Workflow Example:
  [SAM3CreateBox] → [SAM3CombineBoxes] → [SAM3Segmentation] ← [LoadImage]
  [SAM3CreatePoint] → [SAM3CombinePoints] ↗

All geometric refinement uses SAM3's grounding model approach.
No JSON typing required - pure visual node-based workflow!

Author: ComfyUI-SAM3
Version: 2.1.0
License: MIT
"""

# Only run initialization and imports when loaded by ComfyUI, not during pytest
# This prevents relative import errors when pytest collects test modules
import sys
import traceback

# Version info
__version__ = "2.1.0"

# Track initialization status
INIT_SUCCESS = False
INIT_ERRORS = []

if 'pytest' not in sys.modules:
    print(f"[SAM3] ComfyUI-SAM3 v{__version__} initializing...")

    # Step 1: Import node classes
    try:
        from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        print("[SAM3] [OK] Node classes imported successfully")
        INIT_SUCCESS = True
    except Exception as e:
        error_msg = f"Failed to import node classes: {str(e)}"
        INIT_ERRORS.append(error_msg)
        print(f"[SAM3] [WARNING] {error_msg}")
        print(f"[SAM3] Traceback:\n{traceback.format_exc()}")

        # Set empty mappings if import failed
        NODE_CLASS_MAPPINGS = {}
        NODE_DISPLAY_NAME_MAPPINGS = {}

    # Step 2: Import server to register API endpoints
    try:
        from . import sam3_server
        print("[SAM3] [OK] API endpoints registered")
    except Exception as e:
        error_msg = f"Failed to register API endpoints: {str(e)}"
        INIT_ERRORS.append(error_msg)
        print(f"[SAM3] [WARNING] {error_msg}")
        print(f"[SAM3] Traceback:\n{traceback.format_exc()}")

    # Report final status
    if INIT_SUCCESS:
        print(f"[SAM3] [OK] Loaded successfully!")
        print(f"[SAM3] Available nodes: {', '.join(NODE_CLASS_MAPPINGS.keys())}")
        print(f"[SAM3] Interactive SAM3 Detector: Right-click any IMAGE/MASK node → 'Open in SAM3 Detector'")
    else:
        print(f"[SAM3] [ERROR] Failed to load ({len(INIT_ERRORS)} error(s)):")
        for error in INIT_ERRORS:
            print(f"  - {error}")
        print("[SAM3] Please check the errors above and your installation.")

else:
    # During testing, import from conftest mocks
    print(f"[SAM3] ComfyUI-SAM3 v{__version__} running in pytest mode - skipping initialization")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

# Web directory for custom UI (interactive SAM3 detector)
WEB_DIRECTORY = "./web"

# Export for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
