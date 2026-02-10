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
import os
import sys
import traceback

# Version info
__version__ = "3.0.0"  # Major refactor for memory management

# Track initialization status
INIT_SUCCESS = False
INIT_ERRORS = []

# Detect if running under pytest with robust checks
# Only skip initialization when we're CERTAIN tests are actively running:
# 1. PYTEST_CURRENT_TEST env var (only set during actual pytest execution)
# 2. _pytest.config in sys.modules (pytest runtime configuration - more specific)
# Note: We don't check 'pytest' in sys.modules alone because ComfyUI may import
#       pytest as a dependency, causing false positives (see issue #7)
# Allow override with SAM3_FORCE_INIT=1 for users who need to force initialization
force_init = os.environ.get('SAM3_FORCE_INIT') == '1'
is_pytest = (
    'PYTEST_CURRENT_TEST' in os.environ or
    '_pytest.config' in sys.modules
)
skip_init = is_pytest and not force_init

if not skip_init:
    print(f"[SAM3] ComfyUI-SAM3 v{__version__} initializing...")

    # Step 0: Register sam3 model folder with ComfyUI
    try:
        import folder_paths
        sam3_model_dir = os.path.join(folder_paths.models_dir, "sam3")
        os.makedirs(sam3_model_dir, exist_ok=True)
        folder_paths.add_model_folder_path("sam3", sam3_model_dir)
        print(f"[SAM3] [OK] Registered model folder: {sam3_model_dir}")
    except Exception as e:
        error_msg = f"Failed to register model folder: {str(e)}"
        INIT_ERRORS.append(error_msg)
        print(f"[SAM3] [WARNING] {error_msg}")

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
    # During testing, skip initialization to prevent import errors
    # Debug info to help diagnose false positives
    reasons = []
    if 'PYTEST_CURRENT_TEST' in os.environ:
        reasons.append("PYTEST_CURRENT_TEST env var detected")
    if '_pytest.config' in sys.modules:
        reasons.append("_pytest.config module in sys.modules")

    print(f"[SAM3] ComfyUI-SAM3 v{__version__} running in pytest mode - skipping initialization")
    print(f"[SAM3] Reason: {', '.join(reasons)}")
    print(f"[SAM3] If this is a false positive, set environment variable: SAM3_FORCE_INIT=1")

    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

# Web directory for custom UI (interactive SAM3 detector)
WEB_DIRECTORY = "./web"

# Export for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
