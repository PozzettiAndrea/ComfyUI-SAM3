"""
ComfyUI-SAM3: SAM3 Integration for ComfyUI

This custom node package provides integration with Meta's SAM3 (Segment Anything Model 3)
for open-vocabulary image segmentation using text prompts and visual geometric refinement.

Main Nodes:
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
- SAM3VideoModelLoader
- SAM3InitVideoSession / SAM3InitVideoSessionAdvanced
- SAM3VideoPromptEditor   (interactive points/boxes editor + frame slider)
- SAM3PropagateVideo
- SAM3VideoOutput
- SAM3CloseVideoSession

Workflow Example (image):
  [SAM3CreateBox] → [SAM3CombineBoxes] → [SAM3Segmentation] ← [LoadImage]
  [SAM3CreatePoint] → [SAM3CombinePoints] ↗

Workflow Example (video):
  [Load Video / VHS] → [SAM3InitVideoSession] → [SAM3VideoPromptEditor]
                                               → [SAM3PropagateVideo]
                                               → [SAM3VideoOutput]
                                               → [SAM3CloseVideoSession]

All geometric refinement uses SAM3's grounding model approach.
No JSON typing required - pure visual node-based workflow!

Author: ComfyUI-SAM3
Version: 2.1.0
License: MIT
"""

import os
import sys
import traceback

__version__ = "2.1.0"

INIT_SUCCESS = False
INIT_ERRORS = []

# Detect pytest so we don't break test collection with heavy relative imports.
# Override with SAM3_FORCE_INIT=1 if this is a false positive.
force_init = os.environ.get("SAM3_FORCE_INIT") == "1"
is_pytest = (
    "PYTEST_CURRENT_TEST" in os.environ
    or "_pytest.config" in sys.modules
)
skip_init = is_pytest and not force_init

# Always declare these so ComfyUI never crashes on missing symbols.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Web directory for custom UI (interactive detector + video prompt editor).
# Path is relative to THIS package root (ComfyUI-SAM3/).
WEB_DIRECTORY = "./web"

if not skip_init:
    print(f"[SAM3] ComfyUI-SAM3 v{__version__} initializing...")

    # Step 1: Import node classes (image + interactive + video)
    try:
        from .nodes import NODE_CLASS_MAPPINGS as _NODE_MAPS
        from .nodes import NODE_DISPLAY_NAME_MAPPINGS as _NODE_NAMES

        NODE_CLASS_MAPPINGS.update(_NODE_MAPS)
        NODE_DISPLAY_NAME_MAPPINGS.update(_NODE_NAMES)

        # Prefer WEB_DIRECTORY exported by nodes package if present
        try:
            from .nodes import WEB_DIRECTORY as _WEB_DIR
            if _WEB_DIR:
                # Resolve relative to package root when nodes export "../web"
                _pkg_dir = os.path.dirname(__file__)
                _candidate = os.path.normpath(os.path.join(_pkg_dir, "nodes", _WEB_DIR))
                if os.path.isdir(_candidate):
                    # Keep ComfyUI-relative form from package root
                    WEB_DIRECTORY = "./web" if os.path.isdir(os.path.join(_pkg_dir, "web")) else _WEB_DIR
                elif os.path.isdir(os.path.join(_pkg_dir, "web")):
                    WEB_DIRECTORY = "./web"
        except ImportError:
            pass

        print("[SAM3] [OK] Node classes imported successfully")
        INIT_SUCCESS = True
    except Exception as e:
        error_msg = f"Failed to import node classes: {str(e)}"
        INIT_ERRORS.append(error_msg)
        print(f"[SAM3] [WARNING] {error_msg}")
        print(f"[SAM3] Traceback:\n{traceback.format_exc()}")
        NODE_CLASS_MAPPINGS = {}
        NODE_DISPLAY_NAME_MAPPINGS = {}

    # Step 2: Import server to register legacy/interactive API endpoints
    try:
        from . import sam3_server  # noqa: F401
        print("[SAM3] [OK] API endpoints registered (sam3_server)")
    except Exception as e:
        error_msg = f"Failed to register API endpoints: {str(e)}"
        INIT_ERRORS.append(error_msg)
        print(f"[SAM3] [WARNING] {error_msg}")
        print(f"[SAM3] Traceback:\n{traceback.format_exc()}")

    # Step 3: Ensure video REST routes are registered
    # (they are decorated at import-time inside sam3_video_nodes;
    #  importing nodes above already pulls them in, this is a safety net)
    try:
        from .nodes import sam3_video_nodes  # noqa: F401
        print("[SAM3] [OK] Video API endpoints available (/sam3/...)")
    except Exception as e:
        error_msg = f"Failed to ensure video API endpoints: {str(e)}"
        INIT_ERRORS.append(error_msg)
        print(f"[SAM3] [WARNING] {error_msg}")

    # Report final status
    if INIT_SUCCESS:
        print("[SAM3] [OK] Loaded successfully!")
        print(f"[SAM3] Available nodes: {', '.join(NODE_CLASS_MAPPINGS.keys())}")
        print("[SAM3] Interactive SAM3 Detector: Right-click any IMAGE/MASK node → 'Open in SAM3 Detector'")
        print(f"[SAM3] WEB_DIRECTORY: {WEB_DIRECTORY}")
    else:
        print(f"[SAM3] [ERROR] Failed to load ({len(INIT_ERRORS)} error(s)):")
        for error in INIT_ERRORS:
            print(f"  - {error}")
        print("[SAM3] Please check the errors above and your installation.")

else:
    reasons = []
    if "PYTEST_CURRENT_TEST" in os.environ:
        reasons.append("PYTEST_CURRENT_TEST env var detected")
    if "_pytest.config" in sys.modules:
        reasons.append("_pytest.config module in sys.modules")

    print(f"[SAM3] ComfyUI-SAM3 v{__version__} running in pytest mode - skipping initialization")
    print(f"[SAM3] Reason: {', '.join(reasons) if reasons else 'unknown'}")
    print("[SAM3] If this is a false positive, set environment variable: SAM3_FORCE_INIT=1")

# Export for ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]