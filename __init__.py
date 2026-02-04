import os
import sys
import traceback

# Track initialization status
INIT_SUCCESS = False
INIT_ERRORS = []

# Detect if running under pytest
# Only skip initialization when PYTEST_CURRENT_TEST env var is set.
# This is the ONLY reliable indicator that pytest is actively running tests.
# Note: We previously also checked '_pytest.config' in sys.modules, but this
# caused false positives when ComfyUI or its dependencies imported pytest
# as a dependency (see issue #7).
# Allow override with SAM3_FORCE_INIT=1 for edge cases.
force_init = os.environ.get('SAM3_FORCE_INIT') == '1'
is_pytest = 'PYTEST_CURRENT_TEST' in os.environ
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
        print(f"[SAM3] Interactive SAM3 Detector: Right-click any IMAGE/MASK node -> 'Open in SAM3 Detector'")
    else:
        print(f"[SAM3] [ERROR] Failed to load ({len(INIT_ERRORS)} error(s)):")
        for error in INIT_ERRORS:
            print(f"  - {error}")
        print("[SAM3] Please check the errors above and your installation.")

else:
    # During testing, skip initialization to prevent import errors
    print(f"[SAM3] ComfyUI-SAM3 v{__version__} running in pytest mode - skipping initialization")
    print(f"[SAM3] Reason: PYTEST_CURRENT_TEST={os.environ.get('PYTEST_CURRENT_TEST')}")
    print(f"[SAM3] If this is a false positive, set environment variable: SAM3_FORCE_INIT=1")

    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

# Web directory for custom UI (interactive SAM3 detector)
WEB_DIRECTORY = "./web"

# Export for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
