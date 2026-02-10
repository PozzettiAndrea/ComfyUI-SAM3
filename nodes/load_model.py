"""
LoadSAM3Model node - Loads SAM3 model with ComfyUI memory management integration

This node integrates with ComfyUI's model_management system for:
- Automatic GPU/CPU offloading based on VRAM pressure
- Proper cleanup when models are unloaded
- Auto-download from HuggingFace if model not found
"""
from pathlib import Path

import comfy.model_management

from .sam3_model_patcher import create_sam3_model_patcher

try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


class LoadSAM3Model:
    """
    Node to load SAM3 model with ComfyUI memory management integration.

    Specify the path to the model checkpoint. If the model doesn't exist
    and an HF token is provided, it will be downloaded automatically.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": "models/sam3/sam3.pt",
                    "tooltip": "Path to SAM3 model checkpoint (relative to ComfyUI root or absolute)"
                }),
            },
            "optional": {
                "hf_token": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "HuggingFace token (for auto-download)",
                    "tooltip": "If model doesn't exist at path and token is provided, downloads from HuggingFace"
                }),
            }
        }

    RETURN_TYPES = ("SAM3_MODEL",)
    RETURN_NAMES = ("sam3_model",)
    FUNCTION = "load_model"
    CATEGORY = "SAM3"

    def load_model(self, model_path, hf_token=""):
        """
        Load SAM3 model with ComfyUI integration.

        Args:
            model_path: Path to model checkpoint (relative or absolute)
            hf_token: Optional HuggingFace token for auto-download if model missing

        Returns:
            Tuple containing SAM3ModelPatcher for ComfyUI memory management
        """
        # Import SAM3 from vendored library
        try:
            from .sam3_lib.model_builder import build_sam3_image_model
            from .sam3_lib.model.sam3_image_processor import Sam3Processor
        except ImportError as e:
            raise ImportError(
                "SAM3 library import failed. This is an internal error.\n"
                f"Please ensure all files are properly installed in ComfyUI-SAM3/nodes/sam3_lib/\n"
                f"Error: {e}"
            )

        # Get devices from ComfyUI's model management
        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()

        print(f"[SAM3] Load device: {load_device}, Offload device: {offload_device}")

        # Resolve checkpoint path
        checkpoint_path = Path(model_path)
        if not checkpoint_path.is_absolute():
            # Relative path - resolve from current working directory (ComfyUI root)
            checkpoint_path = Path.cwd() / checkpoint_path

        # Check if model exists, download if needed
        if not checkpoint_path.exists():
            if hf_token and hf_token.strip():
                print(f"[SAM3] Model not found at {checkpoint_path}, downloading...")
                self._download_from_huggingface(hf_token, checkpoint_path)
            else:
                raise FileNotFoundError(
                    f"[SAM3] Model file not found: {checkpoint_path}\n"
                    f"Either place the model at this path, or provide an hf_token to download it."
                )

        checkpoint_path = str(checkpoint_path)

        print(f"[SAM3] Loading model from: {checkpoint_path}")

        # Build model on offload device initially (ComfyUI will move to GPU when needed)
        print(f"[SAM3] Building SAM3 model...")
        try:
            model = build_sam3_image_model(
                device=str(offload_device),
                checkpoint_path=checkpoint_path,
                load_from_HF=False,
                hf_token=None,
                eval_mode=True,
                enable_segmentation=True,
                enable_inst_interactivity=True,  # Enable SAM2-style point/box segmentation
                compile=False
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"[SAM3] Checkpoint file not found: {checkpoint_path}\n"
                f"Error: {e}"
            )
        except (RuntimeError, ValueError) as e:
            error_msg = str(e)
            if "checkpoint" in error_msg.lower() or "state_dict" in error_msg.lower():
                raise RuntimeError(
                    f"[SAM3] Invalid or corrupted checkpoint file.\n"
                    f"Checkpoint: {checkpoint_path}\n"
                    f"Error: {e}"
                )
            elif "CUDA" in error_msg or "device" in error_msg.lower():
                raise RuntimeError(
                    f"[SAM3] Device error - GPU may not be available or out of memory.\n"
                    f"Error: {e}"
                )
            else:
                raise RuntimeError(f"[SAM3] Failed to load model: {e}")

        print(f"[SAM3] Model loaded successfully")

        # Create processor
        print(f"[SAM3] Creating SAM3 processor...")
        processor = Sam3Processor(
            model=model,
            resolution=1008,
            device=str(offload_device),
            confidence_threshold=0.2
        )

        print(f"[SAM3] Processor created successfully")

        # Create ComfyUI-compatible model patcher
        patcher = create_sam3_model_patcher(model, processor, str(load_device))

        print(f"[SAM3] Model ready (size: {patcher.model_size() / 1024 / 1024:.1f} MB)")

        return (patcher,)

    def _download_from_huggingface(self, hf_token, target_path):
        """Download SAM3 model from HuggingFace to the specified path."""
        if not HF_HUB_AVAILABLE:
            raise ImportError(
                "[SAM3] huggingface_hub is required to download models from HuggingFace.\n"
                "Please install it with: pip install huggingface_hub"
            )

        # Ensure parent directory exists
        target_path = Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[SAM3] Downloading SAM3 model from HuggingFace...")
        print(f"[SAM3] Target: {target_path}")

        try:
            SAM3_MODEL_ID = "facebook/sam3"
            SAM3_CKPT_NAME = "sam3.pt"

            hf_hub_download(
                repo_id=SAM3_MODEL_ID,
                filename=SAM3_CKPT_NAME,
                token=hf_token.strip(),
                local_dir=str(target_path.parent),
                local_dir_use_symlinks=False
            )

            # hf_hub_download saves as filename, rename if needed
            downloaded_file = target_path.parent / SAM3_CKPT_NAME
            if downloaded_file != target_path and downloaded_file.exists():
                downloaded_file.rename(target_path)

            print(f"[SAM3] Model downloaded successfully to: {target_path}")

        except Exception as e:
            if "401" in str(e) or "authentication" in str(e).lower() or "gated" in str(e).lower():
                raise RuntimeError(
                    f"[SAM3] Authentication failed. Please ensure:\n"
                    f"1. You have requested access at: https://huggingface.co/facebook/sam3\n"
                    f"2. Your access has been approved (check your email)\n"
                    f"3. Your token is valid (get it from: https://huggingface.co/settings/tokens)\n"
                    f"Error: {e}"
                )
            else:
                raise RuntimeError(
                    f"[SAM3] Failed to download model from HuggingFace.\n"
                    f"Error: {e}"
                )


# Register the node
NODE_CLASS_MAPPINGS = {
    "LoadSAM3Model": LoadSAM3Model
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSAM3Model": "(down)Load SAM3 Model"
}
