"""
LoadSAM3Model node - Loads SAM3 model with ComfyUI memory management integration

This node integrates with ComfyUI's model_management system for:
- Automatic GPU/CPU offloading based on VRAM pressure
- Proper cleanup when models are unloaded
- Integration with folder_paths for model discovery
"""
import os
import torch
from pathlib import Path

import folder_paths
import comfy.model_management

from .sam3_model_patcher import create_sam3_model_patcher

try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


def get_sam3_models():
    """Get list of available SAM3 models from the sam3 folder."""
    try:
        models = folder_paths.get_filename_list("sam3")
        return models if models else []
    except Exception:
        return []


class LoadSAM3Model:
    """
    Node to load SAM3 model with ComfyUI memory management integration.

    This node:
    - Uses ComfyUI's folder_paths for model discovery
    - Returns a SAM3ModelPatcher for proper VRAM management
    - Supports downloading from HuggingFace if model not found locally
    """

    @classmethod
    def INPUT_TYPES(cls):
        models = get_sam3_models()
        # Add placeholder option for HuggingFace download
        model_choices = models + ["[Download from HuggingFace]"] if models else ["[Download from HuggingFace]"]

        return {
            "required": {
                "model_name": (model_choices, {
                    "default": model_choices[0] if model_choices else "[Download from HuggingFace]",
                    "tooltip": "Select SAM3 model from ComfyUI/models/sam3/ folder, or download from HuggingFace"
                }),
            },
            "optional": {
                "hf_token": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "HuggingFace token (required for download)",
                    "tooltip": "HuggingFace authentication token. Required only when downloading from HuggingFace. Get token from https://huggingface.co/settings/tokens"
                }),
            }
        }

    RETURN_TYPES = ("SAM3_MODEL",)
    RETURN_NAMES = ("sam3_model",)
    FUNCTION = "load_model"
    CATEGORY = "SAM3"

    def load_model(self, model_name, hf_token=""):
        """
        Load SAM3 model with ComfyUI integration.

        Args:
            model_name: Model filename from sam3 folder or "[Download from HuggingFace]"
            hf_token: Optional HuggingFace token for downloading gated models

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

        # Determine checkpoint path
        if model_name == "[Download from HuggingFace]":
            checkpoint_path = self._download_from_huggingface(hf_token)
        else:
            checkpoint_path = folder_paths.get_full_path("sam3", model_name)
            if checkpoint_path is None:
                raise FileNotFoundError(
                    f"[SAM3] Model file not found: {model_name}\n"
                    f"Please ensure the file exists in ComfyUI/models/sam3/"
                )

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

    def _download_from_huggingface(self, hf_token):
        """Download SAM3 model from HuggingFace."""
        if not HF_HUB_AVAILABLE:
            raise ImportError(
                "[SAM3] huggingface_hub is required to download models from HuggingFace.\n"
                "Please install it with: pip install huggingface_hub"
            )

        if not hf_token or not hf_token.strip():
            raise ValueError(
                "[SAM3] HuggingFace token required to download SAM3 model.\n"
                "The SAM3 model is gated and requires authentication.\n"
                "Please:\n"
                "1. Request access at: https://huggingface.co/facebook/sam3\n"
                "2. Get your token at: https://huggingface.co/settings/tokens\n"
                "3. Provide the token in the 'hf_token' input field"
            )

        # Get sam3 models directory
        sam3_folder_paths = folder_paths.get_folder_paths("sam3")
        if not sam3_folder_paths:
            raise RuntimeError("[SAM3] sam3 folder not registered with ComfyUI")

        models_dir = sam3_folder_paths[0]
        local_checkpoint = Path(models_dir) / "sam3.pt"

        if local_checkpoint.exists():
            print(f"[SAM3] Found existing model: {local_checkpoint}")
            return str(local_checkpoint)

        print(f"[SAM3] Downloading SAM3 model from HuggingFace...")
        print(f"[SAM3] Target directory: {models_dir}")

        try:
            SAM3_MODEL_ID = "facebook/sam3"
            SAM3_CKPT_NAME = "sam3.pt"

            hf_hub_download(
                repo_id=SAM3_MODEL_ID,
                filename=SAM3_CKPT_NAME,
                token=hf_token.strip(),
                local_dir=models_dir,
                local_dir_use_symlinks=False
            )

            print(f"[SAM3] Model downloaded successfully to: {local_checkpoint}")
            return str(local_checkpoint)

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
    "LoadSAM3Model": "Load SAM3 Model"
}
