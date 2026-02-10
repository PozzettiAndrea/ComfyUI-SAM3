"""
LoadSAM3Model node - Loads SAM3 model and creates processor
"""
import os
import torch
from pathlib import Path
from .utils import get_comfy_models_dir


# Global cache for loaded models
_MODEL_CACHE = {}


class LoadSAM3Model:
    """
    Node to load SAM3 model and create processor

    This node handles downloading the model from HuggingFace if not found locally,
    and caches the loaded model to avoid reloading.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            },
            "optional": {
                "model_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Leave empty to auto-download from HuggingFace"
                }),
                "hf_token": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "HuggingFace token (get from https://huggingface.co/settings/tokens)"
                }),
            }
        }

    RETURN_TYPES = ("SAM3_MODEL",)
    RETURN_NAMES = ("sam3_model",)
    FUNCTION = "load_model"
    CATEGORY = "SAM3"

    def load_model(self, device="auto", model_path="", hf_token=""):
        """
        Load SAM3 model

        Args:
            device: Device to load model on (auto/cuda/cpu)
            model_path: Optional path to local checkpoint
            hf_token: Optional HuggingFace token for downloading gated models

        Returns:
            Tuple containing model dict with model and processor
        """
        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[SAM3] Loading model on device: {device}")

        # Check cache (use hash of token for privacy in cache key)
        token_hash = str(hash(hf_token))[:8] if hf_token else "notoken"
        cache_key = f"{model_path}_{device}_{token_hash}"
        if cache_key in _MODEL_CACHE:
            print(f"[SAM3] Using cached model")
            return (_MODEL_CACHE[cache_key],)

        # Import SAM3 from vendored library
        try:
            from ..sam3_lib.model_builder import build_sam3_image_model
            from ..sam3_lib.model.sam3_image_processor import Sam3Processor
        except ImportError as e:
            raise ImportError(
                "SAM3 library import failed. This is an internal error.\n"
                f"Please ensure all files are properly installed in ComfyUI-SAM3/sam3_lib/\n"
                f"Error: {e}"
            )

        # Determine checkpoint path
        checkpoint_path = None
        load_from_hf = True

        if model_path and model_path.strip():
            # User provided a custom path
            checkpoint_path = model_path.strip()
            load_from_hf = False
            print(f"[SAM3] Loading from custom path: {checkpoint_path}")
        else:
            # Check if model exists in ComfyUI models folder
            models_dir = get_comfy_models_dir()
            local_checkpoint = Path(models_dir) / "sam3.pt"

            if local_checkpoint.exists():
                checkpoint_path = str(local_checkpoint)
                load_from_hf = False
                print(f"[SAM3] Found model in ComfyUI models folder: {checkpoint_path}")
            else:
                print(f"[SAM3] Model not found locally, will download from HuggingFace")
                if not hf_token or not hf_token.strip():
                    print(f"[SAM3] WARNING: No HuggingFace token provided!")
                    print(f"[SAM3] Please provide a token in the 'hf_token' input field")
                    print(f"[SAM3] Get your token from: https://huggingface.co/settings/tokens")
                    print(f"[SAM3] And request access to: https://huggingface.co/facebook/sam3")
                else:
                    print(f"[SAM3] Using provided HuggingFace token for authentication")
                print(f"[SAM3] Model will be downloaded to: {models_dir}")

                # We'll let SAM3 download it, then copy to our models folder
                checkpoint_path = None
                load_from_hf = True

        # Load model
        print(f"[SAM3] Building SAM3 model...")
        try:
            model = build_sam3_image_model(
                device=device,
                checkpoint_path=checkpoint_path,
                load_from_HF=load_from_hf,
                hf_token=hf_token.strip() if hf_token else None,
                eval_mode=True,
                enable_segmentation=True,
                enable_inst_interactivity=False,  # Disabled - not supported in vendored version
                compile=False  # Disable compile for now (can enable later for speed)
            )
        except Exception as e:
            if "Repo model facebook/sam3" in str(e) or "authentication" in str(e).lower() or "401" in str(e):
                raise RuntimeError(
                    f"Failed to download model from HuggingFace. Please:\n"
                    f"1. Request access at: https://huggingface.co/facebook/sam3\n"
                    f"2. Get your token at: https://huggingface.co/settings/tokens\n"
                    f"3. Paste the token in the 'hf_token' field of the LoadSAM3Model node\n"
                    f"Error: {e}"
                )
            else:
                raise RuntimeError(f"Failed to load SAM3 model: {e}")

        print(f"[SAM3] Model loaded successfully")

        # Create processor
        print(f"[SAM3] Creating SAM3 processor...")
        processor = Sam3Processor(
            model=model,
            resolution=1008,
            device=device,
            confidence_threshold=0.2  # Default lowered for SAM3's presence scoring
        )

        print(f"[SAM3] Processor created successfully")

        # Package model and processor together
        model_dict = {
            "model": model,
            "processor": processor,
            "device": device
        }

        # Cache the model
        _MODEL_CACHE[cache_key] = model_dict

        return (model_dict,)


# Register the node
NODE_CLASS_MAPPINGS = {
    "LoadSAM3Model": LoadSAM3Model
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSAM3Model": "Load SAM3 Model"
}
