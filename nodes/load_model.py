"""
LoadSAM3Model node - Loads SAM3 model and creates processor
"""
import os
import torch
from pathlib import Path
from .utils import get_comfy_models_dir

try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


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
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device to run the model on. 'auto' selects CUDA if available, otherwise CPU. GPU highly recommended for performance."
                }),
                "use_gpu_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model on GPU between inferences (faster but uses more VRAM). Set to False to offload to CPU after each inference."
                }),
            },
            "optional": {
                "model_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Leave empty to auto-download from HuggingFace",
                    "tooltip": "Optional absolute path to local SAM3 checkpoint file (must include filename, e.g., /path/to/sam3.pt). Leave empty to auto-download from HuggingFace directly to ComfyUI/models/sam3/sam3.pt"
                }),
                "hf_token": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "HuggingFace token (get from https://huggingface.co/settings/tokens)",
                    "tooltip": "HuggingFace authentication token required for downloading SAM3 model. Get token from https://huggingface.co/settings/tokens and request access at https://huggingface.co/facebook/sam3"
                }),
            }
        }

    RETURN_TYPES = ("SAM3_MODEL",)
    RETURN_NAMES = ("sam3_model",)
    FUNCTION = "load_model"
    CATEGORY = "SAM3"

    def load_model(self, device="auto", use_gpu_cache=True, model_path="", hf_token=""):
        """
        Load SAM3 model

        Args:
            device: Device to load model on (auto/cuda/cpu)
            use_gpu_cache: Keep model on GPU between inferences (True) or offload to CPU (False)
            model_path: Optional path to local checkpoint
            hf_token: Optional HuggingFace token for downloading gated models

        Returns:
            Tuple containing model dict with model and processor
        """
        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[SAM3] Loading model on device: {device}")
        print(f"[SAM3] GPU cache: {'enabled' if use_gpu_cache else 'disabled (will offload to CPU after inference)'}")

        # Check cache (use hash of token for privacy in cache key)
        token_hash = str(hash(hf_token))[:8] if hf_token else "notoken"
        cache_key = f"{model_path}_{device}_{token_hash}"
        if cache_key in _MODEL_CACHE:
            print(f"[SAM3] Using cached model")
            cached_model = _MODEL_CACHE[cache_key]
            # Update the use_gpu_cache setting in case it changed
            cached_model["use_gpu_cache"] = use_gpu_cache
            return (cached_model,)

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

        # Determine checkpoint path
        checkpoint_path = None
        load_from_hf = True

        if model_path and model_path.strip():
            # User provided a custom path - validate it
            custom_path = Path(model_path.strip())

            # Validate the path exists
            if not custom_path.exists():
                raise FileNotFoundError(
                    f"[SAM3] Model file not found: {custom_path}\n"
                    f"Please provide a valid absolute path to a sam3.pt checkpoint file.\n"
                    f"Expected format: /full/path/to/sam3.pt\n"
                    f"Example: {Path(get_comfy_models_dir()) / 'sam3.pt'}"
                )

            # Validate it's a file, not a directory
            if not custom_path.is_file():
                raise ValueError(
                    f"[SAM3] Path is a directory, not a file: {custom_path}\n"
                    f"Please provide the full path including the filename.\n"
                    f"Example: {custom_path / 'sam3.pt'}"
                )

            # Validate it's a .pt file
            if custom_path.suffix != '.pt':
                print(f"[SAM3] WARNING: File does not have .pt extension: {custom_path}")
                print(f"[SAM3] Expected a PyTorch checkpoint file (.pt)")

            checkpoint_path = str(custom_path)
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
                # Model not found locally - download from HuggingFace
                print(f"[SAM3] Model not found locally, downloading from HuggingFace...")

                # Validate we have huggingface_hub available
                if not HF_HUB_AVAILABLE:
                    raise ImportError(
                        "[SAM3] huggingface_hub is required to download models from HuggingFace.\n"
                        "Please install it with: pip install huggingface_hub"
                    )

                # Validate HF token
                if not hf_token or not hf_token.strip():
                    raise ValueError(
                        "[SAM3] HuggingFace token required to download SAM3 model.\n"
                        "The SAM3 model is gated and requires authentication.\n"
                        "Please:\n"
                        "1. Request access at: https://huggingface.co/facebook/sam3\n"
                        "2. Get your token at: https://huggingface.co/settings/tokens\n"
                        "3. Provide the token in the 'hf_token' input field"
                    )

                print(f"[SAM3] Using provided HuggingFace token for authentication")
                print(f"[SAM3] Downloading to: {models_dir}")

                # Download directly to ComfyUI models folder
                try:
                    SAM3_MODEL_ID = "facebook/sam3"
                    SAM3_CKPT_NAME = "sam3.pt"

                    # Download checkpoint directly to ComfyUI folder
                    downloaded_path = hf_hub_download(
                        repo_id=SAM3_MODEL_ID,
                        filename=SAM3_CKPT_NAME,
                        token=hf_token.strip(),
                        local_dir=models_dir,
                        local_dir_use_symlinks=False  # Copy file instead of symlink
                    )

                    checkpoint_path = str(local_checkpoint)  # Use consistent path
                    load_from_hf = False  # We now have a local checkpoint
                    print(f"[SAM3] Model downloaded successfully to: {checkpoint_path}")

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
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"[SAM3] Checkpoint file not found: {checkpoint_path}\n"
                f"Please ensure the file exists and the path is correct.\n"
                f"Error: {e}"
            )
        except PermissionError as e:
            raise PermissionError(
                f"[SAM3] Permission denied accessing checkpoint: {checkpoint_path}\n"
                f"Please check file permissions and try again.\n"
                f"Error: {e}"
            )
        except (RuntimeError, ValueError) as e:
            error_msg = str(e)
            # Check for common error patterns
            if "checkpoint" in error_msg.lower() or "state_dict" in error_msg.lower():
                raise RuntimeError(
                    f"[SAM3] Invalid or corrupted checkpoint file.\n"
                    f"The checkpoint may be incomplete or from an incompatible version.\n"
                    f"Try re-downloading the model or using a different checkpoint.\n"
                    f"Checkpoint: {checkpoint_path}\n"
                    f"Error: {e}"
                )
            elif "CUDA" in error_msg or "device" in error_msg.lower():
                raise RuntimeError(
                    f"[SAM3] Device error - GPU may not be available or out of memory.\n"
                    f"Try:\n"
                    f"1. Set device to 'cpu' if you don't have a GPU\n"
                    f"2. Free up GPU memory if using CUDA\n"
                    f"3. Restart ComfyUI\n"
                    f"Error: {e}"
                )
            else:
                raise RuntimeError(f"[SAM3] Failed to load model: {e}")
        except Exception as e:
            raise RuntimeError(
                f"[SAM3] Unexpected error loading model.\n"
                f"Checkpoint: {checkpoint_path}\n"
                f"Error: {e}"
            )

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
            "device": device,
            "use_gpu_cache": use_gpu_cache,
            "original_device": device  # Store the original device for re-loading
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
