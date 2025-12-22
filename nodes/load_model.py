"""
LoadSAM3Model node - Loads SAM3 model with ComfyUI memory management integration

This node integrates with ComfyUI's model_management system for:
- Automatic GPU/CPU offloading based on VRAM pressure
- Proper cleanup when models are unloaded
- Auto-download from HuggingFace if model not found

Unified model supports both image segmentation and video tracking workflows.
"""
from pathlib import Path
import gc

import torch
import comfy.model_management
from folder_paths import base_path as comfy_base_path

from .sam3_model_patcher import SAM3ModelWrapper, SAM3ModelPatcher

try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


class SAM3UnifiedModel(SAM3ModelPatcher):
    """
    Unified SAM3 model that supports both image segmentation and video tracking.

    Inherits from SAM3ModelPatcher for ComfyUI integration.
    Exposes both image interface (processor, sam3_wrapper) and video interface
    (handle_stream_request, model).
    """

    def __init__(self, video_predictor, processor, load_device, offload_device):
        """
        Args:
            video_predictor: Sam3VideoPredictor instance (for video tracking)
            processor: Sam3Processor instance (for image segmentation)
            load_device: Device to load model on for inference
            offload_device: Device to offload model to when not in use
        """
        self._video_predictor = video_predictor
        self._processor = processor
        self._load_device = load_device
        self._offload_device = offload_device
        self._model = None  # For parent class compatibility

        # Create wrapper for the detector model (used for image segmentation)
        # The video predictor's model.detector is Sam3ImageOnVideoMultiGPU (inherits from Sam3Image)
        detector_model = video_predictor.model.detector
        wrapper = SAM3ModelWrapper(detector_model, processor, load_device, offload_device)

        # Initialize parent SAM3ModelPatcher
        super().__init__(wrapper)

    # === Image Interface (for SAM3Segmentation, SAM3Grounding, etc.) ===

    @property
    def processor(self):
        """Sam3Processor for image segmentation."""
        return self._processor

    # sam3_wrapper is inherited from SAM3ModelPatcher

    # === Video Interface (for SAM3Propagate, etc.) ===

    @property
    def model(self):
        """
        Access underlying video model for video tracking.

        Video nodes use: sam3_model.model.to(device), sam3_model.model.cpu()
        """
        return self._video_predictor.model

    @model.setter
    def model(self, value):
        """Allow parent class to set model attribute."""
        self._model = value

    def __getattr__(self, name):
        """Forward unknown attributes to video predictor for compatibility."""
        # Avoid infinite recursion - check if _video_predictor exists
        if name == '_video_predictor':
            raise AttributeError(name)
        # Forward to video predictor
        if hasattr(self._video_predictor, name):
            return getattr(self._video_predictor, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def start_session(self, *args, **kwargs):
        """Forward to video predictor."""
        return self._video_predictor.start_session(*args, **kwargs)

    def close_session(self, *args, **kwargs):
        """Forward to video predictor."""
        return self._video_predictor.close_session(*args, **kwargs)

    def handle_stream_request(self, request):
        """
        Video tracking API - dispatch streaming requests.

        Used by SAM3Propagate for frame-by-frame propagation.
        """
        return self._video_predictor.handle_stream_request(request)

    def handle_request(self, request):
        """
        Video tracking API - dispatch single requests.

        Used by video nodes for session management, adding prompts, etc.
        """
        return self._video_predictor.handle_request(request)

    # === Memory Management ===

    def patch_model(self, device_to=None, lowvram_model_memory=0, load_weights=True, force_patch_weights=False):
        """Load model to GPU for inference."""
        result = super().patch_model(device_to, lowvram_model_memory, load_weights, force_patch_weights)
        # Also move video model components
        if device_to is None:
            device_to = self._load_device
        self._video_predictor.model.to(device_to)
        return result

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        """Unload model from GPU."""
        super().unpatch_model(device_to, unpatch_weights)
        # Also offload video model components
        if device_to is None:
            device_to = self._offload_device
        self._video_predictor.model.to(device_to)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class LoadSAM3Model:
    """
    Node to load SAM3 model with ComfyUI memory management integration.

    Specify the path to the model checkpoint. If the model doesn't exist,
    it will be automatically downloaded from HuggingFace.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": "models/sam3/sam3.pt",
                    "tooltip": "Path to SAM3 model checkpoint (relative to ComfyUI root or absolute). Auto-downloads if not found."
                }),
            },
        }

    RETURN_TYPES = ("SAM3_MODEL",)
    RETURN_NAMES = ("sam3_model",)
    FUNCTION = "load_model"
    CATEGORY = "SAM3"

    def load_model(self, model_path):
        """
        Load SAM3 unified model with ComfyUI integration.

        Builds a unified model that supports both image segmentation and video tracking.
        Instance interactivity (SAM2-style points/boxes) is always enabled.
        Auto-downloads from HuggingFace if model not found.

        Args:
            model_path: Path to model checkpoint (relative or absolute)

        Returns:
            Tuple containing SAM3UnifiedModel for ComfyUI memory management
        """
        # Always enable instance interactivity (required by checkpoint)
        enable_inst_interactivity = True
        # Import SAM3 from vendored library
        try:
            from .sam3_lib.sam3_video_predictor import Sam3VideoPredictor
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
            # Always resolve relative paths against the ComfyUI root directory
            checkpoint_path = Path(comfy_base_path) / checkpoint_path

        # Check if model exists, download if needed (public repo, no token required)
        if not checkpoint_path.exists():
            print(f"[SAM3] Model not found at {checkpoint_path}, downloading from HuggingFace...")
            self._download_from_huggingface(checkpoint_path)

        checkpoint_path_str = str(checkpoint_path)

        # BPE path for tokenizer
        bpe_path = Path(__file__).parent / "sam3_lib" / "bpe_simple_vocab_16e6.txt.gz"
        bpe_path_str = str(bpe_path)

        print(f"[SAM3] Loading unified model from: {checkpoint_path_str}")
        print(f"[SAM3] Using BPE tokenizer: {bpe_path_str}")

        # Build video predictor (contains both image and video capabilities)
        print(f"[SAM3] Building SAM3 unified model...")
        try:
            video_predictor = Sam3VideoPredictor(
                checkpoint_path=checkpoint_path_str,
                bpe_path=bpe_path_str,
                enable_inst_interactivity=enable_inst_interactivity,
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"[SAM3] Checkpoint file not found: {checkpoint_path_str}\n"
                f"Error: {e}"
            )
        except (RuntimeError, ValueError) as e:
            error_msg = str(e)
            if "checkpoint" in error_msg.lower() or "state_dict" in error_msg.lower():
                raise RuntimeError(
                    f"[SAM3] Invalid or corrupted checkpoint file.\n"
                    f"Checkpoint: {checkpoint_path_str}\n"
                    f"Error: {e}"
                )
            elif "CUDA" in error_msg or "device" in error_msg.lower():
                raise RuntimeError(
                    f"[SAM3] Device error - GPU may not be available or out of memory.\n"
                    f"Error: {e}"
                )
            else:
                raise RuntimeError(f"[SAM3] Failed to load model: {e}")

        print(f"[SAM3] Video predictor loaded successfully")

        # Create processor for image segmentation using the detector
        # The detector is Sam3ImageOnVideoMultiGPU which inherits from Sam3Image
        print(f"[SAM3] Creating SAM3 processor...")
        detector = video_predictor.model.detector
        processor = Sam3Processor(
            model=detector,
            resolution=1008,
            device=str(load_device),
            confidence_threshold=0.2
        )

        print(f"[SAM3] Processor created successfully")

        # Create unified model
        unified_model = SAM3UnifiedModel(
            video_predictor=video_predictor,
            processor=processor,
            load_device=load_device,
            offload_device=offload_device
        )

        print(f"[SAM3] Unified model ready (size: {unified_model.model_size() / 1024 / 1024:.1f} MB)")

        return (unified_model,)

    def _download_from_huggingface(self, target_path):
        """Download SAM3 model from HuggingFace (public repo, no token needed)."""
        if not HF_HUB_AVAILABLE:
            raise ImportError(
                "[SAM3] huggingface_hub is required to download models.\n"
                "Please install it with: pip install huggingface_hub"
            )

        # Ensure parent directory exists
        target_path = Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        SAM3_MODEL_ID = "1038lab/sam3"
        SAM3_CKPT_NAME = "sam3.pt"

        print(f"[SAM3] Downloading from {SAM3_MODEL_ID}...")
        print(f"[SAM3] Target: {target_path}")

        try:
            hf_hub_download(
                repo_id=SAM3_MODEL_ID,
                filename=SAM3_CKPT_NAME,
                local_dir=str(target_path.parent),
            )
            print(f"[SAM3] Model downloaded successfully to: {target_path.parent / SAM3_CKPT_NAME}")

        except Exception as e:
            raise RuntimeError(
                f"[SAM3] Failed to download model from HuggingFace.\n"
                f"Repo: {SAM3_MODEL_ID}\n"
                f"Error: {e}"
            )


# Register the node
NODE_CLASS_MAPPINGS = {
    "LoadSAM3Model": LoadSAM3Model
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSAM3Model": "(down)Load SAM3 Model"
}
