"""
SAM3 Model Patcher - Wraps SAM3 models for ComfyUI's model management system

This module provides ComfyUI-compatible wrappers that integrate with:
- comfy.model_management.load_models_gpu()
- ComfyUI's automatic VRAM management
- Device offloading when memory pressure is detected
"""

import torch
import logging
import gc
import comfy.model_management
from comfy.model_patcher import ModelPatcher


class SAM3ModelWrapper:
    """
    Lightweight wrapper that makes SAM3 models compatible with ComfyUI's ModelPatcher.

    This provides the interface expected by ModelPatcher without the full complexity
    of diffusion models (no patches, hooks, etc. needed for SAM3).
    """

    def __init__(self, model, processor, load_device, offload_device):
        """
        Args:
            model: The SAM3 model (torch.nn.Module)
            processor: Sam3Processor instance
            load_device: Device to load model on for inference (e.g., cuda:0)
            offload_device: Device to offload model to when not in use (e.g., cpu)
        """
        self.model = model
        self.processor = processor
        self.load_device = load_device
        self.offload_device = offload_device
        self.device = offload_device  # Current device

        # Required attributes for ModelPatcher compatibility
        self.model_loaded_weight_memory = 0
        self.lowvram_patch_counter = 0
        self.model_lowvram = False
        self.current_weight_patches_uuid = None

        # Calculate and cache model size
        self._model_size = None

    def model_size(self):
        """Return model size in bytes (cached for performance)."""
        if self._model_size is None:
            self._model_size = comfy.model_management.module_size(self.model)
        return self._model_size

    def to(self, device):
        """Move model to specified device with full processor sync."""
        device_str = str(device)
        if str(self.device) != device_str:
            logging.debug(f"[SAM3] Moving model from {self.device} to {device}")
            self.model.to(device)
            self.device = device

            # Sync processor device using dedicated method if available
            if hasattr(self.processor, 'sync_device_with_model'):
                self.processor.sync_device_with_model()
            elif hasattr(self.processor, 'device'):
                self.processor.device = device_str
                # Also sync find_stage tensors if present
                if hasattr(self.processor, 'find_stage') and self.processor.find_stage is not None:
                    fs = self.processor.find_stage
                    if hasattr(fs, 'img_ids') and fs.img_ids is not None:
                        fs.img_ids = fs.img_ids.to(device)
                    if hasattr(fs, 'text_ids') and fs.text_ids is not None:
                        fs.text_ids = fs.text_ids.to(device)
        return self

    def get_dtype(self):
        """Return model dtype."""
        try:
            return next(self.model.parameters()).dtype
        except StopIteration:
            return torch.float32


class SAM3ModelPatcher(ModelPatcher):
    """
    ModelPatcher subclass for SAM3 models.

    This integrates SAM3 with ComfyUI's model management system, enabling:
    - Automatic GPU/CPU offloading based on VRAM pressure
    - Proper cleanup when models are unloaded
    - Integration with load_models_gpu()
    """

    def __init__(self, sam3_wrapper):
        """
        Args:
            sam3_wrapper: SAM3ModelWrapper instance
        """
        self.sam3_wrapper = sam3_wrapper

        # Initialize ModelPatcher with the wrapper
        super().__init__(
            model=sam3_wrapper,
            load_device=sam3_wrapper.load_device,
            offload_device=sam3_wrapper.offload_device,
            size=sam3_wrapper.model_size(),
            weight_inplace_update=False
        )

    @property
    def sam3_model(self):
        """Access the underlying SAM3 model."""
        return self.sam3_wrapper.model

    @property
    def processor(self):
        """Access the SAM3 processor."""
        return self.sam3_wrapper.processor

    def model_size(self):
        """Override to use cached size from wrapper."""
        return self.sam3_wrapper.model_size()

    def clone(self):
        """Clone this patcher (shares underlying model)."""
        # For SAM3, we don't need complex cloning - just reference the same wrapper
        n = SAM3ModelPatcher(self.sam3_wrapper)
        n.patches = {}
        n.object_patches = {}
        n.model_options = {"transformer_options": {}}
        return n

    def patch_model(self, device_to=None, lowvram_model_memory=0, load_weights=True, force_patch_weights=False):
        """
        Load model to GPU for inference.

        SAM3 doesn't use LoRA patches, so this is simplified.
        """
        if device_to is None:
            device_to = self.load_device

        # Move the actual model to device
        self.sam3_wrapper.to(device_to)
        self.sam3_wrapper.model_loaded_weight_memory = self.model_size()

        return self.sam3_wrapper.model

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        """
        Unload model from GPU.

        Moves model to offload device and cleans up VRAM.
        """
        if device_to is None:
            device_to = self.offload_device

        # Move model to offload device
        self.sam3_wrapper.to(device_to)
        self.sam3_wrapper.model_loaded_weight_memory = 0

        # Force cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def model_patches_to(self, device):
        """SAM3 doesn't use patches, so this is a no-op."""
        pass

    def model_patches_models(self):
        """SAM3 doesn't have sub-models in patches."""
        return []

    def current_loaded_device(self):
        """Return the device the model is currently on."""
        return self.sam3_wrapper.device

    def loaded_size(self):
        """Return currently loaded size (0 if offloaded)."""
        if "cuda" in str(self.sam3_wrapper.device):
            return self.model_size()
        return 0

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        """Load model to device."""
        self.patch_model(device_to)
        return self.model_size()

    def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
        """Unload model to free memory."""
        self.unpatch_model(device_to)
        return self.model_size()

    def memory_required(self, input_shape=None):
        """
        Estimate memory required for inference.

        SAM3 uses ~1008x1008 internal resolution plus activations.
        """
        base_memory = self.model_size()
        # Estimate activation memory (rough approximation)
        activation_memory = 1008 * 1008 * 256 * 4 * 10  # ~10 intermediate tensors
        return base_memory + activation_memory

    def cleanup(self):
        """Cleanup resources."""
        self.unpatch_model()
        super().cleanup()

    def __del__(self):
        """Ensure cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass


def create_sam3_model_patcher(model, processor, device="cuda"):
    """
    Factory function to create a SAM3ModelPatcher.

    Args:
        model: SAM3 model instance
        processor: Sam3Processor instance
        device: Target inference device

    Returns:
        SAM3ModelPatcher instance ready for ComfyUI
    """
    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()

    wrapper = SAM3ModelWrapper(model, processor, load_device, offload_device)
    patcher = SAM3ModelPatcher(wrapper)

    return patcher
