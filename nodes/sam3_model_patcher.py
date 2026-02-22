"""
SAM3 Model Patcher - Wraps SAM3 models for ComfyUI's model management system.

Provides a single ModelPatcher subclass that integrates with:
- comfy.model_management.load_models_gpu()
- ComfyUI's automatic VRAM management
- Device offloading when memory pressure is detected
"""

import gc
import torch
import comfy.model_management
from comfy.model_patcher import ModelPatcher


class SAM3UnifiedModel(ModelPatcher):
    """
    Unified SAM3 model patcher for ComfyUI.

    Wraps the video predictor (detector + tracker) and image processor
    behind ComfyUI's ModelPatcher for automatic VRAM management.
    All layers use comfy.ops.manual_cast, so dtype is handled per-layer.
    """

    def __init__(self, video_predictor, processor, load_device, offload_device, dtype=None):
        self._video_predictor = video_predictor
        self._processor = processor
        self._load_device = load_device
        self._offload_device = offload_device
        self._model_dtype = dtype or torch.float32
        # Set inference dtype on processor immediately so set_image() always
        # casts image inputs to bf16/fp16 before backbone forward.
        self._processor._inference_dtype = self._model_dtype
        import logging, os
        if os.environ.get("DEBUG_COMFYUI_SAM3", "").lower() in ("1", "true", "yes"):
            logging.getLogger("sam3").warning("SAM3UnifiedModel: dtype=%s, processor._inference_dtype=%s", self._model_dtype, self._processor._inference_dtype)

        # The full model (detector + tracker) is the nn.Module we manage
        full_model = video_predictor.model
        model_size = comfy.model_management.module_size(full_model)

        # ModelPatcher assigns self.model = full_model
        super().__init__(
            model=full_model,
            load_device=load_device,
            offload_device=offload_device,
            size=model_size,
            weight_inplace_update=False,
        )

    # -- Properties -----------------------------------------------------------

    @property
    def processor(self):
        return self._processor

    @property
    def current_device(self):
        """Current device the model is on."""
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self._offload_device

    # -- Video predictor delegation -------------------------------------------

    def start_session(self, *args, **kwargs):
        return self._video_predictor.start_session(*args, **kwargs)

    def close_session(self, *args, **kwargs):
        return self._video_predictor.close_session(*args, **kwargs)

    def handle_stream_request(self, request):
        return self._video_predictor.handle_stream_request(request)

    def handle_request(self, request):
        return self._video_predictor.handle_request(request)

    def __getattr__(self, name):
        # Delegate unknown attributes to video predictor (e.g. _ALL_INFERENCE_STATES)
        if name == '_video_predictor':
            raise AttributeError(name)
        if hasattr(self._video_predictor, name):
            return getattr(self._video_predictor, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    # -- ModelPatcher overrides -----------------------------------------------

    def patch_model(self, device_to=None, lowvram_model_memory=0, load_weights=True, force_patch_weights=False):
        if device_to is None:
            device_to = self._load_device
        self.model.to(device_to)
        # Native ComfyUI pattern: weights stay in checkpoint dtype (fp32).
        # manual_cast handles per-layer casting to match input dtype.
        self._sync_processor_device(device_to)
        return self.model

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if device_to is None:
            device_to = self._offload_device
        self.model.to(device_to)
        self._sync_processor_device(device_to)
        gc.collect()
        comfy.model_management.soft_empty_cache()

    def clone(self):
        n = SAM3UnifiedModel(
            self._video_predictor, self._processor,
            self._load_device, self._offload_device,
            dtype=self._model_dtype,
        )
        n.patches = {}
        n.object_patches = {}
        n.model_options = {"transformer_options": {}}
        return n

    def model_size(self):
        return comfy.model_management.module_size(self.model)

    def memory_required(self, input_shape=None):
        base_memory = self.model_size()
        activation_memory = 1008 * 1008 * 256 * 4 * 10
        return base_memory + activation_memory

    def model_patches_to(self, device):
        pass

    def model_patches_models(self):
        return []

    def current_loaded_device(self):
        return self.current_device

    def loaded_size(self):
        device = self.current_device
        if device is not None and device.type != "cpu":
            return self.model_size()
        return 0

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        self.patch_model(device_to)
        return self.model_size()

    def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
        self.unpatch_model(device_to)
        return self.model_size()

    def cleanup(self):
        self.unpatch_model()
        super().cleanup()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    # -- Internal helpers -----------------------------------------------------

    def _sync_processor_device(self, device):
        """Sync processor's cached device state after model movement."""
        # Tell the processor what dtype to cast image inputs to.
        # manual_cast layers will then match their weights to the input dtype.
        self._processor._inference_dtype = self._model_dtype
        if hasattr(self._processor, 'sync_device_with_model'):
            self._processor.sync_device_with_model()
        elif hasattr(self._processor, 'device'):
            self._processor.device = str(device)
            if hasattr(self._processor, 'find_stage') and self._processor.find_stage is not None:
                fs = self._processor.find_stage
                if hasattr(fs, 'img_ids') and fs.img_ids is not None:
                    fs.img_ids = fs.img_ids.to(device)
                if hasattr(fs, 'text_ids') and fs.text_ids is not None:
                    fs.text_ids = fs.text_ids.to(device)
