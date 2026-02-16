"""
LoadSAM3Model node - Loads SAM3 model with ComfyUI memory management integration
"""
import logging
from pathlib import Path

log = logging.getLogger("sam3")

import torch
from folder_paths import base_path as comfy_base_path

try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


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
            "optional": {
                "precision": (["auto", "bf16", "fp16", "fp32"], {
                    "default": "auto",
                    "tooltip": "Model precision. auto: best for your GPU (bf16 on Ampere+, fp16 on Volta/Turing, fp32 on older)."
                }),
                "attention": (["auto", "sdpa", "flash_attn", "sage"], {
                    "default": "auto",
                    "tooltip": "Attention backend. auto: best available (sage > flash_attn > sdpa). sdpa: PyTorch native. flash_attn: Tri Dao's FlashAttention (FA2/FA3, requires flash-attn package). sage: SageAttention (auto-detects v3 for Blackwell or v2, requires sageattention/sageattn3 package)."
                }),
                "memory_mode": (["cpu_offload", "cache_gpu", "unload"], {
                    "default": "cpu_offload",
                    "tooltip": "Memory after inference. cpu_offload: move to CPU (default). cache_gpu: keep in VRAM (fastest repeated runs). unload: aggressively free all VRAM."
                }),
                "compile": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable torch.compile for faster inference. Model loading takes longer (pre-compiles all code paths), but inference is significantly faster on every run."
                }),
            },
        }

    RETURN_TYPES = ("SAM3_MODEL",)
    RETURN_NAMES = ("sam3_model",)
    FUNCTION = "load_model"
    CATEGORY = "SAM3"

    def load_model(self, model_path, precision="auto", attention="auto",
                   memory_mode="cpu_offload", compile=False):
        from .sam3_model_patcher import SAM3UnifiedModel
        from .sam3_lib.sam3_video_predictor import Sam3VideoPredictor
        from .sam3_lib.model.sam3_image_processor import Sam3Processor
        import comfy.model_management

        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()

        # Resolve checkpoint path
        checkpoint_path = Path(model_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path(comfy_base_path) / checkpoint_path

        # Auto-download if needed
        if not checkpoint_path.exists():
            log.info(f"Model not found at {checkpoint_path}, downloading from HuggingFace...")
            self._download_from_huggingface(checkpoint_path)

        # BPE path for tokenizer
        bpe_path = str(Path(__file__).parent / "sam3_lib" / "bpe_simple_vocab_16e6.txt.gz")

        log.info(f"Loading model from: {checkpoint_path}")
        if compile:
            log.info("torch.compile enabled")

        video_predictor = Sam3VideoPredictor(
            checkpoint_path=str(checkpoint_path),
            bpe_path=bpe_path,
            enable_inst_interactivity=True,
            attention_backend=attention,
            compile=compile,
        )

        # Run compilation warmup to pre-compile all code paths
        if compile:
            log.info("Running compilation warmup (this may take a few minutes on first run)...")
            video_predictor.model.warm_up_compilation()
            log.info("Compilation warmup complete")

        detector = video_predictor.model.detector
        processor = Sam3Processor(
            model=detector,
            resolution=1008,
            device=str(load_device),
            confidence_threshold=0.2
        )

        # Resolve dtype
        if precision == "auto":
            if comfy.model_management.should_use_bf16(load_device):
                dtype = torch.bfloat16
            elif comfy.model_management.should_use_fp16(load_device):
                dtype = torch.float16
            else:
                dtype = torch.float32
        elif precision == "bf16":
            dtype = torch.bfloat16
        elif precision == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        unified_model = SAM3UnifiedModel(
            video_predictor=video_predictor,
            processor=processor,
            load_device=load_device,
            offload_device=offload_device,
            memory_mode=memory_mode,
            dtype=dtype,
        )

        log.info(f"Model ready ({unified_model.model_size() / 1024 / 1024:.1f} MB)")

        return (unified_model,)

    def _download_from_huggingface(self, target_path):
        if not HF_HUB_AVAILABLE:
            raise ImportError(
                "[SAM3] huggingface_hub is required to download models.\n"
                "Please install it with: pip install huggingface_hub"
            )

        target_path = Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        hf_hub_download(
            repo_id="1038lab/sam3",
            filename="sam3.pt",
            local_dir=str(target_path.parent),
        )
        log.info(f"Model downloaded to: {target_path.parent / 'sam3.pt'}")


NODE_CLASS_MAPPINGS = {
    "LoadSAM3Model": LoadSAM3Model
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSAM3Model": "(down)Load SAM3 Model"
}
