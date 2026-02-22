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
    Auto-downloads the model from HuggingFace if not found.
    """

    MODEL_DIR = "models/sam3"
    MODEL_FILENAME = "sam3.safetensors"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "precision": (["auto", "bf16", "fp16", "fp32"], {
                    "default": "auto",
                    "tooltip": "Model precision. auto: best for your GPU (bf16 on Ampere+, fp16 on Volta/Turing, fp32 on older)."
                }),
                "attention": (["auto", "sdpa", "flash_attn", "sage"], {
                    "default": "auto",
                    "tooltip": "Attention backend. auto: best available (sage > flash_attn > sdpa). sdpa: PyTorch native. flash_attn: Tri Dao's FlashAttention (FA2/FA3, requires flash-attn package). sage: SageAttention (auto-detects v3 for Blackwell or v2, requires sageattention/sageattn3 package)."
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

    def load_model(self, precision="auto", attention="auto", compile=False):
        from .sam3_model_patcher import SAM3UnifiedModel
        from .sam3.predictor import Sam3VideoPredictor
        from .sam3.utils import Sam3Processor
        import comfy.model_management

        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()

        # Fixed checkpoint path
        checkpoint_path = Path(comfy_base_path) / self.MODEL_DIR / self.MODEL_FILENAME

        # Auto-download if needed
        if not checkpoint_path.exists():
            log.info(f"Model not found at {checkpoint_path}, downloading from HuggingFace...")
            self._download_from_huggingface()

        # BPE path for tokenizer
        bpe_path = str(Path(__file__).parent / "sam3" / "bpe_simple_vocab_16e6.txt.gz")

        # Resolve dtype before model creation
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

        # Tell sam3_attention() what half-precision dtype to target.
        # This enables centralized dtype normalization for every attention call
        # site — including self-attention on fp32 token embeddings and
        # cross-attention after bf16+fp32 positional-encoding type promotion.
        # Must be called after set_sam3_backend() (inside Sam3VideoPredictor).
        from .sam3.attention import set_sam3_dtype
        set_sam3_dtype(dtype if dtype != torch.float32 else None)

        # Selective weight casting: cast only parameters (not buffers) to target
        # dtype for VRAM savings and half-precision attention.
        #
        # Buffers like freqs_cis (complex RoPE tensor in the ViT backbone) must
        # stay in their original dtype — .to(bf16) on a complex tensor discards
        # the imaginary part, destroying position information.
        #
        # inst_interactive_predictor must also be cast: its no_mem_embed
        # (nn.Parameter) is added directly to bf16 vision features in
        # predict_inst(), causing fp32 type promotion. comfy.ops.manual_cast
        # can't intercept plain + operations — it only casts linear layer
        # weights. Keeping inst_interactive_predictor in fp32 makes all Q/K/V
        # tensors fp32, causing flash attention to fall back to SDPA.
        if dtype != torch.float32:
            import os
            detector = video_predictor.model.detector
            for param in detector.backbone.parameters():
                param.data = param.data.to(dtype=dtype)
            if detector.inst_interactive_predictor is not None:
                for param in detector.inst_interactive_predictor.parameters():
                    param.data = param.data.to(dtype=dtype)
            if os.environ.get("DEBUG_COMFYUI_SAM3", "").lower() in ("1", "true", "yes"):
                log.warning(
                    "LoadSAM3Model: backbone dtype=%s, inst_interactive_predictor dtype=%s",
                    next(detector.backbone.parameters()).dtype,
                    next(detector.inst_interactive_predictor.parameters()).dtype
                    if detector.inst_interactive_predictor is not None else "N/A",
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

        unified_model = SAM3UnifiedModel(
            video_predictor=video_predictor,
            processor=processor,
            load_device=load_device,
            offload_device=offload_device,
            dtype=dtype,
        )

        log.info(f"Model ready ({unified_model.model_size() / 1024 / 1024:.1f} MB)")

        return (unified_model,)

    def _download_from_huggingface(self):
        if not HF_HUB_AVAILABLE:
            raise ImportError(
                "[SAM3] huggingface_hub is required to download models.\n"
                "Please install it with: pip install huggingface_hub"
            )

        model_dir = Path(comfy_base_path) / self.MODEL_DIR
        model_dir.mkdir(parents=True, exist_ok=True)

        hf_hub_download(
            repo_id="apozz/sam3-safetensors",
            filename=self.MODEL_FILENAME,
            local_dir=str(model_dir),
        )
        log.info(f"Model downloaded to: {model_dir / self.MODEL_FILENAME}")


NODE_CLASS_MAPPINGS = {
    "LoadSAM3Model": LoadSAM3Model
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSAM3Model": "(down)Load SAM3 Model"
}
