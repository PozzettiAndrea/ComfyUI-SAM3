"""
LoadSAM3Model node - Loads SAM3 model with ComfyUI memory management integration
"""
import logging
from pathlib import Path

log = logging.getLogger("sam3")

import os
import torch
import folder_paths
from folder_paths import base_path as comfy_base_path
from comfy_api.latest import io

# Register model folder with ComfyUI's folder_paths system
_sam3_models_dir = os.path.join(folder_paths.models_dir, "sam3")
os.makedirs(_sam3_models_dir, exist_ok=True)
folder_paths.add_model_folder_path("sam3", _sam3_models_dir)

try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


class LoadSAM3Model(io.ComfyNode):
    """
    Node to load SAM3 model with ComfyUI memory management integration.
    Auto-downloads the model from HuggingFace if not found.
    """

    MODEL_DIR = "models/sam3"
    MODEL_FILENAME = "sam3.safetensors"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadSAM3Model",
            display_name="(Down)Load SAM3 Model",
            category="SAM3",
            inputs=[
                io.Combo.Input("precision", options=["auto", "bf16", "fp16", "fp32"],
                               default="auto", optional=True,
                               tooltip="Model precision. auto: best for your GPU (bf16 on Ampere+, fp16 on Volta/Turing, fp32 on older)."),
                io.Boolean.Input("compile", default=False, optional=True,
                                 tooltip="Enable torch.compile for faster inference. Model loading takes longer (pre-compiles all code paths), but inference is significantly faster on every run."),
            ],
            outputs=[
                io.Custom("SAM3_MODEL").Output(display_name="sam3_model"),
            ],
        )

    @classmethod
    def execute(cls, precision="auto", compile=False):
        from .sam3_model_patcher import SAM3UnifiedModel
        from .sam3.predictor import Sam3VideoPredictor
        from .sam3.utils import Sam3Processor
        from .sam3 import build_sam3_video_model, _load_checkpoint_file, remap_video_checkpoint
        import comfy.model_management
        import comfy.utils

        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()

        # Fixed checkpoint path
        checkpoint_path = Path(comfy_base_path) / cls.MODEL_DIR / cls.MODEL_FILENAME

        # Auto-download if needed
        if not checkpoint_path.exists():
            log.info(f"Model not found at {checkpoint_path}, downloading from HuggingFace...")
            cls._download_from_huggingface()

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

        # ---- Meta-device construction (avoids 2x RAM) ----
        # 1. Build the model graph on the meta device (zero memory).
        log.info("Constructing model on meta device (zero memory)...")
        with torch.device("meta"):
            model = build_sam3_video_model(
                checkpoint_path=None,
                load_from_HF=False,
                bpe_path=bpe_path,
                enable_inst_interactivity=True,
                compile=compile,
                skip_checkpoint=True,
            )

        # 2. Load checkpoint and remap keys.
        log.info("Loading checkpoint into meta model with assign=True...")
        ckpt = _load_checkpoint_file(str(checkpoint_path))
        remapped_ckpt = remap_video_checkpoint(ckpt, enable_inst_interactivity=True)
        del ckpt  # free raw checkpoint memory immediately

        # 3. Load weights directly into the model (assign=True replaces meta
        #    tensors with real tensors without allocating a second copy).
        missing_keys, unexpected_keys = model.load_state_dict(
            remapped_ckpt, strict=False, assign=True,
        )
        del remapped_ckpt
        if missing_keys:
            log.info(f"Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            log.info(f"Unexpected keys: {len(unexpected_keys)}")

        # 4. Fix any leftover meta-device buffers (e.g. registered buffers
        #    that were not in the checkpoint, like zero-init masks).
        for name, buf in model.named_buffers():
            if buf.device.type == "meta":
                parts = name.split(".")
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                parent._buffers[parts[-1]] = torch.zeros_like(buf, device="cpu")

        model.eval()

        # Wrap in predictor (pass the pre-built model so it skips construction).
        video_predictor = Sam3VideoPredictor(
            bpe_path=bpe_path,
            enable_inst_interactivity=True,
            compile=compile,
            model=model,
        )

        # Tell sam3_attention() what half-precision dtype to target.
        # This enables centralized dtype normalization for every attention call
        # site -- including self-attention on fp32 token embeddings and
        # cross-attention after bf16+fp32 positional-encoding type promotion.
        from .sam3.attention import set_sam3_dtype
        set_sam3_dtype(dtype if dtype != torch.float32 else None)

        # Selective weight casting: cast only parameters (not buffers) to target
        # dtype for VRAM savings and half-precision attention.
        #
        # Buffers like freqs_cis (complex RoPE tensor in the ViT backbone) must
        # stay in their original dtype -- .to(bf16) on a complex tensor discards
        # the imaginary part, destroying position information.
        #
        # inst_interactive_predictor must also be cast: its no_mem_embed
        # (nn.Parameter) is added directly to bf16 vision features in
        # predict_inst(), causing fp32 type promotion. comfy.ops.manual_cast
        # can't intercept plain + operations -- it only casts linear layer
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

        return io.NodeOutput(unified_model)

    @staticmethod
    def _download_from_huggingface():
        if not HF_HUB_AVAILABLE:
            raise ImportError(
                "[SAM3] huggingface_hub is required to download models.\n"
                "Please install it with: pip install huggingface_hub"
            )

        model_dir = Path(comfy_base_path) / LoadSAM3Model.MODEL_DIR
        model_dir.mkdir(parents=True, exist_ok=True)

        hf_hub_download(
            repo_id="apozz/sam3-safetensors",
            filename=LoadSAM3Model.MODEL_FILENAME,
            local_dir=str(model_dir),
        )
        log.info(f"Model downloaded to: {model_dir / LoadSAM3Model.MODEL_FILENAME}")


NODE_CLASS_MAPPINGS = {
    "LoadSAM3Model": LoadSAM3Model
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSAM3Model": "(Down)Load SAM3 Model"
}
