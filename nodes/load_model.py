"""
LoadSAM3Model node - Returns model config for subprocess-based loading.
Actual model construction happens inside consumer nodes (in isolation env).
"""
import logging
import os
from pathlib import Path

log = logging.getLogger("sam3")

import torch
import folder_paths
from folder_paths import base_path as comfy_base_path
from comfy_api.latest import io

# Register folder and ensure it exists
_sam3_models_dir = os.path.join(folder_paths.models_dir, "sam3")
os.makedirs(_sam3_models_dir, exist_ok=True)
folder_paths.add_model_folder_path("sam3", _sam3_models_dir)

try:
    from huggingface_hub import hf_hub_url
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


class LoadSAM3Model(io.ComfyNode):
    MODEL_DIR = "models/sam3"
    MODEL_FILENAME = "sam3.safetensors"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadSAM3Model",
            display_name="(Down)Load SAM3 Model",
            category="SAM3",
            outputs=[
                io.Custom("SAM3_MODEL_CONFIG").Output(display_name="sam3_model_config"),
            ],
            inputs=[
                io.Combo.Input(
                    "precision",
                    options=["auto", "bf16", "fp16", "fp32"],
                    default="auto",
                    optional=True,
                    tooltip="Model precision. auto picks bf16/fp16/fp32 based on GPU.",
                ),
                io.Boolean.Input(
                    "compile",
                    default=False,
                    optional=True,
                    tooltip="Enable torch.compile (slower first load, faster runs).",
                ),
            ],
        )

    @classmethod
    def execute(cls, precision="auto", compile=False):
        import comfy.utils
        import comfy.model_management

        load_device = comfy.model_management.get_torch_device()

        # Fixed checkpoint path inside ComfyUI base
        checkpoint_path = Path(comfy_base_path) / cls.MODEL_DIR / cls.MODEL_FILENAME

        # Auto-download if missing
        if not checkpoint_path.exists():
            log.info(f"[SAM3] Checkpoint not found at {checkpoint_path}, downloading...")
            try:
                cls._download_from_huggingface()
            except Exception as e:
                log.error(f"[SAM3] Download failed: {e}")
                raise RuntimeError(f"[SAM3] Failed to download model: {e}")

            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"[SAM3] Download reported success but file missing: {checkpoint_path}"
                )

        # Resolve BPE/tokenizer path relative to plugin root
        # Adjust the relative path if your plugin folder structure differs
        plugin_root_candidates = [
            Path(__file__).resolve().parent.parent,            # nodes/ -> root
            Path(__file__).resolve().parent / "sam3_lib",       # inside nodes
            Path(__file__).resolve().parent,
        ]
        bpe_path = None
        for root in plugin_root_candidates:
            candidate = root / "sam3" / "bpe_simple_vocab_16e6.txt.gz"
            if candidate.exists():
                bpe_path = candidate
                break
        if bpe_path is None:
            # Final fallback: try to construct path but warn
            bpe_path = plugin_root_candidates[0] / "sam3" / "bpe_simple_vocab_16e6.txt.gz"
            log.warning(f"[SAM3] BPE file not found at expected paths. Using fallback: {bpe_path}")

        # Resolve dtype
        if precision == "auto":
            if comfy.model_management.should_use_bf16(load_device):
                dtype_str = "bf16"
            elif comfy.model_management.should_use_fp16(load_device):
                dtype_str = "fp16"
            else:
                dtype_str = "fp32"
        else:
            mapping = {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32"}
            if precision not in mapping:
                raise ValueError(f"[SAM3] Invalid precision '{precision}'. Choose auto/bf16/fp16/fp32.")
            dtype_str = mapping[precision]

        log.info(f"[SAM3] Config ready: device={load_device}, precision={dtype_str}, compile={compile}, checkpoint={checkpoint_path}")

        config = {
            "checkpoint_path": str(checkpoint_path.resolve()),
            "bpe_path": str(bpe_path.resolve()) if bpe_path and bpe_path.exists() else "",
            "precision": precision,
            "dtype": dtype_str,
            "compile": bool(compile),
        }
        return io.NodeOutput(config)

    @classmethod
    def _download_from_huggingface(cls, pbar=None):
        if not HF_HUB_AVAILABLE:
            raise ImportError(
                "[SAM3] huggingface_hub and requests are required. "
                "Install: pip install huggingface_hub requests"
            )

        import requests
        from huggingface_hub import hf_hub_url

        model_dir = Path(comfy_base_path) / cls.MODEL_DIR
        model_dir.mkdir(parents=True, exist_ok=True)

        url = hf_hub_url("apozz/sam3-safetensors", cls.MODEL_FILENAME)
        dest = model_dir / cls.MODEL_FILENAME
        tmp = dest.with_suffix(".tmp")

        log.info(f"[SAM3] Downloading {cls.MODEL_FILENAME} ...")

        resp = requests.get(url, stream=True, timeout=120, allow_redirects=True)
        resp.raise_for_status()
        total_size = int(resp.headers.get("content-length", 0))

        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB

        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = int(downloaded * 100 / total_size)
                        # Simple progress log every ~10%
                        if pct % 10 == 0 and pct > 0:
                            log.info(f"[SAM3] Download progress: {pct}%")

        # Atomic rename
        if tmp.exists():
            tmp.rename(dest)
            log.info(f"[SAM3] Saved checkpoint to {dest}")
        else:
            raise RuntimeError(f"[SAM3] Download failed: temp file missing ({tmp})")


NODE_CLASS_MAPPINGS = {
    "LoadSAM3Model": LoadSAM3Model,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSAM3Model": "(Down)Load SAM3 Model",
}