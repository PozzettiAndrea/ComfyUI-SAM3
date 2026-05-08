"""
LoadSAM3Model node - Returns model config for subprocess-based loading.

Actual model construction happens inside consumer nodes (in the isolation
env subprocess), following the Trellis2 pattern.  This node resolves
precision, downloads the checkpoint if needed, and returns a JSON-safe
config dict.
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
    Node to load SAM3 model configuration.
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
                io.Custom("SAM3_MODEL_CONFIG").Output(display_name="sam3_model_config"),
            ],
        )

    @classmethod
    def execute(cls, precision="auto", compile=False):
        import comfy.utils
        import comfy.model_management

        load_device = comfy.model_management.get_torch_device()

        # Fixed checkpoint path
        checkpoint_path = Path(comfy_base_path) / cls.MODEL_DIR / cls.MODEL_FILENAME

        # Auto-download if needed
        if not checkpoint_path.exists():
            print(f"[SAM3] Model not found at {checkpoint_path}, downloading from HuggingFace...", flush=True)
            pbar = comfy.utils.ProgressBar(100)
            cls._download_from_huggingface(pbar)
            print("[SAM3] Model downloaded successfully", flush=True)

        # BPE path for tokenizer
        bpe_path = str(Path(__file__).parent / "sam3" / "bpe_simple_vocab_16e6.txt.gz")

        # Resolve dtype
        if precision == "auto":
            if comfy.model_management.should_use_bf16(load_device):
                dtype = torch.bfloat16
            elif comfy.model_management.should_use_fp16(load_device):
                dtype = torch.float16
            else:
                dtype = torch.float32
        else:
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        # Store dtype as string for JSON-safe IPC across isolation boundary
        dtype_str = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}[dtype]
        log.info(f"Resolved precision: {precision} -> {dtype_str}")

        config = {
            "checkpoint_path": str(checkpoint_path),
            "bpe_path": bpe_path,
            "precision": precision,
            "dtype": dtype_str,
            "compile": compile,
        }
        return io.NodeOutput(config)

    @staticmethod
    def _download_from_huggingface(pbar=None):
        if not HF_HUB_AVAILABLE:
            raise ImportError(
                "[SAM3] huggingface_hub is required to download models.\n"
                "Please install it with: pip install huggingface_hub"
            )
        import requests
        from huggingface_hub import hf_hub_url

        model_dir = Path(comfy_base_path) / LoadSAM3Model.MODEL_DIR
        model_dir.mkdir(parents=True, exist_ok=True)

        url = hf_hub_url("apozz/sam3-safetensors", LoadSAM3Model.MODEL_FILENAME)
        dest = model_dir / LoadSAM3Model.MODEL_FILENAME
        tmp = dest.with_suffix(".tmp")

        print(f"[SAM3] Downloading {LoadSAM3Model.MODEL_FILENAME} from apozz/sam3-safetensors...", flush=True)
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        last_pct = 0

        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = int(downloaded * 100 / total)
                    if pbar and pct > last_pct:
                        pbar.update_absolute(pct, 100)
                    if pct >= last_pct + 10:
                        print(f"[SAM3] Download progress: {pct}%", flush=True)
                        last_pct = pct

        tmp.rename(dest)
        if pbar:
            pbar.update_absolute(100, 100)
        print(f"[SAM3] Downloaded {LoadSAM3Model.MODEL_FILENAME}", flush=True)


NODE_CLASS_MAPPINGS = {
    "LoadSAM3Model": LoadSAM3Model
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSAM3Model": "(Down)Load SAM3 Model"
}
