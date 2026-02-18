"""
Convert sam3.pt checkpoint to sam3.safetensors and upload to HuggingFace.

Usage:
    python scripts/convert_sam3_safetensors.py

Downloads sam3.pt from 1038lab/sam3, unwraps the "model" key,
converts to safetensors, and uploads to apozz/sam3-safetensors.
"""

import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from huggingface_hub import HfApi, hf_hub_download

SRC_REPO = "1038lab/sam3"
SRC_FILE = "sam3.pt"
DST_REPO = "apozz/sam3-safetensors"
DST_FILE = "sam3.safetensors"
HF_TOKEN_PATH = Path.home() / "hf_token.txt"


def main():
    token = HF_TOKEN_PATH.read_text().strip()
    api = HfApi(token=token)

    # Download original checkpoint
    print(f"Downloading {SRC_FILE} from {SRC_REPO}...")
    pt_path = hf_hub_download(repo_id=SRC_REPO, filename=SRC_FILE)
    print(f"  -> {pt_path}")

    # Load and unwrap
    print("Loading checkpoint...")
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        state_dict = ckpt["model"]
        print(f"  Unwrapped 'model' key ({len(state_dict)} tensors)")
    else:
        state_dict = ckpt
        print(f"  Flat state dict ({len(state_dict)} tensors)")

    # Check all values are tensors
    non_tensor = {k: type(v).__name__ for k, v in state_dict.items() if not isinstance(v, torch.Tensor)}
    if non_tensor:
        print(f"  WARNING: {len(non_tensor)} non-tensor entries will be dropped:")
        for k, t in list(non_tensor.items())[:10]:
            print(f"    {k}: {t}")
        state_dict = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}

    # Clone tensors (shared memory causes RuntimeError in safetensors)
    print("Cloning tensors...")
    state_dict = {k: v.clone() for k, v in state_dict.items()}

    # Save safetensors
    out_path = Path(pt_path).parent / DST_FILE
    print(f"Saving {DST_FILE}...")
    save_file(state_dict, str(out_path))
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  -> {out_path} ({size_mb:.1f} MB)")

    # Create repo and upload
    print(f"Creating repo {DST_REPO}...")
    api.create_repo(repo_id=DST_REPO, exist_ok=True)

    print(f"Uploading {DST_FILE}...")
    api.upload_file(
        path_or_fileobj=str(out_path),
        path_in_repo=DST_FILE,
        repo_id=DST_REPO,
        repo_type="model",
    )
    print(f"Done! https://huggingface.co/{DST_REPO}")


if __name__ == "__main__":
    main()
