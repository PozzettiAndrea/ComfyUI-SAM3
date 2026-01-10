"""
Prestartup script for ComfyUI-SAM3
Copies assets to input folder and workflows to user workflows folder
"""
import os
import shutil
from pathlib import Path

def copy_files_skip_existing(src_dir, dst_dir):
    """Copy files from src to dst, skipping files that already exist"""
    if not os.path.exists(src_dir):
        print(f"[SAM3] Source directory not found: {src_dir}")
        return

    os.makedirs(dst_dir, exist_ok=True)

    copied = 0
    skipped = 0

    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dst_path = os.path.join(dst_dir, item)

        if os.path.isfile(src_path):
            if os.path.exists(dst_path):
                skipped += 1
            else:
                shutil.copy(src_path, dst_path)
                copied += 1
        elif os.path.isdir(src_path):
            # Recursively copy directories
            if not os.path.exists(dst_path):
                shutil.copytree(src_path, dst_path)
                copied += 1
            else:
                skipped += 1

    return copied, skipped

def main():
    # Get the directory where this script is located (custom_nodes/ComfyUI-SAM3/)
    script_dir = Path(__file__).parent.absolute()

    # Get ComfyUI root directory (two levels up from custom_nodes/ComfyUI-SAM3/)
    comfy_root = script_dir.parent.parent

    print(f"[SAM3] ComfyUI-SAM3 prestartup script running...")
    print(f"[SAM3] Script directory: {script_dir}")
    print(f"[SAM3] ComfyUI root: {comfy_root}")

    # Copy image assets to input folder (skip .txt.gz files)
    assets_src = script_dir / "assets"
    assets_dst = comfy_root / "input"

    if assets_src.exists():
        print(f"[SAM3] Copying image assets to {assets_dst}...")
        copied = 0
        skipped = 0
        for item in os.listdir(str(assets_src)):
            src_path = assets_src / item
            if src_path.is_file():
                # Only copy image files (jpg, png, etc), skip .txt.gz
                if not item.endswith('.txt.gz'):
                    dst_path = assets_dst / item
                    if dst_path.exists():
                        skipped += 1
                    else:
                        shutil.copy(str(src_path), str(dst_path))
                        copied += 1
        print(f"[SAM3] Image assets: {copied} copied, {skipped} skipped")
    else:
        print(f"[SAM3] No assets folder found")
    
    print(f"[SAM3] Prestartup script completed")

if __name__ == "__main__":
    main()

# Run on import (ComfyUI calls prestartup scripts by importing them)
try:
    main()
except Exception as e:
    print(f"[SAM3] Error in prestartup script: {e}")
    import traceback
    traceback.print_exc()
