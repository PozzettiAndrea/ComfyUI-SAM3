#!/usr/bin/env python3
# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""
Installation script for ComfyUI-SAM3.

Uses comfy-env for declarative dependency management via comfy-env.toml.
"""

import sys
from pathlib import Path


def main():
    print("\n" + "=" * 60)
    print("ComfyUI-SAM3 Installation")
    print("=" * 60)

    from comfy_env import install

    node_root = Path(__file__).parent.absolute()

    # Run comfy-env install (local mode - installs into ComfyUI's Python)
    try:
        install(config=node_root / "comfy-env.toml", mode="local", node_dir=node_root)
    except Exception as e:
        print(f"\n[SAM3] Installation FAILED: {e}")
        print("[SAM3] Report issues at: https://github.com/PozzettiAndrea/ComfyUI-SAM3/issues")
        return 1

    print("\n" + "=" * 60)
    print("[SAM3] Installation completed!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
