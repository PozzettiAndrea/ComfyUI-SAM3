"""
Installation script for ComfyUI-SAM3 required dependencies.
Called by ComfyUI Manager during installation/update.

For GPU acceleration (optional), run speedup.py after installation.
"""
import os
import subprocess
import sys


def install_requirements():
    """
    Install dependencies from requirements.txt.
    """
    print("[ComfyUI-SAM3] Installing requirements.txt dependencies...")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_path = os.path.join(script_dir, "requirements.txt")

    if not os.path.exists(requirements_path):
        print("[ComfyUI-SAM3] [WARNING] requirements.txt not found, skipping")
        return False

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", requirements_path],
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode == 0:
            print("[ComfyUI-SAM3] [OK] Requirements installed successfully")
        else:
            print("[ComfyUI-SAM3] [WARNING] Requirements installation had issues")
            if result.stderr:
                print(f"[ComfyUI-SAM3] Error details: {result.stderr[:500]}")
            return False

    except Exception as e:
        print(f"[ComfyUI-SAM3] [WARNING] Requirements installation error: {e}")
        return False

    # Install ninja for faster CUDA compilation (used by speedup.py if run later)
    print("[ComfyUI-SAM3] Installing ninja build system...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "ninja"],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            print("[ComfyUI-SAM3] [OK] Ninja installed successfully")
        else:
            print("[ComfyUI-SAM3] [WARNING] Ninja installation failed (optional, only needed for GPU acceleration)")
    except Exception as e:
        print(f"[ComfyUI-SAM3] [WARNING] Ninja installation error: {e}")

    return True


if __name__ == "__main__":
    print("[ComfyUI-SAM3] Running installation script...")
    print("="*80)

    # Install requirements.txt
    install_requirements()

    print("="*80)
    print("[ComfyUI-SAM3] Installation script completed")
    print("")
    print("[ComfyUI-SAM3] [INFO] For GPU acceleration (optional, 5-10x faster video tracking):")
    print("[ComfyUI-SAM3]    python speedup.py")
    print("[ComfyUI-SAM3]")
    print("[ComfyUI-SAM3] GPU acceleration is optional. ComfyUI-SAM3 works fine without it.")
