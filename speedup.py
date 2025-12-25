#!/usr/bin/env python3
"""
GPU Acceleration Setup for ComfyUI-SAM3

This script installs GPU-accelerated CUDA extensions for faster video tracking:
- torch_generic_nms: 5-10x faster Non-Maximum Suppression
- cc_torch: GPU-accelerated connected components

These are OPTIONAL. ComfyUI-SAM3 works fine without them using CPU fallbacks.

Usage:
    python speedup.py                           # Install GPU extensions
    python speedup.py --compile-only            # Compile even without GPU present (for CI)
    python speedup.py --compile-only --cuda-arch 8.9  # Compile for specific architecture (for CI)

For ComfyUI Portable (Windows):
    cd path\\to\\ComfyUI\\custom_nodes\\ComfyUI-SAM3
    path\\to\\python_embeded\\python.exe speedup.py

Requirements:
- NVIDIA GPU with CUDA support
- PyTorch with CUDA enabled
- Pre-built wheels available for most CUDA/PyTorch combinations
- Falls back to compilation if no wheel available (requires nvcc)
"""
import os
import subprocess
import sys
import argparse

# =============================================================================
# Pre-built Wheel Configuration
# =============================================================================

WHEEL_INDEX = "https://pozzettiandrea.github.io/sam3-speedup-wheels"

# Map (CUDA major.minor, PyTorch major.minor) -> wheel directory
WHEEL_DIRS = {
    ("12.4", "2.5"): "cu124-torch251",
    ("12.6", "2.6"): "cu126-torch260",
    ("12.6", "2.8"): "cu126-torch280",
    ("12.8", "2.7"): "cu128-torch271",
    ("12.8", "2.8"): "cu128-torch280",
    ("12.8", "2.9"): "cu128-torch291",
}

# Global flags
COMPILE_ONLY = False
CUDA_ARCH_OVERRIDE = None


# =============================================================================
# Wheel Installation Functions
# =============================================================================

def get_wheels_url():
    """
    Get wheel index URL for current CUDA + PyTorch versions.
    Returns URL string if matching wheels exist, None otherwise.
    """
    try:
        import torch
    except ImportError:
        return None

    cuda = torch.version.cuda
    if not cuda:
        return None

    torch_ver = torch.__version__.split('+')[0]  # Remove +cu128 suffix if present
    cuda_mm = '.'.join(cuda.split('.')[:2])       # "12.8"
    torch_mm = '.'.join(torch_ver.split('.')[:2]) # "2.8"

    wheel_dir = WHEEL_DIRS.get((cuda_mm, torch_mm))
    if wheel_dir:
        return f"{WHEEL_INDEX}/{wheel_dir}/"
    return None


def try_install_from_wheels(package_name):
    """
    Try installing a package from pre-built wheels.
    Returns True if successful, False otherwise.
    """
    wheels_url = get_wheels_url()
    if not wheels_url:
        return False

    print(f"[ComfyUI-SAM3] Trying pre-built wheel from {wheels_url}")
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install",
            package_name, "--find-links", wheels_url
        ], capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            print(f"[ComfyUI-SAM3] [OK] Installed {package_name} from pre-built wheel!")
            return True
        else:
            if result.stderr:
                print(f"[ComfyUI-SAM3] Wheel install failed: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"[ComfyUI-SAM3] Wheel install timed out")
        return False
    except Exception as e:
        print(f"[ComfyUI-SAM3] Wheel install error: {e}")
        return False


# =============================================================================
# CUDA Environment Functions
# =============================================================================

def find_cuda_home():
    """
    Find CUDA installation directory from all possible sources.
    Returns path to CUDA installation or None if not found.
    """
    import site
    import shutil

    # Check existing CUDA_HOME environment variable
    if "CUDA_HOME" in os.environ and os.path.exists(os.environ["CUDA_HOME"]):
        cuda_home = os.environ["CUDA_HOME"]
        nvcc_path = os.path.join(cuda_home, "bin", "nvcc")
        if os.path.exists(nvcc_path) or os.path.exists(nvcc_path + ".exe"):
            print(f"[ComfyUI-SAM3] Found CUDA via CUDA_HOME: {cuda_home}")
            return cuda_home

    # Check if nvcc is in system PATH
    nvcc_in_path = shutil.which("nvcc")
    if nvcc_in_path:
        nvcc_dir = os.path.dirname(nvcc_in_path)
        if os.path.basename(nvcc_dir) == "bin":
            cuda_home = os.path.dirname(nvcc_dir)
            if cuda_home != "/usr" or "cuda-12" in cuda_home or "cuda-11" in nvcc_in_path:
                print(f"[ComfyUI-SAM3] Found CUDA via system PATH: {cuda_home}")
                return cuda_home

    # Check system CUDA installations (Linux/Mac)
    system_paths = [
        "/usr/local/cuda-12.8", "/usr/local/cuda-12.6", "/usr/local/cuda-12.4",
        "/usr/local/cuda-12.2", "/usr/local/cuda-12.0", "/usr/local/cuda",
        "/usr/cuda", "/opt/cuda",
    ]
    for cuda_path in system_paths:
        if os.path.exists(cuda_path):
            nvcc_path = os.path.join(cuda_path, "bin", "nvcc")
            if os.path.exists(nvcc_path):
                print(f"[ComfyUI-SAM3] Found system CUDA: {cuda_path}")
                return cuda_path

    # Check Windows locations
    if sys.platform == "win32":
        program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
        cuda_base = os.path.join(program_files, "NVIDIA GPU Computing Toolkit", "CUDA")
        if os.path.exists(cuda_base):
            versions = sorted([d for d in os.listdir(cuda_base) if os.path.isdir(os.path.join(cuda_base, d))], reverse=True)
            for v in versions:
                cuda_path = os.path.join(cuda_base, v)
                if os.path.exists(os.path.join(cuda_path, "bin", "nvcc.exe")):
                    print(f"[ComfyUI-SAM3] Found Windows CUDA: {cuda_path}")
                    return cuda_path

    # Check pip-installed CUDA
    for sp in site.getsitepackages():
        for subdir in ["nvidia/cuda_nvcc", "nvidia_cuda_nvcc", "cuda_nvcc"]:
            cuda_path = os.path.join(sp, subdir)
            if os.path.exists(cuda_path):
                nvcc = os.path.join(cuda_path, "bin", "nvcc")
                if os.path.exists(nvcc) or os.path.exists(nvcc + ".exe"):
                    print(f"[ComfyUI-SAM3] Found pip CUDA: {cuda_path}")
                    return cuda_path

    # Check conda
    if "CONDA_PREFIX" in os.environ:
        nvcc = os.path.join(os.environ["CONDA_PREFIX"], "bin", "nvcc")
        if os.path.exists(nvcc):
            print(f"[ComfyUI-SAM3] Found conda CUDA: {os.environ['CONDA_PREFIX']}")
            return os.environ["CONDA_PREFIX"]

    return None


def check_nvcc_available():
    """Check if CUDA compiler (nvcc) is available."""
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"[ComfyUI-SAM3] [OK] CUDA compiler found")
            return True
    except:
        pass
    return False


def check_msvc_available():
    """Check if MSVC compiler is available on Windows."""
    if sys.platform != "win32":
        return True

    import shutil
    if shutil.which("cl"):
        print("[ComfyUI-SAM3] [OK] MSVC compiler found")
        return True

    vs_paths = [
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC",
    ]
    for p in vs_paths:
        if os.path.exists(p):
            print(f"[ComfyUI-SAM3] Visual Studio found but cl.exe not in PATH")
            print("[ComfyUI-SAM3] Run from Developer Command Prompt for VS")
            return True

    print("[ComfyUI-SAM3] [ERROR] MSVC not found. Install Visual Studio Build Tools.")
    return False


def setup_cuda_environment():
    """Setup CUDA environment variables for compilation."""
    cuda_home = find_cuda_home()
    if not cuda_home:
        return None

    env = os.environ.copy()
    env["CUDA_HOME"] = cuda_home
    os.environ["CUDA_HOME"] = cuda_home

    cuda_bin = os.path.join(cuda_home, "bin")
    if cuda_bin not in env.get("PATH", ""):
        env["PATH"] = cuda_bin + os.pathsep + env.get("PATH", "")
        os.environ["PATH"] = env["PATH"]

    if check_nvcc_available():
        print(f"[ComfyUI-SAM3] CUDA environment configured: {cuda_home}")
        return env

    return None


def get_cuda_arch_list(override=None):
    """Detect GPU compute capability and return TORCH_CUDA_ARCH_LIST string."""
    if override:
        print(f"[ComfyUI-SAM3] Using specified CUDA architecture: {override}")
        return override

    try:
        import torch
        if not torch.cuda.is_available():
            print("[ComfyUI-SAM3] [WARNING] CUDA not available in PyTorch")
            return None

        major, minor = torch.cuda.get_device_capability(0)
        compute_cap = f"{major}.{minor}"
        device_name = torch.cuda.get_device_name(0)
        print(f"[ComfyUI-SAM3] Detected GPU: {device_name} (sm_{major}{minor})")
        return compute_cap

    except Exception as e:
        print(f"[ComfyUI-SAM3] [ERROR] Could not detect GPU: {e}")
        return None


# =============================================================================
# Installation Functions
# =============================================================================

def install_extension(name, repo_url, cuda_arch_list, compile_only=False, timeout=600):
    """
    Install a CUDA extension - tries pre-built wheels first, falls back to compilation.
    """
    print(f"[ComfyUI-SAM3] Checking for {name}...")

    # Check if already installed
    try:
        __import__(name)
        print(f"[ComfyUI-SAM3] [OK] {name} already installed")
        return True
    except ImportError:
        pass

    # Try pre-built wheels first (fast!)
    if try_install_from_wheels(name):
        return True

    # Fall back to compilation
    print(f"[ComfyUI-SAM3] No pre-built wheel available, compiling from source...")

    # Check PyTorch CUDA
    try:
        import torch
        if not compile_only and not torch.cuda.is_available():
            print(f"[ComfyUI-SAM3] [INFO] CUDA not available, skipping {name}")
            return False
        if compile_only and not torch.version.cuda:
            print(f"[ComfyUI-SAM3] [ERROR] PyTorch not built with CUDA")
            return False
    except ImportError:
        print(f"[ComfyUI-SAM3] [WARNING] PyTorch not found")
        return False

    # Setup CUDA environment
    cuda_env = setup_cuda_environment()
    if not cuda_env:
        print(f"[ComfyUI-SAM3] [WARNING] CUDA compiler not found")
        return False

    # Set architecture
    if cuda_arch_list:
        cuda_env["TORCH_CUDA_ARCH_LIST"] = cuda_arch_list
        os.environ["TORCH_CUDA_ARCH_LIST"] = cuda_arch_list

    # Platform-specific compiler setup
    import shutil
    if sys.platform == "win32":
        if not check_msvc_available():
            return False
        msvc_flags = "/permissive- /Zc:__cplusplus"
        cuda_env["CFLAGS"] = cuda_env.get("CFLAGS", "") + " " + msvc_flags
        cuda_env["CXXFLAGS"] = cuda_env.get("CXXFLAGS", "") + " " + msvc_flags
    else:
        gcc = shutil.which("gcc-11") or shutil.which("gcc")
        gxx = shutil.which("g++-11") or shutil.which("g++")
        if gcc:
            cuda_env["CC"] = gcc
        if gxx:
            cuda_env["CXX"] = gxx

    # Compile
    print(f"[ComfyUI-SAM3] Compiling {name} (this may take a few minutes)...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-build-isolation", repo_url],
            capture_output=True, text=True, timeout=timeout, env=cuda_env
        )
        if result.returncode == 0:
            print(f"[ComfyUI-SAM3] [OK] {name} compiled successfully")
            return True
        else:
            print(f"[ComfyUI-SAM3] [WARNING] {name} compilation failed")
            if result.stderr:
                print(f"[ComfyUI-SAM3] Error: {result.stderr[:300]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"[ComfyUI-SAM3] [WARNING] Compilation timed out")
        return False
    except Exception as e:
        print(f"[ComfyUI-SAM3] [WARNING] Error: {e}")
        return False


def try_install_torch_generic_nms():
    """Install torch_generic_nms for GPU-accelerated NMS."""
    cuda_arch_list = get_cuda_arch_list(CUDA_ARCH_OVERRIDE)
    if not cuda_arch_list and not COMPILE_ONLY:
        print("[ComfyUI-SAM3] [INFO] Skipping GPU NMS (no compatible GPU)")
        return False

    success = install_extension(
        name="torch_generic_nms",
        repo_url="git+https://github.com/ronghanghu/torch_generic_nms",
        cuda_arch_list=cuda_arch_list,
        compile_only=COMPILE_ONLY
    )

    if success:
        print("[ComfyUI-SAM3] Video tracking will use GPU-accelerated NMS (5-10x faster)")
    return success


def try_install_cc_torch():
    """Install cc_torch for GPU-accelerated connected components."""
    cuda_arch_list = get_cuda_arch_list(CUDA_ARCH_OVERRIDE)
    if not cuda_arch_list and not COMPILE_ONLY:
        print("[ComfyUI-SAM3] [INFO] Skipping GPU connected components (no compatible GPU)")
        return False

    success = install_extension(
        name="cc_torch",
        repo_url="git+https://github.com/ronghanghu/cc_torch.git",
        cuda_arch_list=cuda_arch_list,
        compile_only=COMPILE_ONLY
    )

    if success:
        print("[ComfyUI-SAM3] Video tracking will use GPU-accelerated connected components")
    return success


def check_python_version():
    """Check Python version compatibility."""
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 10):
        print(f"[ComfyUI-SAM3] [ERROR] Python {major}.{minor} too old (need 3.10+)")
        return False
    if major == 3 and minor >= 13:
        print(f"[ComfyUI-SAM3] [WARNING] Python 3.13+ may have compatibility issues")
    return True


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    if not check_python_version():
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Install GPU acceleration for ComfyUI-SAM3")
    parser.add_argument("--compile-only", action="store_true", help="Compile without GPU present")
    parser.add_argument("--cuda-arch", type=str, help="CUDA architecture (e.g., '8.6')")
    args = parser.parse_args()

    COMPILE_ONLY = args.compile_only
    CUDA_ARCH_OVERRIDE = args.cuda_arch

    print("[ComfyUI-SAM3] GPU Acceleration Setup")
    print("=" * 60)
    print("[ComfyUI-SAM3] Installing GPU-accelerated CUDA extensions")
    print("[ComfyUI-SAM3] These are OPTIONAL - SAM3 works without them")
    print("=" * 60)

    nms_success = try_install_torch_generic_nms()
    print("=" * 60)
    cc_success = try_install_cc_torch()
    print("=" * 60)

    if nms_success and cc_success:
        print("[ComfyUI-SAM3] [OK] GPU acceleration installed!")
    elif nms_success or cc_success:
        print("[ComfyUI-SAM3] [WARNING] Partial installation")
    else:
        print("[ComfyUI-SAM3] [INFO] GPU acceleration not installed (using CPU fallbacks)")
