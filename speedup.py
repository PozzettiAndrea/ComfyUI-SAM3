#!/usr/bin/env python3
"""
GPU Acceleration Setup for ComfyUI-SAM3

This script installs GPU-accelerated CUDA extensions for faster video tracking:
- torch_generic_nms: 5-10x faster Non-Maximum Suppression
- cc_torch: GPU-accelerated connected components

These are OPTIONAL. ComfyUI-SAM3 works fine without them using CPU fallbacks.

Usage:
    python speedup.py              # Install GPU extensions
    python speedup.py --compile-only  # Compile even without GPU present (for CI)
    
For Comfyui_Portable:
    cd H:\AI\ComfyUI_windows_portable\ComfyUI\custom_nodes\comfyui-sam3 # Open Terminal within comfyui-sam3 folder
    H:\AI\ComfyUI_windows_portable\python_embeded\python.exe speedup.py # Run this command, replacing the path with the path correct for your system -> python_embeded\python.exe 
    


Requirements:
- NVIDIA GPU with CUDA support
- PyTorch with CUDA enabled
- CUDA compiler (nvcc) - this script will install it via micromamba if missing
"""
import os
import subprocess
import sys
import argparse

# FIX: Add current directory to path for embedded python support
# This ensures speedup_common can be found regardless of how python is called
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
# -------------------------------------------------------

# Import shared functions
import speedup_common

# Global flag for compile-only mode
COMPILE_ONLY = False


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

    # Check if nvcc is in system PATH (e.g., from apt-get install nvidia-cuda-toolkit)
    nvcc_in_path = shutil.which("nvcc")
    if nvcc_in_path:
        # Derive CUDA_HOME from nvcc location
        nvcc_dir = os.path.dirname(nvcc_in_path)
        if os.path.basename(nvcc_dir) == "bin":
            cuda_home = os.path.dirname(nvcc_dir)
            if cuda_home != "/usr" or "cuda-12" in cuda_home or "cuda-11" in nvcc_in_path:
                print(f"[ComfyUI-SAM3] Found CUDA via system PATH: {cuda_home} (nvcc at {nvcc_in_path})")
                return cuda_home
            else:
                print(f"[ComfyUI-SAM3] Found nvcc at {nvcc_in_path} but deferring to explicit CUDA 12.x path search...")

    # Check system CUDA installations (Linux/Mac)
    system_paths = [
        "/usr/local/cuda-12.4",
        "/usr/local/cuda-12.3",
        "/usr/local/cuda-12.2",
        "/usr/local/cuda-12.1",
        "/usr/local/cuda-12.0",
        "/usr/local/cuda",
        "/usr/cuda",
        "/opt/cuda",
        "/usr/local/cuda-12",
        "/usr/local/cuda-11",
    ]

    for cuda_path in system_paths:
        if os.path.exists(cuda_path):
            nvcc_path = os.path.join(cuda_path, "bin", "nvcc")
            if os.path.exists(nvcc_path):
                print(f"[ComfyUI-SAM3] Found system CUDA installation: {cuda_path}")
                return cuda_path

    # Check Windows common locations
    if sys.platform == "win32":
        program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
        cuda_base = os.path.join(program_files, "NVIDIA GPU Computing Toolkit", "CUDA")
        if os.path.exists(cuda_base):
            versions = [d for d in os.listdir(cuda_base) if os.path.isdir(os.path.join(cuda_base, d))]
            if versions:
                versions.sort(reverse=True)
                cuda_path = os.path.join(cuda_base, versions[0])
                nvcc_path = os.path.join(cuda_path, "bin", "nvcc.exe")
                if os.path.exists(nvcc_path):
                    print(f"[ComfyUI-SAM3] Found Windows CUDA installation: {cuda_path}")
                    return cuda_path

    # Check pip-installed CUDA in site-packages
    site_packages = site.getsitepackages()
    if isinstance(site_packages, str):
        site_packages = [site_packages]

    for sp in site_packages:
        cuda_nvcc_paths = [
            os.path.join(sp, "nvidia", "cuda_nvcc"),
            os.path.join(sp, "nvidia_cuda_nvcc"),
            os.path.join(sp, "cuda_nvcc"),
        ]

        for cuda_path in cuda_nvcc_paths:
            if os.path.exists(cuda_path):
                nvcc_path = os.path.join(cuda_path, "bin", "nvcc")
                if os.path.exists(nvcc_path) or os.path.exists(nvcc_path + ".exe"):
                    print(f"[ComfyUI-SAM3] Found pip-installed CUDA: {cuda_path}")
                    return cuda_path

    # Check conda environment (if running in one)
    if "CONDA_PREFIX" in os.environ:
        conda_prefix = os.environ["CONDA_PREFIX"]
        nvcc_path = os.path.join(conda_prefix, "bin", "nvcc")
        if os.path.exists(nvcc_path):
            print(f"[ComfyUI-SAM3] Found conda CUDA installation: {conda_prefix}")
            return conda_prefix

    return None


def check_nvcc_available():
    """
    Check if CUDA compiler (nvcc) is available.
    Returns True if nvcc is found, False otherwise.
    """
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            output = result.stdout
            if "release" in output:
                version = output.split('release')[1].split(',')[0].strip()
                print(f"[ComfyUI-SAM3] [OK] CUDA compiler found: release {version}")
            else:
                print(f"[ComfyUI-SAM3] [OK] CUDA compiler found")
            print(f"[ComfyUI-SAM3] nvcc output: {output.strip()}")
            return True
        else:
            print(f"[ComfyUI-SAM3] [WARNING] nvcc command failed with return code {result.returncode}")
            if result.stderr:
                print(f"[ComfyUI-SAM3] Error: {result.stderr[:200]}")
            return False
    except FileNotFoundError:
        print("[ComfyUI-SAM3] [WARNING] nvcc command not found in PATH")
        return False
    except subprocess.TimeoutExpired:
        print("[ComfyUI-SAM3] [WARNING] nvcc command timed out")
        return False
    except Exception as e:
        print(f"[ComfyUI-SAM3] [WARNING] Error checking nvcc: {e}")
        return False


def get_cuda_arch_list():
    """
    Detect GPU compute capability and return TORCH_CUDA_ARCH_LIST string.
    Returns None if detection fails.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            print("[ComfyUI-SAM3] [WARNING] CUDA not available in PyTorch")
            return None

        # Get compute capability of current device
        major, minor = torch.cuda.get_device_capability(0)
        compute_cap = f"{major}.{minor}"

        # Get device name for informational purposes
        device_name = torch.cuda.get_device_name(0)
        print(f"[ComfyUI-SAM3] Detected GPU: {device_name} (compute capability {compute_cap})")

        # Check if GPU is too new for current PyTorch version
        if major >= 12:
            print("=" * 80)
            print(f"[ComfyUI-SAM3] [ERROR] GPU compute capability {compute_cap} (sm_{major}{minor}) is not supported")
            print(f"[ComfyUI-SAM3] Current PyTorch version does not support Blackwell architecture (RTX 50-series)")
            print(f"[ComfyUI-SAM3]")
            print(f"[ComfyUI-SAM3] IMPORTANT: This is OPTIONAL - ComfyUI-SAM3 works without GPU acceleration!")
            print(f"[ComfyUI-SAM3]")
            print(f"[ComfyUI-SAM3] WORKAROUNDS:")
            print(f"[ComfyUI-SAM3] 1. Skip GPU acceleration (Recommended)")
            print(f"[ComfyUI-SAM3]    - Video tracking will use CPU fallback (5-10x slower but fully functional)")
            print(f"[ComfyUI-SAM3]    - Image segmentation is unaffected")
            print(f"[ComfyUI-SAM3]    - Simply don't run speedup.py, just use the base installation")
            print(f"[ComfyUI-SAM3]")
            print(f"[ComfyUI-SAM3] 2. Wait for PyTorch to add official sm_{major}{minor} support")
            print(f"[ComfyUI-SAM3]    - Track progress: https://github.com/pytorch/pytorch/issues/159207")
            print(f"[ComfyUI-SAM3]    - Rerun speedup.py once support is available")
            print(f"[ComfyUI-SAM3]")
            print(f"[ComfyUI-SAM3] 3. Advanced users: Build PyTorch from source with sm_{major}{minor} support")
            print(f"[ComfyUI-SAM3]    - See: https://pytorch.org/get-started/locally/#from-source")
            print("=" * 80)
            return None  # Abort compilation

        # OPTIMIZED: Compile ONLY for the detected GPU architecture
        arch_string = compute_cap

        print(f"[ComfyUI-SAM3]")
        print(f"[ComfyUI-SAM3] OPTIMIZED COMPILATION:")
        print(f"[ComfyUI-SAM3] Compiling ONLY for your GPU: {compute_cap} (sm_{major}{minor})")
        print(f"[ComfyUI-SAM3] Previous approach compiled for 5-6 architectures")
        print(f"[ComfyUI-SAM3] New approach compiles for 1 architecture only")
        print(f"[ComfyUI-SAM3]")
        print(f"[ComfyUI-SAM3] TIME SAVINGS:")
        print(f"[ComfyUI-SAM3]   Old: ~15-30 minutes (compiling for 5-6 architectures)")
        print(f"[ComfyUI-SAM3]   New: ~3-5 minutes (compiling for 1 architecture)")
        print(f"[ComfyUI-SAM3]   Improvement: 75-80% faster compilation!")
        print(f"[ComfyUI-SAM3]")
        print(f"[ComfyUI-SAM3] Using CUDA architecture: {arch_string}")

        return arch_string

    except Exception as e:
        print(f"[ComfyUI-SAM3] [ERROR] Could not detect GPU compute capability: {e}")
        import traceback
        traceback.print_exc()
        return None  # Fail safely instead of guessing architectures


def setup_cuda_environment():
    """
    Setup CUDA environment variables (CUDA_HOME and PATH) for compilation.
    Returns dict with environment variables if CUDA found, None otherwise.
    """
    cuda_home = find_cuda_home()

    if cuda_home:
        env = os.environ.copy()

        env["CUDA_HOME"] = cuda_home
        os.environ["CUDA_HOME"] = cuda_home
        print(f"[ComfyUI-SAM3] Set CUDA_HOME={cuda_home}")

        cuda_bin = os.path.join(cuda_home, "bin")
        if cuda_bin not in env.get("PATH", ""):
            env["PATH"] = cuda_bin + os.pathsep + env.get("PATH", "")
            os.environ["PATH"] = env["PATH"]
            print(f"[ComfyUI-SAM3] Added to PATH: {cuda_bin}")

        if check_nvcc_available():
            print(f"[ComfyUI-SAM3] CUDA environment configured:")
            print(f"[ComfyUI-SAM3]   CUDA_HOME: {env.get('CUDA_HOME', 'not set')}")
            print(f"[ComfyUI-SAM3]   PATH includes: {cuda_bin}")
            return env
        else:
            print(f"[ComfyUI-SAM3] [WARNING] nvcc not accessible after setting CUDA_HOME={cuda_home}")

    print("[ComfyUI-SAM3] [WARNING] Could not setup CUDA environment")
    return None


def ensure_micromamba_installed():
    """
    Ensure micromamba is available. If not found, download and install it.
    Returns True if micromamba is available, False otherwise.
    """
    import shutil

    # Check if micromamba, mamba, or conda already exists
    for cmd in ["micromamba", "mamba", "conda"]:
        if shutil.which(cmd):
            print(f"[ComfyUI-SAM3] Found {cmd}")
            return True

    print("[ComfyUI-SAM3] No conda-based package manager found")
    print("[ComfyUI-SAM3] Installing micromamba...")

    try:
        import tempfile
        import tarfile
        import urllib.request

        # Determine platform
        if sys.platform.startswith('linux'):
            platform = 'linux-64'
        elif sys.platform == 'darwin':
            import platform as plat
            if plat.machine() == 'arm64':
                platform = 'osx-arm64'
            else:
                platform = 'osx-64'
        else:
            print(f"[ComfyUI-SAM3] [WARNING] Unsupported platform for micromamba: {sys.platform}")
            return False

        # Download micromamba
        url = f"https://micro.mamba.pm/api/micromamba/{platform}/latest"
        print(f"[ComfyUI-SAM3] Downloading micromamba from {url}...")

        with tempfile.NamedTemporaryFile(delete=False, suffix='.tar.bz2') as tmp_file:
            urllib.request.urlretrieve(url, tmp_file.name)
            tmp_path = tmp_file.name

        # Extract to a local directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        micromamba_dir = os.path.join(script_dir, "bin")
        os.makedirs(micromamba_dir, exist_ok=True)

        print(f"[ComfyUI-SAM3] Extracting micromamba to {micromamba_dir}...")
        with tarfile.open(tmp_path, 'r:bz2') as tar:
            for member in tar.getmembers():
                if member.name == 'bin/micromamba':
                    member.name = 'micromamba'
                    tar.extract(member, micromamba_dir)
                    break

        # Make executable
        micromamba_path = os.path.join(micromamba_dir, "micromamba")
        os.chmod(micromamba_path, 0o755)

        # Add to PATH for current session
        os.environ["PATH"] = micromamba_dir + os.pathsep + os.environ.get("PATH", "")

        # Clean up
        os.unlink(tmp_path)

        # Verify installation
        if shutil.which("micromamba"):
            print(f"[ComfyUI-SAM3] [OK] micromamba installed successfully at {micromamba_path}")
            return True
        else:
            print("[ComfyUI-SAM3] [WARNING] micromamba installed but not found in PATH")
            return False

    except Exception as e:
        print(f"[ComfyUI-SAM3] [WARNING] Failed to install micromamba: {e}")
        return False


def try_install_cuda_micromamba(cuda_version=None):
    """
    Try to install CUDA toolkit using micromamba/conda/mamba.
    Returns True if successful, False otherwise.
    """
    import shutil

    print("[ComfyUI-SAM3] Attempting conda-based installation...")

    # Ensure micromamba is available
    if not ensure_micromamba_installed():
        print("[ComfyUI-SAM3] Could not setup micromamba, skipping conda-based installation")
        return False

    # Check for micromamba, mamba, or conda (in preference order)
    installer = None
    for cmd in ["micromamba", "mamba", "conda"]:
        if shutil.which(cmd):
            installer = cmd
            print(f"[ComfyUI-SAM3] Using {cmd} for CUDA toolkit installation...")
            break

    if not installer:
        print("[ComfyUI-SAM3] No conda-based package manager found after installation attempt")
        return False

    # Only proceed if we're in a conda environment
    if "CONDA_PREFIX" not in os.environ:
        print("[ComfyUI-SAM3] Not running in conda environment, skipping conda installation")
        return False

    try:
        # Determine package to install
        package_spec = "cuda-toolkit"
        if cuda_version:
            try:
                parts = cuda_version.split('.')
                cuda_major = int(parts[0])
                cuda_minor = int(parts[1]) if len(parts) > 1 else 0

                package_spec = f"cuda-toolkit={cuda_major}.{cuda_minor}"
                print(f"[ComfyUI-SAM3] Targeting CUDA version: {package_spec}")
            except (ValueError, IndexError):
                print(f"[ComfyUI-SAM3] Using generic cuda-toolkit package")

        # Install CUDA toolkit
        print(f"[ComfyUI-SAM3] Running: {installer} install -y {package_spec} -c conda-forge")
        result = subprocess.run(
            [installer, "install", "-y", package_spec, "-c", "conda-forge"],
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode == 0:
            print(f"[ComfyUI-SAM3] [OK] CUDA toolkit installed via {installer}")

            if setup_cuda_environment():
                return True
            else:
                print("[ComfyUI-SAM3] [WARNING] Installation succeeded but nvcc not accessible")
                conda_prefix = os.environ.get("CONDA_PREFIX", "")
                nvcc_path = os.path.join(conda_prefix, "bin", "nvcc")
                print(f"[ComfyUI-SAM3] Expected nvcc at: {nvcc_path}")
                print(f"[ComfyUI-SAM3] Exists: {os.path.exists(nvcc_path)}")
                return False
        else:
            print(f"[ComfyUI-SAM3] [WARNING] {installer} installation failed")
            if result.stderr:
                print(f"[ComfyUI-SAM3] Error: {result.stderr[:300]}")

            # Try fallback to major version only
            if cuda_version and "=" in package_spec:
                cuda_major = cuda_version.split('.')[0]
                fallback_spec = f"cuda-toolkit={cuda_major}.*"
                print(f"[ComfyUI-SAM3] Trying fallback: {fallback_spec}")
                result = subprocess.run(
                    [installer, "install", "-y", fallback_spec, "-c", "conda-forge"],
                    capture_output=True,
                    text=True,
                    timeout=600
                )
                if result.returncode == 0 and setup_cuda_environment():
                    print(f"[ComfyUI-SAM3] [OK] CUDA toolkit installed via {installer} with fallback spec")
                    return True

            return False

    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"[ComfyUI-SAM3] {installer} installation failed: {e}")
        return False
    except Exception as e:
        print(f"[ComfyUI-SAM3] Unexpected error during {installer} installation: {e}")
        return False


def try_install_cuda_toolkit():
    """
    Attempt to install CUDA toolkit if nvcc is missing.
    Returns True if successful or already installed, False otherwise.
    """
    print("[ComfyUI-SAM3] Attempting to install CUDA toolkit...")

    try:
        import torch

        cuda_version = torch.version.cuda
        if not cuda_version:
            print("[ComfyUI-SAM3] [WARNING] PyTorch not built with CUDA support")
            return False

        cuda_major = cuda_version.split('.')[0]
        print(f"[ComfyUI-SAM3] PyTorch CUDA version: {cuda_version} (major: cu{cuda_major})")

        # Try micromamba/conda first
        if try_install_cuda_micromamba(cuda_version):
            return True

        # If that fails, provide manual instructions
        print("[ComfyUI-SAM3] [WARNING] Could not install CUDA compiler automatically")
        print("[ComfyUI-SAM3] Manual installation required:")
        print("[ComfyUI-SAM3]   https://developer.nvidia.com/cuda-downloads")
        print("[ComfyUI-SAM3] Select your platform and follow the instructions")
        return False

    except Exception as e:
        print(f"[ComfyUI-SAM3] [WARNING] CUDA toolkit installation error: {e}")
        return False


def try_install_torch_generic_nms():
    """
    Install torch_generic_nms for GPU-accelerated video tracking.
    Provides 5-10x faster NMS compared to CPU fallback.
    """
    # Get CUDA architecture list
    cuda_arch_list = get_cuda_arch_list()
    if not cuda_arch_list:
        print("[ComfyUI-SAM3] [INFO] Skipping GPU NMS compilation (no compatible GPU detected)")
        print("[ComfyUI-SAM3] Video tracking will use CPU fallback (slower but functional)")
        return False

    # Use unified installation function from speedup_common
    success = speedup_common.install_extension(
        name="torch_generic_nms",
        repo_url="git+https://github.com/ronghanghu/torch_generic_nms",
        cuda_arch_list=cuda_arch_list,
        compile_only=COMPILE_ONLY,
        timeout=600,
        verbose=False
    )

    if success:
        print("[ComfyUI-SAM3] Video tracking will use GPU-accelerated NMS (5-10x faster)")
    else:
        print("[ComfyUI-SAM3] This is optional - video tracking will use CPU NMS fallback")

    return success


def try_install_cc_torch():
    """
    Install cc_torch for GPU-accelerated connected components.
    Used for hole filling in video tracking masks.
    """
    # Get CUDA architecture list
    cuda_arch_list = get_cuda_arch_list()
    if not cuda_arch_list:
        print("[ComfyUI-SAM3] [INFO] Skipping GPU connected components (no compatible GPU detected)")
        print("[ComfyUI-SAM3] Video tracking will use CPU fallback (slower but functional)")
        return False

    # Use unified installation function from speedup_common
    success = speedup_common.install_extension(
        name="cc_torch",
        repo_url="git+https://github.com/ronghanghu/cc_torch.git",
        cuda_arch_list=cuda_arch_list,
        compile_only=COMPILE_ONLY,
        timeout=600,
        verbose=False
    )

    if success:
        print("[ComfyUI-SAM3] Video tracking will use GPU-accelerated connected components")
    else:
        print("[ComfyUI-SAM3] This is optional - video tracking will use CPU fallback")

    return success


def check_python_version():
    """Check Python version compatibility."""
    major, minor = sys.version_info[:2]

    if major < 3 or (major == 3 and minor < 10):
        print(f"[ComfyUI-SAM3] [ERROR] Python {major}.{minor} is too old")
        print("[ComfyUI-SAM3] Minimum required: Python 3.10")
        return False

    if major == 3 and minor >= 13:
        print(f"[ComfyUI-SAM3] [WARNING] Python {major}.{minor} detected")
        print("[ComfyUI-SAM3] Python 3.13+ may have compatibility issues with CUDA extensions")
        print("[ComfyUI-SAM3] Recommended: Python 3.10, 3.11, or 3.12")
        # Continue anyway, just warn

    return True


if __name__ == "__main__":
    # Check Python version first
    if not check_python_version():
        sys.exit(1)

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Install GPU acceleration for ComfyUI-SAM3",
        epilog="This script is OPTIONAL. ComfyUI-SAM3 works fine without it using CPU fallbacks."
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Compile CUDA extensions even without GPU present (for CI testing)"
    )
    args = parser.parse_args()

    COMPILE_ONLY = args.compile_only

    if COMPILE_ONLY:
        print("[ComfyUI-SAM3] Running in COMPILE-ONLY mode (will skip GPU runtime checks)")

    print("[ComfyUI-SAM3] GPU Acceleration Setup")
    print("="*80)
    print("[ComfyUI-SAM3] This will install GPU-accelerated CUDA extensions")
    print("[ComfyUI-SAM3] These are OPTIONAL - ComfyUI-SAM3 works without them")
    print("="*80)

    # Try to install GPU-accelerated dependencies
    nms_success = try_install_torch_generic_nms()

    print("="*80)

    cc_success = try_install_cc_torch()

    print("="*80)

    if nms_success and cc_success:
        print("[ComfyUI-SAM3] ✓ GPU acceleration setup completed successfully!")
        print("[ComfyUI-SAM3] Video tracking will use GPU-accelerated operations (5-10x faster)")
    elif nms_success or cc_success:
        print("[ComfyUI-SAM3] ⚠ GPU acceleration partially installed")
        print("[ComfyUI-SAM3] Some operations will use CPU fallback")
    else:
        print("[ComfyUI-SAM3] ℹ GPU acceleration not installed")
        print("[ComfyUI-SAM3] ComfyUI-SAM3 will use CPU fallbacks (slower but functional)")
