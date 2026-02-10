"""
Installation script for ComfyUI-SAM3 optional dependencies.
Called by ComfyUI Manager during installation/update.
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
            return True
        else:
            print("[ComfyUI-SAM3] [WARNING] Requirements installation had issues")
            if result.stderr:
                print(f"[ComfyUI-SAM3] Error details: {result.stderr[:500]}")
            return False

    except Exception as e:
        print(f"[ComfyUI-SAM3] [WARNING] Requirements installation error: {e}")
        return False


def find_cuda_home():
    """
    Find CUDA installation directory from all possible sources.
    Returns path to CUDA installation or None if not found.
    """
    import site

    # Check existing CUDA_HOME environment variable
    if "CUDA_HOME" in os.environ and os.path.exists(os.environ["CUDA_HOME"]):
        cuda_home = os.environ["CUDA_HOME"]
        nvcc_path = os.path.join(cuda_home, "bin", "nvcc")
        if os.path.exists(nvcc_path) or os.path.exists(nvcc_path + ".exe"):
            print(f"[ComfyUI-SAM3] Found CUDA via CUDA_HOME: {cuda_home}")
            return cuda_home

    # Check system CUDA installations (Linux/Mac)
    system_paths = [
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
            # Find latest version
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
        # Check nvidia/cuda_nvcc structure
        cuda_nvcc_paths = [
            os.path.join(sp, "nvidia", "cuda_nvcc"),
            os.path.join(sp, "nvidia_cuda_nvcc"),
            os.path.join(sp, "cuda_nvcc"),
        ]

        for cuda_path in cuda_nvcc_paths:
            if os.path.exists(cuda_path):
                # Check if nvcc binary exists
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
            # Extract CUDA version from nvcc output
            output = result.stdout
            if "release" in output:
                print(f"[ComfyUI-SAM3] [OK] CUDA compiler found: {output.split('release')[1].split(',')[0].strip()}")
                return True
        return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
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

        # Common architecture list covering most modern GPUs
        # Include current GPU and common architectures
        arch_list = []

        # Add current GPU architecture
        arch_list.append(compute_cap)

        # Add common architectures for compatibility
        common_archs = ["7.5", "8.0", "8.6", "8.9", "9.0"]
        for arch in common_archs:
            if arch not in arch_list:
                arch_list.append(arch)

        arch_string = " ".join(arch_list)
        print(f"[ComfyUI-SAM3] Using CUDA architectures: {arch_string}")

        return arch_string

    except Exception as e:
        print(f"[ComfyUI-SAM3] [WARNING] Could not detect GPU compute capability: {e}")
        # Fallback to common architectures
        return "7.5 8.0 8.6 8.9 9.0"


def setup_cuda_environment():
    """
    Setup CUDA environment variables (CUDA_HOME and PATH) for compilation.
    Searches for CUDA installation from all possible sources.
    Returns True if CUDA found and environment set up, False otherwise.
    """
    # First try to find existing CUDA installation
    cuda_home = find_cuda_home()

    if cuda_home:
        # Set CUDA_HOME environment variable
        os.environ["CUDA_HOME"] = cuda_home
        print(f"[ComfyUI-SAM3] Set CUDA_HOME={cuda_home}")

        # Add CUDA bin directory to PATH
        cuda_bin = os.path.join(cuda_home, "bin")
        if cuda_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")
            print(f"[ComfyUI-SAM3] Added to PATH: {cuda_bin}")

        # Verify nvcc is accessible
        if check_nvcc_available():
            return True

    return False


def try_install_cuda_toolkit():
    """
    Attempt to install CUDA toolkit if nvcc is missing.
    Supports Linux, Windows, and Mac (though Mac doesn't support CUDA anymore).
    Returns True if successful or already installed, False otherwise.
    """
    print("[ComfyUI-SAM3] Attempting to install CUDA toolkit...")

    try:
        import torch

        cuda_version = torch.version.cuda
        if not cuda_version:
            print("[ComfyUI-SAM3] [WARNING] PyTorch not built with CUDA support")
            return False

        # Extract major version from CUDA version (e.g., "12" from "12.8")
        cuda_major = cuda_version.split('.')[0]
        print(f"[ComfyUI-SAM3] PyTorch CUDA version: {cuda_version} (major: cu{cuda_major})")

        # Try pip install first (works on all platforms but may be incomplete)
        package_name = f"nvidia-cuda-nvcc-cu{cuda_major}"
        print(f"[ComfyUI-SAM3] Trying to install {package_name} via pip...")

        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_name],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            print("[ComfyUI-SAM3] [OK] CUDA compiler package installed via pip")

            # Try to setup environment with newly installed CUDA
            if setup_cuda_environment():
                print("[ComfyUI-SAM3] [OK] CUDA environment configured successfully")
                return True
            else:
                print("[ComfyUI-SAM3] [WARNING] pip package installed but nvcc not accessible")
                print("[ComfyUI-SAM3] Note: pip nvidia-cuda-nvcc packages are often incomplete")
                print("[ComfyUI-SAM3] Trying system package manager as fallback...")
        else:
            print(f"[ComfyUI-SAM3] [WARNING] pip install of {package_name} failed")
            if result.stderr:
                print(f"[ComfyUI-SAM3] Error: {result.stderr[:300]}")

        # Platform-specific installation fallbacks
        if sys.platform.startswith('linux'):
            return try_install_cuda_linux()
        elif sys.platform == 'win32':
            return try_install_cuda_windows(cuda_version)
        elif sys.platform == 'darwin':
            return try_install_cuda_mac()
        else:
            print(f"[ComfyUI-SAM3] [WARNING] Unsupported platform: {sys.platform}")
            return False

    except Exception as e:
        print(f"[ComfyUI-SAM3] [WARNING] CUDA toolkit installation error: {e}")
        return False


def try_install_cuda_linux():
    """Try to install CUDA toolkit on Linux using package managers."""
    print("[ComfyUI-SAM3] Attempting Linux package manager installation...")

    # Try apt-get (Debian/Ubuntu)
    try:
        print("[ComfyUI-SAM3] Trying apt-get (Debian/Ubuntu)...")
        result = subprocess.run(
            ["sudo", "apt-get", "update"],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode == 0:
            result = subprocess.run(
                ["sudo", "apt-get", "install", "-y", "nvidia-cuda-toolkit"],
                capture_output=True,
                text=True,
                timeout=600
            )
            if result.returncode == 0:
                print("[ComfyUI-SAM3] [OK] CUDA compiler installed via apt-get")
                if setup_cuda_environment():
                    return True
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"[ComfyUI-SAM3] apt-get not available or failed: {e}")

    # Try yum (RedHat/CentOS/Fedora)
    try:
        print("[ComfyUI-SAM3] Trying yum (RedHat/CentOS)...")
        result = subprocess.run(
            ["sudo", "yum", "install", "-y", "cuda-toolkit"],
            capture_output=True,
            text=True,
            timeout=600
        )
        if result.returncode == 0:
            print("[ComfyUI-SAM3] [OK] CUDA compiler installed via yum")
            if setup_cuda_environment():
                return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Try dnf (Fedora)
    try:
        print("[ComfyUI-SAM3] Trying dnf (Fedora)...")
        result = subprocess.run(
            ["sudo", "dnf", "install", "-y", "cuda-toolkit"],
            capture_output=True,
            text=True,
            timeout=600
        )
        if result.returncode == 0:
            print("[ComfyUI-SAM3] [OK] CUDA compiler installed via dnf")
            if setup_cuda_environment():
                return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    print("[ComfyUI-SAM3] [WARNING] Could not install CUDA compiler via package managers")
    print("[ComfyUI-SAM3] Manual installation required:")
    print("[ComfyUI-SAM3]   https://developer.nvidia.com/cuda-downloads")
    print("[ComfyUI-SAM3] Select your Linux distribution and follow the instructions")
    return False


def try_install_cuda_windows(cuda_version):
    """Try to install CUDA toolkit on Windows."""
    print("[ComfyUI-SAM3] Attempting Windows CUDA installation...")

    # Try chocolatey if available
    try:
        print("[ComfyUI-SAM3] Checking for Chocolatey package manager...")
        result = subprocess.run(
            ["choco", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print("[ComfyUI-SAM3] Chocolatey found, attempting installation...")
            result = subprocess.run(
                ["choco", "install", "cuda", "-y"],
                capture_output=True,
                text=True,
                timeout=600
            )
            if result.returncode == 0:
                print("[ComfyUI-SAM3] [OK] CUDA installed via Chocolatey")
                if setup_cuda_environment():
                    return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("[ComfyUI-SAM3] Chocolatey not available")

    # Windows requires manual installation - provide instructions
    print("[ComfyUI-SAM3] [WARNING] Automatic CUDA installation not available on Windows")
    print("[ComfyUI-SAM3] Manual installation required:")
    print("[ComfyUI-SAM3]")
    print("[ComfyUI-SAM3] 1. Visit: https://developer.nvidia.com/cuda-downloads")
    print("[ComfyUI-SAM3] 2. Select: Windows > x86_64 > your Windows version")
    print("[ComfyUI-SAM3] 3. Download and run the installer")
    print(f"[ComfyUI-SAM3] 4. Make sure to install CUDA {cuda_version} to match your PyTorch version")
    print("[ComfyUI-SAM3] 5. After installation, restart your terminal and run install.py again")
    return False


def try_install_cuda_mac():
    """Handle Mac CUDA installation (not supported)."""
    print("[ComfyUI-SAM3] [WARNING] CUDA is not supported on macOS")
    print("[ComfyUI-SAM3] Apple Silicon Macs do not support NVIDIA GPUs")
    print("[ComfyUI-SAM3] Intel Macs with NVIDIA GPUs are no longer officially supported by NVIDIA")
    print("[ComfyUI-SAM3]")
    print("[ComfyUI-SAM3] GPU-accelerated video tracking is not available on Mac")
    print("[ComfyUI-SAM3] CPU fallback will be used (slower but functional)")
    return False


def try_install_torch_generic_nms():
    """
    Attempt to install torch_generic_nms for GPU-accelerated video tracking.
    This provides 5-10x faster NMS compared to CPU fallback.
    """
    print("[ComfyUI-SAM3] Checking for torch_generic_nms (GPU-accelerated NMS for video)...")

    # Check if already installed
    try:
        import torch_generic_nms
        print("[ComfyUI-SAM3] [OK] torch_generic_nms already installed")
        return True
    except ImportError:
        pass

    # Check if CUDA is available in PyTorch
    try:
        import torch
        if not torch.cuda.is_available():
            print("[ComfyUI-SAM3] [INFO] CUDA not available, skipping GPU NMS compilation")
            print("[ComfyUI-SAM3] Video tracking will use CPU fallback (slower but functional)")
            return False
    except ImportError:
        print("[ComfyUI-SAM3] [WARNING] PyTorch not found, cannot install torch_generic_nms")
        return False

    # Setup CUDA environment (find CUDA from all possible sources)
    if not setup_cuda_environment():
        print("[ComfyUI-SAM3] [WARNING] CUDA compiler not found in system")
        print("[ComfyUI-SAM3] Attempting to install CUDA toolkit...")

        if not try_install_cuda_toolkit():
            print("[ComfyUI-SAM3] [WARNING] Could not install CUDA compiler")
            print("[ComfyUI-SAM3] Video tracking will use CPU NMS fallback")
            print("[ComfyUI-SAM3] To install manually, see: https://developer.nvidia.com/cuda-downloads")
            return False

    # Get CUDA architecture list
    cuda_arch_list = get_cuda_arch_list()

    if cuda_arch_list:
        os.environ["TORCH_CUDA_ARCH_LIST"] = cuda_arch_list
        print(f"[ComfyUI-SAM3] Set TORCH_CUDA_ARCH_LIST={cuda_arch_list}")

    # Install torch_generic_nms from GitHub
    print("[ComfyUI-SAM3] Compiling torch_generic_nms from source (may take 1-2 minutes)...")

    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                "git+https://github.com/ronghanghu/torch_generic_nms"
            ],
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode == 0:
            print("[ComfyUI-SAM3] [OK] torch_generic_nms compiled and installed successfully")
            print("[ComfyUI-SAM3] Video tracking will use GPU-accelerated NMS (5-10x faster)")
            return True
        else:
            print("[ComfyUI-SAM3] [WARNING] torch_generic_nms compilation failed")
            print("[ComfyUI-SAM3] This is optional - video tracking will use CPU NMS fallback")
            if result.stderr:
                print(f"[ComfyUI-SAM3] Error details: {result.stderr[:500]}")
            return False

    except subprocess.TimeoutExpired:
        print("[ComfyUI-SAM3] [WARNING] torch_generic_nms compilation timed out")
        print("[ComfyUI-SAM3] Video tracking will use CPU NMS fallback")
        return False
    except Exception as e:
        print(f"[ComfyUI-SAM3] [WARNING] torch_generic_nms installation error: {e}")
        print("[ComfyUI-SAM3] Continuing without GPU NMS (video tracking will still work)")
        return False


def try_install_cc_torch():
    """
    Attempt to install cc_torch for GPU-accelerated connected components.
    Used for hole filling in video tracking masks.
    """
    print("[ComfyUI-SAM3] Checking for cc_torch (GPU-accelerated connected components)...")

    # Check if already installed
    try:
        import cc_torch
        print("[ComfyUI-SAM3] [OK] cc_torch already installed")
        return True
    except ImportError:
        pass

    # Check if CUDA is available in PyTorch
    try:
        import torch
        if not torch.cuda.is_available():
            print("[ComfyUI-SAM3] [INFO] CUDA not available, skipping GPU connected components")
            print("[ComfyUI-SAM3] Video tracking will use CPU fallback (slower but functional)")
            return False
    except ImportError:
        print("[ComfyUI-SAM3] [WARNING] PyTorch not found, cannot install cc_torch")
        return False

    # Setup CUDA environment (find CUDA from all possible sources)
    if not setup_cuda_environment():
        print("[ComfyUI-SAM3] [WARNING] CUDA compiler not found in system")
        print("[ComfyUI-SAM3] Attempting to install CUDA toolkit...")

        if not try_install_cuda_toolkit():
            print("[ComfyUI-SAM3] [WARNING] Could not install CUDA compiler")
            print("[ComfyUI-SAM3] Video tracking will use CPU connected components fallback")
            return False

    # Get CUDA architecture list
    cuda_arch_list = get_cuda_arch_list()

    if cuda_arch_list:
        os.environ["TORCH_CUDA_ARCH_LIST"] = cuda_arch_list
        print(f"[ComfyUI-SAM3] Set TORCH_CUDA_ARCH_LIST={cuda_arch_list}")

    # Install cc_torch from GitHub
    print("[ComfyUI-SAM3] Compiling cc_torch from source (may take 1-2 minutes)...")

    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                "git+https://github.com/ronghanghu/cc_torch.git"
            ],
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode == 0:
            print("[ComfyUI-SAM3] [OK] cc_torch compiled and installed successfully")
            print("[ComfyUI-SAM3] Video tracking will use GPU-accelerated connected components")
            return True
        else:
            print("[ComfyUI-SAM3] [WARNING] cc_torch compilation failed")
            print("[ComfyUI-SAM3] This is optional - video tracking will use CPU fallback")
            if result.stderr:
                print(f"[ComfyUI-SAM3] Error details: {result.stderr[:500]}")
            return False

    except subprocess.TimeoutExpired:
        print("[ComfyUI-SAM3] [WARNING] cc_torch compilation timed out")
        print("[ComfyUI-SAM3] Video tracking will use CPU fallback")
        return False
    except Exception as e:
        print(f"[ComfyUI-SAM3] [WARNING] cc_torch installation error: {e}")
        print("[ComfyUI-SAM3] Continuing without GPU connected components")
        return False


if __name__ == "__main__":
    print("[ComfyUI-SAM3] Running installation script...")
    print("="*80)

    # Install requirements.txt first
    install_requirements()

    print("="*80)

    # Then try to install GPU-accelerated dependencies
    try_install_torch_generic_nms()

    print("="*80)

    try_install_cc_torch()

    print("="*80)
    print("[ComfyUI-SAM3] Installation script completed")
