#!/usr/bin/env python3
"""
Common shared functions for GPU acceleration setup in ComfyUI-SAM3.

This module contains shared code used by both speedup.py and speedup_blackwell.py
to avoid duplication and ensure consistent behavior.
"""
import os
import subprocess
import sys


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
        "/usr/local/cuda-12.8",
        "/usr/local/cuda-12.7",
        "/usr/local/cuda-12.6",
        "/usr/local/cuda-12.5",
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


def check_msvc_available():
    """
    Check if MSVC compiler (cl.exe) is available on Windows.
    Returns True if found or not on Windows, False if missing on Windows.
    """
    if sys.platform != "win32":
        return True  # Not needed on non-Windows

    import shutil

    # Check if cl.exe is in PATH
    if shutil.which("cl"):
        print("[ComfyUI-SAM3] [OK] MSVC compiler (cl.exe) found in PATH")
        return True

    # Check common Visual Studio installation paths
    vs_paths = [
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC",
    ]

    for vs_path in vs_paths:
        if os.path.exists(vs_path):
            print(f"[ComfyUI-SAM3] [OK] Visual Studio found at: {vs_path}")
            print("[ComfyUI-SAM3] [WARNING] cl.exe not in PATH - run from Developer Command Prompt")
            return True

    print("[ComfyUI-SAM3] [ERROR] MSVC compiler not found")
    print("[ComfyUI-SAM3]")
    print("[ComfyUI-SAM3] GPU acceleration requires Visual Studio Build Tools on Windows.")
    print("[ComfyUI-SAM3] Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
    print("[ComfyUI-SAM3] During installation, select 'Desktop development with C++'")
    print("[ComfyUI-SAM3]")
    print("[ComfyUI-SAM3] After installing, run speedup.py from 'Developer Command Prompt for VS'")
    return False


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


def install_extension(name, repo_url, cuda_arch_list, compile_only=False, timeout=600, verbose=False):
    """
    Unified function to install a CUDA extension from GitHub.

    Args:
        name: Extension name (e.g., "torch_generic_nms")
        repo_url: GitHub repository URL
        cuda_arch_list: TORCH_CUDA_ARCH_LIST string
        compile_only: Compile even without GPU present
        timeout: Compilation timeout in seconds
        verbose: Show real-time compilation output

    Returns:
        True if successful, False otherwise
    """
    print(f"[ComfyUI-SAM3] Checking for {name}...")

    # Check if already installed
    try:
        __import__(name)
        print(f"[ComfyUI-SAM3] [OK] {name} already installed")
        return True
    except ImportError:
        pass

    # Check if CUDA is available in PyTorch
    try:
        import torch
        if not compile_only and not torch.cuda.is_available():
            print(f"[ComfyUI-SAM3] [INFO] CUDA not available, skipping {name} compilation")
            return False
        elif compile_only:
            print(f"[ComfyUI-SAM3] [INFO] Compile-only mode: checking PyTorch CUDA support...")
            if not torch.version.cuda:
                print(f"[ComfyUI-SAM3] [ERROR] PyTorch was not built with CUDA support")
                return False
            print(f"[ComfyUI-SAM3] [OK] PyTorch built with CUDA {torch.version.cuda}")
    except ImportError:
        print(f"[ComfyUI-SAM3] [WARNING] PyTorch not found, cannot install {name}")
        return False

    # Setup CUDA environment
    cuda_env = setup_cuda_environment()
    if not cuda_env:
        print(f"[ComfyUI-SAM3] [WARNING] CUDA compiler not found in system")
        print(f"[ComfyUI-SAM3] Attempting to install CUDA toolkit...")

        if not try_install_cuda_toolkit():
            print(f"[ComfyUI-SAM3] [WARNING] Could not install CUDA compiler")
            return False

        cuda_env = setup_cuda_environment()
        if not cuda_env:
            print(f"[ComfyUI-SAM3] [WARNING] CUDA environment setup failed after toolkit installation")
            return False

    # Set CUDA architecture list
    if cuda_arch_list:
        cuda_env["TORCH_CUDA_ARCH_LIST"] = cuda_arch_list
        os.environ["TORCH_CUDA_ARCH_LIST"] = cuda_arch_list
        print(f"[ComfyUI-SAM3] Set TORCH_CUDA_ARCH_LIST={cuda_arch_list}")

    # Set C/C++ compiler paths explicitly
    import shutil

    if sys.platform == "win32":
        # Windows: Check MSVC availability and set flags
        if not check_msvc_available():
            return False

        # MSVC flags to fix C2872 'std' ambiguous symbol error (PyTorch 2.8+)
        # /permissive- enables stricter C++ conformance
        # /Zc:__cplusplus reports correct C++ standard version
        msvc_flags = "/permissive- /Zc:__cplusplus"
        existing_cflags = cuda_env.get("CFLAGS", "")
        existing_cxxflags = cuda_env.get("CXXFLAGS", "")
        cuda_env["CFLAGS"] = f"{existing_cflags} {msvc_flags}".strip()
        cuda_env["CXXFLAGS"] = f"{existing_cxxflags} {msvc_flags}".strip()
        os.environ["CFLAGS"] = cuda_env["CFLAGS"]
        os.environ["CXXFLAGS"] = cuda_env["CXXFLAGS"]
        print(f"[ComfyUI-SAM3] Set MSVC flags: {msvc_flags}")
    else:
        # Linux/Mac: Use GCC
        gcc_path = shutil.which("gcc-11") or shutil.which("gcc")
        gxx_path = shutil.which("g++-11") or shutil.which("g++")
        if gcc_path:
            cuda_env["CC"] = gcc_path
            os.environ["CC"] = gcc_path
            print(f"[ComfyUI-SAM3] Set CC={gcc_path}")
        if gxx_path:
            cuda_env["CXX"] = gxx_path
            os.environ["CXX"] = gxx_path
            print(f"[ComfyUI-SAM3] Set CXX={gxx_path}")

    # Install extension from GitHub
    print(f"[ComfyUI-SAM3] Compiling {name} from source (estimated time: ~60 seconds per architecture)...")

    pip_args = [
        sys.executable, "-m", "pip", "install",
        "--no-build-isolation",
    ]

    if verbose:
        pip_args.append("--verbose")

    pip_args.append(repo_url)

    try:
        result = subprocess.run(
            pip_args,
            capture_output=not verbose,
            text=True,
            timeout=timeout,
            env=cuda_env
        )

        if result.returncode == 0:
            print(f"[ComfyUI-SAM3] [OK] {name} compiled and installed successfully")
            return True
        else:
            print(f"[ComfyUI-SAM3] [WARNING] {name} compilation failed")
            if result.stderr and not verbose:
                print(f"[ComfyUI-SAM3] Error details: {result.stderr[:500]}")
            return False

    except subprocess.TimeoutExpired:
        print(f"[ComfyUI-SAM3] [WARNING] {name} compilation timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"[ComfyUI-SAM3] [WARNING] {name} installation error: {e}")
        return False
