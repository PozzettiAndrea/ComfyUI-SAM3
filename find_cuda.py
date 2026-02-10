"""
Diagnostic script to find CUDA installations on the system.
Run this to help debug install.py CUDA detection.
"""
import os
import sys
import site
import subprocess
from pathlib import Path


print("="*80)
print("CUDA Installation Diagnostic")
print("="*80)

# Check environment variables
print("\n1. Environment Variables:")
print(f"   CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
print(f"   CUDA_PATH: {os.environ.get('CUDA_PATH', 'Not set')}")
print(f"   PATH: {os.environ.get('PATH', 'Not set')[:200]}...")

# Check for nvcc in PATH
print("\n2. Checking for nvcc in PATH:")
try:
    result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print(f"   [OK] nvcc found: {result.stdout.split('release')[1].split(',')[0].strip() if 'release' in result.stdout else 'version unknown'}")
    else:
        print("   [X] nvcc not found in PATH")
except FileNotFoundError:
    print("   [X] nvcc command not found")
except Exception as e:
    print(f"   [X] Error running nvcc: {e}")

# Check site-packages
print("\n3. Site-packages locations:")
sp_list = site.getsitepackages()
if isinstance(sp_list, str):
    sp_list = [sp_list]
for sp in sp_list:
    print(f"   - {sp}")

# Search for nvidia CUDA packages in site-packages
print("\n4. Searching for NVIDIA CUDA packages in site-packages:")
for sp in sp_list:
    sp_path = Path(sp)
    if sp_path.exists():
        # Look for nvidia directories
        nvidia_dir = sp_path / "nvidia"
        if nvidia_dir.exists():
            print(f"   Found nvidia directory: {nvidia_dir}")
            for item in nvidia_dir.iterdir():
                if item.is_dir() and 'cuda' in item.name.lower():
                    print(f"     - {item.name}/")
                    # Check for bin directory
                    bin_dir = item / "bin"
                    if bin_dir.exists():
                        print(f"       [OK] bin/ exists")
                        nvcc_path = bin_dir / "nvcc"
                        if nvcc_path.exists():
                            print(f"       [OK] nvcc found: {nvcc_path}")
                        elif (bin_dir / "nvcc.exe").exists():
                            print(f"       [OK] nvcc.exe found: {bin_dir / 'nvcc.exe'}")
                        else:
                            print(f"       [X] nvcc not found in bin/")
                            # List what's actually in bin/
                            if list(bin_dir.iterdir()):
                                print(f"       Files in bin/:")
                                for f in list(bin_dir.iterdir())[:10]:  # Show first 10 files
                                    print(f"         - {f.name}")
                    else:
                        print(f"       [X] bin/ not found")

# Check system CUDA installations
print("\n5. Checking system CUDA installations:")
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
            print(f"   [OK] Found: {cuda_path}")
        else:
            print(f"   ~ Exists but no nvcc: {cuda_path}")
    else:
        print(f"   [X] Not found: {cuda_path}")

# Check conda environment
print("\n6. Conda environment:")
if "CONDA_PREFIX" in os.environ:
    conda_prefix = os.environ["CONDA_PREFIX"]
    print(f"   Running in conda env: {conda_prefix}")
    nvcc_path = os.path.join(conda_prefix, "bin", "nvcc")
    if os.path.exists(nvcc_path):
        print(f"   [OK] nvcc found in conda: {nvcc_path}")
    else:
        print(f"   [X] nvcc not found in conda bin/")
else:
    print("   Not running in conda environment")

# Check PyTorch CUDA version
print("\n7. PyTorch CUDA info:")
try:
    import torch
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        major, minor = torch.cuda.get_device_capability(0)
        print(f"   Compute capability: {major}.{minor}")
except ImportError:
    print("   PyTorch not installed")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "="*80)
print("Diagnostic complete")
print("="*80)
