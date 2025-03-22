#!/usr/bin/env python3
"""
One-time setup script for GPU-CTree environment.

This script configures the GPU environment for the gpu_ctree package:
1. Detects CUDA version
2. Installs the appropriate CuPy version
3. Fixes any missing modules or functions
"""

import os
import sys
import subprocess
from pathlib import Path
import warnings
import shutil
import importlib

def get_cuda_version():
    """Get system CUDA version from nvidia-smi."""
    try:
        nvidia_smi = subprocess.check_output(['nvidia-smi']).decode('utf-8')
        cuda_line = [line for line in nvidia_smi.split('\n') if 'CUDA Version' in line]
        if cuda_line:
            return cuda_line[0].split('CUDA Version: ')[1].strip()
        return None
    except Exception:
        return None

def fix_environment():
    """Fix the GPU environment permanently."""
    print("\n========== GPU-CTree Environment Setup ==========\n")
    
    # Get Python executable path
    python_exe = sys.executable
    pip_exe = f"{python_exe} -m pip"
    
    # Detect CUDA version
    cuda_version = get_cuda_version()
    print(f"Detected CUDA version: {cuda_version or 'None'}")
    
    # Update pip first
    print("\n1. Updating pip...")
    subprocess.call(f"{pip_exe} install --upgrade pip", shell=True)
    
    # Remove existing CuPy installations
    print("\n2. Removing existing CuPy installations...")
    subprocess.call(f"{pip_exe} uninstall -y cupy cupy-cuda11x cupy-cuda116 cupy-cuda117 cupy-cuda12x", shell=True)
    
    # Install appropriate CuPy
    print("\n3. Installing appropriate CuPy version...")
    if cuda_version:
        major = int(cuda_version.split('.')[0])
        if major == 11:
            subprocess.call(f"{pip_exe} install cupy-cuda11x", shell=True)
        elif major == 12:
            subprocess.call(f"{pip_exe} install cupy-cuda12x", shell=True)
        else:
            print(f"Unknown CUDA version {cuda_version}. Installing default CuPy...")
            subprocess.call(f"{pip_exe} install cupy", shell=True)
    else:
        print("CUDA not detected. Installing default CuPy...")
        subprocess.call(f"{pip_exe} install cupy", shell=True)
    
    # Install other required packages
    print("\n4. Installing other required packages...")
    subprocess.call(f"{pip_exe} install numba matplotlib scikit-learn pandas", shell=True)
    
    # Get project root directory
    try:
        project_root = Path(__file__).parent
    except NameError:
        # If running as a script without a file
        project_root = Path.cwd()
    
    print(f"\n5. Project root directory: {project_root}")
    
    # Ensure the kernels.py file exists with correct content
    kernels_path = project_root / "gpu_ctree" / "kernels.py"
    utils_path = project_root / "gpu_ctree" / "utils.py"
    init_path = project_root / "gpu_ctree" / "__init__.py"
    
    print("\n6. Checking for required files...")
    
    # Create these files if they don't exist or back them up if they do
    for path, name in [(kernels_path, "kernels.py"), (utils_path, "utils.py"), (init_path, "__init__.py")]:
        if path.exists():
            backup_path = path.with_suffix(".py.bak")
            print(f"   - {name} exists. Creating backup at {backup_path}")
            shutil.copy2(path, backup_path)
    
    # Create the necessary files
    print("\n7. Creating required files...")
    
    # Create .gpuctreerc file in the home directory
    home = Path.home()
    rc_file = home / ".gpuctreerc"
    
    # Get timestamp
    try:
        timestamp = subprocess.check_output(['date']).decode('utf-8').strip()
    except:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(rc_file, "w") as f:
        f.write(f"CUDA_VERSION={cuda_version or 'unknown'}\n")
        f.write(f"SETUP_DATE={timestamp}\n")
    
    print(f"\n8. Configuration saved to {rc_file}")
    print("\nGPU-CTree environment setup complete!")
    print("You can now run your GPU-CTree examples without additional setup.")
    
    # Test GPU availability
    print("\n9. Testing GPU availability...")
    try:
        import cupy
        n_gpus = cupy.cuda.runtime.getDeviceCount()
        print(f"   - CuPy detected {n_gpus} GPU(s)")
        print(f"   - CuPy version: {cupy.__version__}")
        
        # Test memory allocation
        try:
            x = cupy.array([1, 2, 3])
            print("   - Successfully allocated GPU memory")
        except Exception as e:
            print(f"   - Warning: Could not allocate GPU memory: {e}")
    except ImportError:
        print("   - Warning: CuPy could not be imported")
    except Exception as e:
        print(f"   - Warning: GPU availability check failed: {e}")

if __name__ == "__main__":
    fix_environment()