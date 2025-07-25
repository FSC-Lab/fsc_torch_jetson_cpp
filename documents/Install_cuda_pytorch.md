# CUDA and PyTorch Installation Guide

This guide provides step-by-step instructions for installing CUDA and PyTorch for the DataDriven_Quadrotor project.

## Prerequisites

- NVIDIA GPU with compute capability 3.5 or higher
- Linux/Windows operating system
- Administrator/root privileges for installation

## Step 1: Install Python 3.9+

PyTorch 2.0 requires Python 3.9 or higher. We recommend using Python 3.9 for optimal compatibility.

### Verify Python Installation
```bash
python3 --version
```

If Python 3.9+ is not installed, download it from [python.org](https://www.python.org/downloads/) or use your system's package manager.

## Step 2: Install CUDA 12.8

### Download and Install
1. Download CUDA 12.8 from the [official NVIDIA archive](https://developer.nvidia.com/cuda-12-8-0-download-archive)
2. Follow the installation instructions for your operating system
3. Choose the appropriate installer (runfile, deb, rpm, or exe)

### Verify Installation
After installation, verify CUDA is properly installed:
```bash
nvcc --version
```

### Configure Environment Variables (if needed)
If the `nvcc --version` command fails, manually add CUDA to your system path:

```bash
# Add to ~/.bashrc or ~/.zshrc
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

# Reload your shell configuration
source ~/.bashrc
```

### Test CUDA Installation
```bash
# Check CUDA compiler
nvcc --version

# Check NVIDIA driver
nvidia-smi
```

## Step 3: Install PyTorch with CUDA Support

Install PyTorch with CUDA 12.8 support using the official PyTorch installation command:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Verify PyTorch Installation
Test that PyTorch can access your GPU:

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

## Additional Resources

- [PyTorch Official Installation Guide](https://pytorch.org/get-started/locally/)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [NVIDIA Driver Downloads](https://www.nvidia.com/drivers/)

