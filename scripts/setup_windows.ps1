$ErrorActionPreference = "Stop"

Write-Host "Creating Python virtual environment..."
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

Write-Host ""
Write-Host "STOP HERE before installing dependencies."
Write-Host "Install the correct CUDA-enabled PyTorch build for YOUR NVIDIA driver using"
Write-Host "the official selector documented in docs/gpu.md."
Write-Host ""
Write-Host "After PyTorch is installed and 'torch.cuda.is_available()' is True, run:"
Write-Host "    python -m pip install -r requirements.txt"
Write-Host "    python -m pip install -r requirements-dev.txt"
Write-Host "    python src/gpu_check.py --require-cuda"
Write-Host "    python src/verify_stack.py --require-cuda"
