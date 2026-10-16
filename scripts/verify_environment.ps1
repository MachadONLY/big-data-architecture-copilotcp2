$ErrorActionPreference = "Continue"

Write-Host "=== Windows / PowerShell ==="
Get-ComputerInfo | Select-Object WindowsProductName, WindowsVersion, OsArchitecture, CsTotalPhysicalMemory

Write-Host "`n=== NVIDIA DRIVER / GPU ==="
nvidia-smi

Write-Host "`n=== CUDA TOOLKIT (if installed) ==="
nvcc --version

Write-Host "`n=== PYTHON ==="
python --version
python -c "import sys; print(sys.executable); print(sys.version)"

Write-Host "`n=== PIP ==="
python -m pip --version

Write-Host "`n=== GIT ==="
git --version

Write-Host "`n=== PYTORCH CUDA CHECK ==="
python src/gpu_check.py --require-cuda --output assets/screenshots/gpu_environment.json
