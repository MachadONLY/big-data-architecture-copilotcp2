param(
    [string]$ModelOverride = ""
)
$ErrorActionPreference = "Stop"
$configs = @("configs/experiment_01.yaml", "configs/experiment_02.yaml", "configs/experiment_03.yaml")
New-Item -ItemType Directory -Force -Path "experiments/logs" | Out-Null
foreach ($config in $configs) {
    $name = [System.IO.Path]::GetFileNameWithoutExtension($config)
    Write-Host "`n=== RUNNING $name ==="
    Write-Host "Keep second VSCode terminal open with: nvidia-smi -l 2"
    if ($ModelOverride) {
        & python src/train.py --config $config --model-override $ModelOverride
    } else {
        & python src/train.py --config $config
    }
    if ($LASTEXITCODE -ne 0) { throw "$name FAILED with exit code $LASTEXITCODE" }
}
Write-Host "All experiments finished. Check experiments/results.csv"
