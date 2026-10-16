param(
    [string]$ModelOverride = "",
    [string[]]$Member = @()
)
$ErrorActionPreference = "Stop"
function RunPython([string[]] $Argv) {
    & python @Argv
    if ($LASTEXITCODE -ne 0) { throw "Python failed (exit code $LASTEXITCODE): $($Argv -join ' ')" }
}
Write-Host "=== 1/8 GPU CHECK ==="
RunPython @('src/gpu_check.py','--require-cuda','--output','assets/screenshots/gpu_environment.json')
Write-Host "=== 2/8 PINNED QLORA STACK + NF4 GPU SMOKE ==="
RunPython @('src/verify_stack.py','--require-cuda','--output','assets/screenshots/stack_environment.json')
Write-Host "=== 3/8 DATASET ==="
RunPython @('src/prepare_dataset.py')
RunPython @('src/validate_dataset.py')
Write-Host "=== 4/8 TESTS ==="
RunPython @('-m','pytest','-q')
Write-Host "=== 5/8 TRAINING EXPERIMENTS ==="
& .\scripts\run_experiments.ps1 -ModelOverride $ModelOverride
Write-Host "=== 6/8 SELECT ADAPTER BY VALIDATION LOSS ==="
RunPython @('src/select_adapter.py')
$adapter = (Get-Content 'experiments/best_adapter.txt' -Raw).Trim()
Write-Host "Selected adapter: $adapter"
Write-Host "=== 7/8 BASE VS FINE-TUNED TEST BENCHMARK ==="
RunPython @('src/evaluate.py','--adapter',$adapter)
Write-Host "=== 8/8 FREEZE ENVIRONMENT + CHARTS ==="
& python -m pip freeze | Out-File -Encoding utf8 environment.lock.txt
if ($LASTEXITCODE -ne 0) { throw 'pip freeze failed' }
$latestEval = Get-ChildItem 'experiments/evaluation' -Directory | Sort-Object Name -Descending | Select-Object -First 1
if ($latestEval) { RunPython @('src/plot_results.py','--evaluation-summary',(Join-Path $latestEval.FullName 'summary.json')) }
Write-Host "=== REPORT (not final until screenshot/team names) ==="
$reportArgs = @('src/finalize_submission.py','--allow-incomplete')
foreach ($m in $Member) { $reportArgs += @('--member', $m) }
RunPython $reportArgs
Write-Host "Pipeline finished. Do the human review, take the real GPU screenshot, check Streamlit, then package."
