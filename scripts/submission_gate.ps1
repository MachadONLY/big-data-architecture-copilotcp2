param([string[]]$Member = @())
$ErrorActionPreference = "Stop"
Write-Host "=== FINAL SUBMISSION GATE ==="
$cmd = @('src/finalize_submission.py')
foreach ($m in $Member) { $cmd += @('--member', $m) }
& python @cmd
if ($LASTEXITCODE -ne 0) { throw "Submission gate FAILED (experimental evidence or team data missing). See FINAL_REPORT.md." }
& python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Tests FAILED. Do not deliver." }
& python src/validate_dataset.py
if ($LASTEXITCODE -ne 0) { throw "Dataset QA FAILED. Do not deliver." }
& python src/gpu_check.py --require-cuda
if ($LASTEXITCODE -ne 0) { throw "GPU CUDA validation FAILED. Do not deliver." }
& python src/verify_stack.py --require-cuda
if ($LASTEXITCODE -ne 0) { throw "QLoRA stack / bitsandbytes NF4 validation FAILED. Do not deliver." }
Write-Host "SUBMISSION GATE: PASS"
