param([switch]$RequireGpu)
$ErrorActionPreference = 'Stop'
Write-Host '=== QUICK PREFLIGHT ==='
& python -m compileall -q src app tests
if ($LASTEXITCODE -ne 0) { throw 'Python syntax compilation FAILED.' }
& python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'pytest FAILED.' }
& python src/qa_static.py
if ($LASTEXITCODE -ne 0) { throw 'Static QA FAILED.' }
& python src/validate_dataset.py
if ($LASTEXITCODE -ne 0) { throw 'Dataset QA FAILED.' }
if ($RequireGpu) {
    & python src/gpu_check.py --require-cuda --output assets/screenshots/gpu_environment.json
    if ($LASTEXITCODE -ne 0) { throw 'GPU CUDA check FAILED.' }
    & python src/verify_stack.py --require-cuda --output assets/screenshots/stack_environment.json
    if ($LASTEXITCODE -ne 0) { throw 'Pinned QLoRA stack or bitsandbytes NF4 GPU smoke FAILED.' }
} else {
    & python src/verify_stack.py
    if ($LASTEXITCODE -ne 0) { throw 'Pinned QLoRA stack API check FAILED.' }
}
Write-Host 'PREFLIGHT: PASS (static checks only unless -RequireGpu supplied)'
