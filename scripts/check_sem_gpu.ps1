$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Push-Location $projectRoot
try {
    python src/qa_static.py
    if ($LASTEXITCODE -ne 0) { throw 'QA estático falhou' }
    python src/validate_dataset.py
    if ($LASTEXITCODE -ne 0) { throw 'Validação do dataset falhou' }
    python src/offline_demo.py
    if ($LASTEXITCODE -ne 0) { throw 'Exemplo de interface falhou' }
    Write-Host 'QA OFFLINE PASS — esta verificação NÃO substitui CUDA/treinamento.'
} finally { Pop-Location }
