param(
    [string]$RepositoryUrl = "https://github.com/MachadONLY/big-data-architecture-copilot.git",
    [string]$CommitMessage = "Entrega FIAP CP2-CP3 — Big Data Architecture Copilot"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $repoRoot
try {
    & git --version
    if ($LASTEXITCODE -ne 0) { throw "Git não está disponível no terminal." }
    if (-not (Test-Path ".git")) {
        & git init
        if ($LASTEXITCODE -ne 0) { throw "git init falhou." }
    }

    # Identificação do commit informada pelo grupo. Alterável com git config.
    & git config user.name "Gabriel Teixeira Machado"
    & git config user.email "rm557@fia.com"

    & git add .
    if ($LASTEXITCODE -ne 0) { throw "git add falhou." }
    & git diff --cached --quiet
    if ($LASTEXITCODE -eq 1) {
        & git commit -m $CommitMessage
        if ($LASTEXITCODE -ne 0) { throw "git commit falhou." }
    } elseif ($LASTEXITCODE -ne 0) {
        throw "Não foi possível verificar as alterações staged."
    }

    & git branch -M main
    if ($LASTEXITCODE -ne 0) { throw "git branch falhou." }
    $remotes = @(& git remote)
    if ($remotes -contains "origin") {
        & git remote set-url origin $RepositoryUrl
    } else {
        & git remote add origin $RepositoryUrl
    }
    if ($LASTEXITCODE -ne 0) { throw "Não foi possível configurar o remoto." }

    & git push -u origin main
    if ($LASTEXITCODE -ne 0) { throw "Push falhou. Verifique se o repositório vazio existe e sua autenticação GitHub." }
    Write-Host "PUSH CONFIRMADO: $($RepositoryUrl -replace '\.git$', '')"
} finally {
    Pop-Location
}
