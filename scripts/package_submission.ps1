param([string[]]$Member = @())
$ErrorActionPreference = "Stop"
& .\scripts\submission_gate.ps1 -Member $Member
& python src/package_submission.py
if ($LASTEXITCODE -ne 0) { throw 'Submission packaging FAILED' }
Write-Host 'Attach BOTH ZIPs from dist/ to Teams (code + adapter) and the GitHub URL.'
