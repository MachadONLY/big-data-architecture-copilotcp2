param(
    [Parameter(Mandatory=$true)]
    [string]$Adapter
)

$ErrorActionPreference = "Stop"
python src/evaluate.py --adapter $Adapter
