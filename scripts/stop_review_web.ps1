$ErrorActionPreference = 'Stop'
$reviewRoot = Split-Path -Parent $PSScriptRoot
$reviewStateFile = Join-Path $reviewRoot 'data\review_web\server.json'
if (-not (Test-Path -LiteralPath $reviewStateFile)) { Write-Output 'No recorded background service.'; exit }
$reviewState = Get-Content -LiteralPath $reviewStateFile -Raw | ConvertFrom-Json
$reviewEntry = Join-Path $PSScriptRoot 'serve_review_web.py'
$reviewProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $([int]$reviewState.pid)"
if (-not $reviewProcess -or -not $reviewProcess.CommandLine.Contains($reviewEntry)) { Write-Output 'The recorded service is no longer running.'; exit }
$reviewChildEntry = Join-Path $PSScriptRoot 'analyze_single_review_emotions.py'
Get-CimInstance Win32_Process -Filter "ParentProcessId = $([int]$reviewState.pid)" | Where-Object { $_.CommandLine -and $_.CommandLine.Contains($reviewChildEntry) } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Stop-Process -Id ([int]$reviewState.pid) -Force
Write-Output 'Review analysis service stopped.'
