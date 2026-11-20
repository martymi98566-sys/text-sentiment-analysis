param([int]$Port = 8765, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$reviewRoot = Split-Path -Parent $PSScriptRoot
$reviewPython = Join-Path $reviewRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $reviewPython)) {
    $reviewLegacyPython = Join-Path (Split-Path -Parent $reviewRoot) 'text-sentiment-analysis-local-env\Scripts\python.exe'
    if (Test-Path -LiteralPath $reviewLegacyPython) { $reviewPython = $reviewLegacyPython }
    else { $reviewPython = (Get-Command python -ErrorAction Stop).Source }
}
$reviewEntry = Join-Path $PSScriptRoot 'serve_review_web.py'
$reviewRuntime = Join-Path $reviewRoot 'data\review_web'
$reviewUrl = "http://127.0.0.1:$Port"
if (-not (Test-Path -LiteralPath $reviewPython)) { throw "Python environment not found: $reviewPython" }
try { $reviewHealth = Invoke-RestMethod "$reviewUrl/api/health" -TimeoutSec 2 } catch { $reviewHealth = $null }
if ($reviewHealth.application -ne 'movie-review-local') {
    New-Item -ItemType Directory -Path $reviewRuntime -Force | Out-Null
    $env:PYTHONIOENCODING = 'utf-8'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $reviewProcess = Start-Process -FilePath $reviewPython -ArgumentList @('-B','-u',('"' + $reviewEntry + '"'),'--port',$Port) -WorkingDirectory $reviewRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $reviewRuntime 'server.stdout.log') -RedirectStandardError (Join-Path $reviewRuntime 'server.stderr.log')
    $reviewStarted = $false
    for ($reviewAttempt = 0; $reviewAttempt -lt 30; $reviewAttempt++) {
        Start-Sleep -Milliseconds 500
        try { $reviewHealth = Invoke-RestMethod "$reviewUrl/api/health" -TimeoutSec 1; if ($reviewHealth.application -eq 'movie-review-local') { $reviewStarted = $true; break } } catch {}
        if ($reviewProcess.HasExited) { break }
    }
    if (-not $reviewStarted) { throw "Cannot start the service. Check data\review_web\server.stderr.log or use another port." }
    @{ pid = $reviewProcess.Id; port = $Port; entry = $reviewEntry } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $reviewRuntime 'server.json') -Encoding UTF8
}
Write-Output "Review analysis is ready: $reviewUrl"
if (-not $NoBrowser) { Start-Process $reviewUrl }
