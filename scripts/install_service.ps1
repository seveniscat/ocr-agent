# Register and start the ocr-agent uvicorn service via NSSM.
# Must be run as Administrator (UAC).

$ErrorActionPreference = 'Stop'

$NSSM     = 'C:\Users\zhangbing\AppData\Local\Microsoft\WinGet\Packages\NSSM.NSSM_Microsoft.Winget.Source_8wekyb3d8bbwe\nssm-2.24-101-g897c7ad\win64\nssm.exe'
$SERVICE  = 'ocr-agent'
$APPDIR   = 'D:\bzdev\ocr-agent'
$PY       = "$APPDIR\.venv\Scripts\python.exe"
$LOGDIR   = "$APPDIR\logs"

# Sanity checks
if (-not (Test-Path $NSSM))  { Write-Error "nssm.exe not found at $NSSM"; exit 1 }
if (-not (Test-Path $PY))    { Write-Error "python.exe not found at $PY"; exit 1 }
if (-not (Test-Path $APPDIR)) { Write-Error "app dir not found at $APPDIR"; exit 1 }

# Create log dir
if (-not (Test-Path $LOGDIR)) { New-Item -ItemType Directory -Path $LOGDIR | Out-Null }

# Remove any pre-existing service with the same name (idempotent).
$existing = Get-Service -Name $SERVICE -ErrorAction SilentlyContinue
if ($existing) {
    Write-Output "Service $SERVICE exists — removing the old one first."
    if ($existing.Status -eq 'Running') {
        & $NSSM stop $SERVICE 2>$null | Out-Null
        Start-Sleep -Seconds 2
    }
    & $NSSM remove $SERVICE confirm
    Write-Output "Old service removed."
}

# Register the new service.
Write-Output "Installing service $SERVICE ..."
& $NSSM install $SERVICE $PY '-m uvicorn app.main:app --host 0.0.0.0 --port 48763'
Write-Output "Installed."

# Configure the service.
& $NSSM set $SERVICE AppDirectory       $APPDIR
& $NSSM set $SERVICE AppEnvironmentExtra 'PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True'
& $NSSM set $SERVICE DisplayName        'ocr-agent (uvicorn :48763)'
& $NSSM set $SERVICE Description        'High-resolution OCR + polygon bbox engine (PaddleOCR 3.7.0 / PP-OCRv6). HTTP on port 48763.'
& $NSSM set $SERVICE Start              SERVICE_AUTO_START   # boot auto-start
& $NSSM set $SERVICE AppStdout          "$LOGDIR\stdout.log"
& $NSSM set $SERVICE AppStderr          "$LOGDIR\stderr.log"
& $NSSM set $SERVICE AppRotateFiles     1
& $NSSM set $SERVICE AppRotateOnline    1
& $NSSM set $SERVICE AppRotateBytes     10485760   # rotate at 10 MB

# Crash auto-restart (restart after 5s delay, no matter the exit code).
& $NSSM set $SERVICE AppExit Default Restart
& $NSSM set $SERVICE AppRestartDelay    5000

# Run as LocalSystem (default). LocalSystem has no network cred issues for
# outbound HTTP and can read D:\bzdev. Change if you need a specific account.
Write-Output "Configuration done. Starting service..."

& $NSSM start $SERVICE
Start-Sleep -Seconds 8

$s = Get-Service -Name $SERVICE -ErrorAction SilentlyContinue
Write-Output ("SERVICE_STATUS: " + $s.Status)
Write-Output ("SERVICE_STARTTYPE: " + $s.StartType)
