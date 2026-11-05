# Native Windows MobileGym preview server (serial GUIAGENT runner).
param([switch]$Rebuild)
$ErrorActionPreference="Stop"
if ($env:OS -ne "Windows_NT") { throw "Native Windows PowerShell required." }
$Root=(Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Mobile=Join-Path $Root "external\mobilegym"
if (-not (Test-Path (Join-Path $Mobile "package.json"))) {
  throw "Run scripts/windows/setup_windows.ps1 -Benchmark MobileGym first."
}
Push-Location $Mobile
try {
  if ($Rebuild -or -not (Test-Path (Join-Path $Mobile "dist\index.html"))) {
    npm run build
    if ($LASTEXITCODE -ne 0) { throw "MobileGym build failed." }
  }
  Write-Host "MobileGym preview: http://127.0.0.1:4173"
  npm run preview -- --host 127.0.0.1 --port 4173
  if ($LASTEXITCODE -ne 0) { throw "MobileGym preview terminated with an error." }
} finally { Pop-Location }
