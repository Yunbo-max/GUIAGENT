# Prepare a SEPARATE native Windows Git worktree for AndroidWorld optimization.
# No overwrites: only create data/models junctions to EXISTING assets. Junctions are writable.
# Keep original active MobileGym process/source tree completely untouched.
param([Parameter(Mandatory=$true)][string]$OriginalRepo)
$ErrorActionPreference="Stop"
if($env:OS -ne "Windows_NT"){throw "Native Windows required"}
$Root=(Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Original=(Resolve-Path $OriginalRepo).Path
if($Root -eq $Original){throw "Optimization must run from a separate worktree, not the original active checkout."}
if(!(Test-Path (Join-Path $Original "src\improving\mobilegym_feedback.py"))){
  throw "Original source checkout not found at $Original"
}
foreach($name in @("data","models")){
  $target=Join-Path $Original $name
  $newpath=Join-Path $Root $name
  if(!(Test-Path $target)){throw "Required original asset directory missing: $target"}
  if(Test-Path $newpath){throw "Refusing to replace existing path: $newpath"}
  New-Item -ItemType Junction -Path $newpath -Target $target | Out-Null
  Write-Host "Shared writable asset junction: $newpath -> $target" -ForegroundColor Cyan
}
$Py=Join-Path $Original ".venv-guiagent\Scripts\python.exe"
if(!(Test-Path $Py)){throw "Original Python 3.11 environment missing: $Py"}
Write-Host "Asset links ready. No active files were modified by setup." -ForegroundColor Green
Write-Host "Junctions are NOT read-only! Do not rerun prepare/download or change model files from the new worktree." -ForegroundColor Yellow
Write-Host "Use this venv interpreter for new-worktree commands: & '$Py' ..."
Write-Host "Do not run AndroidWorld on an AVD already controlled by the original collector." -ForegroundColor Yellow
