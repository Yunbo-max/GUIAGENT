# Native Windows launcher, pilot by default; explicit -Full starts full benchmark.
param(
  [ValidateSet("AndroidWorld","MobileGym")]
  [string]$Benchmark = "AndroidWorld",
  [ValidateSet("plan","fetch","prepare","setup-env","env-check","run","report")]
  [string]$Stage = "plan",
  [switch]$Full,
  [switch]$Samples16,
  [switch]$Resume
)
$ErrorActionPreference="Stop"
$Root=(Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
$Python=Join-Path $Root ".venv-guiagent\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Missing virtualenv; run scripts/windows/setup_windows.ps1 first." }
if ($Benchmark -eq "AndroidWorld") {
  if ($Samples16) {
    # The 16-sample experiment uses a new output root and 2 independent AVDs,
    # but deliberately reuses the original frozen train/eval manifest.
    if ($Stage -eq "prepare") {
      throw "The 16-sample AndroidWorld config reuses the frozen manifest. Prepare it with the original 4-sample config if it does not exist; NEVER regenerate an existing manifest."
    }
    $file=if ($Full) {"configs/android/androidworld_feedback_learned_windows_full_parallel2_samples16.json"} else {
      "configs/android/androidworld_feedback_learned_windows_pilot_parallel2_samples16.json"
    }
  } else {
    $file=if ($Full) {"configs/android/androidworld_feedback_learned_windows_96gb.json"} else {
      "configs/android/androidworld_feedback_learned_windows_pilot_96gb.json"
    }
  }
  $entry="scripts/run_androidworld_feedback.py"
} else {
  if ($Stage -notin @("plan","env-check","run")) {
    throw "MobileGym only supports plan, env-check or run in this launcher"
  }
  if ($Resume) { throw "MobileGym runner does not support --resume. Use a new experiment directory." }
  if ($Samples16) {
    $file=if ($Full) {"configs/mobilegym/feedback_only_qwen4b_parallel4_samples16.json"} else {
      "configs/mobilegym/feedback_only_qwen4b_parallel4_samples16_pilot.json"
    }
  } else {
    $file=if ($Full) {"configs/mobilegym/feedback_only_qwen4b_learned.json"} else {
      "configs/mobilegym/feedback_only_qwen4b_pilot.json"
    }
  }
  $entry="scripts/run_mobilegym_feedback.py"
}
Write-Host "$Benchmark / $Stage / $(if ($Full) {'FULL'} else {'PILOT'}) / $(if ($Samples16) {'Round0=16; Stage4=16'} else {'Original group size'})" -ForegroundColor Cyan
$argv=@($entry,"--config",$file,$Stage)
if ($Resume) { $argv+= "--resume" }
& $Python @argv
if ($LASTEXITCODE -ne 0) { throw "Benchmark stage failed (exit $LASTEXITCODE). Inspect partial receipts; do not infer zero reward." }
