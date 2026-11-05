# Native Windows PowerShell 5.1+ installer for GUIAGENT / NVIDIA 96GB class.
# Does not install WSL, Linux, Docker, vLLM or system CUDA Toolkit.
# -InstallSystemTools and -SetupAndroidSDK are explicit OS modification switches.
param(
  [ValidateSet("Both","AndroidWorld","MobileGym")]
  [string]$Benchmark = "Both",
  [switch]$InstallSystemTools,
  [switch]$SetupAndroidSDK,
  [switch]$DownloadModel,
  [switch]$DownloadMobileGymData
)
$ErrorActionPreference = "Stop"
if ($env:OS -ne "Windows_NT") { throw "Run from native Windows PowerShell, not WSL." }
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
$Python = Join-Path $Root ".venv-guiagent\Scripts\python.exe"
$WantAndroid = $Benchmark -in @("Both","AndroidWorld")
$WantMobile = $Benchmark -in @("Both","MobileGym")
function Run([string]$Label,[scriptblock]$Action) {
  Write-Host "==> $Label" -ForegroundColor Cyan
  $global:LASTEXITCODE = 0
  & $Action
  if ($LASTEXITCODE -ne 0) { throw "$Label failed with exit code $LASTEXITCODE" }
}
function Require([string]$Name,[string]$Message) {
  if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) { throw $Message }
}
if ($InstallSystemTools) {
  Require "winget" "Install Windows Package Manager (winget) first."
  $packages = @("Git.Git","Python.Python.3.11","Gyan.FFmpeg")
  if ($WantMobile) { $packages += "OpenJS.NodeJS.LTS" }
  if ($WantAndroid) { $packages += "Google.AndroidStudio" }
  foreach ($id in $packages) {
    Run "winget $id" { winget install --id $id --exact --accept-source-agreements --accept-package-agreements }
  }
  Write-Host "System packages installed. Close/reopen PowerShell and rerun without -InstallSystemTools." -ForegroundColor Yellow
  return
}
Require "git" "Install Git for Windows: winget install --id Git.Git --exact"
Require "py" "Install 64-bit Python 3.11: winget install --id Python.Python.3.11 --exact"
Run "Python 3.11" { py -3.11 --version }
if (-not (Test-Path $Python)) {
  Run "Create shared Python 3.11 virtualenv" {
    py -3.11 -m venv (Join-Path $Root ".venv-guiagent")
  }
}
Run "pip tooling" { & $Python -m pip install --upgrade pip setuptools wheel }
# Uses official 2.8.0 CUDA 12.8 Windows wheels; bundled CUDA runtime is sufficient.
# Windows NVIDIA graphics driver is mandatory; a separate CUDA Toolkit is not.
Run "PyTorch 2.8 CUDA 12.8" {
  & $Python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
}
Run "Qwen vision / AndroidWorld training requirements" {
  & $Python -m pip install -r (Join-Path $Root "requirements\android.txt")
}
Run "GUIAGENT editable package without legacy text-only dependencies" {
  & $Python -m pip install --no-deps -e $Root
}
Run "GPU BF16 verification" {
  & $Python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; assert torch.cuda.is_bf16_supported(), 'BF16 unsupported'; print(torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory/(1024**3),1), 'GiB', torch.version.cuda)"
}
if ($WantAndroid) {
  # Pinned upstream via GUIAGENT existing fetch; model weight not fetched here.
  Run "Pinned AndroidWorld / UI-Voyager source" {
    & $Python scripts/run_androidworld_feedback.py --config configs/android/androidworld_feedback_learned_windows_96gb.json fetch
  }
}
if ($WantMobile) {
  Require "node" "Install Node.js 22+: winget install --id OpenJS.NodeJS.LTS --exact"
  Require "npm" "Install npm with Node.js >=22"
  $nodeVersion = (& node --version).Trim().TrimStart("v")
  if ([int]$nodeVersion.Split(".")[0] -lt 22) { throw "MobileGym requires Node 22+" }
  $Mobile = Join-Path $Root "external\mobilegym"
  if (-not (Test-Path $Mobile)) {
    New-Item -ItemType Directory -Path (Join-Path $Root "external") -Force | Out-Null
    Run "Clone MobileGym" { git clone https://github.com/Purewhiter/mobilegym.git $Mobile }
  }
  Run "Checkout frozen MobileGym" {
    git -C $Mobile checkout --detach 5fbd3f34be0e9780ae4b1e9d5089963ddbd05d38
  }
  Run "MobileGym Python dependencies" {
    & $Python -m pip install -r (Join-Path $Mobile "bench_env\requirements.txt")
  }
  Run "Playwright Chromium on Windows" { & $Python -m playwright install chromium }
  Push-Location $Mobile
  try {
    $env:PUPPETEER_SKIP_DOWNLOAD = "1"
    if (Test-Path (Join-Path $Mobile "package-lock.json")) {
      Run "MobileGym npm ci" { & npm.cmd ci }
    } else {
      Run "MobileGym npm install" { & npm.cmd install }
    }
    Run "Build MobileGym simulator" { & npm.cmd run build }
    if ($DownloadMobileGymData) {
      $Archive = Join-Path $Mobile "mobilegym-data-v0.1.0.tar.gz"
      if (-not (Test-Path $Archive)) {
        Write-Host "Downloading ~1.9GB official companion data..." -ForegroundColor Yellow
        Invoke-WebRequest -Uri "https://github.com/Purewhiter/mobilegym/releases/download/data-v0.1.0/mobilegym-data-v0.1.0.tar.gz" -OutFile $Archive
      }
      Run "Extract MobileGym companion data" { tar.exe -xzf $Archive -C $Mobile }
    } else {
      Write-Host "Companion data not downloaded: use -DownloadMobileGymData." -ForegroundColor Yellow
    }
  } finally { Pop-Location }
}
if ($DownloadModel) {
  Run "Download pinned Qwen3-VL-4B once for both benchmarks" {
    & $Python scripts/windows/download_qwen3vl.py --output models/qwen3-vl-4b-instruct
  }
} else {
  Write-Host "No weights downloaded: use -DownloadModel to fetch pinned Qwen3-VL-4B." -ForegroundColor Yellow
}
if ($SetupAndroidSDK -and $WantAndroid) {
  $Sdk = if ($env:ANDROID_SDK_ROOT) { $env:ANDROID_SDK_ROOT } elseif ($env:ANDROID_HOME) {
    $env:ANDROID_HOME
  } else { Join-Path $env:LOCALAPPDATA "Android\Sdk" }
  $SdkManager = Join-Path $Sdk "cmdline-tools\latest\bin\sdkmanager.bat"
  $AvdManager = Join-Path $Sdk "cmdline-tools\latest\bin\avdmanager.bat"
  $Emulator = Join-Path $Sdk "emulator\emulator.exe"
  if (-not (Test-Path $SdkManager) -or -not (Test-Path $AvdManager)) {
    throw "Android Studio SDK Manager: install Android SDK Command-line Tools (latest) first, then rerun -SetupAndroidSDK."
  }
  $env:ANDROID_SDK_ROOT = $Sdk
  $env:ANDROID_HOME = $Sdk
  if (-not $env:JAVA_HOME) {
    $jbr = Join-Path $env:ProgramFiles "Android\Android Studio\jbr"
    if (Test-Path $jbr) { $env:JAVA_HOME = $jbr }
  }
  Write-Host "SDK licenses may ask for interactive acceptance." -ForegroundColor Yellow
  Run "Accept Android SDK licenses" { & $SdkManager "--sdk_root=$Sdk" --licenses }
  Run "Install native Android API33 SDK, Google APIs image, adb, emulator" {
    & $SdkManager "--sdk_root=$Sdk" "platform-tools" "emulator" "platforms;android-33" "system-images;android-33;google_apis;x86_64"
  }
  $known = @(& $Emulator -list-avds)
  if ($known -notcontains "AndroidWorldAvd") {
    Run "Create Pixel 6 / API33 native AVD" {
      "no" | & $AvdManager create avd -n AndroidWorldAvd -k "system-images;android-33;google_apis;x86_64" -d pixel_6
    }
  } else {
    Write-Host "AndroidWorldAvd exists; existing device preserved."
  }
}
Run "Package dependency consistency" { & $Python -m pip check }
Write-Host "Installation completed. Run native preflight; AndroidWorld also requires WHPX/virtualization." -ForegroundColor Green
Write-Host "& '$Python' scripts/windows/preflight_guiagent.py --benchmark $($Benchmark.ToLower())"
