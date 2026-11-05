# Start the pinned Pixel6/API33 AndroidWorldAvd from native Windows.
# In emulator-specific research: use grpc 8554 and console port 5554.
param([switch]$Headless)
$ErrorActionPreference="Stop"
if ($env:OS -ne "Windows_NT") { throw "This script is for native Windows only." }
$Sdk=if ($env:ANDROID_SDK_ROOT) {$env:ANDROID_SDK_ROOT} elseif ($env:ANDROID_HOME) {$env:ANDROID_HOME} else {Join-Path $env:LOCALAPPDATA "Android\Sdk"}
$emulator=Join-Path $Sdk "emulator\emulator.exe"
$adb=Join-Path $Sdk "platform-tools\adb.exe"
if (-not (Test-Path $emulator) -or -not (Test-Path $adb)) { throw "Android SDK missing. Run setup_windows.ps1 -SetupAndroidSDK first." }
$available=@(& $emulator -list-avds)
if ($available -notcontains "AndroidWorldAvd") { throw "Pixel 6 API33 AVD named AndroidWorldAvd missing." }
& $emulator -accel-check
if ($LASTEXITCODE -ne 0) { throw "WHPX acceleration unavailable; Windows GPU VRAM cannot replace a CPU hypervisor." }
$argsList=@("-avd","AndroidWorldAvd","-no-snapshot","-port","5554","-grpc","8554")
if ($Headless) { $argsList+=@("-no-window","-gpu","swiftshader_indirect","-no-audio") }
$proc=Start-Process -FilePath $emulator -ArgumentList $argsList -PassThru
Write-Host "AndroidWorld native emulator started: PID $($proc.Id)."
Write-Host "Run: & '$adb' -s emulator-5554 wait-for-device"
Write-Host "Then: python scripts/run_androidworld_feedback.py --config configs/android/androidworld_feedback_learned_windows_pilot_96gb.json setup-env"
