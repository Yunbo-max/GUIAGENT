# Native Windows: provision/start an isolated second Pixel6 AndroidWorld AVD.
param([switch]$InstallSecondAVD,[switch]$Headless)
$ErrorActionPreference="Stop"
if($env:OS -ne "Windows_NT"){throw "Native Windows required"}
$Sdk=if($env:ANDROID_SDK_ROOT){$env:ANDROID_SDK_ROOT} elseif($env:ANDROID_HOME){$env:ANDROID_HOME} else {Join-Path $env:LOCALAPPDATA "Android\Sdk"}
$emulator=Join-Path $Sdk "emulator\emulator.exe"
$manager=Join-Path $Sdk "cmdline-tools\latest\bin\avdmanager.bat"
$adb=Join-Path $Sdk "platform-tools\adb.exe"
if(!(Test-Path $emulator) -or !(Test-Path $adb)){throw "Install Android Studio SDK/emulator/platform-tools"}
$avds=@(& $emulator -list-avds)
if($InstallSecondAVD -and $avds -notcontains "AndroidWorldAvd2"){
  if(!(Test-Path $manager)){throw "Install Android SDK Command-line Tools (latest)"}
  "no" | & $manager create avd -n AndroidWorldAvd2 -k "system-images;android-33;google_apis;x86_64" -d pixel_6
  if($LASTEXITCODE -ne 0){throw "AVD creation failed"}
  $avds=@(& $emulator -list-avds)
}
foreach($name in @("AndroidWorldAvd","AndroidWorldAvd2")){
  if($avds -notcontains $name){throw "Missing $name. Create Pixel 6 API33 AVD first."}
}
& $emulator -accel-check
if($LASTEXITCODE -ne 0){throw "WHPX acceleration unavailable"}
$existing=@(& $adb devices)
$slots=@(
  @{name="AndroidWorldAvd";port=5554;grpc=8554;serial="emulator-5554"},
  @{name="AndroidWorldAvd2";port=5556;grpc=8556;serial="emulator-5556"}
)
foreach($slot in $slots){
  if($existing -match ([regex]::Escape($slot.serial)+"\s+device")){
    Write-Host "$($slot.serial) already running - NOT taking over its tasks." -ForegroundColor Yellow
    continue
  }
  $arguments=@("-avd",$slot.name,"-no-snapshot","-port",[string]$slot.port,
    "-grpc",[string]$slot.grpc)
  if($Headless){$arguments+=@("-no-window","-gpu","swiftshader_indirect","-no-audio")}
  $p=Start-Process -FilePath $emulator -ArgumentList $arguments -PassThru
  Write-Host "Started $($slot.name) PID=$($p.Id) adb=$($slot.serial) grpc=$($slot.grpc)"
}
Write-Host "Check device readiness using adb devices. Do not start a second research collector against an AVD already owned by the current run." -ForegroundColor Yellow
