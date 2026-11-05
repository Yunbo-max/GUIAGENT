# Pure Windows: AndroidWorld + MobileGym on NVIDIA 96GB

**Status:** Native Windows installation and benchmark integration code is
prepared. No Windows emulator, PyTorch GPU training, or full benchmark has
been run by this GitHub-editing session.

The repository now supports two independent verified GUI RL environments:
MobileGym (browser simulation) and AndroidWorld (Pixel 6 / Android 13
emulator). Both use pinned Qwen3-VL-4B-Instruct BF16 checkpoints and
feedback-only Learned SPECTRUM with an unspectrally modulated Feedback SD
control. **Do not pool their success rates.**

## 1. Machine requirements

- Windows 10/11 x64, Python 3.11 x64, NVIDIA driver supporting CUDA 12.8
  PyTorch wheels, a GPU with 96GB VRAM, preferably at least 64GB SYSTEM
  RAM, and 500GB free disk (1TB preferred for all checkpoints and images).
- Git, ffmpeg, and Node.js 22+ for MobileGym; Playwright Chromium and
  the companion MobileGym assets. AndroidWorld additionally needs Android
  Studio/SDK, Pixel 6 Android 13 Tiramisu API33 x86_64 and the AVD named
  AndroidWorldAvd.
- Windows Hypervisor Platform (WHPX) and CPU virtualization must work for
  AndroidWorld. 96GB NVIDIA VRAM does not substitute for nested
  virtualization in a Windows cloud VM. Check emulator -accel-check. If the
  cloud host blocks virtualization, MobileGym can still be run in pure Windows.
- AndroidEnv's upstream Windows support is less maintained than on Unix.
  Host-native pilot results, not this source delivery, qualify compatibility.
- No WSL, Docker, Linux CUDA image, vLLM, external LLM API or separate CUDA
  Toolkit is needed for these two local benchmark runners.

Upstream references:
[AndroidWorld](https://github.com/google-research/android_world) |
[AndroidEmulator command line](https://developer.android.com/studio/run/emulator-commandline) |
[WHPX setup](https://developer.android.com/studio/run/emulator-acceleration) |
[AndroidEnv Windows note](https://github.com/google-deepmind/android_env) |
[MobileGym](https://github.com/Purewhiter/mobilegym)

## 2. Installation from native Windows PowerShell

Clone the repository, or pull the branch once merged. If Git is missing,
install it first through the Windows Package Manager.

    winget install --id Git.Git -e
    git clone https://github.com/Yunbo-max/GUIAGENT.git
    cd GUIAGENT

Optional initial system tools installer (Git, Python3.11, FFmpeg, Node,
Android Studio). It completes the system-install stage and exits; **restart
PowerShell** before the next command so PATH is refreshed:

    powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\setup_windows.ps1 -Benchmark Both -InstallSystemTools

In Android Studio SDK Manager, enable "Android SDK Command-line Tools
(latest)" under SDK Tools. In Windows Features, enable "Windows Hypervisor
Platform" and reboot if necessary. On cloud Windows, verify the hosting
provider allows nested virtualization. Then run the full installer:

    powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\setup_windows.ps1 -Benchmark Both -SetupAndroidSDK -DownloadModel -DownloadMobileGymData

The installer creates .venv-guiagent using Python3.11, installs the official
PyTorch 2.8 / TorchVision 0.23 CUDA 12.8 Windows wheels, then the pinned
AndroidWorld, Qwen vision, and MobileGym Python requirements. It downloads
one pinned Qwen3-VL-4B checkpoint, the pinned UI-Voyager/AndroidWorld
source, the pinned MobileGym source, Chromium, Node dependencies and
MobileGym's approximately 1.9GB archive, builds the MobileGym site, and
creates the AndroidWorldAvd only if missing. It never silently deletes an
existing AVD or changes your GPU driver, and does not start training.

System-AVD setup uses the native SDK manager and may require you to accept
SDK licenses interactively. If command-line tools are missing, the installer
fails explicitly instead of claiming the AVD was installed.

## 3. Preflight before starting either benchmark

    $py = ".\.venv-guiagent\Scripts\python.exe"
    & $py scripts\windows\static_check.py
    & $py scripts\windows\preflight_guiagent.py --benchmark both
    & $py scripts\windows\download_qwen3vl.py --verify-only
    & $py -m pip check
    & $py -m pytest tests\test_android_windows_native.py tests\test_androidworld_feedback.py tests\test_windows_gui_contract.py tests\test_mobilegym_feedback.py -q

Verify actual GPU model, VRAM in GiB, torch CUDA/BF16 support, SDK
paths, WHPX acceleration, pinned sources and Chromium/Node. A configured
96GB card is not a guarantee of actual LoRA training memory headroom.

## 4. Native AndroidWorld: Pilot first, full benchmark second

Start AndroidWorld in a separate PowerShell window:

    powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\start_androidworld.ps1

Optional -Headless works without a desktop window but still requires WHPX.
This launches the emulator with console port 5554 and gRPC port 8554.
Use a second PowerShell in this repository:

    .\scripts\windows\run_benchmark.ps1 -Benchmark AndroidWorld -Stage setup-env
    .\scripts\windows\run_benchmark.ps1 -Benchmark AndroidWorld -Stage env-check
    .\scripts\windows\run_benchmark.ps1 -Benchmark AndroidWorld -Stage prepare
    .\scripts\windows\run_benchmark.ps1 -Benchmark AndroidWorld -Stage plan
    .\scripts\windows\run_benchmark.ps1 -Benchmark AndroidWorld -Stage run

The default pilot has 4 train templates, 8 native evaluation tasks, one
iteration, and its own private manifest/output. It may halt with a signal
shortage if native rollouts produce insufficient true execution feedback;
never fabricate successful actions or treat infrastructure errors as 0 reward.

For the full 116-task benchmark only after the pilot succeeds:

    .\scripts\windows\run_benchmark.ps1 -Benchmark AndroidWorld -Full -Stage prepare
    .\scripts\windows\run_benchmark.ps1 -Benchmark AndroidWorld -Full -Stage plan
    .\scripts\windows\run_benchmark.ps1 -Benchmark AndroidWorld -Full -Stage run
    .\scripts\windows\run_benchmark.ps1 -Benchmark AndroidWorld -Full -Stage report

**Full protocol:** 16 task templates x2 train instances, Round 0
native-only 128 trajectories with NO training, then three arms:
Feedback SD (4 native attempts per instance per round), Feedback
SPECTRUM Fixed (2 native +2 spectral), Feedback SPECTRUM Learned
(2 native +2 learned spectral). Three refinement rounds, 116 verified
eval task instances per arm per round. Each arm receives the same extra
Round-0 episodes and the same number of attempted training rollouts,
not necessarily identical tokens or wall clock FLOPs.

The AndroidWorld manifest retains a disjoint calibration split to
support the original parser, but **the new feedback-only runner never
collects or reads it for learning**. Spectrum gains use real training-side
feedback moments and a training-task-separated guard. Evaluator episodes
are exclusively for reporting. AndroidWorld requires both explicit
termination and a true task.is_successful score of exactly 1.0.

The historical UI-Voyager GRSD experiment, which uses a different RFT
initialization, remains available but must not be mixed into the new
three-arm protocol as a compute-matched result.

## 5. Native Windows MobileGym

In a separate PowerShell window start the preview server:

    powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\start_mobilegym.ps1

It binds to http://127.0.0.1:4173. Then:

    .\scripts\windows\run_benchmark.ps1 -Benchmark MobileGym -Stage plan
    .\scripts\windows\run_benchmark.ps1 -Benchmark MobileGym -Stage env-check
    .\scripts\windows\run_benchmark.ps1 -Benchmark MobileGym -Stage run

For the full existing 32-training-template / 256-test-task 3-round
protocol, explicitly add -Full. Native Windows MobileGym does not
require AVD, Android SDK, ADB or CPU hypervisor. The existing MobileGym
runner does not support resuming incomplete roots; preserve any partial
files and choose a new output root after diagnosis.

## 6. Reporting and limits

Report MobileGym and AndroidWorld as separate simulated/interactive GUI
benchmarks, each with per-round verified environment success rates,
trajectory discovery counts, real action/step budgets, training examples
and GPU resources, learned gain acceptance/fallback and KL diagnostics.
Reserve adequate space for every merged Qwen checkpoint and screenshot.
Only actual host execution can qualify the Windows emulator, full
backprop memory and all-task native verifier paths.

This integration does not claim identical scores on two environments,
official UI-Voyager 81% reproduction, multi-run statistical stability,
or a proof of long-horizon reward credit assignment.

PowerShell execution policy may block direct .ps1 launches; run
Set-ExecutionPolicy -Scope Process Bypass in the **current PowerShell window**
if allowed by your organization, rather than changing the permanent policy.
