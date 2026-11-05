# AndroidWorld 双 AVD 加速 — 纯 Windows

**注意：已经运行的 Python 进程无法通过 GitHub 更新而突然加速。**
这套代码在独立 Git 分支 / 新的输出目录中。不要覆盖当前 AndroidWorld
已采集的轨迹，不要修改正在运行 MobileGym 的原始源码目录。

**最安全的获取方式：新建 Windows worktree，通过 NTFS Junction 共享既有模型/数据。**

在当前原仓库的 PowerShell 窗口执行以下命令；这里不修改原 worktree：

    $old = (Get-Location).Path
    git fetch origin feature/androidworld-two-avd-throughput-20261105
    git worktree add ..\GUIAGENT-AVD2 origin/feature/androidworld-two-avd-throughput-20261105
    Set-Location ..\GUIAGENT-AVD2
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts/windows/link_parallel_assets.ps1 -OriginalRepo $old
    $py = Join-Path $old ".venv-guiagent\Scripts\python.exe"

上述脚本只为新 worktree 创建 data 和 models 的 NTFS Junction；Junction **本身不是只读权限**，不要从新 worktree 重新 prepare/download 或修改共享模型与数据；不复制权重、
不修改旧实验输出、不触碰活跃 MobileGym 进程。生成结果放在新 worktree
下的 runs，不与旧结果混用。如果原目录或新的 data/models 路径有冲突，
脚本会直接报错，禁止覆盖。

**先让原 AndroidWorld collector 在安全边界停止并释放 AVD1，再准备 AVD2；
MobileGym 可以继续运行。** 随后用新 worktree 的 Python 依次执行：

    powershell -NoProfile -ExecutionPolicy Bypass -File scripts/windows/start_androidworld_parallel2.ps1 -InstallSecondAVD
    & $py scripts/run_androidworld_feedback.py --config configs/android/androidworld_feedback_learned_windows_pilot_parallel2.json setup-env
    & $py scripts/run_androidworld_feedback.py --config configs/android/androidworld_feedback_learned_windows_pilot_parallel2.json env-check
    & $py scripts/windows/bench_androidworld_throughput.py --mode both

A/B 的原生成功数、动作量和平均耗时写入
runs/androidworld_windows/avd_ab_probe_v1/report.json。先判断是否真有吞吐收益，
不要先启动数千条轨迹的正式实验。只有 A/B 通过，再运行新的 2-AVD
Pilot / Full。原来已经完成的记录仍保存在旧目录。

# AndroidWorld CPU/AVD throughput pilot — pure Windows

This additive experiment compares the original single AndroidWorld
emulator with TWO independent Pixel6/API33 emulator processes. Both
retain the same benchmark tasks, original Qwen3-VL-4B architecture,
30-step limit, fixed post-action 1.5s wait, screenshots, native reward,
feedback KL, SPECTRUM gains, and unchanged LoRA updates.

The new runner uses Windows multiprocessing spawn, not threads, since
upstream task initialization sets process-global Python/NumPy RNGs.
Each child has its own emulator name, ADB serial (console+1), gRPC
port, screenshot directory and Qwen inference copy. The parent remains
the owner of gradient geometry, gain learning and SFT checkpoints.
All worker episodes are reassembled into the original frozen
instance/repeat order; missing or invalid native verification fails.

This has NOT been GPU/emulator stress-tested on the host. It is
not a hot patch: editing GitHub cannot accelerate a running Python
process, and the original study remains untouched.

Capacity: Windows host 16 physical CPU cores / 32 threads, ~255 GiB
system RAM, NVIDIA GPU1 ~95.6 GiB total VRAM. The optional 2-worker
configuration uses 2 CPU threads per worker and a 20 GiB PyTorch
allocator limit per child, not preallocated VRAM. MobileGym may
still be sharing GPU1. Measure combined VRAM and CPU before increasing
worker count to 3 or 4.

Opt-in configs:

- configs/android/androidworld_feedback_learned_windows_pilot_parallel2.json
- configs/android/androidworld_feedback_learned_windows_full_parallel2.json

Both reuse the same frozen original pilot/full instance manifests,
but write to NEW output directories. Do not rerun "prepare" against
an existing manifest. Original results/partial screenshots must
remain intact, including Round 0 episodes already completed.

For a smaller and faster throughput-only A/B (2 train instances,
2 repeated trajectories each, no SFT), before committing the full
pilot:

    python scripts/windows/bench_androidworld_throughput.py --config configs/android/androidworld_feedback_learned_windows_pilot_parallel2.json --mode both

The output compares real episode completions and actions per minute,
along with wall-clock components model_generation, ADB actuation,
state stabilization, PNG encoding and explicit 1.5s wait.
Run it only when both AVDs are fully installed and not controlled
by an older active collector. A/B outputs live under a separate
runs/androidworld_windows/avd_ab_probe_v1 root.

1. On the Windows machine, finish or intentionally pause the original
   AndroidWorld Python collector at a safe boundary. Do not let two
   collectors operate the same AVD. Do not stop MobileGym if it is healthy.
2. Create another independent Pixel6/API33 x86_64 AVD and start both:

       powershell -NoProfile -ExecutionPolicy Bypass -File scripts/windows/start_androidworld_parallel2.ps1 -InstallSecondAVD

3. With no competing collector, install setup apps/forwarder on BOTH
   emulators; ensure both return native screenshots:

       python scripts/run_androidworld_feedback.py --config configs/android/androidworld_feedback_learned_windows_pilot_parallel2.json setup-env
       python scripts/run_androidworld_feedback.py --config configs/android/androidworld_feedback_learned_windows_pilot_parallel2.json env-check

4. Do a strictly NEW one-round pilot, not full experiment:

       python scripts/run_androidworld_feedback.py --config configs/android/androidworld_feedback_learned_windows_pilot_parallel2.json plan
       python scripts/run_androidworld_feedback.py --config configs/android/androidworld_feedback_learned_windows_pilot_parallel2.json run

5. Inspect output round0_native/complete.json and completed train/eval
   receipts, elapsed minutes, verifier numerator/denominator, per-step
   action counts, CUDA peak and Windows CPU usage. Compare with an
   equivalent 1-AVD pilot. Different sampling RNG streams mean they
   are not bit-identical; reject any unverified environment result.
6. Only if 2 workers win on measured task throughput without abnormal
   native failures, run the full config in a separate output root:

       python scripts/run_androidworld_feedback.py --config configs/android/androidworld_feedback_learned_windows_full_parallel2.json plan
       python scripts/run_androidworld_feedback.py --config configs/android/androidworld_feedback_learned_windows_full_parallel2.json run

The expected speedup is UNKNOWN before the actual A/B. The previous
1-AVD 7 episodes / 20 minutes is only a rough point estimate. Two
AVDs can overlap simulator waits but add GPU model copies and CUDA
contention. Do not promise 2x or a revised finish date without proof.

The current original run cannot be transparently continued by this
new process layout. Any already-complete Round 0 episodes remain
preserved in the original output; the opt-in accelerated experiment
starts its own Round 0. Never merge the two runs into one official
reproducibility trajectory without an explicit provenance policy.
