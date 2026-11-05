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
