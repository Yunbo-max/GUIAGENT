# GUIAGENT: Round 0 = 16 Native, Stage 4 = 8 Native + 8 Spectral

**2026-11-06 configuration-only study.** These four configs have fresh output
roots. The current 4-rollout experiments, private frozen manifests, local
screenshots and trained checkpoints are deliberately untouched. No active
Windows Python process will automatically adopt these settings.

## 1. Which model generates Native trajectories?

The "Native" agent in EVERY refinement round is that METHOD'S current,
unmodified post-distillation Qwen3-VL-4B student:

- Shared Round 0: base theta_0 -> **16 fresh Native trials per train instance**;
  no gradient updates and no spectral operator
- Round 1: theta_0 produces **8 fresh Native**, spectrally steered theta_0
  produces **8 Spectral**; environment feedback -> reverse KL student theta_1
- Round 2: theta_1 produces **8 fresh Native + 8 Spectral**; student theta_2
- Round 3: theta_2 produces **8 fresh Native + 8 Spectral**; student theta_3
- Baseline Feedback SD does **16 Native per instance** per round.
- Fixed SPECTRUM and Learned SPECTRUM use 8 Native + 8 Spectral,
  within the same 16 episode-attempt budget. All three methods begin
  from the SAME Round 0 seed/episodes, then evolve separate checkpoints.
- Stage 6 evaluates ONLY the latest unsteered NATIVE checkpoint from that
  arm. Eval episodes are **never fed into feedback, geometry or SFT**.

Importantly, Round 1 has the same theta_0 base as Round 0, but the
implementation samples fresh independent Native trajectories rather
than silently reusing the 16 Round-0 episodes. Later rounds never
mislabel old theta_0 trajectories as current theta_r samples.

Spectral trajectories are evidence, not a direct supervised SFT set.
The feedback builder takes sampled **Native action prefixes** and
appends same-instance actually successful paths (including spectral
ones) and/or native parser/action errors. The frozen same-checkpoint
teacher conditions on hindsight evidence, and the unsteered student
receives reverse-KL updates. If no informative feedback exists, this
runner stops with signal_shortage; 16 samples do not guarantee success.

## 2. Complete matched comparison

| Stage | Feedback SD | Fixed SPECTRUM | Learned SPECTRUM |
|---|---|---|---|
| Shared Round 0 | 16 Native | Same 16 Native | Same 16 Native |
| Each refinement round | 16 Native | 8 Native + 8 Spectral | 8 Native + 8 Learned Spectral |
| Student objective | Feedback Reverse KL | Same | Same |
| Student init per round | Own latest theta_r | Own latest theta_r | Own latest theta_r |
| Final evaluation | Native latest student | Native latest student | Native latest student |

There are three refinement rounds in the full study and one in the
pilot. Both benchmarks use the same Qwen3-VL-4B BF16 model revision.
The learned-gain settings, 16 maximum feedback moment examples,
8 fit / 4 guard examples, 32 maximum optimizer updates for each
FULL training stage and original effective training batch size are
UNCHANGED from the source full configurations. This is intentional
to isolate the change in collected rollout count.

**Important constraint:** collecting 16 attempts does NOT mean the
optimizer now trains on every one of those attempts. In particular,
only admitted Native prefixes enter the response-KL minibatch
schedule, which still caps FULL training at 32 optimizer steps.
Extra successful attempts can enrich hindsight feedback, but each
native prefix currently uses up to ONE same-instance verified
successful path as evidence (feedback.max_successes = 1). This
sampling-only study should be followed by a distinct ablation if
you want to increase teacher evidence diversity or SFT token budget.

## 3. Full task counts (per Benchmark)

| Count | MobileGym | AndroidWorld |
|---|---:|---:|
| Train instances | 32 tasks | 16 templates x 2 = 32 instances |
| Shared Round 0 native | 512 | 512 |
| Per arm, per round stage 4 | 512 | 512 |
| All arms x 3 rounds stage 4 | 4,608 | 4,608 |
| Total TRAIN rollouts incl Round 0 | **5,120** | **5,120** |
| Evaluation per arm, per round | 256 | 116 |
| Total EVAL across arms/rounds | 2,304 | 1,044 |
| Total requested episodes | 7,424 | 6,164 |

Training-rollout attempts increase **4x** compared with the existing
4-sample study: 1,280 -> 5,120 in EACH benchmark. This can lengthen
the already long AndroidWorld schedule considerably. 2 AVDs may help
but have NOT been shown to make the new 16-sample experiment finish
within the old ETA. These are *requested* counts, not a claim of
completed episodes, GPU efficiency or performance gains.

## 4. Safe opt-in Windows execution

For MobileGym, Windows uses 4 independent Chromium environments (one
serialized inference thread) and BF16 Qwen; AndroidWorld uses the
opt-in 2 independent AVD protocol (separate GPU inference processes).
Both still target GPU1 as configured by the host's
CUDA_VISIBLE_DEVICES, not by an implicit GPU selection in these JSONs.
Keep each experiment root separate from any currently running study.
Do NOT change the underlying source code of an active Python process.

Inspect the budget **without downloading models or launching tasks**:

    python scripts/gui_sample_budget16.py --benchmark all

For MobileGym (requires pinned MobileGym service/model already ready):

    python scripts/run_mobilegym_feedback.py --config configs/mobilegym/feedback_only_qwen4b_parallel4_samples16_pilot.json plan
    python scripts/run_mobilegym_feedback.py --config configs/mobilegym/feedback_only_qwen4b_parallel4_samples16_pilot.json run

When its pilot passes and a full new run is intended:

    python scripts/run_mobilegym_feedback.py --config configs/mobilegym/feedback_only_qwen4b_parallel4_samples16.json plan
    python scripts/run_mobilegym_feedback.py --config configs/mobilegym/feedback_only_qwen4b_parallel4_samples16.json run

For AndroidWorld (must have independent WHPX-ready AVDs at console
5554 / 5556 and gRPC 8554 / 8556, with full native app setup):

    python scripts/run_androidworld_feedback.py --config configs/android/androidworld_feedback_learned_windows_pilot_parallel2_samples16.json env-check
    python scripts/run_androidworld_feedback.py --config configs/android/androidworld_feedback_learned_windows_pilot_parallel2_samples16.json plan
    python scripts/run_androidworld_feedback.py --config configs/android/androidworld_feedback_learned_windows_pilot_parallel2_samples16.json run

Only after its pilot and 1-vs-2 AVD throughput check succeed:

    python scripts/run_androidworld_feedback.py --config configs/android/androidworld_feedback_learned_windows_full_parallel2_samples16.json plan
    python scripts/run_androidworld_feedback.py --config configs/android/androidworld_feedback_learned_windows_full_parallel2_samples16.json run

The 16-sample AndroidWorld configs reuse the ORIGINAL frozen
pilot/full train/eval manifests; do **not** call prepare on an already
frozen manifest. The -Samples16 switch in
scripts/windows/run_benchmark.ps1 chooses the corresponding configs
for plan/run and blocks the potentially destructive AndroidWorld
prepare operation.

The MobileGym runner is still single-attempt/nonresumable.
The AndroidWorld parallel runner validates completed stage receipts
and refuses to silently repair partial results. Keep all original
4-sample data and process state; this is a separate experimental
trajectory, NOT a seamless resume. Never operate a second collector
on an AVD that the original run still owns.

## 5. What the study actually tests

Primary hypotheses:
1. At a fixed 16-episode budget, does adding 8 Learned Spectral
   explorations improve **verified success discovery** relative
   to 16 Native, or 8 Native + 8 fixed spectral?
2. Does richer actual environment feedback yield more valid
   native-prefix reverse-KL training examples and improve the latest
   unsteered student after each round?
3. Is the improvement worth the additional environmental sampling and
   spectral/KL calibration compute?

Report separately: native/spectral verified successes, terminal status,
feedback admissibility rate, actual native training-prefix count,
gain identity fallbacks, fitted and held-out KL, CPU/ADB/Chromium
wall-clock time, GPU VRAM, and per-round native eval success.
No claim that spectral is always more successful than native, or
that this automatically beats GRPO, is supported without results.
