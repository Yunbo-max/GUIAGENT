# SPECTRUM current agent entry

## This branch: Android experiment

For `android-feedback-spectrum`, use [the Android runbook](docs/ANDROID_UI_VOYAGER_ZH.md)
and [current tau/gated handoff](rounds/20261101-android-variants/WEB_HANDOFF.md),
then [variant commands and semantics](docs/ANDROID_VARIANTS_ZH.md). Scope: one Qwen3-VL-4B
AndroidWorld study; shared RFT, UI-Voyager GRSD, fixed/decayed/gated feedback-only arms.
New four-arm config uses a fresh output root; legacy both remains the old pair.
The latest integration is generated_unexecuted; [older review evidence](docs/ANDROID_REVIEW_20261029.md)
belongs to its original revision, not this port. Gate reference entropy is per arm
and bound to completed first-refinement geometry; do not reset it during resume.
Preserve old experiments and results. CPU tests/tiny models do not qualify real4B
CUDA or emulator runs. Native-only Android setup, immutable upstream/model pins,
private parameter manifests, real verifier, no fabricated reference trajectories.
Do not install legacy `[train]` in the Android environment. Publish only this branch
with expected-head readback; no main merge is authorized by this task.

## Existing studies (unchanged)

Read [LOCAL_AGENT_RUNBOOK.md](LOCAL_AGENT_RUNBOOK.md), then
[the current handoff](rounds/20261027-multidomain/WEB_HANDOFF.md) at the delivered commit.
The acquisition and qualification sections are part of the entry.

- Current scope: code on Qwen2.5-Coder7B, Llama3.1-8B and DeepSeekCoder6.7B;
  independent math/QA/ARC-AGI-3 on the same original Qwen3.8-27B checkpoint.
- One training trajectory per method; no mandatory temperature/training-seed
  sweeps. The old40-job/1.5B matrix is historical, not the default new queue.
- Delivery is generated_unexecuted. Source review/authored tests are not actual
  software acceptance or scientific PASS. Qualify on the real native host.
- Separate legacy-code and27B native Conda environments. No container/unsafe-local
  fallback. Program/math scoring uses the qualified native bubblewrap boundary.
- Preserve historical data/results, running checkouts, sealed protocols, private
  randomization and unfavorable outcomes. Never conflate MBPP500, MBPP+378,
  HumanEval+164 or the historical author-LCB protocols.
- Qwen3.8 is hybrid: explicit full-attention K/V targets only; this is not GDN or
  vision adaptation. ARC uses deterministic all-frame text observations.
- ARC calibration requires real successful independent-game trajectories and native
  replay. Heldout games never supply training/calibration references.
- Restore actual SSH/GPU/process/lease state before native Local-harness execution.
  Do not infer liveness from saved status or restart unknown historical watchers.
- Return raw receipts/outputs/native replay. Keep secrets, bulk gated data/model
  weights and seed values out of public Git. Publish with expected-head readback.
