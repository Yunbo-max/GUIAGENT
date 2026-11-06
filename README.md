# SPECTRUM

## 16-sample GUI self-improvement protocol (Round 0 and every refinement round)

[中文/English 16-sample experiment guide](docs/GUI_SAMPLES16_ROUND0.md) ·
[Offline budget and fidelity verifier](scripts/gui_sample_budget16.py) ·
[MobileGym Full 16-sample config](configs/mobilegym/feedback_only_qwen4b_parallel4_samples16.json) ·
[AndroidWorld Full 16-sample config](configs/android/androidworld_feedback_learned_windows_full_parallel2_samples16.json)

Round 0 collects 16 native episodes per training instance; every subsequent
round samples 16 attempts, split **8 native + 8 spectral** (while Feedback SD
uses **16 native**). The "native" policy always uses each method's **current
post-distillation checkpoint**, not the initial Round-0 model. All configs are
opt-in and have separate output roots; they do NOT modify active 4-sample runs.
The training-rollout budget rises 4x, and actual completion time is unknown.


## Opt-in native Windows AndroidWorld dual-AVD throughput experiment

[Two-emulator CPU/sampling plan](docs/ANDROIDWORLD_PARALLEL2_ZH.md) ·
[Dual-AVD pilot configuration](configs/android/androidworld_feedback_learned_windows_pilot_parallel2.json) ·
[Windows AVD2 launcher](scripts/windows/start_androidworld_parallel2.ps1)

The native AndroidWorld collection can now run in two separate, isolated Pixel6
Android 13 emulator processes and validate every native verdict. This **does
not alter an active Windows experiment**, and no speedup or numerical benchmark
score has yet been measured on the dual-AVD host. The current MobileGym
four-Chromium optimization remains untouched.

## GPU 1 throughput and memory qualification — 5 November 2026

[显存参数、1/2/4/24 环境实测与限制](docs/GPU_MEMORY_AND_THROUGHPUT_ZH.md) ·
[Opt-in four-environment config](configs/mobilegym/feedback_only_qwen4b_parallel4.json)

MobileGym can overlap independent browser environments around one serialized Qwen inference thread.
Short native tests found 4 environments faster than 1; 24 showed no additional gain over 4.
Completed episodes can be explicitly reused from a matching sealed original study.
These are hardware/throughput qualifications, not completed full benchmark accuracy results.
The earlier generated-only status notes below describe the original delivery; see the dated guide for current evidence.

## Native Windows: MobileGym + AndroidWorld on NVIDIA 96GB

[中文 Windows 安装指南](docs/WINDOWS_96GB_SETUP_ZH.md) ·
[Windows installation and two-benchmark guide](docs/WINDOWS_NATIVE_96GB_GUI_BENCHMARKS.md) ·
[AndroidWorld Learned config](configs/android/androidworld_feedback_learned_windows_96gb.json) ·
[PowerShell installer](scripts/windows/setup_windows.ps1) ·
[Benchmark launcher](scripts/windows/run_benchmark.ps1)

AndroidWorld is now an additional native Windows-capable GUI benchmark
alongside MobileGym. Both use Qwen3-VL-4B BF16 feedback-only Learned SPECTRUM
and a no-spectrum Feedback SD control. The existing UI-Voyager protocol and
MobileGym historical experimental records are preserved. Windows emulator
and GPU end-to-end qualification still need to run on the target host.


## MobileGym learned-feedback spectrum (new opt-in study)

[Execution guide](docs/MOBILEGYM_FEEDBACK_LEARNED_ZH.md) ·
[Config](configs/mobilegym/feedback_only_qwen4b_learned.json) ·
[Entry](scripts/run_mobilegym_feedback.py)

This adds a shared native Round 0, feedback-only visual K/V gradient geometry,
and the latest full-spectrum learned gains adapted to multimodal feedback KL.
The comparison includes matched Feedback SD (no spectrum) and fixed spectrum.
It is **code delivered, not GPU or native MobileGym benchmark tested**.
Legacy AndroidWorld/UI-Voyager results and configs are unchanged.


**Proximal Spectral Modulation for Looped Self-Distillation**

## Android GUI branch — 1 November 2026

[**GUI tau/gated guide / 当前运行指南**](docs/ANDROID_VARIANTS_ZH.md)
· [Four-arm configuration](configs/android/ui_voyager_qwen4b_variants.json)
· [Handoff](rounds/20261101-android-variants/WEB_HANDOFF.md)
· [Original environment and assets](docs/ANDROID_UI_VOYAGER_ZH.md)

One Qwen3-VL-4B / AndroidWorld study now supports UI-Voyager GRSD, fixed
feedback SPECTRUM, SPECTRUM with per-round tau decay 0.6, and spectral-entropy-gated
SPECTRUM (gamma 5). All share three RFT rounds, then run three refinement rounds
per arm. The three SPECTRUM arms keep the same feedback-only KL objective.

The new config uses a fresh result root and `run --arm all`; `both` retains the
original two-arm meaning. The original config/results remain unchanged. This port
reuses the existing main-branch tau/gated definitions, with receipt-bound adaptive
reference state. A gate can increase tau; it is not guaranteed collapse prevention.

This revision is **generated_unexecuted**: source/static review and authored tests
are separate from pending Local tests and real 4B/CUDA/emulator qualification.
The [older 82 CPU passes](docs/ANDROID_REVIEW_20261029.md) do not certify this port.
Non-GUI studies below are this branch's historical snapshot; current main has
separate updates and is not modified by the GUI delivery.

## Historical non-GUI experiment entry — 27 October 2026

[**Run the current study / 当前运行指南**](LOCAL_AGENT_RUNBOOK.md)
· [Web → Local handoff](rounds/20261027-multidomain/WEB_HANDOFF.md)
· [Code](docs/CROSSFAMILY_CODE.md) · [Math / QA](docs/TEXT_TASKS.md)
· [ARC-AGI-3](docs/ARC_AGI3.md) · [27B model adapter](docs/MODEL_COMPATIBILITY_27B.md)

| Experiment | Models | Evaluation |
|---|---|---|
| Cross-family code | **Qwen2.5-Coder-7B-Instruct**, **Llama-3.1-8B-Instruct**, **DeepSeek-Coder-6.7B-Instruct** | MBPP500 main + native HumanEval+164 transfer |
| Mathematics | **Qwen3.8-27B** | Complete MATH500 |
| QA | Same original **Qwen3.8-27B** | Complete MMLU-Pro + GPQA Diamond |
| Interactive agents | Same original **Qwen3.8-27B** | ARC-AGI-3, frozen public game holdout |

Base + Vanilla SD + SSD adaptation + SPECTRUM; five rounds and **one independent
training trajectory per method**, with fixed decoding and no temperature/seed grid.
Math, QA and ARC start independently from the same original27B checkpoint.
The three code models are different families. Their parameters are approximately
7–8B; Qwen3.8 uses an explicitly supported hybrid text decoder.

The new source includes model/data acquisition and verification, recursive training,
native scoring, raw result retention and bound resume. Delivery remains
**generated_unexecuted**: source/static review is complete; project tests and actual
GPU/native execution must be qualified on the user's host.
ARC SPECTRUM additionally requires a real replay-verified successful calibration
trajectory; collection/import commands are supplied. A27B model is not a guarantee
of ARC success. Default public holdout results must not be called full competition scores.

The [current runbook](LOCAL_AGENT_RUNBOOK.md) is the entry to use. The older release
matrix below is retained as historical documentation, not the new default queue.
Existing model results, data and active runs have not been rewritten.

SPECTRUM studies what a model retains when it repeatedly learns from its own
generated code. Each round estimates loss-sensitive key/value geometry using
the same fixed reference anchor, temporarily modulates the generator, and trains
one native student on every raw completion. Generation weights are restored
before LoRA training; final inference needs no spectral intervention.

The fixed anchor supplies initial reference information that is reused for
calibration. Generated training completions are not checked, ranked, filtered,
or given quality-dependent loss weights. Evaluation tests are separate. The
anchor-replay **control** deliberately uses the references as supervised training
examples and is labeled separately.

## Earlier release design (historical)

This repository continues the implementation from
[`yuhanlydia/improving`](https://github.com/yuhanlydia/improving) at commit
`45195e866162df1cdf9de0169d2bf17347eb6abf`. The release keeps the original SPECTRUM
definition and adds a fixed validation matrix, controls, pinned assets and a
complete result exporter. Historical results are developmental evidence; the
new experiment cells remain unrun until the commands below are executed.

**Release reference:** [原版运行指南（中文）](docs/RUN_FINAL_ZH.md)
· [Experiment contract](docs/FINAL_EXPERIMENTS.md)
· [Exact models and datasets](docs/DATA_MODELS.md)
· [Research audit](docs/AUTORESEARCH_CHECK.md)
· [Existing evidence](docs/EXISTING_EVIDENCE.md)
· [Related-work audit](docs/LITERATURE_AUDIT.md)

### Earlier four-backbone plan

| Alias | Hugging Face model | Main comparison | Independent training replicates |
|---|---|---|---:|
| `qwen1.5b` | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | Vanilla SD / SSD / SPECTRUM, five rounds | 3 |
| `qwen3b` | `Qwen/Qwen2.5-Coder-3B-Instruct` | Same three methods, five rounds | 1 |
| `qwen7b` | `Qwen/Qwen2.5-Coder-7B-Instruct` | Same three methods, five rounds | 1 |
| `deepseek6.7b` | `deepseek-ai/deepseek-coder-6.7b-instruct` | Same three methods, five rounds | 1 |

All four are **planned trained-method comparisons**, not an initial-model-only
matrix. Qwen uses grouped-query attention; DeepSeek uses multi-head attention.
Full model revisions and layer/head dimensions are in
[`configs/final/assets.json`](configs/final/assets.json).

### Five evaluation datasets

| Dataset | Evaluation tasks | Role |
|---|---:|---|
| MBPP | 500 | In-domain, rounds 0–5; separate 291 training / 50 anchor / 30 validation tasks |
| HumanEval+ | 164 | Frozen-student transfer, official EvalPlus checks |
| APPS Intro | 200 | Frozen transfer, fixed adapted evaluation |
| CodeContests | 165 | Frozen transfer, full official test population with adapted evaluation |
| LiveCodeBench | 200 | Frozen transfer, fixed release/time window with adapted evaluation |

Every final evaluation uses 64 samples per task. All reported pass@k values and
correct-implementation metrics come from the same pool. The transfer benchmarks
provide no training or calibration signal. The three adapted competition
protocols are not official leaderboard reproductions.

## Earlier release execution notes (historical)

Follow the [ordered Local command cards](LOCAL_AGENT_RUNBOOK.md#calibrate-debug-then-dispatch).
The current host route is native Conda. The earlier release guide documents a Docker
path; it is retained as historical implementation reference, not the current host's
launch instruction. Native isolation/scorer integration must be qualified before
release execution. `--allow-unsafe-local` does not establish isolation.
The SPECTRUM training path requires a compatible BF16 GPU; the research branch's
FP16 author-checkpoint evaluation is a different protocol. The `improving` import/CLI
name is retained for compatibility.

The private randomization file is generated once, is ignored by Git, and must be
kept with local run artifacts. Public report tables use `r1/r2/r3` labels.
Planning loads no model and launches no experiment. **There is no automatic
full-grid execution.** Use `run --block all` only to request the entire fixed
matrix. Other blocks are `scale`, `mechanism`, `token_matched`, `diagnostic`,
`transfer`, and `decoder`; transfer/decoder require their source students.

Schedule the independent core jobs on the actual allocated devices through the
Local harness; three replicates do not imply three available GPUs. Never give two
processes the same job; file locks reject that conflict. Repeating a compatible command
resumes sealed work. An interrupted SFT stage restarts that stage, rather than
restoring optimizer steps. Do not change code, data, hardware profile or configs
inside an active plan; create a new run root for changed conditions.

## What the controls answer

- **Projection / random orientation / isotropic gain:** does full-rank,
  loss-sensitive orientation matter beyond generic attenuation?
- **Fixed initial geometry:** is per-round recalibration useful?
- **Fixed-anchor replay:** is directly learning the same reference answers an
  adequate simpler alternative?
- **Matched SFT token exposure:** do benefits persist when every arm sees the
  same prescribed supervised-token budget, with all raw records retained?
- **Decoder robustness:** do round-five comparisons survive alternative fixed
  native decoding policies? Initial-model decoding-only scores are retained.
- **Strength and loss scope:** report the predeclared sensitivity results without
  replacing the original method after seeing test outcomes.

## Outputs and completion

`runs/spectrum_release_v1/release_report/` contains editable CSV tables and a
Markdown/JSON index: endpoints, trajectories, paired differences, correct-count
eligibility, retention, independent-replicate summaries, ablations, decoder
effects, generator-to-student diagnostics, resource use and operator diagnostics.
Missing results are `x`, never zero. All effect signs are retained.

The full fixed plan has **40 jobs: 13 training jobs, 24 transfer jobs, 3 decoder
jobs; 134 method–replicate–rounds and 7,106,080 fresh-run generated candidates**.
It is a substantial study, not a smoke run. Run the core block first; inspect
the frozen plan before launching other blocks. Candidate counts are not measured
GPU hours. Keep all checkpoints; see the storage discussion in the run guide.

Completion means the planned artifacts exist and validate, not that SPECTRUM
wins. It does not trigger extra hyperparameter searches or favorable-result
selection. Three independent core replicates and one replicate per scale model
must not be pooled as if all tasks or training runs were independent.

## Verification status of this code release

For the 2026-10-27 amendment, source review and static syntax/JSON checks are
separate from pending Local tests. Primary metrics now follow the manifest;
paired task eligibility is explicit; new run fingerprints bind installed training
dependencies and runtime identity. These changes do not retroactively certify old
runs or provide bitwise reproducibility across arbitrary hosts.

The following describes the original release preparation:

Source review, Python/JSON parsing and CLI help checks were performed. Regression
test sources were added. In accordance with the request for code without running
experiments, **no pytest suite, model download, training, benchmark, or generated
program execution was performed during this release**. GPU memory profiles are
conservative configurations, not measured memory guarantees.

Historical scripts/configs are retained for provenance and compatibility. The
supported final entrypoints are `prepare_final_data.py`, `run_release_study.py`,
and `report_release_study.py`; earlier `run_final_study.py` plans are a different
protocol and must not be mixed with this release.
