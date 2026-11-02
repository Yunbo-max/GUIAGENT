# MobileGym — Feedback-only SPECTRUM with Learned Spectral Gains

Status: CODE DELIVERED, **native MobileGym/Qwen3-VL GPU experiments NOT RUN**.
The current project keeps the old AndroidWorld and UI-Voyager comparisons
unchanged. This new experiment is a separate optional entrypoint, output root,
and protocol. The original Coding/Math Learned SPECTRUM remains on main.

## Scope, methods and why we adapted Learned

The official MobileGym platform is a browser-hosted GUI simulator with a
deterministic state-based judge; it is NOT the AndroidWorld emulator. We pin
Purewhiter/mobilegym to 5fbd3f34be0e9780ae4b1e9d5089963ddbd05d38.
Only the actual screenshot, user goal and past action descriptions enter
the policy. The native judge receives privileged JSON state after execution.

All arms begin with one shared Qwen3-VL-4B checkpoint (pretrained or separately
prepared RFT checkpoint; this runner does NOT reproduce AndroidWorld RFT).
Official MobileGym train and test task whitelists stay disjoint.

Each train task receives 4 *native* rollouts in Round 0, producing verified
success/failure and genuine action-format errors. Round 0 has NO parameter
updates, NO calibration examples, NO spectral modification and NO SFT.
The native-prefix feedback builder uses successful same-instance paths,
possibly from spectral rollouts in later rounds, and actual action errors.
A failed terminal reward alone does not identify which step was wrong.

In each of 3 refinement rounds:
  1. Compute training-only feedback-conditioned KL activation-gradient
     second moments in the current Qwen checkpoint's final two K/V layers.
     The source is feedback only, not 50 correct answer references.
  2. Fixed arm: spectral gain controlled by tau=1, otherwise same operator.
     Learned arm: use the SAME main-branch full-spectrum gain wrapper and
     equal-eigenvalue grouping, with feedback-token KL as the fit objective,
     a different training-task guard, and native-to-steered full-vocabulary KL.
     The final candidate is rechecked with actual BF16 weight folding.
     Min gain=0.1, max gain=1, 8 bounded search steps, 0.02-nat probe budget
     are developmental defaults, not verified performance settings.
  3. Sample 2 original + 2 spectral GUI trajectories per training task.
     The no-spectrum Feedback SD control samples 4 original trajectories.
  4. Use native prefixes, their actual feedback and a frozen version of
     this SAME model to perform response-token reverse-KL LoRA updates.
     Spectral rollouts supply success evidence; they are NOT directly SFT'd.
  5. Evaluate the newly merged NATIVE checkpoint with MobileGym's state-based
     verifier, requiring both COMPLETE and a clean programmatic judge.

Arms included in the default main comparison:
  - feedback_sd (no spectrum, same feedback distillation)
  - feedback_spectrum_fixed (feedback-only geometry, tau=1)
  - feedback_spectrum_learned (feedback-only geometry, learned full gains)
Optional: feedback_spectrum_tau, feedback_spectrum_gated.

The learned gain optimize-stage objective is Feedback KL at FIXED native
response-token prefixes. It is NOT the non-differentiable environment return
and does NOT yet implement counterfactual long-horizon credit assignment.
It can also select the identity operator, which must be reported rather than
relabelled as a gain. No claim that the learned gains outperform fixed tau,
reduce collapse or improve long-horizon success has been established.

## Capacity and installation

Official MobileGym: Node >=22, Python >=3.11, Playwright Chromium,
~1.9GB downloaded companion data, ~400MB system memory per parallel
simulator instance. Our runner uses one headless browser at a time.
Reserve >=64GB RAM and >=500GB available disk for multiple full merged
checkpoints and screenshots. An 80GB H100 is recommended for Qwen3-VL-4B
BF16 gradients; a 24GB 4090 requires actual measured short-run qualification.
These are planning allowances, not measured peaks.

From the SPECTRUM repository root:

    mkdir -p external
    git clone https://github.com/Purewhiter/mobilegym.git external/mobilegym
    git -C external/mobilegym checkout 5fbd3f34be0e9780ae4b1e9d5089963ddbd05d38
    cd external/mobilegym
    npm install
    python -m pip install -r bench_env/requirements.txt
    python -m playwright install chromium
    curl -L -o mobilegym-data.tar.gz https://github.com/Purewhiter/mobilegym/releases/download/data-v0.1.0/mobilegym-data-v0.1.0.tar.gz
    tar -xzf mobilegym-data.tar.gz
    npm run build
    npm run preview -- --port 4173

Keep preview running in a separate shell. From the SPECTRUM root install
the Android vision-model dependency set (NOT the obsolete older text extra):

    python -m pip install -r requirements/android.txt
    python -m pip install -e . --no-deps

Download the Qwen3-VL-4B checkpoint pinned in the JSON (or select an
existing checkpoint with full config/processor files); EDIT model.checkpoint
in configs/mobilegym/feedback_only_qwen4b_learned.json to the actual local
absolute directory before running. The code does not download the model.

Run CPU unit checks first:

    python -m pytest tests/test_mobilegym_feedback.py tests/test_android_geometry.py tests/test_android_variants.py -q

A separate browser environment check is available before GPU work:

    python scripts/run_mobilegym_feedback.py --config configs/mobilegym/feedback_only_qwen4b_learned.json env-check

This is not a task-success score. Start MobileGym preview first.

Plan checks source revision and official split IDs but loads NO model:

    python scripts/run_mobilegym_feedback.py --config configs/mobilegym/feedback_only_qwen4b_learned.json plan

Only after checking the plan, browser service, actual CUDA memory and new
empty output root, explicitly start the experiment:

    python scripts/run_mobilegym_feedback.py --config configs/mobilegym/feedback_only_qwen4b_learned.json run

There is NO hidden scheduled/background launch. An incomplete stage is
retained. The current runner deliberately refuses to resume into an existing
root, rather than silently reuse checkpoints with changed sources.

For FIRST GPU/browser qualification, copy JSON to a different file and
use a new output_dir with train_task_limit=8, test_task_limit=8,
rounds=1, arms=["feedback_sd","feedback_spectrum_learned"].
This tiny pilot may stop with signal_shortage if it finds no usable same-task
feedback. Never fabricate success labels to force the experiment.

## Comparison and reporting

Official MobileGym GRPO with Qwen3-VL-4B is an external reference, NOT
reproduced by this custom runner; its action prompts, train data and
budget can differ. Our primary internal controls share the identical
backbone, same prompt schema and sampled train/test population. Also report
native rollout attempts including Round 0, successful train episodes,
actual environment steps, actions/tokens, learned-gain identity fallbacks
and final deterministic success rate.

The official public test split is used for independent evaluation and is
NEVER used for feedback, gradients, learned gain selection or checkpoint
selection. Model-selection research needs a separate development split.
Multiple seeds and confidence intervals are needed before a paper claim.
One run is NOT proof of scientific superiority.

Failure modes: even an executable model can have zero successes on 8
tasks; sparse terminal rewards have poor localization; a feedback-KL gradient
can be unrelated to environmental advantage. A 0.02-nat native-probe budget
may cause excessive identity fallback, and long image-conditioned forwards
can exceed GPU memory. Preserve every negative and failed result.

Relevant source:
  improving.mobilegym_feedback (control loop),
  improving.mobilegym_visual_gains (gain optimization),
  improving.android.geometry (feedback-only moments),
  improving.learned_gains (unchanged main-branch gain primitives),
  improving.mobilegym_bridge (visual policy/action schema).
