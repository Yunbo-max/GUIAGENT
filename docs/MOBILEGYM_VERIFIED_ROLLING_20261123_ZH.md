# MobileGym: Verified Rolling Action Distillation (VRAD) — 2026-11-23

**Status: new code committed, CPU checks authored; GPU/native end-to-end execution
NOT yet qualified on the user's 97GB host.** No results claimed. This is an
opt-in separate branch and output root; old SPECTRUM / AndroidWorld studies
and sealed experiment artifacts are unchanged.

## Research question

Can a Qwen3-VL-4B GUI policy improve next-action selection through *verified
future interaction evidence*, without GRPO or a learned world model?

In a real trajectory, the policy sees only (goal, current screenshot, action
history) and outputs ONE executable action. The simulator executes it. During
**training-collection only**, the native MobileGym grounded state verifier also
records progress BEFORE and AFTER each executed action. An offline builder looks
ahead 8 actual steps, admits only prefixes with observed progress gains, and
creates current-action and horizon-conditioned future-action-type targets.

The privileged frozen teacher sees verified hindsight (executed action types
and observed progress), while the student sees the original screenshot/history
plus a horizon request. No held-out task, simulator state or hidden answers are
in the student input. Neither teacher nor student simulates a future screenshot.

This V1 is **horizon-conditioned semantic action anticipation, not a parallel
MTP-head implementation**, and it does NOT predict future button coordinates.
Current action training retains the exact sampled next-action token IDs. Future
auxiliary labels are action TYPE only (CLICK/SWIPE/TYPE/...).

Teacher and student are the same starting checkpoint with/without LoRA.
Training loss is weighted **reverse KL** plus a small future-type CE term:

    L = mean_{verified t,k} w_{t,k} [
          KL(p_student(.|s_t,k) || p_frozen_teacher(.|s_t,k,F_future))
          + 0.1 * 1[k>0] * CE(future_action_type)
        ]

Progress is a verifier-derived *prefix* signal, NOT a causal per-action reward.
It cannot prove an action was optimal. This is not GRPO, not a critic, not a
world model, and not a counterfactual branch experiment.

## Protocol (formal, not a pilot)

- Pin: Purewhiter/mobilegym at 5fbd3f34be0e9780ae4b1e9d5089963ddbd05d38.
- Qwen/Qwen3-VL-4B-Instruct at ebb281ec70b05090aa6165b016eac8ec08e71b17.
- All **160 official train templates**, each with 4 native attempts: requested
  640 train episodes (actual success and action lengths are measured).
- All **256 official held-out test templates**, one native episode each.
- Use MobileGym RL per-difficulty step budgets 15/30/45/60 (+10 for grounded
  answer-sheet tasks); never cap all L3/L4 tasks at 30 steps.
- Fixed seed 11, screenshot + text/action-history input only, 30-action context.
- Single GPU device cuda:0 BF16, environment_workers=4, serial
  HuggingFace generation; no dynamic batching, no unmeasured speedup claim.
- A full native held-out evaluation uses the same parser/prompt and budgets.
  Official MobileGym GRPO uses a DIFFERENT GenericAgentV2 history format, so
  action/prompt agreement must be addressed before calling it compute matched.
- Store all failures and negative verdicts. No training on test tasks,
  no checkpoint selection using official test scores.
- Formal comparison must match rollout attempts, screenshots/vision tokens,
  optimiser updates, task instance IDs and evaluation stop rules; report
  exceptions and GPU-hours rather than calling unmatched arms equivalent.

## Run the standalone V1 pipeline

Install the existing pinned MobileGym source and GPU requirements according to
docs/MOBILEGYM_FEEDBACK_LEARNED_ZH.md or the native Windows guide.
Use one empty new output root. Start the MobileGym preview service first.
From the GUIAGENT repository root (PowerShell or Linux shell):

    python -m pip install -e . --no-deps
    python -m pip install -r requirements/android.txt
    python -m pytest tests/test_mobilegym_rolling.py -q

    python scripts/run_mobilegym_verified_rolling.py --config configs/mobilegym/verified_rolling_4b_97gb.json plan
    python scripts/run_mobilegym_verified_rolling.py --config configs/mobilegym/verified_rolling_4b_97gb.json collect
    python scripts/run_mobilegym_verified_rolling.py --config configs/mobilegym/verified_rolling_4b_97gb.json prepare
    python scripts/run_mobilegym_verified_rolling.py --config configs/mobilegym/verified_rolling_4b_97gb.json train
    python scripts/run_mobilegym_verified_rolling.py --config configs/mobilegym/verified_rolling_4b_97gb.json eval

Separate inspection (does not start inference):

    python scripts/run_mobilegym_verified_rolling.py --config configs/mobilegym/verified_rolling_4b_97gb.json report --episodes runs/mobilegym/verified_rolling_97gb_v1/heldout_eval

Outputs (never overwrite): train_episodes/ (original screenshots, steps and
per-step verifier snapshots); train_verified_rolling.jsonl and its receipt;
checkpoint/ with rolling_training_receipt.json; heldout_eval/ with all episodes
and verified_rolling_eval_summary.json.

**Existing 2026-11-14 MobileGym episode files lack interim verification**, so
they cannot serve as verified-step targets. The builder fails with
signal_shortage instead of inventing them. If no new train windows gain progress,
the method is NOT trainable under the current evidence gate; diagnose the
native policy, task state checks and the progress distribution.

## SOTA baseline integration contract

The machine-readable baseline manifest is
configs/mobilegym/sota_baselines_20261123.json. It distinguishes available
source from faithful reproduction and adapted or pending algorithms.

| Arm | Source | What can be executed now? |
|---|---|---|
| Native 4B | this repo | Existing original policy evaluator |
| Feedback SD | this repo | Existing online loop (different data receipt) |
| MobileGym GRPO | Purewhiter/mobilegym mobilegym-rl | External training script, 2-GPU default, single-97GB port unqualified |
| MobileRL (AdaGRPO) | THUDM/MobileRL | Evaluation/model released; training adaptation NOT implemented |
| K2-Agent (C-GRPO) | k2-agent/k2-agent | AndroidWorld result records; training code NOT released |
| GUI-SD OPSD | zhangyan-ucas/GUI-SD-code | External 8-GPU grounding code; multistep/1-GPU port unqualified |
| VRAD V1 | this branch | Full collect->prepare->train->eval code; GPU run pending |
| Latent world ablation | planned | Not implemented |

Comparison metrics: SR, PR, FC, OT, USE, L1-L4 SR, steps/task, action/s,
GPU peak VRAM, full training time and environment interactions. For future
world-model work report transition retrieval and decision value separately.

**Do not copy MobileRL/K2 reported AndroidWorld percentages into a MobileGym
comparison table; same 4B and same test tasks are required.**

## Design limitations and next changes

1. Verify interim judge correctness on task classes and collect step
   verdict coverage. A state verifier on an unfinished task is a progress
   signal, not official terminal pass/fail.
2. Future predictions use semantic ACTION TYPES, not exact future coordinates.
   A real parallel MTP multi-head plus spatial action/region distribution
   requires separate implementation and matched ablation.
3. Teacher may not improve when future progress is weak or hindsight is
   ambiguous. Measure student-teacher KL/gating coverage before an improvement
   claim. A small CE component stabilizes typed future label supervision.
4. No replay of four counterfactual branches from a frozen state in V1.
   True action-value comparison needs independently executed matched branches.
5. One optimiser schedule, one seed; do not claim paper-level robustness or
   accuracy superiority until actual full experiments and baselines finish.

External links:
- https://github.com/Purewhiter/mobilegym
- https://github.com/THUDM/MobileRL
- https://github.com/k2-agent/k2-agent
- https://github.com/zhangyan-ucas/GUI-SD-code
- https://github.com/bcwang-sjtu/GUI-JEPA
