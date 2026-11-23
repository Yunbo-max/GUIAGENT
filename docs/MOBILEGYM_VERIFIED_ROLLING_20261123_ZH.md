# MobileGym: Verified Rolling Action Distillation (VRAD) — 2026-11-23

**状态：已提交代码，CPU 合约测试通过；GPU 训练和 MobileGym 端到端正式运行尚未验证。**

## 中文说明：我们现在真正要跑什么

研究主线是 **Verified Rolling Action Distillation（VRAD）**，即在 MobileGym 训练任务中执行真实动作、读取每一步程序化 Verifier 的进度变化，再把真实执行后的截图、动作与 Verifier 进度作为冻结 Teacher 的特权上下文，通过 Soft-KL 蒸馏到只看当前截图的 Student。

- 输入与推理：Qwen3-VL-4B 输入当前截图、任务目标、历史动作；一次只执行一个可操作 Action，然后获取真实下一张截图。
- 训练数据：160 个正式训练模板 × 每任务 8 条 Native 轨迹，计划采集 1280 条完整训练尝试。训练过程绝不使用 256 个测试任务的 Verifier。
- 第一版软监督：当前动作保留完整 Action JSON；未来步只预测 Action Type（不伪造未来页面的坐标）。本版不是原生多预测头 MTP，也不训练 World Model。
- 训练：单张约 97GB GPU，Qwen3-VL-4B BF16 + LoRA，Freeze Teacher、Reverse KL，未来 Action Type 辅以少量 CE；不用 GRPO。
- 评测：256 个测试任务，全量 SR/PR/FC/USE、步数和资源记录。仓库提供按相同 Task Instance 配对的对比脚本。
- SOTA 对照：MobileGym 官方 GRPO 有开源训练代码；MobileRL AdaGRPO 和 K²-Agent C-GRPO 仍需适配，不能把 AndroidWorld 成绩视作我们的 MobileGym 重跑结果。

### Windows / Linux 运行指令

```bash
python -m pytest tests/test_mobilegym_rolling.py tests/test_mobilegym_eval_compare.py -q
python scripts/run_mobilegym_verified_rolling.py --config configs/mobilegym/verified_rolling_4b_97gb.json plan
python scripts/run_mobilegym_verified_rolling.py --config configs/mobilegym/verified_rolling_4b_97gb.json collect
python scripts/run_mobilegym_verified_rolling.py --config configs/mobilegym/verified_rolling_4b_97gb.json prepare
python scripts/run_mobilegym_verified_rolling.py --config configs/mobilegym/verified_rolling_4b_97gb.json train
python scripts/run_mobilegym_verified_rolling.py --config configs/mobilegym/verified_rolling_4b_97gb.json eval
```

要做 Next-Action-only 消融，复用同一训练轨迹，只把配置改成 `configs/mobilegym/verified_next_action_ablation_4b_97gb.json`，执行 prepare、train、eval，不需要重复 collect。

正式对照的命令：

```bash
python scripts/compare_mobilegym_evals.py --baseline /path/to/native_eval --candidate /path/to/verified_rolling_eval
```

---

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
The collector also writes each post-action screenshot (as a training artifact),
so the teacher may condition on the actual future screenshot while the student
never receives it.

The privileged frozen teacher sees verified hindsight (executed action types
and observed progress), including actual post-action screenshots (if present), while the student sees the original screenshot/history
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
- All **160 official train templates**, each with 8 native attempts: requested
  1,280 train episodes (actual success and action lengths are measured).
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
