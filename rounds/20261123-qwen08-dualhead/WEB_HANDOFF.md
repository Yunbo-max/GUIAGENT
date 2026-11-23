# Qwen0.8 GUI dual-head handoff

**Status: generated_unexecuted.** Selected task: migrate GUI-JEPA's AndroidControl
candidate interface to Qwen3.5-0.8B, then retain the original LM head for planning
and combine it with the trained decision head through explicit prompt routing.
No model, training, native evaluation or project tests were executed by Web.
Actual static checks and pending acceptance are recorded in
[SOURCE_REVIEW.md](SOURCE_REVIEW.md).

## Local Codex: start here

Read [AGENTS](../../AGENTS.md), then the scoped
[Local guide](../../docs/GUI_DECISION_QWEN08_ZH.md), including
[downloads](../../docs/GUI_DECISION_QWEN08_ZH.md#4-download-datasets-and-models) and
[calibration/commands](../../docs/GUI_DECISION_QWEN08_ZH.md#5-calibrate-debug-then-dispatch).
Use the exact delivered commit from the later publication receipt. A file does not
contain its own future commit SHA. Preserve unrelated historical experiments.

Delivery target: existing **Yunbo-max/GUIAGENT**, `main`, additive scoped paths.
The current research-autopilot repository policy delivers new Web code to main;
the old Android branch-only entry applies to that historical study, whose source
and results are preserved. Parent inspected: `0f9a3c67dab7ef6aa3eaf58a79af6095a4668163`.
No new repository, visibility change, HF output target or training job is created.

## Source and experiment mapping

| Construction | Source | Check / expected evidence |
|---|---|---|
| AndroidControl PNG + current-a11y candidates | `androidcontrol.py`, `actions.py`, `prepare.py`, `dataset.py` | official split coverage, no current/future GT input, JSONL hashes and failure manifest |
| One frozen Qwen + original LM head | `model.QwenDecisionModel` | native exact model loading, only one wrapper, frozen backbone/LM, ACT avoids vocabulary logits |
| Variable-N decision output | `model.CandidateScoringHead`, `processor.py`, `spatial.py` | correct merged rectangular grid, offset alignment, masks, gradients, complete candidate set |
| Head pretraining | `runner.extract/train`, `cache.py`, `checkpoint.py` | full train, val-only selection, model/processor/data/checkpoint binding, complete resume state |
| Prompt-gated composition | `router.PromptGatedPolicy`, `runner.evaluate_online` | PLAN input ignores oracle instruction; generated plan enters ACT; same weights, two prefills |
| Native step evaluation | `arguments.py`, `evaluation.py` | parameters resolved from predicted type, no missing-candidate drops, correct conditional vs overall denominators |

All Python modules above are under `src/improving/gui_decision/`; CLI is
[`scripts/gui_decision.py`](../../scripts/gui_decision.py), settings and immutable
input pins are under [`configs/gui_decision`](../../configs/gui_decision/).
Authored acceptance fixtures are under [`tests/gui_decision`](../../tests/gui_decision/).

## Primary sources inspected

- [GUI-JEPA source](https://github.com/bcwang-sjtu/GUI-JEPA/tree/dae79af4598962b9a5112b0bef297b588592192d):
  actions, raw data reader, candidate text export, policy training/judge and parameter ranker.
  Actual commit is `dae79af4598962b9a5112b0bef297b588592192d`; its tree is
  `914b405017a061cb654607122124702f18ded5ce`, a distinct identity.
- [Released GUI-JEPA model bundle](https://huggingface.co/LilCC/GUI-JEPA/tree/a2265ca26dbea66fc59c46240a90d3be5defdb9e):
  frozen span ranker and metadata; the original predictor/ViT/cache/click reranker are not loaded.
- [Qwen official configuration](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/config.json)
  and processor/template at the same pinned revision.
- Transformers **5.3.0** installed source: Qwen3.5 base/conditional-generation forward,
  Qwen3VLProcessor, image preprocessing/merger ordering and tokenizer offset API.
- Official AndroidControl GCS splits and object metadata; the 22 pinned generations,
  sizes and MD5s are retained in `configs/gui_decision/assets.json`.
- [PyTorch official version installation commands](https://pytorch.org/get-started/previous-versions/)
  and PyPI release metadata for the independent candidate environment.

MIT attribution and algorithm differences are in
[third_party/gui_jepa](../../third_party/gui_jepa/README.md).

## Claims and pending acceptance

This is an implementation/migration of a user-selected construction. It is not
a newly validated paper method or a learned gate. Gate kind is `explicit_prompt`.
The first stage trains only the candidate scorer; the original planning head remains frozen.
Output space selection alone does not demonstrate improved reasoning or planning.

Pending Local: dependency/environment installation, authored fixtures, real Qwen
checkpoint/processor forward, resource measurements on the actual RTX 2060,
complete data preparation, training convergence, Low/High evaluation, self-planned
High evaluation, and original Qwen coordinate baseline. CPU fixtures cannot close
the native GPU/data obligations. No scores, speedups or VRAM fit are asserted.

Scientific comparison uses the same trained head for Low ACT, High ACT and High
PLAN_ACT. Complete full test is primary; optional IDD/OOD keeps original membership.
Native scorer is offline step accuracy with recorded history, not emulator task success.
Do not evaluate PLAN_ACT with GT Low instructions or use test to tune the planner.

## Return

Use the scoped guide's actual commands and finite diagnosis policy. Resolve real
host paths, live resources and remaining authority before native dispatch; this
source delivery itself launches no jobs. Keep raw predictions/plans, all failures,
metrics/denominators, receipts, effective configs and actual execution revision.
Return a verified repo/commit/path locator for a small readable packet under this
round's `returns/` directory. Large assets and private seed stay off public Git.
