#!/usr/bin/env python3
"""Paired MobileGym evaluation comparison; refuses unmatched task instances.

Use for Native, Feedback SD, GRPO or VRAD only AFTER each was evaluated with
the same test instances. Different AndroidWorld benchmark scores are excluded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_eval(root: Path) -> dict[str, dict]:
    files = sorted(Path(root).rglob("episode.json"))
    if not files:
        raise FileNotFoundError(f"No episode.json records under {root}")
    rows = {}
    for file in files:
        ep = json.loads(file.read_text(encoding="utf-8"))
        if ep.get("split") != "eval" or ep.get("status") != "complete":
            raise ValueError(f"Not a completed MobileGym evaluation episode: {file}")
        task = ep["task_name"]
        if task in rows:
            raise ValueError(f"Repeated evaluation task: {task}")
        if not isinstance(ep.get("success"), bool):
            raise ValueError(f"No verified success boolean: {file}")
        if not isinstance(ep.get("judge"), dict):
            raise ValueError(f"No native judge: {file}")
        rows[task] = ep
    return rows


def compare(a: dict[str, dict], b: dict[str, dict], *, require_full: bool = True) -> dict:
    if set(a) != set(b):
        raise ValueError("The two evaluations have different test task sets")
    if require_full and len(a) != 256:
        raise ValueError("Formal comparison requires all 256 MobileGym test tasks")
    ids = sorted(a)
    for task in ids:
        if a[task].get("instance_id") != b[task].get("instance_id"):
            raise ValueError(f"Different parameters/instance for task {task}")
        if (a[task].get("max_steps_effective") is not None
                and b[task].get("max_steps_effective") is not None
                and a[task]["max_steps_effective"] != b[task]["max_steps_effective"]):
            raise ValueError(f"Different step budget for task {task}")
    n = len(ids)
    wins = sum(not a[t]["success"] and b[t]["success"] for t in ids)
    losses = sum(a[t]["success"] and not b[t]["success"] for t in ids)
    successes_a = sum(a[t]["success"] for t in ids)
    successes_b = sum(b[t]["success"] for t in ids)
    prog_a, prog_b = [], []
    for task in ids:
        pa = a[task]["judge"].get("progress")
        pb = b[task]["judge"].get("progress")
        if type(pa) not in (float, int) or type(pb) not in (float, int):
            raise ValueError(f"Missing progress for {task}")
        prog_a.append(float(pa))
        prog_b.append(float(pb))
    return {
        "tasks": n,
        "baseline_sr": successes_a / n,
        "candidate_sr": successes_b / n,
        "sr_delta_pp": (successes_b - successes_a) * 100 / n,
        "baseline_pr": sum(prog_a) / n,
        "candidate_pr": sum(prog_b) / n,
        "pr_delta_pp": (sum(prog_b) - sum(prog_a)) * 100 / n,
        "candidate_only_successes": wins,
        "baseline_only_successes": losses,
        "paired_disagreements": wins + losses,
        "note": "Matched task instances and recorded budgets only. External prompt, model and compute equivalence requires separate receipts.",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    result = compare(load_eval(args.baseline), load_eval(args.candidate),
                     require_full=not args.allow_partial)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
