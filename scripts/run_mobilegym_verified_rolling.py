#!/usr/bin/env python3
"""Opt-in MobileGym verified rolling collection, data, training, evaluation.

All commands are explicit; importing this module never starts a GUI or GPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _episodes_summary(episodes: list[dict]) -> dict:
    total = len(episodes)
    if not total:
        raise ValueError("No completed evaluation episodes")
    success = sum(bool(e.get("success")) for e in episodes)
    progress = [
        float(e["judge"]["progress"])
        for e in episodes
        if isinstance(e.get("judge"), dict)
        and isinstance(e["judge"].get("progress"), (float, int))
    ]
    errors = sum(e.get("status") != "complete" for e in episodes)
    false_complete = sum(
        not e.get("success") and e.get("steps")
        and isinstance(e["steps"][-1].get("action"), dict)
        and e["steps"][-1]["action"].get("action_type") == "COMPLETE"
        for e in episodes
    )
    side_effects = sum(
        isinstance(e.get("judge"), dict)
        and not e["judge"].get("clean", True)
        for e in episodes
    )
    return {
        "test_tasks": total, "success": success,
        "sr": success / total,
        "progress_mean": sum(progress) / len(progress) if progress else None,
        "progress_covered": len(progress),
        "false_complete_count": false_complete,
        "unexpected_side_effect_episode_count": side_effects,
        "incomplete": errors,
        "total_steps": sum(len(e.get("steps", [])) for e in episodes),
        "note": "Task progress and native completion are distinct; no test feedback enters learning",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/mobilegym/verified_rolling_4b_97gb.json")
    subs = ap.add_subparsers(dest="command", required=True)
    subs.add_parser("plan")
    subs.add_parser("collect")
    prep = subs.add_parser("prepare")
    prep.add_argument("--episodes", default=None, help="Training-only collection directory")
    prep.add_argument("--out", default=None, help="New train-only examples JSONL")
    subs.add_parser("train")
    ev = subs.add_parser("eval")
    ev.add_argument("--checkpoint", default=None, help="Trained checkpoint directory")
    ev.add_argument("--out", default=None, help="New held-out output directory")
    report = subs.add_parser("report")
    report.add_argument("--episodes", required=True, help="Completed evaluation directory")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    rolling = cfg.get("rolling") or {}
    if args.command == "plan":
        from improving.mobilegym_feedback import plan
        payload = plan(cfg)
    elif args.command == "collect":
        from improving.mobilegym_feedback import plan, _tasks, _collect
        from improving.android.model import QwenPolicy
        if rolling.get("capture_step_verifier") is not True:
            raise ValueError("collect requires rolling.capture_step_verifier=true")
        plan(cfg)
        checkpoint = str(Path(cfg["model"]["checkpoint"]).expanduser().resolve())
        out = Path(rolling["episodes_dir"]).expanduser().resolve()
        if out.exists():
            raise FileExistsError(f"Refusing to overwrite training episodes: {out}")
        policy = QwenPolicy(checkpoint, cfg)
        try:
            episodes = _collect(cfg, checkpoint, _tasks(cfg, "train"), out,
                                cfg["study"]["round0_repeats"], "native",
                                policy, r=0)
        finally:
            policy.close()
        payload = {
            "status": "collected", "split": "train",
            "episodes": len(episodes), "output": str(out),
            "verified_steps": sum(
                s.get("verification", {}).get("status") == "verified"
                for ep in episodes for s in ep["steps"]),
        }
    elif args.command == "prepare":
        from improving.mobilegym_rolling import RollingSettings, prepare_file
        settings = RollingSettings(
            horizon=int(rolling.get("horizon", 3)),
            verify_lookahead=int(rolling.get("verify_lookahead", 8)),
            min_progress_gain=float(rolling.get("min_progress_gain", .001)),
            future_loss_weight=float(rolling.get("future_loss_weight", .3)),
            require_clean=bool(rolling.get("require_clean", True)),
        )
        payload = prepare_file(
            args.episodes or rolling["episodes_dir"],
            args.out or rolling["examples_path"], settings,
        )
    elif args.command == "train":
        from improving.mobilegym_rolling_train import train_rolling
        payload = train_rolling(cfg, rolling["examples_path"])
    elif args.command == "eval":
        from improving.mobilegym_feedback import plan, _tasks, _collect
        from improving.android.model import QwenPolicy
        plan(cfg)
        checkpoint = str(Path(args.checkpoint or cfg["output_dir"]).expanduser().resolve())
        out = Path(args.out or rolling["eval_dir"]).expanduser().resolve()
        if out.exists():
            raise FileExistsError(f"Refusing to overwrite held-out evaluation: {out}")
        eval_cfg = json.loads(json.dumps(cfg))
        eval_cfg["model"]["checkpoint"] = checkpoint
        eval_cfg["rolling"]["capture_step_verifier"] = False
        policy = QwenPolicy(checkpoint, eval_cfg)
        try:
            episodes = _collect(eval_cfg, checkpoint, _tasks(eval_cfg, "test"), out,
                                1, "eval", policy, r=0)
        finally:
            policy.close()
        payload = _episodes_summary(episodes)
        payload["output"] = str(out)
        (out / "verified_rolling_eval_summary.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        root = Path(args.episodes).expanduser().resolve()
        files = sorted(root.rglob("episode.json"))
        episodes = [json.loads(p.read_text(encoding="utf-8")) for p in files]
        if any(e.get("split") != "eval" for e in episodes):
            raise ValueError("Report expects held-out eval episodes only")
        payload = _episodes_summary(episodes)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
