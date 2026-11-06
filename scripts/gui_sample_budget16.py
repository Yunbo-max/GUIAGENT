#!/usr/bin/env python3
"""Offline plan/audit for the 16-sample GUI self-improvement protocol.

No CUDA, emulator, browser or benchmark dependencies are required. Counts are
requested episodes, *not* measured completed trajectories or wall-clock time.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("feedback_sd", "feedback_spectrum_fixed", "feedback_spectrum_learned")
CONFIGS = {
    "mobilegym_full": "configs/mobilegym/feedback_only_qwen4b_parallel4_samples16.json",
    "mobilegym_pilot": "configs/mobilegym/feedback_only_qwen4b_parallel4_samples16_pilot.json",
    "androidworld_full": "configs/android/androidworld_feedback_learned_windows_full_parallel2_samples16.json",
    "androidworld_pilot": "configs/android/androidworld_feedback_learned_windows_pilot_parallel2_samples16.json",
}
BASELINES = {
    "mobilegym_full": "configs/mobilegym/feedback_only_qwen4b_parallel4.json",
    "androidworld_full": "configs/android/androidworld_feedback_learned_windows_full_parallel2.json",
    "androidworld_pilot": "configs/android/androidworld_feedback_learned_windows_pilot_parallel2.json",
}


def config(label: str, root: Path = ROOT) -> dict:
    return json.loads((root / CONFIGS[label]).read_text(encoding="utf-8"))


def training_instances(cfg: dict, benchmark: str) -> int:
    n = cfg["study"]["train_task_limit"]
    if benchmark == "androidworld":
        n *= cfg["data"]["split_counts"]["train"]
    return n


def plan(label: str, *, root: Path = ROOT) -> dict:
    if label not in CONFIGS:
        raise ValueError(f"Unsupported 16-sample benchmark protocol {label}")
    cfg = config(label, root)
    benchmark = label.split("_")[0]
    study = cfg["study"]
    if study.get("round0_repeats") != 16 or study.get("group_size") != 16:
        raise ValueError(f"{label}: Round 0 and per-round total must both be 16")
    if study["arms"] != list(ARMS):
        raise ValueError(f"{label}: must include three matched baseline/method arms")
    if cfg["model"]["repo_id"] != "Qwen/Qwen3-VL-4B-Instruct":
        raise ValueError("Model architecture changed outside sampling study")
    if cfg["spectral"].get("geometry_source") != "feedback_only":
        raise ValueError("No fixed supervised reference answers in this protocol")
    workers = (cfg["mobilegym"] if benchmark == "mobilegym" else cfg["android"]).get("environment_workers", 1)
    if workers != (4 if benchmark == "mobilegym" else 2):
        raise ValueError(f"{label}: intended independent environment concurrency changed")
    if benchmark == "androidworld" and (cfg["data"].get("eval_tasks") != "all"):
        raise ValueError("AndroidWorld must retain full task registry eval")
    n = training_instances(cfg, benchmark)
    r = study["rounds"]
    eval_n = study["test_task_limit"]
    per_round = {}
    for arm in ARMS:
        native = 16 if arm == "feedback_sd" else 8
        spectral = 0 if arm == "feedback_sd" else 8
        per_round[arm] = {"native_per_instance": native,
                          "spectral_per_instance": spectral,
                          "training_episodes_per_round": n * (native + spectral),
                          "native_training_episodes_per_round": n * native,
                          "spectral_training_episodes_per_round": n * spectral,
                          "native_eval_episodes_per_round": eval_n}
    round0 = n * study["round0_repeats"]
    train = round0 + n * study["group_size"] * len(ARMS) * r
    eval_count = eval_n * len(ARMS) * r
    result = {
        "label": label,
        "benchmark": benchmark,
        "output_dir": cfg["output_dir"],
        "model_checkpoint": cfg["model"]["checkpoint"],
        "training_instances": n,
        "rounds": r,
        "round0": {
            "checkpoint": "shared_original_theta0",
            "native_per_instance": 16,
            "spectral_per_instance": 0,
            "native_episodes": round0,
            "training_updates": 0
        },
        "per_round": per_round,
        "all_training_episodes": train,
        "all_eval_episodes": eval_count,
        "all_requested_episodes": train + eval_count,
        "eval_success_feedback_used_for_training": False,
        "native_checkpoint_per_round": "current_post_distillation_checkpoint_theta_r",
        "round1_native_repeat_of_round0": False,
        "notes": [
            "Round 1 uses theta0 but collects FRESH native trajectories, not a resampled or silently copied Round 0 record.",
            "Round 2+ native sampling uses each arm's own latest merged student model theta_r.",
            "Stage-4 success paths/errors supply hindsight context; spectral actions are not direct SFT labels.",
            "More rollout attempts do not increase fixed optimizer max_updates, fit/guard examples or LoRA batch size.",
            "The 16-sample train-rollout budget is 4x the original 4-sample budget; not an ETA or expected 4x improvement.",
        ]
    }
    return result


def compare_only_sampling_changes(root: Path = ROOT) -> None:
    """Ensure full configs differ from original study ONLY in rollout counts/results."""
    for label, baseline in BASELINES.items():
        new = config(label, root)
        old = json.loads((root / baseline).read_text(encoding="utf-8"))
        expected = deepcopy(old)
        expected["output_dir"] = new["output_dir"]
        expected["study"]["group_size"] = 16
        expected["study"]["round0_repeats"] = 16
        if expected != new:
            raise ValueError(f"{label}: non-sampling methodology or model settings were modified")
        if old["output_dir"] == new["output_dir"]:
            raise ValueError(f"{label}: must have a separate output directory")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", choices=list(CONFIGS) + ["all"], default="all")
    args = p.parse_args()
    compare_only_sampling_changes()
    names = CONFIGS if args.benchmark == "all" else (args.benchmark,)
    print(json.dumps({name: plan(name) for name in names},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
