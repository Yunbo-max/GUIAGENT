"""AndroidWorld: verified-feedback full-spectrum learning on real GUI trajectories.

Same comparison as the MobileGym feedback-only study, but uses the *native*
AndroidWorld Pixel6/API33 emulator and official task.is_successful verifier.

This is a separate opt-in protocol, not the legacy UI-Voyager/GRSD experiment.
Only training rollouts build feedback/spectrum; evaluation traces are never used
for student updates. No RFT warmup, no fixed reference-answer calibration, no
GRPO, no external model/teacher. No native Windows GPU run is claimed here.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


ARMS = ("feedback_sd", "feedback_spectrum_fixed", "feedback_spectrum_learned")
PROTOCOL = "androidworld-native-feedback-learned-v1"


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False,
        separators=(",", ":")).encode()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _load_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing complete native rollout evidence: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _verify_episode_population(episodes: list[dict], instances: list[dict],
                               repeats: int, split: str) -> None:
    expected = {i["instance_id"] for i in instances}
    if len(episodes) != len(instances) * repeats:
        raise ValueError("Incomplete emulator rollout population")
    counts = {key: 0 for key in expected}
    for ep in episodes:
        if ep.get("split") != split or ep.get("instance_id") not in counts:
            raise ValueError("AndroidWorld split/instance identity mismatch")
        if type(ep.get("success")) is not bool or ep.get("status") != "complete":
            raise ValueError("Only completed native-verifier outcomes may enter a study")
        counts[ep["instance_id"]] += 1
    if any(count != repeats for count in counts.values()):
        raise ValueError("Some frozen task instances lack complete repeats")


def _config(path: str | Path) -> dict:
    from improving.android.study import load_config
    return load_config(path)


def validate(cfg: dict) -> dict:
    if cfg.get("protocol") != PROTOCOL:
        raise ValueError(f"Expected {PROTOCOL}")
    for name in ("model", "android", "data", "study", "spectral", "gain_learning"):
        if not isinstance(cfg.get(name), dict):
            raise ValueError(f"Missing {name} section")
    study = cfg["study"]
    for name in ("seed", "train_task_limit", "test_task_limit",
                 "round0_repeats", "group_size", "rounds"):
        v = study.get(name)
        if type(v) is not int or v < (0 if name == "seed" else 1):
            raise ValueError(f"study.{name} must be a positive integer (seed >=0)")
    if study["round0_repeats"] != study["group_size"] or study["group_size"] % 2:
        raise ValueError("Require equal, even native exploration budget across study arms")
    if study.get("arms") != list(ARMS):
        raise ValueError("Use exactly the matched Feedback SD / Fixed / Learned arms")
    if cfg["spectral"].get("geometry_source") != "feedback_only":
        raise ValueError("No independent calibration answers in this protocol")
    if not cfg["model"].get("checkpoint"):
        raise ValueError("Use one explicit local Qwen3-VL-4B checkpoint for both benchmarks")
    if cfg["model"].get("device") != "cuda:0" or cfg["model"].get("dtype") != "bfloat16":
        raise ValueError("Default qualified protocol uses native single-GPU BF16 CUDA")
    if cfg["model"].get("repo_id") != "Qwen/Qwen3-VL-4B-Instruct":
        raise ValueError("Do not silently change the shared multimodal backbone")
    if cfg.get("data", {}).get("eval_tasks") != "all":
        raise ValueError("Keep full AndroidWorld native eval registry in the manifest")
    from improving.android.tau import validate_tau_settings
    from improving.learned_gains import validate_gain_settings
    validate_tau_settings(cfg["spectral"])
    validate_gain_settings(cfg["gain_learning"])
    return cfg


def _inputs(cfg: dict) -> tuple[dict, list[dict], list[dict]]:
    from improving.android.study import load_manifest
    from improving.android.assets import verify_upstream
    verify_upstream(cfg["android"]["upstream_root"])
    manifest = load_manifest(cfg["manifest"])
    train = [x for x in manifest["instances"] if x["split"] == "train"]
    evaluation = [x for x in manifest["instances"] if x["split"] == "eval"]
    train = train[:cfg["study"]["train_task_limit"] * cfg["data"]["split_counts"]["train"]]
    if len(train) != cfg["study"]["train_task_limit"] * cfg["data"]["split_counts"]["train"]:
        raise ValueError("Not enough prepared AndroidWorld train instances")
    evaluation = evaluation[:cfg["study"]["test_task_limit"]]
    if len(evaluation) != cfg["study"]["test_task_limit"]:
        raise ValueError("Not enough prepared AndroidWorld evaluation instances")
    if {x["instance_id"] for x in train} & {x["instance_id"] for x in evaluation}:
        raise ValueError("Training/evaluation identity overlap")
    return manifest, train, evaluation


def plan(cfg: dict) -> dict:
    validate(cfg)
    root = Path(cfg["android"]["upstream_root"])
    source_ready = root.joinpath("androidworld", "android_world", "registry.py").is_file()
    manifest_ready = Path(cfg["manifest"]).is_file()
    checkpoint = Path(cfg["model"]["checkpoint"]).expanduser()
    info: dict = {
        "protocol": PROTOCOL, "benchmark": "AndroidWorld", "host": "native Windows or POSIX",
        "output_dir": cfg["output_dir"], "source_ready": source_ready,
        "manifest_ready": manifest_ready,
        "checkpoint_ready": (checkpoint / "config.json").is_file(),
        "model": cfg["model"]["repo_id"], "arms": list(ARMS),
        "no_rft_warmup": True, "calibration_used_for_learning": False,
        "geometry": "feedback_only", "native_reward": "terminated_and_native_score_equals_one",
        "train_task_templates": cfg["study"]["train_task_limit"],
        "test_task_instances": cfg["study"]["test_task_limit"],
        "refinement_rounds": cfg["study"]["rounds"],
        "additional_calibration_reference_budget": 0,
    }
    if source_ready and manifest_ready:
        manifest, train, test = _inputs(cfg)
        info.update(
            frozen_manifest_sha256=_hash(manifest),
            train_instance_ids=[row["instance_id"] for row in train],
            test_instance_ids=[row["instance_id"] for row in test],
            shared_round0_rollouts=len(train) * cfg["study"]["round0_repeats"],
            refinement_rollouts_per_arm=len(train)*cfg["study"]["group_size"]*cfg["study"]["rounds"],
            native_eval_rollouts_per_arm=len(test)*cfg["study"]["rounds"],
        )
    return info


def _collection(cfg, checkpoint: str, instances: list[dict], path: Path,
                *, repeats: int, policy_name: str, policy,
                operators: dict | None = None, round_index: int = 0) -> list[dict]:
    from contextlib import nullcontext
    from improving.android.environment import collect_episodes
    from improving.spectral import folded_operators
    import torch

    if path.exists():
        raise FileExistsError(f"Existing AndroidWorld collection: {path}")
    collection_cfg = deepcopy(cfg)
    collection_cfg["collection"] = {"repeats": repeats}
    seed = int(_hash([cfg["study"]["seed"], checkpoint,
                      policy_name, round_index, str(path)])[:15], 16)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with (folded_operators(policy.model, operators)
          if operators is not None else nullcontext()):
        episodes = collect_episodes(
            instances, policy.generate, path, collection_cfg, policy=policy_name,
            checkpoint=checkpoint, round_index=round_index)
    split = instances[0]["split"]
    _verify_episode_population(episodes, instances, repeats, split)
    return episodes


def _feedback(episodes: list[dict], cfg: dict) -> tuple[list[dict], dict]:
    from improving.android.data import feedback_examples
    examples, evidence = feedback_examples(episodes, cfg)
    if not examples:
        raise RuntimeError(
            "signal_shortage: no verified same-instance successful paths or "
            "genuine action errors on native train prefixes; no fabricated labels")
    return examples, evidence


def _fit_and_guard(feedback: list[dict]):
    from improving.mobilegym_visual_gains import split_by_task
    # Same split is used for fixed spectrum and Learned, avoiding using guard
    # tasks for K/V moment construction. The guard is train-side only.
    fit, guard = split_by_task(feedback, fit_limit=8, guard_limit=4)
    if {x["instance_id"] for x in fit} & {x["instance_id"] for x in guard}:
        raise AssertionError("Learned guard contaminated the geometry fit")
    return fit, guard


def _settings(cfg: dict, arm: str, r: int) -> dict:
    clone = deepcopy(cfg)
    clone["runtime_seed"] = int(_hash([cfg["study"]["seed"], arm, r])[:15], 16)
    if arm == "feedback_spectrum_learned":
        clone["spectral"]["tau"] = 0.0  # covariance only; learned operator replaces I
    return clone


def run(cfg: dict, *, resume: bool = False) -> dict:
    from improving.android.model import QwenPolicy
    from improving.android.geometry import collect_operators
    from improving.android.training import train_feedback
    from improving.mobilegym_visual_gains import fit_visual_feedback_gains
    from improving.android.assets import file_sha256
    from improving.android.portable_lock import exclusive_run_lock
    import torch

    validate(cfg)
    manifest, train, test = _inputs(cfg)
    checkpoint = str(Path(cfg["model"]["checkpoint"]).resolve())
    if not (Path(checkpoint) / "config.json").is_file():
        raise FileNotFoundError(f"Missing Qwen3-VL-4B checkpoint: {checkpoint}")
    root = Path(cfg["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    # Outside the output tree, do not accidentally include a changing lock file
    # in the study identity.
    with exclusive_run_lock(root / ".run.lock"):
        identity = {
            "protocol": PROTOCOL, "manifest_sha256": _hash(manifest),
            "config_sha256": _hash(cfg), "checkpoint": checkpoint,
            "model_config_sha256": file_sha256(Path(checkpoint) / "config.json"),
            "code_version": "androidworld_feedback_v1",
            "plan": plan(cfg),
        }
        manifest_path = root / "study.json"
        if manifest_path.exists():
            if not resume or json.loads(manifest_path.read_text()) != identity:
                raise ValueError("Frozen study exists or identity changed; use --resume or new output_dir")
        else:
            if resume or any(root.iterdir()) and set(p.name for p in root.iterdir()) != {".run.lock"}:
                raise ValueError("Existing unsealed experiment directory")
            _write_json(manifest_path, identity)

        round0_path = root / "round0_native"
        if (round0_path / "episodes.jsonl").is_file():
            round0 = _load_rows(round0_path / "episodes.jsonl")
            _verify_episode_population(round0, train, cfg["study"]["round0_repeats"], "train")
        else:
            if round0_path.exists():
                raise RuntimeError("Incomplete Round 0 collection; do not silently regenerate history")
            initial = QwenPolicy(checkpoint, cfg)
            try:
                round0 = _collection(
                    cfg, checkpoint, train, round0_path,
                    repeats=cfg["study"]["round0_repeats"], policy_name="native",
                    policy=initial)
            finally:
                initial.close()
        feedback, receipt = _feedback(round0, cfg)
        if not (root / "round0_feedback.json").is_file():
            _write_json(root / "round0_feedback.json", receipt)
        elif json.loads((root / "round0_feedback.json").read_text()) != receipt:
            raise ValueError("Round0 evidence changed since freezing study")

        final = {}
        for arm in ARMS:
            arm_settings = _settings(cfg, arm, 0)
            current = checkpoint
            history = []
            source_feedback = feedback
            for r in range(1, cfg["study"]["rounds"] + 1):
                stage = root / arm / f"round_{r:02d}"
                result_file = stage / "round.json"
                if result_file.is_file():
                    if not resume:
                        raise FileExistsError(f"Completed arm already exists: {result_file}")
                    entry = json.loads(result_file.read_text())
                    current = entry["checkpoint"]
                    if not (Path(current) / "config.json").is_file():
                        raise FileNotFoundError(f"Resumed model checkpoint is missing: {current}")
                    raw = _load_rows(stage / "train_native" / "episodes.jsonl")
                    if arm != "feedback_sd":
                        raw += _load_rows(stage / "train_spectral" / "episodes.jsonl")
                    source_feedback, _ = _feedback(raw, arm_settings)
                    history.append(entry)
                    continue
                if stage.exists():
                    raise RuntimeError(f"Partial AndroidWorld stage at {stage}; inspect before retrying")
                stage.mkdir(parents=True)
                settings = _settings(cfg, arm, r)
                policy = QwenPolicy(current, settings)
                operators, gain_receipt, geometry = None, None, None
                try:
                    if arm != "feedback_sd":
                        fit, guard = _fit_and_guard(source_feedback)
                        if not fit or not guard:
                            raise RuntimeError("signal_shortage: missing independent fit/guard task instances")
                        ops, geometry = collect_operators(
                            policy, [], fit, settings, stage / "geometry", round_index=r)
                        if arm == "feedback_spectrum_learned":
                            stored = torch.load(stage / "geometry" / "operators.pt",
                                                map_location="cpu", weights_only=True)
                            operators, gain_receipt = fit_visual_feedback_gains(
                                policy, source_feedback, stored["feedback"], cfg["gain_learning"])
                            torch.save({"operators": operators}, stage / "learned_operators.pt")
                            _write_json(stage / "learned_gains.json", gain_receipt)
                        else:
                            operators = ops
                    if arm == "feedback_sd":
                        episodes = _collection(
                            settings, current, train, stage / "train_native",
                            repeats=cfg["study"]["group_size"],
                            policy_name="native", policy=policy, round_index=r)
                    else:
                        half = cfg["study"]["group_size"] // 2
                        episodes = _collection(
                            settings, current, train, stage / "train_native",
                            repeats=half, policy_name="native",
                            policy=policy, round_index=r)
                        episodes += _collection(
                            settings, current, train, stage / "train_spectral",
                            repeats=half, policy_name="spectral",
                            policy=policy, operators=operators, round_index=r)
                finally:
                    policy.close()
                source_feedback, evidence = _feedback(episodes, settings)
                _write_json(stage / "feedback.json", evidence)
                output = stage / "checkpoint"
                train_receipt = train_feedback(current, source_feedback, output, settings)
                _write_json(stage / "train_receipt.json", train_receipt)
                current = str(output.resolve())
                agent = QwenPolicy(current, settings)
                try:
                    evaluated = _collection(
                        settings, current, test, stage / "evaluation",
                        repeats=1, policy_name="native",
                        policy=agent, round_index=r)
                finally:
                    agent.close()
                wins = sum(bool(row["success"]) for row in evaluated)
                entry = {
                    "arm": arm, "round": r, "checkpoint": current,
                    "train_episodes": len(episodes),
                    "train_success": sum(bool(row["success"]) for row in episodes),
                    "eval_episodes": len(evaluated), "eval_success": wins,
                    "eval_success_rate": wins / len(evaluated),
                    "feedback_examples": len(source_feedback),
                    "geometry_source": geometry["geometry_source"] if geometry else None,
                    "learned_gain_status": gain_receipt["selection"] if gain_receipt else None,
                    "learned_gain_fallback": gain_receipt["fallback_reason"] if gain_receipt else None,
                    "score_definition": "explicit_terminate_and_native_verifier_one",
                }
                _write_json(result_file, entry)
                history.append(entry)
            final[arm] = history
        result = {
            "protocol": PROTOCOL, "benchmark": "AndroidWorld",
            "round0_extra_train_episodes": len(round0),
            "arms": final,
            "warning": ("Matched rollout attempts, not matched FLOPs. AndroidWorld "
                        "native success is verified; test trajectories never become training feedback."),
            "status": "complete",
        }
        report = root / "report.json"
        if not report.exists():
            _write_json(report, result)
        elif json.loads(report.read_text()) != result:
            raise ValueError("Previous report differs; do not overwrite")
        return {"output_dir": str(root), "last_round": {a: rows[-1] for a, rows in final.items()}}


def environment_check(cfg: dict, *, setup: bool = False) -> dict:
    from improving.android.environment import _load_components, _open_env
    validate(cfg)
    _, launcher, _ = _load_components(cfg["android"]["upstream_root"])
    opts = dict(cfg["android"], emulator_setup=setup)
    env = _open_env(launcher, opts)
    try:
        image = env.get_state(wait_to_stabilize=True).pixels
        if image is None or image.size == 0:
            raise RuntimeError("AndroidWorld native screenshot unavailable")
        return {"status": "androidworld_native_environment_checked",
                "native_emulator": True, "setup_apps": setup,
                "image_shape": list(image.shape), "model_loaded": False}
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",
                        default="configs/android/androidworld_feedback_learned_windows_96gb.json")
    parser.add_argument("command", choices=(
        "plan", "fetch", "prepare", "setup-env", "env-check", "run", "report"))
    parser.add_argument("--resume", action="store_true",
                        help="Skip only completed, validated stages; never silently repair partial stages")
    args = parser.parse_args()
    cfg = _config(args.config)
    validate(cfg)
    if args.command == "plan":
        result = plan(cfg)
    elif args.command == "fetch":
        from improving.android.assets import acquire
        result = acquire(cfg, source_only=True)
    elif args.command == "prepare":
        from improving.android.environment import prepare_manifest
        from improving.android.assets import verify_upstream
        verify_upstream(cfg["android"]["upstream_root"])
        manifest = prepare_manifest(cfg["android"]["upstream_root"], cfg["manifest"], cfg)
        result = {"status": "prepared", "instance_count": len(manifest["instances"]),
                  "path": cfg["manifest"]}
    elif args.command in ("setup-env", "env-check"):
        result = environment_check(cfg, setup=args.command == "setup-env")
    elif args.command == "run":
        result = run(cfg, resume=args.resume)
    else:
        path = Path(cfg["output_dir"]) / "report.json"
        result = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
