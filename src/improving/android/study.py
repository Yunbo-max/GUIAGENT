"""One AndroidWorld study: shared RFT, UI-Voyager GRSD, feedback SPECTRUM.

Every completed stage has a receipt. Partial collection/training is kept and is
never silently retried or converted into a zero score. No GPU import for plan.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import time

from .assets import atomic_json, file_sha256, resolve_checkpoint, verify_upstream
from .tau import resolve_tau, validate_tau_settings
from .windows_sdk import resolve_sdk_settings
from .portable_lock import exclusive_run_lock

LEGACY_ARMS = ("ui_voyager", "feedback_spectrum")
ARMS = LEGACY_ARMS + ("feedback_spectrum_tau", "feedback_spectrum_gated")
FEEDBACK_ARMS = ARMS[1:]
TAU_KEYS = frozenset(("tau", "tau_decay", "tau_mode", "tau_gamma"))
VARIANT_DEFAULTS = {
    "feedback_spectrum_tau": {"tau_decay": .6, "tau_mode": None},
    "feedback_spectrum_gated": {"tau_decay": None, "tau_mode": "adaptive", "tau_gamma": 5.},
}


def configured_arms(config: dict) -> tuple[str, ...]:
    """Resolve the declared comparison without importing a model or touching disk."""
    arms = config["study"].get("arms", LEGACY_ARMS)
    if (not isinstance(arms, (list, tuple)) or not arms
            or any(not isinstance(arm, str) or arm not in ARMS for arm in arms)
            or len(set(arms)) != len(arms)):
        raise ValueError("study.arms must be a nonempty list of distinct supported arms")
    return tuple(arms)


def selected_arms(config: dict, selection: str) -> tuple[str, ...]:
    """Keep 'both' backward compatible; 'all' means all configured arms."""
    available = configured_arms(config)
    requested = available if selection == "all" else LEGACY_ARMS if selection == "both" else (selection,)
    for arm in requested:
        if arm not in ARMS:
            raise ValueError(f"Unknown arm {arm}")
        if arm not in available:
            raise ValueError(f"Arm {arm} is not configured in study.arms")
    return requested


def arm_config(config: dict, arm: str) -> dict:
    """Isolate only tau-policy controls; all other experiment settings are shared."""
    selected_arms(config, arm)
    variants = config.get("spectral_variants", {})
    if not isinstance(variants, dict):
        raise ValueError("spectral_variants must be an object")
    for name, override in variants.items():
        if name not in FEEDBACK_ARMS or name not in configured_arms(config):
            raise ValueError(f"Unconfigured spectral variant {name}")
        if not isinstance(override, dict) or set(override) - TAU_KEYS:
            raise ValueError(f"spectral_variants.{name} may only override {sorted(TAU_KEYS)}")
    result = deepcopy(config)
    if arm in FEEDBACK_ARMS:
        result["spectral"].update(VARIANT_DEFAULTS.get(arm, {}))
        result["spectral"].update(variants.get(arm, {}))
    validate_tau_settings(result["spectral"])
    return result


def validate_geometry_tau(receipt: dict, spectral: dict, round_index: int,
                          reference_entropy: float | None = None) -> dict:
    """Check persisted policy against this arm and its frozen round-one reference."""
    policy = receipt.get("tau_policy")
    if not isinstance(policy, dict):
        raise ValueError("Geometry receipt has no tau_policy")
    expected = resolve_tau(spectral, round_index, entropy=policy.get("entropy"),
                           reference_entropy=reference_entropy)
    if any(key not in policy or policy[key] != value for key, value in expected.items()):
        raise ValueError("Geometry tau policy disagrees with arm, round, or frozen reference entropy")
    if receipt.get("tau") != expected["effective_tau"]:
        raise ValueError("Geometry effective tau disagrees with tau_policy")
    return expected


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_config(config: dict) -> None:
    study = config["study"]
    for key in ("rounds", "group_size"):
        value = study[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"study.{key} must be a positive integer")
    if study["group_size"] % 2:
        raise ValueError("group_size must be even for equal native/spectral collection")
    if study.get("spectral_fraction", 0.5) != 0.5:
        raise ValueError("The first study uses spectral_fraction=0.5")
    warmup = study.get("warmup_rounds", 3)
    if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0:
        raise ValueError("warmup_rounds must be a nonnegative integer")
    if not 0 <= config["spectral"].get("alpha", .5) <= 1:
        raise ValueError("spectral.alpha must lie in [0,1]")
    validate_tau_settings(config["spectral"])
    for arm in configured_arms(config):
        arm_config(config, arm)


def load_config(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text())
    root = Path(__file__).resolve().parents[3]
    for key in ("output_dir", "manifest", "assets_receipt"):
        value = Path(config[key]).expanduser()
        config[key] = str(value.resolve() if value.is_absolute() else (root / value).resolve())
    config["android"] = resolve_sdk_settings(config["android"])
    for key in ("upstream_root",):
        value = Path(config["android"][key]).expanduser()
        config["android"][key] = str(value.resolve() if value.is_absolute() else (root / value).resolve())
    if config["model"].get("checkpoint"):
        value = Path(config["model"]["checkpoint"]).expanduser()
        config["model"]["checkpoint"] = str(value.resolve() if value.is_absolute() else (root / value).resolve())
    validate_config(config)
    return config


def load_manifest(path: str | Path) -> dict:
    """Validate the whole frozen population before selecting any split.

    Individual collectors check each instance fingerprint, but only the full
    manifest can establish disjoint training/calibration/evaluation instances.
    Rehashing a modified file does not make duplicate semantic instances valid.
    """
    from .environment import UPSTREAM_COMMIT, _digest, _semantic_params

    manifest = json.loads(Path(path).expanduser().read_text())
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("Unsupported Android manifest schema")
    expected_hash = _digest({k: v for k, v in manifest.items() if k != "manifest_hash"})
    if manifest.get("manifest_hash") != expected_hash:
        raise ValueError("Frozen manifest hash mismatch")
    if manifest.get("upstream_commit") != UPSTREAM_COMMIT:
        raise ValueError("Frozen manifest upstream revision mismatch")
    if manifest.get("protocol") != "parameter_disjoint_same_templates":
        raise ValueError("Unsupported frozen manifest protocol")
    if type(manifest.get("seed")) is not int or manifest["seed"] < 0:
        raise ValueError("Frozen manifest seed must be a nonnegative integer")
    splits = {"train", "calibration", "eval"}
    rows = manifest.get("instances")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Frozen manifest instances must be a nonempty list")
    observed = {split: set() for split in splits}
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("split"), str) or row["split"] not in splits:
            raise ValueError("Frozen manifest has an invalid instance split")
        if any(not isinstance(row.get(key), str) or not row[key] for key in ("task_name", "goal")):
            raise ValueError("Frozen manifest instance needs task_name and goal")
        if not isinstance(row.get("params"), dict):
            raise ValueError("Frozen manifest instance parameters must be an object")
        if type(row.get("initialization_seed")) is not int or row["initialization_seed"] < 0:
            raise ValueError("Frozen manifest initialization_seed must be a nonnegative integer")
        actual_id = _digest({"task_name": row["task_name"], "params": _semantic_params(row["params"])})
        if row.get("instance_id") != actual_id:
            raise ValueError("Frozen manifest instance fingerprint mismatch")
        if actual_id in seen:
            raise ValueError("Duplicate semantic instance in frozen manifest; all splits must be disjoint")
        seen.add(actual_id)
        observed[row["split"]].add(row["task_name"])
    if any(not names for names in observed.values()):
        raise ValueError("Frozen manifest needs nonempty train, calibration and eval splits")
    selected = manifest.get("selected_tasks")
    if not isinstance(selected, dict) or set(selected) != splits:
        raise ValueError("Frozen manifest selected_tasks must describe all three splits")
    for split, names in selected.items():
        if (not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != len(names) or set(names) != observed[split]):
            raise ValueError(f"Frozen manifest selected_tasks mismatch for {split}")
    count = manifest.get("registered_task_count")
    if type(count) is not int or count < len(set().union(*observed.values())):
        raise ValueError("Frozen manifest registered_task_count is inconsistent")
    return manifest


def source_fingerprint() -> str:
    here = Path(__file__).resolve().parent
    files = sorted(here.glob("*.py")) + [here.parent / "spectral.py"]
    return digest({str(p.relative_to(here.parent)): file_sha256(p) for p in files})


def runtime_fingerprint() -> dict:
    versions = {}
    for package in ("torch", "torchvision", "transformers", "peft", "accelerate", "numpy",
                    "Pillow", "android_env", "scikit-image", "opencv-python", "opencv-python-headless"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": sys.version, "packages": versions}


def stage_config(config: dict, identity: str) -> dict:
    """Private, stable stage randomization; no extra training-seed sweep."""
    result = deepcopy(config)
    seed = int(config.get("runtime_seed", 0))
    result["runtime_seed"] = int(digest([seed, identity])[:15], 16)
    return result


def load_operators(path: Path) -> dict:
    import torch
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(bundle, dict) or not isinstance(bundle.get("operators"), dict) or not bundle["operators"]:
        raise ValueError("Invalid saved geometry bundle")
    return bundle["operators"]


def seal_study(root: Path, config: dict, manifest: dict, source: str, inputs: dict | None = None) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    value = {"schema_version": 1, "config": config, "manifest_sha256": digest(manifest),
             "source_sha256": source, "inputs": inputs or {}}
    value["identity"] = digest(value)
    path = root / "study.json"
    if path.exists():
        if json.loads(path.read_text()).get("identity") != value["identity"]:
            raise ValueError("Study identity changed; start a new output_dir, do not overwrite prior runs")
    else:
        atomic_json(path, value)
    return value


@contextmanager
def study_lock(root: Path):
    """Advisory nonblocking lock on Windows and POSIX."""
    with exclusive_run_lock(root / ".run.lock"):
        yield


def read_complete_stage(path: Path) -> dict | None:
    if not path.exists():
        return None
    receipt = path / "complete.json"
    if not receipt.is_file():
        raise RuntimeError(f"Incomplete stage preserved at {path}; inspect it before explicit recovery")
    value = json.loads(receipt.read_text())
    for name, expected in value.get("artifacts", {}).items():
        target = path / name
        if not target.is_file() or file_sha256(target) != expected:
            raise ValueError(f"Completed stage artifact changed: {target}")
    for name, expected in value.get("external_artifacts", {}).items():
        target = Path(name)
        if not target.is_file() or file_sha256(target) != expected:
            raise ValueError(f"External stage artifact changed: {target}")
    return value


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize_episodes(episodes: list[dict], *, expected: int | None = None) -> dict:
    if expected is not None and len(episodes) != expected:
        raise ValueError(f"Received {len(episodes)} episodes, expected {expected}; incomplete evaluation")
    if len({row["episode_id"] for row in episodes}) != len(episodes):
        raise ValueError("Duplicate episode IDs")
    if not episodes:
        raise ValueError("No episodes to summarize")
    if any(type(row.get("success")) is not bool for row in episodes):
        raise ValueError("Episode success must be a native verifier boolean")
    steps = [step for row in episodes for step in row["steps"]]
    paths = {digest([step.get("action") for step in row["steps"]]) for row in episodes if row["success"]}
    per_instance: dict[str, list[bool]] = {}
    for row in episodes:
        per_instance.setdefault(row.get("instance_id", row["episode_id"]), []).append(row["success"])
    return {"episodes": len(episodes), "successes": sum(row["success"] for row in episodes),
            "success_rate": sum(row["success"] for row in episodes) / len(episodes),
            "instances": len(per_instance), "instances_with_success": sum(any(v) for v in per_instance.values()),
            "environment_steps": len(steps),
            "response_tokens": sum(len(s.get("response_token_ids", [])) for s in steps),
            "invalid_actions": sum(bool(s.get("action_error")) for s in steps),
            "exact_success_action_sequences": len(paths),
            "diversity_note": "Exact serialized action sequence proxy; not semantic path richness or matched-correct coverage"}


def checkpoint_files(checkpoint: str) -> list[Path]:
    root = Path(checkpoint)
    if not (root / "config.json").is_file():
        raise FileNotFoundError(f"Checkpoint config missing: {root}")
    return sorted(p.resolve() for p in root.iterdir() if p.is_file() and
                  p.suffix in {".json", ".safetensors", ".txt", ".jinja", ".model"})


def example_images(rows: list[dict]) -> list[Path]:
    result = set()
    for row in rows:
        for step in row.get("steps", [row]):
            for key in ("screenshot", "after_screenshot"):
                if isinstance(step.get(key), str):
                    result.add(Path(step[key]).expanduser().resolve())
            for message in step.get("messages", []):
                content = message.get("content", [])
                if not isinstance(content, list):
                    continue
                for block in content:
                    if block.get("type") == "image" and isinstance(block.get("image"), str):
                        result.add(Path(block["image"]).expanduser().resolve())
    return sorted(result)


def _finish(path: Path, receipt: dict, artifacts: list[str] = (), external_paths=()) -> dict:
    receipt = dict(receipt, artifacts={name: file_sha256(path / name) for name in artifacts},
                   external_artifacts={str(p): file_sha256(p) for p in sorted(set(external_paths))})
    atomic_json(path / "complete.json", receipt)
    return receipt


def _instances(manifest: dict, split: str) -> list[dict]:
    rows = [row for row in manifest["instances"] if row["split"] == split]
    if not rows:
        raise ValueError(f"No {split} instances in frozen manifest")
    return rows


def _collect(checkpoint: str, instances: list[dict], path: Path, config: dict,
             repeats: int, *, policy_name: str = "native", round_index: int = 0,
             policy=None, operators=None) -> list[dict]:
    done = read_complete_stage(path)
    if done is not None:
        if done["checkpoint"] != checkpoint or done["policy"] != policy_name or done["repeats"] != repeats:
            raise ValueError("Collection receipt identity mismatch")
        return load_jsonl(path / "records.jsonl")
    from .model import QwenPolicy
    from .environment import collect_episodes
    from contextlib import nullcontext
    own_policy = policy is None
    path.mkdir(parents=True)
    started = time.monotonic()
    settings = stage_config(config, str(path))
    settings.setdefault("collection", {})["repeats"] = repeats
    if own_policy:
        policy = QwenPolicy(checkpoint, settings)
    try:
        import torch
        torch.manual_seed(settings["runtime_seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(settings["runtime_seed"])
        if operators is not None:
            from ..spectral import folded_operators
            context = folded_operators(policy.model, operators)
        else:
            context = nullcontext()
        with context:
            episodes = collect_episodes(instances, policy.generate, path / "raw", settings,
                                        policy=policy_name, checkpoint=checkpoint, round_index=round_index)
        metrics = summarize_episodes(episodes, expected=len(instances) * repeats)
        write_jsonl(path / "records.jsonl", episodes)
        _finish(path, {"checkpoint": checkpoint, "policy": policy_name, "repeats": repeats,
                       "round": round_index, "elapsed_seconds": time.monotonic() - started,
                       "metrics": metrics}, ["records.jsonl"],
                checkpoint_files(checkpoint) + example_images(episodes))
        return episodes
    finally:
        if own_policy and policy is not None:
            policy.close()


def _train(checkpoint: str, examples: list[dict], path: Path, config: dict, kind: str) -> str:
    done = read_complete_stage(path)
    if done is not None:
        if done["input_checkpoint"] != checkpoint or done["kind"] != kind:
            raise ValueError("Training receipt identity mismatch")
        target = done["checkpoint"]
        if not (Path(target) / "config.json").exists():
            raise FileNotFoundError(f"Completed checkpoint missing: {target}")
        return target
    path.mkdir(parents=True)
    write_jsonl(path / "examples.jsonl", examples)
    if not examples:
        _finish(path, {"kind": kind, "input_checkpoint": checkpoint, "checkpoint": checkpoint,
                       "training": {"status": "no_update", "reason": "no_usable_supervision",
                                    "examples": 0, "optimizer_steps": 0}},
                ["examples.jsonl"], checkpoint_files(checkpoint))
        return checkpoint
    from .training import train_sft, train_feedback
    target = path / "checkpoint"
    function = train_feedback if kind == "feedback" else train_sft
    result = function(checkpoint, examples, str(target), stage_config(config, str(path)))
    _finish(path, {"kind": kind, "input_checkpoint": checkpoint, "checkpoint": str(target),
                   "training": result}, ["examples.jsonl"],
            checkpoint_files(checkpoint) + checkpoint_files(str(target)) + example_images(examples))
    return str(target)


def run_study(config: dict, arm: str = "both", *, warmup_only: bool = False) -> dict:
    validate_config(config)
    methods = selected_arms(config, arm)
    verify_upstream(config["android"]["upstream_root"])
    manifest = load_manifest(config["manifest"])
    train_instances, eval_instances = _instances(manifest, "train"), _instances(manifest, "eval")
    initial = resolve_checkpoint(config)
    root = Path(config["output_dir"])
    from .data import success_examples, feedback_examples
    with study_lock(root):
        seal_study(root, config, manifest, source_fingerprint(),
                   inputs={"checkpoint": {str(p): file_sha256(p) for p in checkpoint_files(initial)},
                           "runtime": runtime_fingerprint()})
        config = deepcopy(config)
        config["runtime_seed"] = int(manifest["seed"])
        checkpoint = initial
        group_size = config["study"]["group_size"]
        for r in range(1, config["study"].get("warmup_rounds", 3) + 1):
            stage = root / "shared" / f"rft_{r:02d}"
            episodes = _collect(checkpoint, train_instances, stage / "collect", config, group_size, round_index=r)
            checkpoint = _train(checkpoint, success_examples(episodes), stage / "train", config, "rft")
        atomic_json(root / "shared_initialization.json", {"checkpoint": checkpoint, "initial_checkpoint": initial})
        if warmup_only:
            return {"status": "warmup_complete", "checkpoint": checkpoint}
        _collect(checkpoint, eval_instances, root / "shared" / "evaluation", config, 1)
        shared_checkpoint = checkpoint
        anchors: list[dict] | None = None
        for method in methods:
            settings = arm_config(config, method)
            checkpoint = shared_checkpoint
            previous_feedback: list[dict] = []
            reference_entropy: float | None = None
            if method in FEEDBACK_ARMS and anchors is None:
                calibration = _collect(shared_checkpoint, _instances(manifest, "calibration"),
                                       root / "shared" / "anchor_collection", config,
                                       config["study"].get("anchor_repeats", group_size))
                anchors = success_examples(calibration, split="calibration")
                if not anchors:
                    raise RuntimeError("No native verified calibration success; spectral arm cannot fabricate its anchor")
                random.Random(stage_config(config, "fixed_anchor_order")["runtime_seed"]).shuffle(anchors)
            for r in range(1, settings["study"]["rounds"] + 1):
                stage = root / method / f"round_{r:02d}"
                tau_policy = None
                if method == "ui_voyager":
                    episodes = _collect(checkpoint, train_instances, stage / "collect", settings, group_size, round_index=r)
                    from .baseline import build_grsd_examples
                    examples, evidence = build_grsd_examples(episodes, settings["android"]["upstream_root"], settings)
                    atomic_json(stage / "supervision.json", evidence)
                    checkpoint = _train(checkpoint, examples, stage / "train", settings, "grsd")
                else:
                    from .model import QwenPolicy
                    from .geometry import collect_operators
                    policy = QwenPolicy(checkpoint, settings)
                    try:
                        geometry = stage / "geometry"
                        completed = read_complete_stage(geometry)
                        if completed is None:
                            calibration_started = time.monotonic()
                            operators, receipt = collect_operators(policy, anchors, previous_feedback, settings, geometry,
                                                                   round_index=r, reference_entropy=reference_entropy)
                            tau_policy = validate_geometry_tau(receipt, settings["spectral"], r, reference_entropy)
                            _finish(geometry, {"checkpoint": checkpoint, "calibration": receipt,
                                               "elapsed_seconds": time.monotonic() - calibration_started},
                                    ["operators.pt", "calibration.json"],
                                    checkpoint_files(checkpoint) + example_images(anchors) + example_images(previous_feedback))
                        else:
                            if completed["checkpoint"] != checkpoint:
                                raise ValueError("Geometry checkpoint mismatch")
                            if not {"operators.pt", "calibration.json"}.issubset(completed.get("artifacts", {})):
                                raise ValueError("Geometry receipt must hash operators.pt and calibration.json")
                            receipt = json.loads((geometry / "calibration.json").read_text())
                            if receipt != completed["calibration"]:
                                raise ValueError("Geometry calibration receipt mismatch")
                            tau_policy = validate_geometry_tau(receipt, settings["spectral"], r, reference_entropy)
                            operators = load_operators(geometry / "operators.pt")
                        if r == 1 and tau_policy["mode"] == "adaptive":
                            reference_entropy = tau_policy["reference_entropy"]
                        native = _collect(checkpoint, train_instances, stage / "collect_native", settings, group_size // 2,
                                          round_index=r, policy=policy)
                        spectral = _collect(checkpoint, train_instances, stage / "collect_spectral", settings, group_size // 2,
                                            round_index=r, policy_name="spectral", policy=policy, operators=operators)
                    finally:
                        policy.close()
                    examples, evidence = feedback_examples(native + spectral, settings)
                    atomic_json(stage / "supervision.json", evidence)
                    checkpoint = _train(checkpoint, examples, stage / "train", settings, "feedback")
                    previous_feedback = examples
                _collect(checkpoint, eval_instances, stage / "evaluation", settings, 1, round_index=r)
                atomic_json(stage / "round.json", {"arm": method, "round": r, "checkpoint": checkpoint,
                                                   "tau_policy": tau_policy})
        return report(config)


def report(config: dict) -> dict:
    validate_config(config)
    arms = configured_arms(config)
    root = Path(config["output_dir"])
    rows = []
    for path in [root / "shared" / "evaluation"] + [root / arm / f"round_{r:02d}" / "evaluation"
                                                    for arm in arms for r in range(1, config["study"]["rounds"] + 1)]:
        marker = path / "complete.json"
        if marker.is_file():
            value = read_complete_stage(path)
            rows.append({"stage": str(path.relative_to(root)), **value["metrics"], "checkpoint": value["checkpoint"]})
        else:
            rows.append({"stage": str(path.relative_to(root)), "status": "incomplete" if path.exists() else "not_run"})
    costs = {}
    cost_roots = {"shared": root / "shared", **{arm: root / arm for arm in arms},
                  "spectral_reference_acquisition": root / "shared" / "anchor_collection"}
    for category, cost_root in cost_roots.items():
        total = {"collection_episodes": 0, "environment_steps": 0, "generated_response_tokens": 0,
                 "optimizer_steps": 0, "trained_response_token_presentations": 0,
                 "teacher_prompt_token_presentations": 0, "no_update_stages": 0,
                 "collection_seconds": 0.0, "training_seconds": 0.0, "calibration_seconds": 0.0}
        for marker in cost_root.glob("**/complete.json"):
            if category == "shared" and "anchor_collection" in marker.relative_to(cost_root).parts:
                continue
            value = json.loads(marker.read_text())
            if "metrics" in value:
                total["collection_episodes"] += value["metrics"]["episodes"]
                total["environment_steps"] += value["metrics"]["environment_steps"]
                total["generated_response_tokens"] += value["metrics"]["response_tokens"]
                total["collection_seconds"] += value.get("elapsed_seconds", 0.)
            if "training" in value:
                train = value["training"]
                total["optimizer_steps"] += train.get("optimizer_steps", 0)
                total["trained_response_token_presentations"] += train.get("response_token_presentations", 0)
                total["teacher_prompt_token_presentations"] += train.get("teacher_prompt_token_presentations", 0)
                total["training_seconds"] += train.get("elapsed_seconds", 0.)
                total["no_update_stages"] += train.get("status") == "no_update"
            if "calibration" in value:
                total["calibration_seconds"] += value.get("elapsed_seconds", 0.)
        costs[category] = total
    tau_history = []
    for arm in arms:
        if arm not in FEEDBACK_ARMS:
            continue
        settings = arm_config(config, arm)
        reference_entropy = None
        for r in range(1, config["study"]["rounds"] + 1):
            path = root / arm / f"round_{r:02d}" / "geometry"
            entry = {"arm": arm, "round": r}
            if not (path / "complete.json").is_file():
                entry["status"] = "incomplete" if path.exists() else "not_run"
            else:
                value = read_complete_stage(path)
                receipt = value["calibration"]
                if ("tau_policy" not in receipt and arm == "feedback_spectrum"
                        and settings["spectral"].get("tau_mode") is None
                        and settings["spectral"].get("tau_decay") is None):
                    # Older fixed-arm receipts did not record entropy or scheduling.
                    entry.update(status="legacy_receipt", effective_tau=receipt.get("tau"))
                else:
                    if not {"operators.pt", "calibration.json"}.issubset(value.get("artifacts", {})):
                        raise ValueError("Geometry receipt must hash operators.pt and calibration.json")
                    if receipt != json.loads((path / "calibration.json").read_text()):
                        raise ValueError("Geometry calibration receipt mismatch")
                    policy = validate_geometry_tau(receipt, settings["spectral"], r, reference_entropy)
                    if r == 1 and policy["mode"] == "adaptive":
                        reference_entropy = policy["reference_entropy"]
                    entry.update(status="complete", **policy)
            tau_history.append(entry)
    result = {"protocol": "controlled_ui_voyager_comparison", "arms": list(arms),
              "evaluations": rows, "costs": costs, "tau_history": tau_history,
              "budget_note": "Refinement rollout attempts and available optimizer updates are matched. Actual actions/tokens/FLOPs are not matched. shared/anchor_collection is additional spectral-reference acquisition; it is not used by GRSD. Calibration/teacher costs must be reported separately.",
              "note": "One training trajectory per arm; no statistical claim of replicated training runs. Missing cells remain missing."}
    if root.exists():
        atomic_json(root / "report.json", result)
    return result
