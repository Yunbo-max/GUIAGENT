"""Pinned native AndroidWorld collection; no mock scoring or fallback actions.

The evaluator requires explicit termination and a native task score of exactly
one. Partial native scores are retained separately. This controlled protocol is
shared by both arms and is not a claim to reproduce the author's paper numbers.
"""
from __future__ import annotations

from contextlib import contextmanager
import base64
import copy
import dataclasses
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import random
import secrets
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable
import urllib.request

from .prompting import ActionParseError, action_description, build_messages, load_system_prompt, parse_mobile_action

UPSTREAM_COMMIT = "67b65e2be093753ecaa2964f48739339b870813e"
ACCESSIBILITY_FORWARDER_URL = "https://storage.googleapis.com/android_env-tasks/2024.05.13-accessibility_forwarder.apk"
ACCESSIBILITY_FORWARDER_SHA256 = "97a56a544e44d79f9b3181fc7dbdd72cffa908efd3d53c82afad1773061a350a"
_TYPE_TAG = "__android_type__"
_ROW_TYPES = {"Expense", "Recipe", "CalendarEvent"}


def _encode_params(value: Any) -> Any:
    """Encode the pinned registry's concrete parameter types without pickle.

    AndroidWorld includes SQL-row dataclasses and one PIL target image. A plain
    json round-trip loses the attributes its native validators require.
    """
    from PIL import Image
    if value is None or type(value) in (str, bool, int, float):
        return value
    if isinstance(value, dict):
        if _TYPE_TAG in value or any(not isinstance(k, str) for k in value):
            raise ValueError("Native parameter dictionaries require string keys and no reserved type tag")
        return {k: _encode_params(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_encode_params(v) for v in value]
    if isinstance(value, tuple):
        return {_TYPE_TAG: "tuple", "items": [_encode_params(v) for v in value]}
    if dataclasses.is_dataclass(value):
        cls = type(value)
        if cls.__module__ != "android_world.task_evals.utils.sqlite_schema_utils" or cls.__name__ not in _ROW_TYPES:
            raise TypeError(f"Unregistered native parameter dataclass: {cls.__module__}.{cls.__name__}")
        return {_TYPE_TAG: cls.__name__, "fields": {f.name: _encode_params(getattr(value, f.name)) for f in dataclasses.fields(value)}}
    if isinstance(value, Image.Image):
        stream = io.BytesIO()
        value.save(stream, format="PNG")
        return {_TYPE_TAG: "png", "base64": base64.b64encode(stream.getvalue()).decode("ascii")}
    raise TypeError(f"Unsupported native parameter type: {type(value)!r}")


def _decode_params(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_params(v) for v in value]
    if not isinstance(value, dict):
        return value
    if _TYPE_TAG not in value:
        return {k: _decode_params(v) for k, v in value.items()}
    tag = value[_TYPE_TAG]
    if not isinstance(tag, str):
        raise ValueError("Native parameter type tag must be a string")
    if tag == "tuple" and set(value) == {_TYPE_TAG, "items"}:
        return tuple(_decode_params(v) for v in value["items"])
    if tag == "png" and set(value) == {_TYPE_TAG, "base64"}:
        from PIL import Image
        with Image.open(io.BytesIO(base64.b64decode(value["base64"], validate=True))) as image:
            if image.format != "PNG":
                raise ValueError("Native image parameter must be PNG")
            return image.copy()
    if tag in _ROW_TYPES and set(value) == {_TYPE_TAG, "fields"}:
        schema = importlib.import_module("android_world.task_evals.utils.sqlite_schema_utils")
        cls = getattr(schema, tag)
        if set(value["fields"]) != {f.name for f in dataclasses.fields(cls)}:
            raise ValueError(f"Native row field mismatch for {tag}")
        return cls(**{k: _decode_params(v) for k, v in value["fields"].items()})
    raise ValueError(f"Unknown or malformed native parameter type tag: {tag!r}")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _private_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive else os.O_TRUNC)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


@contextmanager
def _seeded(seed: int):
    import numpy as np
    py_state, np_state = random.getstate(), np.random.get_state()
    random.seed(seed)
    np.random.seed(seed % 2**32)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def _load_components(upstream_root: str | Path) -> tuple[dict, Any, Any]:
    root = Path(upstream_root).expanduser().resolve()
    expected = root / "androidworld/android_world"
    if not expected.is_dir():
        raise FileNotFoundError(f"Missing UI-Voyager AndroidWorld checkout at {root}")
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != UPSTREAM_COMMIT:
        raise ValueError(f"UI-Voyager revision mismatch: expected {UPSTREAM_COMMIT}, got {head}")
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise ValueError("Pinned UI-Voyager checkout has modified tracked files")
    sys.path.insert(0, str(root / "androidworld"))
    registry = importlib.import_module("android_world.registry")
    actual = Path(registry.__file__).resolve()
    if not actual.is_relative_to(expected):
        raise RuntimeError(f"Another AndroidWorld installation was imported: {actual}")
    registry_object = registry.TaskRegistry()
    tasks = registry_object.get_registry(registry_object.ANDROID_WORLD_FAMILY)
    launcher = importlib.import_module("android_world.env.env_launcher")
    actions = importlib.import_module("android_world.env.json_action")
    return tasks, launcher, actions.JSONAction


def _semantic_params(params: dict) -> dict:
    # A changed randomization seed alone must not bypass split collision checks.
    return {k: v for k, v in params.items() if k != "seed"}


def prepare_manifest(upstream_root: str | Path, output_path: str | Path, config: dict) -> dict:
    """Prepare a private fixed-instance manifest without opening an emulator.

    Defaults select 16 parameterizable training/calibration templates and retain
    every registered template in evaluation. This is parameter generalization,
    not held-out-template generalization. No population is silently dropped.
    """
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Manifest already exists: {output_path}")
    tasks, _, _ = _load_components(upstream_root)
    data = config.get("data", {})
    counts = data.get("split_counts", {"train": 2, "calibration": 1, "eval": 1})
    if set(counts) != {"train", "calibration", "eval"} or any(type(v) is not int or v < 1 for v in counts.values()):
        raise ValueError("split_counts must contain positive train/calibration/eval integers")
    seed = data.get("seed", secrets.randbits(63))
    if type(seed) is not int or seed < 0:
        raise ValueError("data.seed must be a nonnegative integer")
    max_attempts = int(data.get("max_param_attempts", 128))
    if max_attempts < sum(counts.values()):
        raise ValueError("max_param_attempts must cover the requested split counts")
    names = sorted(tasks)
    pools: dict[str, list[dict]] = {}
    used: set[str] = set()
    excluded: list[dict] = []
    instances: list[dict] = []

    def candidates(name: str, needed: int) -> list[dict]:
        pool = pools.setdefault(name, [])
        seen = {_digest(_semantic_params(p["params"])) for p in pool}
        for attempt in range(len(pool), max_attempts):
            if len(pool) >= needed:
                break
            candidate_seed = int(_digest([seed, name, attempt])[:15], 16)
            with _seeded(candidate_seed):
                params = tasks[name].generate_random_params()
            # Typed, allowlisted JSON preserves native SQL rows and target images.
            params = json.loads(_canonical(_encode_params(params)))
            identity = _digest(_semantic_params(params))
            if identity in seen:
                continue
            seen.add(identity)
            with _seeded(candidate_seed):
                task = tasks[name](_decode_params(copy.deepcopy(params)))
                goal = str(task.goal)
            pool.append({"task_name": name, "params": params, "goal": goal,
                         "initialization_seed": candidate_seed,
                         "instance_id": _digest({"task_name": name, "params": _semantic_params(params)})})
        return pool

    def select(spec: Any, split: str) -> list[str]:
        if spec == "all":
            return names
        if spec == "parameterized":
            limit = int(data.get(f"{split}_task_limit", 16))
            if limit < 1:
                raise ValueError(f"{split}_task_limit must be positive")
            selected = []
            for name in names:
                pool = candidates(name, sum(counts.values()))
                if len(pool) < sum(counts.values()):
                    excluded.append({"task_name": name, "split": split, "reason": "insufficient_distinct_parameters", "distinct_found": len(pool)})
                    continue
                selected.append(name)
                if len(selected) == limit:
                    break
            if len(selected) < limit:
                raise ValueError(f"Only {len(selected)} parameterized templates available; requested {limit}")
            return selected
        if not isinstance(spec, list) or not spec or len(set(spec)) != len(spec):
            raise ValueError(f"{split}_tasks must be all, parameterized, or a nonempty unique list")
        if any(name not in tasks for name in spec):
            raise ValueError(f"Unknown {split} task names: {sorted(set(spec)-set(tasks))}")
        return sorted(spec)

    common = data.get("tasks")
    selected = {split: select(data.get(f"{split}_tasks", common if common is not None else ("all" if split == "eval" else "parameterized")), split)
                for split in ("train", "calibration", "eval")}
    # Reserve evaluation first, then allocate independent training and anchor instances.
    for split in ("eval", "calibration", "train"):
        for name in selected[split]:
            total_needed = sum(counts[s] for s in selected if name in selected[s])
            pool = candidates(name, total_needed)
            available = [p for p in pool if p["instance_id"] not in used]
            if len(available) < counts[split]:
                raise ValueError(f"Task {name} cannot provide disjoint {split} parameters; choose parameterized training templates")
            for candidate in available[:counts[split]]:
                used.add(candidate["instance_id"])
                instances.append({**candidate, "split": split})
    result = {"schema_version": 1, "upstream_commit": UPSTREAM_COMMIT,
              "protocol": "parameter_disjoint_same_templates", "seed": seed,
              "reproducibility": "reuse_frozen_manifest; seed_alone_does_not_freeze_native_wall_clock_parameters",
              "native_wall_clock_parameters": ["MarkorCreateFolder.folder_name"],
              "registered_task_count": len(names), "selected_tasks": selected,
              "excluded_from_training": excluded, "instances": instances}
    result["manifest_hash"] = _digest(result)
    _private_json(output_path, result, exclusive=True)
    return result


def _install_accessibility_forwarder(settings: dict) -> None:
    """Bootstrap the pinned native a11y dependency on the selected emulator.

    UI-Voyager's controller disables AndroidEnv's automatic APK installation,
    yet immediately starts its accessibility services. A fresh AVD therefore
    needs the APK before the launcher can create an environment or set up apps.
    The native wrapper still owns service activation and tree-log permissions.
    This runs only for the user's explicit environment-setup operation.
    """
    console = settings.get("console_port", 5554)
    server = settings.get("adb_server_port", 5037)
    if type(console) is not int or not 1 <= console < 65535 or type(server) is not int or not 1 <= server <= 65535:
        raise ValueError("Android console and ADB server ports must be valid integers")
    adb = os.path.expanduser(settings.get("adb_path", "~/android/platform-tools/adb"))
    command = [adb, "-P", str(server), "-s", f"emulator-{console}"]
    device = subprocess.run(command + ["get-state"], check=True, capture_output=True, text=True, timeout=30)
    if device.stdout.strip() != "device":
        raise RuntimeError(f"Configured emulator-{console} is not ready: {device.stdout.strip()}")
    with urllib.request.urlopen(ACCESSIBILITY_FORWARDER_URL, timeout=60) as response:
        apk = response.read(16 * 1024 * 1024)
    if hashlib.sha256(apk).hexdigest() != ACCESSIBILITY_FORWARDER_SHA256:
        raise ValueError("Accessibility forwarder APK SHA256 does not match the pinned native artifact")
    with tempfile.TemporaryDirectory(prefix="android-accessibility-") as directory:
        path = Path(directory) / "accessibility_forwarder.apk"
        path.write_bytes(apk)
        installed = subprocess.run(command + ["install", "-r", str(path)], check=True,
                                   capture_output=True, text=True, timeout=120)
        if "Success" not in installed.stdout.splitlines():
            raise RuntimeError(f"ADB did not confirm accessibility forwarder installation: {installed.stdout.strip()}")
    package = subprocess.run(command + ["shell", "pm", "path", "com.google.androidenv.accessibilityforwarder"],
                             check=True, capture_output=True, text=True, timeout=30)
    if not any(line.startswith("package:") for line in package.stdout.splitlines()):
        raise RuntimeError("Accessibility forwarder package is absent after installation")


def _open_env(launcher: Any, settings: dict) -> Any:
    kwargs = {"console_port": 5554, "grpc_port": 8554, "adb_server_port": 5037,
              "emulator_setup": False, "freeze_datetime": True}
    for key in ("console_port", "grpc_port", "adb_server_port", "emulator_setup", "freeze_datetime",
                "adb_path", "emulator_path", "avd_name", "android_sdk_root", "android_avd_home"):
        if key in settings:
            kwargs[key] = os.path.expanduser(settings[key]) if isinstance(settings[key], str) else settings[key]
    if kwargs["emulator_setup"]:
        _install_accessibility_forwarder(kwargs)
    return launcher.load_and_setup_env(**kwargs)


def _execute_native_action(env: Any, action: Any, state: Any) -> None:
    if action.action_type in ("answer", "status"):
        env.execute_action(action)
        return
    # AsyncAndroidEnv.execute_action discards this result. Inspect it explicitly:
    # upstream actuation catches ADB exceptions and reports result.success=False.
    actuation = importlib.import_module("android_world.env.actuation")
    result = actuation.execute_adb_action(action, state.ui_elements, env.logical_screen_size, env.controller)
    if not result.success:
        raise RuntimeError(f"Native Android action failed: {_canonical(result.to_dict())}")


def collect_episodes(instances: list[dict], generate: Callable[[list[dict]], dict],
                     output_dir: str | Path, config: dict, policy: str,
                     checkpoint: str, round_index: int) -> list[dict]:
    """Execute every requested episode, retaining raw outputs and actual verifier scores.

    The emulator must already be running. Unexpected runtime/scorer errors abort
    with a partial receipt whose success is null; they never count as task failure.
    Output directories are single-attempt, preventing accidental overwrite/resume.
    """
    if policy not in ("native", "spectral"):
        raise ValueError("policy must be native or spectral")
    if not instances:
        raise ValueError("No instances to collect")
    settings = config.get("android", {})
    upstream_root = settings.get("upstream_root")
    if not upstream_root:
        raise ValueError("android.upstream_root is required")
    tasks, launcher, action_class = _load_components(upstream_root)
    system_prompt = load_system_prompt(upstream_root)
    max_steps = int(settings.get("max_steps", 30))
    history_len = int(settings.get("history_len", 100))
    pause = float(settings.get("wait_after_action_seconds", 2.0))
    record_timing = settings.get("collect_timing", False)
    if type(record_timing) is not bool:
        raise ValueError("android.collect_timing must be a boolean")
    repeats = config.get("collection", {}).get("repeats", 1)
    if type(repeats) is not int or repeats < 1:
        raise ValueError("collection.repeats must be a positive integer")
    if max_steps < 1 or history_len < 1 or not math.isfinite(pause) or pause < 0:
        raise ValueError("Invalid Android step/history/pause limits")
    for instance in instances:
        if instance.get("task_name") not in tasks or instance.get("split") not in ("train", "calibration", "eval"):
            raise ValueError("Unknown task or split in instance manifest")
        actual_id = _digest({"task_name": instance["task_name"], "params": _semantic_params(instance["params"])})
        if actual_id != instance.get("instance_id"):
            raise ValueError("Instance fingerprint mismatch")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    episodes = []
    env = _open_env(launcher, settings)
    try:
        requests = [(instance, repeat) for instance in instances for repeat in range(repeats)]
        for request_index, (instance, repeat) in enumerate(requests):
            episode_id = _digest([str(output), instance["instance_id"], request_index, repeat, policy, checkpoint, round_index])[:24]
            episode_dir = output / episode_id
            episode_dir.mkdir(mode=0o700)
            with _seeded(int(instance["initialization_seed"])):
                task = tasks[instance["task_name"]](_decode_params(copy.deepcopy(instance["params"])))
            episode = {"schema_version": 1, "episode_id": episode_id,
                       "instance_id": instance["instance_id"], "task_name": instance["task_name"],
                       "goal": str(task.goal), "split": instance["split"], "policy": policy,
                       "checkpoint": str(checkpoint), "round": round_index,
                       "repeat_id": repeat,
                       "success": None, "native_score": None, "terminated": False,
                       "steps": [], "status": "collecting", "upstream_commit": UPSTREAM_COMMIT,
                       "scoring_rule": "terminated_and_native_score_equals_one"}
            partial = episode_dir / "episode.partial.json"
            _private_json(partial, episode)
            started = time.monotonic()
            initialized = False
            try:
                with _seeded(int(instance["initialization_seed"])):
                    env.reset(go_home=True)
                    task.initialize_task(env)
                    # Native information-retrieval tasks reword date parameters
                    # during initialize_task; use their actual post-init goal.
                    episode["goal"] = str(task.goal)
                    episode["manifest_goal"] = instance.get("goal")
                initialized = True
                history = []
                for step_index in range(max_steps):
                    from PIL import Image
                    loop_begin = time.perf_counter()
                    state = env.get_state(wait_to_stabilize=True)
                    before_state_elapsed = time.perf_counter() - loop_begin
                    pixels = state.pixels
                    screenshot = episode_dir / f"step_{step_index:04d}.png"
                    screenshot_start = time.perf_counter()
                    Image.fromarray(pixels).save(screenshot)
                    save_before_elapsed = time.perf_counter() - screenshot_start
                    messages = build_messages(episode["goal"], screenshot, history, system_prompt, history_len)
                    inference_begin = time.perf_counter()
                    sampled = generate(messages)
                    inference_elapsed = time.perf_counter() - inference_begin
                    response, ids = sampled["response"], sampled["response_token_ids"]
                    if not isinstance(response, str) or not isinstance(ids, list) or any(type(t) is not int or t < 0 for t in ids):
                        raise TypeError("generate must return response text and exact nonnegative token IDs")
                    step = {"index": step_index, "messages": messages, "response": response,
                            "response_token_ids": ids, "screenshot": str(screenshot),
                            "action": None, "action_error": None, "feedback": "",
                            "feedback_kind": "none", "done": False}
                    episode["steps"].append(step)
                    _private_json(partial, episode)
                    try:
                        action = parse_mobile_action(response, pixels.shape[1], pixels.shape[0])
                    except ActionParseError as error:
                        step.update(action_error=str(error), feedback=str(error), feedback_kind="action_error")
                        history.append(f"Invalid action; no operation executed: {error}")
                        _private_json(partial, episode)
                        continue
                    step["action"] = action
                    adb_begin = time.perf_counter()
                    _execute_native_action(env, action_class(**action), state)
                    adb_elapsed = time.perf_counter() - adb_begin
                    terminal = action["action_type"] in ("status", "answer")
                    step.update(done=terminal, feedback="Action submitted to AndroidWorld.", feedback_kind="action_executed")
                    history.append(action_description(response))
                    wait_begin = time.perf_counter()
                    if pause:
                        time.sleep(pause)
                    wait_elapsed = time.perf_counter() - wait_begin
                    after_begin = time.perf_counter()
                    after = env.get_state(wait_to_stabilize=True)
                    after_elapsed = time.perf_counter() - after_begin
                    after_path = episode_dir / f"step_{step_index:04d}_after.png"
                    after_save_begin = time.perf_counter()
                    Image.fromarray(after.pixels).save(after_path)
                    after_save_elapsed = time.perf_counter() - after_save_begin
                    step["after_screenshot"] = str(after_path)
                    if record_timing:
                        step["perf_seconds"] = {
                            "before_state": before_state_elapsed,
                            "before_png": save_before_elapsed,
                            "model_generation": inference_elapsed,
                            "adb_actuation": adb_elapsed,
                            "explicit_wait": wait_elapsed,
                            "after_state": after_elapsed,
                            "after_png": after_save_elapsed,
                            "total_step": time.perf_counter() - loop_begin,
                        }
                    _private_json(partial, episode)
                    if terminal:
                        episode["terminated"] = True
                        break
                score = float(task.is_successful(env))
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError(f"Native verifier returned invalid score: {score}")
                episode.update(native_score=score, success=bool(episode["terminated"] and score == 1.0),
                               status="complete", elapsed_seconds=time.monotonic()-started)
                if record_timing:
                    labels = ("before_state","before_png","model_generation","adb_actuation",
                              "explicit_wait","after_state","after_png","total_step")
                    episode["timing_totals_seconds"] = {
                        key: sum(step.get("perf_seconds", {}).get(key,0.0)
                                 for step in episode["steps"]) for key in labels
                    }
                    episode["timing_definition"] = (
                        "wall_clock_per_step; model_generation_includes_processor;"
                        "ADB_env_get_state_may_include_simulator_CPU_and_network;"
                        "no_CUDA_sync_side_effect_introduced")
                _private_json(episode_dir / "episode.json", episode, exclusive=True)
                partial.unlink()
                episodes.append(episode)
                fd = os.open(output / "episodes.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                with os.fdopen(fd, "a", encoding="utf-8") as handle:
                    handle.write(_canonical(episode) + "\n")
            except BaseException as error:
                episode.update(status="aborted", success=None, runtime_error=f"{type(error).__name__}: {error}",
                               elapsed_seconds=time.monotonic()-started)
                _private_json(partial, episode)
                raise
            finally:
                if initialized or getattr(task, "initialized", False):
                    task.tear_down(env)
    finally:
        env.close()
    return episodes
