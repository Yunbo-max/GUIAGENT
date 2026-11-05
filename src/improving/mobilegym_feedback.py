"""Opt-in MobileGym Feedback-only SPECTRUM study.

One native Round 0 collects verified trajectories WITHOUT training.
Each subsequent round: feedback-gradient spectrum, 2 native + 2 spectral
rollouts, feedback-conditioned reverse-KL LoRA, independent native evaluation.
No extra calibration split, GRPO or external teacher model.

Research prototype: host GPU/browser qualification and real task-success
measurements have NOT been executed as part of source delivery.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

from improving.mobilegym_bridge import action_data, action_object, messages, balanced_examples


ARMS = (
    "feedback_sd", "feedback_spectrum_fixed", "feedback_spectrum_learned",
    "feedback_spectrum_tau", "feedback_spectrum_gated",
)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, default=str,
        separators=(",", ":")).encode()).hexdigest()


def _json_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, default=str)
        f.write("\n")


def _jsonl_new(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        for item in values:
            f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")


def validate(cfg: dict) -> dict:
    for section in ("model", "mobilegym", "study", "spectral", "train"):
        if not isinstance(cfg.get(section), dict):
            raise ValueError(f"Missing {section} object")
    workers = cfg["mobilegym"].get("environment_workers", 1)
    if type(workers) is not int or workers not in (1, 2, 4, 8, 16, 24):
        raise ValueError("mobilegym.environment_workers must be 1, 2, 4, 8, 16 or 24")
    study = cfg["study"]
    for name in ("train_task_limit", "test_task_limit", "round0_repeats",
                 "group_size", "rounds", "max_steps", "history_len"):
        if type(study.get(name)) is not int or study[name] < 1:
            raise ValueError(f"study.{name} must be a positive integer")
    if study["group_size"] % 2 or study["round0_repeats"] != study["group_size"]:
        raise ValueError("Round 0 must match an even refinement per-task rollout budget")
    arms = study.get("arms")
    if not isinstance(arms, list) or not arms or any(a not in ARMS for a in arms):
        raise ValueError("study.arms must be a nonempty list of supported methods")
    if len(set(arms)) != len(arms):
        raise ValueError("Duplicate study arms")
    if cfg["spectral"].get("geometry_source") != "feedback_only":
        raise ValueError("MobileGym protocol requires feedback_only geometry")
    if not cfg["model"].get("checkpoint"):
        raise ValueError("A local Qwen3-VL checkpoint is required")
    if not cfg["mobilegym"].get("pinned_commit"):
        raise ValueError("MobileGym source SHA must be pinned")
    from improving.android.tau import validate_tau_settings
    validate_tau_settings(cfg["spectral"])
    if "feedback_spectrum_learned" in arms:
        from improving.learned_gains import validate_gain_settings
        validate_gain_settings(cfg.get("gain_learning", {}))
    return cfg


def _vendor(cfg: dict) -> str:
    root = Path(cfg["mobilegym"]["root"]).expanduser().resolve()
    if not (root / "bench_env" / "task" / "registry.py").is_file():
        raise FileNotFoundError(f"MobileGym source unavailable: {root}")
    sha = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if sha != cfg["mobilegym"]["pinned_commit"]:
        raise ValueError(f"Wrong MobileGym commit {sha}; expected pinned {cfg['mobilegym']['pinned_commit']}")
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        text=True).strip()
    if dirty:
        raise ValueError("Pinned official MobileGym has modified tracked files")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return sha


def _tasks(cfg: dict, split: str) -> list:
    from bench_env.task.registry import load_tasks
    from bench_env.splits import resolve_split, base_task_id
    whitelist = resolve_split(split)
    seed = int(cfg["study"].get("seed", 11))
    tasks = sorted(
        (t for t in load_tasks(sample_n=1, seed=seed) if base_task_id(t.id) in whitelist),
        key=lambda t: _digest(t.id))
    limit = cfg["study"]["train_task_limit" if split == "train" else "test_task_limit"]
    if len(tasks) < limit:
        raise ValueError(f"Only {len(tasks)} distinct {split} tasks available; need {limit}")
    return tasks[:limit]


def plan(cfg: dict) -> dict:
    validate(cfg)
    sha = _vendor(cfg)
    train, test = _tasks(cfg, "train"), _tasks(cfg, "test")
    if {t.id for t in train} & {t.id for t in test}:
        raise ValueError("Official MobileGym train/test template overlap")
    return {
        "protocol": "mobilegym-feedback-only-learned-v1",
        "vendor_commit": sha,
        "model_checkpoint": cfg["model"]["checkpoint"],
        "train_ids": [t.id for t in train],
        "test_ids": [t.id for t in test],
        "train_seed": cfg["study"].get("seed", 11),
        "arms": cfg["study"]["arms"], "rounds": cfg["study"]["rounds"],
        "round0_native": len(train) * cfg["study"]["round0_repeats"],
        "episodes_per_refinement_arm": len(train) * cfg["study"]["group_size"],
        "evaluations_per_round_arm": len(test),
        "calibration_set": None,
        "note": "plan only; original train task IDs are disjoint from official held-out test",
    }


async def _environment_probe_async(cfg: dict) -> dict:
    """Read one official task screenshot/state without loading Qwen or training."""
    from bench_env.env.mobile_gym import MobileGymEnv
    from bench_env.runner.base import Controller
    from bench_env.env.base import Action
    env = MobileGymEnv(
        url=cfg["mobilegym"]["url"], headless=True, verbose=False,
        coord_space="norm_0_1000", delay_after_action=0.05)
    task = _fresh_task(_tasks(cfg, "train")[0])
    await env.start()
    await env.wait_ready()
    try:
        obs, _ = await Controller.setup(env, task, eval_mode="grounded")
        image = obs.get_screenshot_bytes()
        if not image:
            raise RuntimeError("MobileGym env-check received an empty screenshot")
        state = await env.get_state()
        if not isinstance(state, dict) or not state:
            raise RuntimeError("MobileGym env-check received no state for verifier")
        action = await env.step(Action.wait(0.01))
        if not action.observation.get_screenshot_bytes():
            raise RuntimeError("MobileGym env-check failed to observe post-action screen")
        return {"status": "env_check_passed", "task_id": task.id,
                "screenshot_bytes": len(image), "state_keys": sorted(state),
                "gpu_loaded": False, "benchmark_score": None}
    finally:
        task.teardown(env)
        await env.close()


def environment_probe(cfg: dict) -> dict:
    plan(cfg)
    return asyncio.run(_environment_probe_async(cfg))


def _fresh_task(proto):
    task = type(proto)(_seed=proto._seed)
    task._template_index = getattr(proto, "_template_index", None)
    if hasattr(proto, "_instance_id"):
        task._instance_id = proto._instance_id
    return task


async def _setup_with_retry(env, task, max_retries: int = 4):
    """Setup a task with retries: MobileGym app stores are lazily hydrated after
    page load, so the first get_state can race the store registration."""
    from bench_env.runner.base import Controller
    import asyncio
    last = None
    for retry in range(max_retries):
        try:
            return await Controller.setup(env, task, eval_mode="grounded")
        except Exception as exc:
            last = exc
            if retry < max_retries - 1:
                await asyncio.sleep(2.0 + retry * 1.5)
    raise last


async def _collect_async(cfg, checkpoint: str, tasks: list, out: Path, repeats: int,
                         policy_name: str, policy, operators: dict | None, round_index: int):
    from contextlib import nullcontext
    from bench_env.env.mobile_gym import MobileGymEnv
    from bench_env.runner.base import Evaluator
    from improving.spectral import folded_operators

    workers = cfg["mobilegym"].get("environment_workers", 1)
    if type(workers) is not int or workers not in (1, 2, 4, 8, 16, 24):
        raise ValueError("mobilegym.environment_workers must be 1, 2, 4, 8, 16 or 24")
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite episodes: {out}")
    from improving.mobilegym_reuse import collection_reuse
    reused, reuse_receipt = collection_reuse(
        cfg, out, tasks, repeats, checkpoint, policy_name, round_index)
    out.mkdir(parents=True)
    if reuse_receipt is not None:
        _json_new(out / "reuse_receipt.json", reuse_receipt)

    # Disable chromium GPU to avoid clashing with the loaded Qwen CUDA context.
    _orig_launch = MobileGymEnv.get_launch_args
    @staticmethod
    def _patched_launch(headless, proxy=None, browser_type="chromium"):
        args = _orig_launch(headless, proxy, browser_type)
        if browser_type == "chromium":
            args.setdefault("args", []).append("--disable-gpu")
        return args
    MobileGymEnv.get_launch_args = _patched_launch

    # Some tasks (e.g. alipay.SendMessageToContact) hydrate their app store lazily
    # after reset+warm, slower than the native 5-attempt get_state retry. Extend the
    # retry window so the required app store has time to register before sampling.
    import asyncio as _asyncio
    _orig_get_state = MobileGymEnv.get_state
    async def _patched_get_state(self, *, required_apps=None):
        for attempt in range(40):
            state = await self._get_state()
            if state and "os" in state:
                apps = state.get("apps") or {}
                if required_apps is None or all(a in apps for a in required_apps):
                    return state
            await _asyncio.sleep(0.5 + attempt * 0.25)
        return await _orig_get_state(self, required_apps=required_apps)
    MobileGymEnv.get_state = _patched_get_state

    # open_app (used by single-app tasks) does NOT preload the app's Zustand store;
    # after a page reload the store is lazily created, so alipay's store never
    # registers and get_state(required_apps=['alipay']) fails. Preload it first.
    _orig_open_app = MobileGymEnv.open_app
    async def _patched_open_app(self, app_name, timeout_ms=8000, wait_stable=False):
        try:
            await self.page.evaluate(
                "async (id) => { try { await window.__SIM__?.preloadAppStores?.([id]); } catch {} }",
                app_name)
        except Exception:
            pass
        return await _orig_open_app(self, app_name, timeout_ms, wait_stable)
    MobileGymEnv.open_app = _patched_open_app

    # Pre-warm every involved app store so their Zustand stores register before
    # the per-task reset+warm path; some apps (alipay) hydrate slowly on cold
    # preview servers and miss the native get_state retry window.
    all_apps = sorted({a for t in tasks for a in (getattr(t, "apps", None) or [])})

    async def _fresh_env():
        e = MobileGymEnv(
            url=cfg["mobilegym"]["url"], headless=True, verbose=False,
            coord_space="norm_0_1000",
            delay_after_action=float(cfg["mobilegym"].get("delay_after_action", 0.2)))
        try:
            await e.start()
            await e.wait_ready()
            if all_apps:
                try:
                    await e.warm_apps(all_apps)
                except Exception:
                    pass
        except BaseException:
            await e.close()
            raise
        return e

    from improving.rollout_workers import ordered_workers, SerialInference
    workers = cfg["mobilegym"].get("environment_workers", 1)
    requests = [(proto, attempt) for proto in tasks for attempt in range(repeats)]

    async def worker(worker_index, assignments):
        env = None
        evaluator = Evaluator(judge_mode="state", eval_mode="grounded")
        local_records = []
        episode_count = 0
        try:
            for slot, (proto, attempt) in assignments:
                if slot in reused:
                    previous = reused[slot]
                    _json_new(out / previous["episode_id"] / "episode.json", previous)
                    local_records.append((slot, previous))
                    continue
                if env is None:
                    env = await _fresh_env()
                task = _fresh_task(proto)
                episode_id = _digest([proto.id, attempt, policy_name, round_index, str(out)])[:24]
                directory = out / episode_id
                directory.mkdir(mode=0o700)
                steps, history = [], []
                ep = {
                    "episode_id": episode_id, "task_name": proto.id,
                    "instance_id": None, "goal": None,
                    "split": "eval" if policy_name == "eval" else "train",
                    "policy": "native" if policy_name == "eval" else policy_name,
                    "checkpoint": checkpoint, "round": round_index,
                    "steps": steps, "success": None, "native_score": None,
                    "judge": None, "status": "collecting",
                }
                try:
                    initial, params = await _setup_with_retry(env, task)
                    ep["instance_id"] = _digest([task.id, params, task.description])
                    ep["goal"] = task.description
                    obs = initial
                    terminated = False
                    max_steps = min(cfg["study"]["max_steps"],
                                    getattr(task, "max_steps", None) or cfg["study"]["max_steps"])
                    for i in range(max_steps):
                        screenshot = directory / f"step_{i:03d}.jpg"
                        image = obs.get_screenshot_bytes()
                        if not image:
                            raise RuntimeError("MobileGym provided no visual observation")
                        screenshot.write_bytes(image)
                        msg = messages(task.description, screenshot, history, cfg["study"]["history_len"])
                        sample = (policy.generate(msg) if workers == 1 else await inference(msg))
                        response, ids = sample["response"], sample["response_token_ids"]
                        if not ids:
                            raise RuntimeError("Empty model continuation")
                        step = {
                            "index": i, "messages": msg, "response": response,
                            "response_token_ids": ids, "screenshot": str(screenshot.resolve()),
                            "action": None, "action_error": None, "feedback": "",
                            "feedback_kind": "none", "done": False,
                        }
                        steps.append(step)
                        try:
                            kind, data = action_data(response)
                            action = action_object(response)
                            step["action"] = {"action_type": kind, "data": data}
                            result = await env.step(action)
                        except (ValueError, TypeError) as exc:
                            # Model-invalid actions count as failures, never repaired.
                            step.update(action_error=str(exc), feedback=str(exc),
                                        feedback_kind="action_error")
                            break
                        step["done"] = bool(result.done)
                        step["feedback_kind"] = "action_executed"
                        history.append(f"{kind}: {data}")
                        obs = result.observation
                        if result.done:
                            terminated = kind == "COMPLETE"
                            break
                    # State-based verifier, including unexpected side effects.
                    final = await env.get_observation()
                    verdict = await evaluator.evaluate(
                        task, initial, final, SimpleNamespace(agent_answer=env.agent_answer))
                    if verdict.judge_error:
                        raise RuntimeError("Native MobileGym judge error: " + verdict.judge_error)
                    ep["success"] = bool(terminated and verdict.passed)
                    ep["native_score"] = 1.0 if ep["success"] else 0.0
                    ep["judge"] = verdict.to_dict()
                    ep["status"] = "complete"
                    _json_new(directory / "episode.json", ep)
                    local_records.append((slot, ep))
                except BaseException as exc:
                    ep["status"] = "aborted"
                    ep["runtime_error"] = f"{type(exc).__name__}: {exc}"
                    _json_new(directory / "episode.partial.json", ep)
                    raise
                finally:
                    task.teardown(env)
                    # Periodic browser restart: qz Chromium accumulates state across
                    # many screenshot-heavy episodes and eventually dies with a silent
                    # browser-close. Recycle it every few episodes.
                    episode_count += 1
                    if episode_count % 4 == 0 and episode_count < len(assignments):
                        await env.close()
                        env = await _fresh_env()
        finally:
            if env is not None:
                await env.close()
        return local_records

    try:
        with (folded_operators(policy.model, operators)
              if operators is not None else nullcontext()):
            async with SerialInference(policy.generate) as inference:
                records = await ordered_workers(requests, workers, worker)
    finally:
        MobileGymEnv.get_launch_args = staticmethod(_orig_launch)
        MobileGymEnv.get_state = _orig_get_state
        MobileGymEnv.open_app = _orig_open_app
    _jsonl_new(out / "records.jsonl", records)
    return records


def _collect(cfg, checkpoint, tasks, output, repeats, policy_name, policy, operators=None, r=0):
    import torch
    seed = int(_digest([cfg["study"].get("seed", 11), r, policy_name])[:15], 16)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return asyncio.run(_collect_async(
        cfg, checkpoint, tasks, output, repeats, policy_name, policy, operators, r))


def _feedback(episodes, cfg):
    from improving.android.data import feedback_examples
    examples, record = feedback_examples(episodes, cfg)
    if not examples:
        raise RuntimeError(
            "signal_shortage: no verified same-instance success or real action error "
            "on native prefixes; do not invent feedback")
    return examples, record


def _settings(cfg, arm):
    c = deepcopy(cfg)
    # Stable per-arm runtime seed (training.py's plan_training_batches requires
    # an integer in [0, 2**63); the feedback-only runner has no manifest stage).
    c.setdefault("runtime_seed", int(_digest(["feedback_runtime", c["study"].get("seed", 11), arm])[:15], 16))
    spectral = c["spectral"]
    spectral.pop("tau_decay", None)
    spectral.pop("tau_mode", None)
    if arm == "feedback_spectrum_tau":
        spectral["tau_decay"] = 0.6
    if arm == "feedback_spectrum_gated":
        spectral.update(tau_mode="adaptive", tau_gamma=5.0)
    if arm == "feedback_spectrum_learned":
        spectral["tau"] = 0.0  # covariance pass only; learned T overrides identity
    return c


def run(cfg: dict) -> dict:
    from improving.android.model import QwenPolicy
    from improving.android.geometry import collect_operators
    from improving.android.training import train_feedback
    from improving.mobilegym_visual_gains import fit_visual_feedback_gains
    import torch

    from improving.mobilegym_reuse import verify_reuse_source
    reuse_root = verify_reuse_source(cfg)
    protocol = plan(cfg)  # real official split and pinned source, before GPU work
    train, test = _tasks(cfg, "train"), _tasks(cfg, "test")
    root = Path(cfg["output_dir"]).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"Experiment root must be new: {root}")
    checkpoint0 = str(Path(cfg["model"]["checkpoint"]).expanduser().resolve())
    if not (Path(checkpoint0) / "config.json").is_file():
        raise FileNotFoundError(f"Missing local model checkpoint: {checkpoint0}")
    root.mkdir(parents=True)
    _json_new(root / "study.json", {
        "plan": protocol, "configuration": cfg, "config_hash": _digest(cfg),
        "starting_checkpoint": checkpoint0, "status": "new_run_no_implicit_resume",
        "explicit_episode_reuse_source": str(reuse_root) if reuse_root is not None else None,
    })

    # Shared Round 0: no spectral generator, no SFT, no calibration references.
    base = QwenPolicy(checkpoint0, cfg)
    try:
        round0 = _collect(cfg, checkpoint0, train, root / "round0_native",
                          cfg["study"]["round0_repeats"], "native", base)
    finally:
        base.close()
    initial_feedback, evidence = _feedback(round0, cfg)
    _json_new(root / "round0_feedback.json", evidence)

    histories = {}
    for arm in cfg["study"]["arms"]:
        settings = _settings(cfg, arm)
        checkpoint = checkpoint0
        feedback = initial_feedback
        reference_entropy = None
        history = []
        for r in range(1, cfg["study"]["rounds"] + 1):
            stage = root / arm / f"round_{r:02d}"
            stage.mkdir(parents=True)
            policy = QwenPolicy(checkpoint, settings)
            geometry = None
            learned_audit = None
            try:
                operators = None
                if arm != "feedback_sd":
                    selected = balanced_examples(
                        feedback, int(settings["spectral"].get("max_feedback_examples", 16)))
                    if not selected:
                        raise RuntimeError("signal_shortage: missing feedback-gradient examples")
                    operators, geometry = collect_operators(
                        policy, [], selected, settings, stage / "geometry",
                        round_index=r, reference_entropy=reference_entropy)
                    if r == 1 and arm == "feedback_spectrum_gated":
                        reference_entropy = geometry["tau_policy"]["reference_entropy"]
                    if arm == "feedback_spectrum_learned":
                        bundle = torch.load(
                            stage / "geometry" / "operators.pt", map_location="cpu", weights_only=True)
                        operators, learned_audit = fit_visual_feedback_gains(
                            policy, feedback, bundle["feedback"], cfg.get("gain_learning"))
                        torch.save({"operators": operators}, stage / "learned_operators.pt")
                        _json_new(stage / "learned_gains.json", learned_audit)
                if arm == "feedback_sd":
                    episodes = _collect(
                        settings, checkpoint, train, stage / "native",
                        settings["study"]["group_size"], "native", policy, r=r)
                else:
                    half = settings["study"]["group_size"] // 2
                    native = _collect(
                        settings, checkpoint, train, stage / "native",
                        half, "native", policy, r=r)
                    steered = _collect(
                        settings, checkpoint, train, stage / "spectral",
                        half, "spectral", policy, operators, r=r)
                    episodes = native + steered
            finally:
                policy.close()

            examples, evidence = _feedback(episodes, settings)
            _json_new(stage / "feedback_receipt.json", evidence)
            output = stage / "checkpoint"
            training = train_feedback(checkpoint, examples, output, settings)
            checkpoint = str(output)
            _json_new(stage / "training_receipt.json", training)
            evaluator = QwenPolicy(checkpoint, settings)
            try:
                evaluated = _collect(
                    settings, checkpoint, test, stage / "eval",
                    1, "eval", evaluator, r=r)
            finally:
                evaluator.close()
            successes = sum(ep["success"] for ep in evaluated)
            row = {
                "arm": arm, "round": r, "checkpoint": checkpoint,
                "train_episodes": len(episodes),
                "train_success": sum(ep["success"] for ep in episodes),
                "evaluation_episodes": len(evaluated),
                "evaluation_success": successes,
                "evaluation_success_rate": successes / len(evaluated),
                "training_feedback_examples": len(examples),
                "geometry_source": geometry.get("geometry_source") if geometry else None,
                "tau": geometry.get("tau") if geometry else None,
                "learned_gain_status": learned_audit.get("selection") if learned_audit else None,
                "learned_gain_fallback": learned_audit.get("fallback_reason") if learned_audit else None,
            }
            _json_new(stage / "round.json", row)
            history.append(row)
        histories[arm] = history
    result = {
        "protocol": "mobilegym-feedback-only-learned-v1",
        "methods": histories,
        "note": "Round0 adds shared verified environment interactions. Equal "
                "rollout attempts and optimizer-step caps do not imply equal FLOPs/tokens. "
                "One training seed only. Native test scores are not training feedback.",
    }
    _json_new(root / "report.json", result)
    return {"root": str(root), "last_round": {k: v[-1] for k, v in histories.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("command", choices=("plan", "env-check", "run"))
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output = (plan(cfg) if args.command == "plan"
              else environment_probe(cfg) if args.command == "env-check"
              else run(cfg))
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
