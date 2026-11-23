"""Verified Rolling Action Distillation: train-only offline dataset construction.

Consumes native MobileGym episode.json records with opt-in interim state checks.
Unlike a world model this module predicts action tokens, not future screenshots.
The future action-type auxiliary task is horizon-conditioned; it is NOT a native
multi-head token predictor. Training and held-out scoring are separate.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ACTION_TYPES = frozenset({
    "CLICK", "LONG_PRESS", "SWIPE", "TYPE", "ANSWER",
    "BACK", "HOME", "ENTER", "WAIT", "COMPLETE", "ABORT",
})


@dataclass(frozen=True)
class RollingSettings:
    horizon: int = 3
    verify_lookahead: int = 8
    min_progress_gain: float = 0.001
    future_loss_weight: float = 0.3
    require_clean: bool = True

    def __post_init__(self) -> None:
        if self.horizon < 1 or self.verify_lookahead < self.horizon:
            raise ValueError("Require 1 <= horizon <= verify_lookahead")
        if not 0 <= self.min_progress_gain <= 1:
            raise ValueError("min_progress_gain must be in [0,1]")
        if not 0 <= self.future_loss_weight <= 1:
            raise ValueError("future_loss_weight must be in [0,1]")


def _verified(step: dict) -> dict | None:
    result = step.get("verification")
    if not isinstance(result, dict) or result.get("status") != "verified":
        return None
    before, after = result.get("progress_before"), result.get("progress_after")
    if any(type(x) not in (int, float) or not 0 <= x <= 1 for x in (before, after)):
        return None
    if type(result.get("clean")) is not bool:
        return None
    return result


def _append_text(messages: list[dict], instruction: str) -> list[dict]:
    out = deepcopy(messages)
    if not out or out[-1].get("role") != "user":
        raise ValueError("Expected recorded screenshot prompt ending in user turn")
    content = out[-1].get("content")
    if isinstance(content, str):
        out[-1]["content"] = content + "\n" + instruction
    elif isinstance(content, list):
        content.append({"type": "text", "text": instruction})
    else:
        raise ValueError("Invalid multimodal prompt content")
    return out


def _action_kind(step: dict) -> str | None:
    action = step.get("action")
    if not isinstance(action, dict):
        return None
    kind = action.get("action_type")
    return kind if isinstance(kind, str) and kind in ACTION_TYPES else None


def _hindsight(window: list[dict], *, start_index: int, gain: float) -> str:
    lines = [
        "Training-only hindsight from the SAME real MobileGym trajectory.",
        "These are executed actions and interim goal-check progress, not oracle",
        "counterfactual outcomes or proof that each individual action was optimal.",
        f"Verified progress gain over observed prefix: {gain:.4f}.",
    ]
    for offset, step in enumerate(window):
        v = _verified(step)
        assert v is not None
        lines.append(
            f"Executed step {start_index + offset}: {_action_kind(step)}; "
            f"goal progress {float(v['progress_before']):.4f} -> "
            f"{float(v['progress_after']):.4f}; clean={v['clean']}."
        )
    return "\n".join(lines)


def build_examples(
    episodes: Iterable[dict],
    settings: RollingSettings = RollingSettings(),
) -> tuple[list[dict], dict]:
    """Create gated next-action and future action-type KL examples.

    A window is admitted only when genuine interim state checks show positive
    task progress with no disallowed side effects. This is prefix progress,
    NOT causal step-level reward. Future coordinates are never pseudo-labelled.
    """
    from improving.android.data import validate_episodes

    rows = validate_episodes(episodes, split="train")
    counts: Counter = Counter()
    examples: list[dict] = []
    for ep in rows:
        counts["episodes"] += 1
        if ep["policy"] != "native":
            counts["skipped_non_native_episode"] += 1
            continue
        steps = ep["steps"]
        for t, current in enumerate(steps):
            counts["candidate_prefixes"] += 1
            if not current.get("response_token_ids") or not _action_kind(current):
                counts["skipped_unusable_action"] += 1
                continue
            v0 = _verified(current)
            if v0 is None:
                counts["skipped_unverified"] += 1
                continue
            horizon_end = min(len(steps), t + settings.verify_lookahead)
            best_index, best_gain = None, 0.0
            for j in range(t, horizon_end):
                vj = _verified(steps[j])
                if vj is None or not _action_kind(steps[j]):
                    break
                if settings.require_clean and not vj["clean"]:
                    break
                gain = float(vj["progress_after"]) - float(v0["progress_before"])
                if gain > best_gain:
                    best_index, best_gain = j, gain
            if best_index is None or best_gain <= settings.min_progress_gain:
                counts["skipped_no_verified_gain"] += 1
                continue

            window = steps[t:best_index + 1]
            evidence = _hindsight(window, start_index=t, gain=best_gain)
            # One shared starting screenshot for every future-position forecast.
            max_offsets = min(settings.horizon, len(window))
            for k in range(max_offsets):
                future_step = steps[t + k]
                future_kind = _action_kind(future_step)
                assert future_kind is not None
                if k == 0:
                    prompt = _append_text(
                        current["messages"],
                        "Execute the next GUI action on the CURRENT real screenshot. "
                        "Return one executable mobile_use action."
                    )
                    target = current["response"]
                    target_ids = list(current["response_token_ids"])
                    objective = "next_action_kl"
                    loss_weight = 1.0
                else:
                    prompt = _append_text(
                        current["messages"],
                        f"Predict ONLY the semantic ACTION TYPE for the future step "
                        f"+{k} from this current screen. Do not invent future "
                        "click coordinates or claim to have observed the future "
                        "page. Answer with exactly one action-type label."
                    )
                    target = future_kind
                    # Tokenise this offline label only at training time using
                    # the pinned policy tokenizer; it is not an on-policy sample.
                    target_ids = None
                    objective = "future_action_type_kl"
                    loss_weight = settings.future_loss_weight
                examples.append({
                    "split": "train",
                    "policy": "native",
                    "objective": objective,
                    "horizon_offset": k,
                    "episode_id": ep["episode_id"],
                    "instance_id": ep["instance_id"],
                    "task_name": ep["task_name"],
                    "step_index": t,
                    "messages": prompt,
                    "response": target,
                    "response_token_ids": target_ids,
                    "feedback": evidence,
                    "verified_progress_gain": best_gain,
                    "loss_weight": loss_weight,
                    "feedback_end_step": best_index,
                    "future_screenshot": steps[best_index].get("next_screenshot"),
                    "evidence_provenance": "actual_train_episode_interim_verifier",
                })
                counts[objective] += 1
            counts["admitted_prefixes"] += 1
    if not examples:
        raise RuntimeError(
            "signal_shortage: no verified train-only progress windows; "
            "enable rolling.capture_step_verifier while collecting NEW train episodes "
            "or use another training collection. Never infer progress from test data."
        )
    receipt = {
        "method": "verified_rolling_horizon_conditioned_soft_kl_v1",
        "status": "prepared",
        "source_split": "train",
        "source_policy": "native",
        "world_model_used": False,
        "step_reward_claim": False,
        "native_multihead_mtp": False,
        "horizon": settings.horizon,
        "verify_lookahead": settings.verify_lookahead,
        "settings": settings.__dict__,
        "counts": dict(counts),
        "examples": len(examples),
    }
    return examples, receipt


def load_episodes(path: Path) -> tuple[list[dict], list[str]]:
    """Strictly consume one training-episode directory, never an evaluation tree."""
    root = Path(path).expanduser().resolve()
    files = sorted(root.rglob("episode.json"))
    if not files:
        raise FileNotFoundError(f"No completed episode.json beneath {root}")
    rows: list[dict] = []
    for item in files:
        episode = json.loads(item.read_text(encoding="utf-8"))
        if episode.get("split") != "train":
            raise ValueError(f"Held-out or invalid episode in training input: {item}")
        if episode.get("status") != "complete":
            raise ValueError(f"Incomplete episode in training input: {item}")
        rows.append(episode)
    return rows, [str(p) for p in files]


def prepare_file(
    input_root: str | Path,
    output_file: str | Path,
    settings: RollingSettings,
) -> dict:
    import os
    from tempfile import NamedTemporaryFile

    output = Path(output_file).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite dataset: {output}")
    episodes, sources = load_episodes(Path(input_root))
    examples, receipt = build_examples(episodes, settings)
    receipt["source_files"] = len(sources)
    receipt["source_digest"] = hashlib.sha256(
        "\n".join(sources).encode("utf-8")).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Atomic new file; never alter the original episode/screenshot artifacts.
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                            prefix=output.name + ".partial-", delete=False) as f:
        tmp = Path(f.name)
        for row in examples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    try:
        os.link(tmp, output)  # exclusive create; fail if concurrent writer exists
    finally:
        tmp.unlink(missing_ok=True)
    report = output.with_suffix(output.suffix + ".receipt.json")
    with report.open("x", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return receipt
