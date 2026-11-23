"""JSON contracts for the additive Qwen GUI decision experiment.

Current-step supervision is deliberately not part of a model observation.
All preparation and evaluation helpers are usable without importing Torch.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

SCHEMA_VERSION = 1
IGNORE_INDEX = -100
GUI_JEPA_COMMIT = "dae79af4598962b9a5112b0bef297b588592192d"
GUI_JEPA_MODEL_REVISION = "a2265ca26dbea66fc59c46240a90d3be5defdb9e"
FLAG_NAMES = ("clickable", "long_clickable", "editable", "scrollable")
ACTION_KINDS = ("click", "input_text", "scroll", "navigate_back", "navigate_home", "wait", "open_app")
RAW_ACTION_KINDS = (*ACTION_KINDS, "long_press")
SCROLL_DIRECTIONS = ("up", "down", "left", "right")
OFFICIAL_SPLIT_COUNTS = {"train": 13603, "validation": 137, "test": 1543}
MODEL_OBSERVATION_KEYS = frozenset({
    "sample_id", "screenshot_path", "screen_width", "screen_height",
    "goal", "instruction", "history", "candidates",
})
CANDIDATE_KEYS = frozenset({
    "candidate_id", "kind", "element_index", "direction", "box", "text", "flags",
})


class Candidate(TypedDict):
    candidate_id: str
    kind: str
    element_index: int | None
    direction: str | None
    box: list[float] | None
    text: str
    flags: list[bool]


class ModelObservation(TypedDict):
    sample_id: str
    screenshot_path: str
    screen_width: int
    screen_height: int
    goal: str
    instruction: str
    history: list[str]
    candidates: list[Candidate]


def finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def validate_box(box: Any, name: str = "box") -> list[float]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError(f"{name} must contain four xyxy coordinates")
    result = [finite_number(x, name) for x in box]
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError(f"{name} must have positive width and height")
    # Native a11y enumeration permits partially off-screen rectangles. Their
    # normalized coordinates can therefore extend outside [0, 1]. Do not clip
    # the action centre or its evaluation target here.
    return result


def validate_candidate(candidate: Mapping[str, Any]) -> None:
    if set(candidate) != CANDIDATE_KEYS:
        raise ValueError(f"Candidate keys must be exactly {sorted(CANDIDATE_KEYS)}")
    if not isinstance(candidate["candidate_id"], str) or not candidate["candidate_id"]:
        raise ValueError("candidate_id must be a nonempty string")
    kind = candidate["kind"]
    if kind not in ACTION_KINDS:
        raise ValueError(f"Unsupported candidate kind: {kind!r}")
    if not isinstance(candidate["text"], str):
        raise ValueError("Candidate text must be a string")
    flags = candidate["flags"]
    if not isinstance(flags, list) or len(flags) != 4 or any(type(x) is not bool for x in flags):
        raise ValueError(f"flags must be four bools in {FLAG_NAMES} order")
    if kind in ("click", "input_text"):
        index = candidate["element_index"]
        if type(index) is not int or index < 0:
            raise ValueError("Element actions require a nonnegative element_index")
        validate_box(candidate["box"])
    elif candidate["element_index"] is not None or candidate["box"] is not None:
        raise ValueError("Global actions must not carry an element or box")
    if kind == "scroll":
        if candidate["direction"] not in SCROLL_DIRECTIONS:
            raise ValueError("Scroll candidates require a valid direction")
    elif candidate["direction"] is not None:
        raise ValueError("Only scroll candidates may carry a direction")


def validate_model_observation(observation: Mapping[str, Any]) -> None:
    """Reject unexpected fields rather than silently forwarding supervision."""
    if set(observation) != MODEL_OBSERVATION_KEYS:
        raise ValueError(f"Model observation keys must be exactly {sorted(MODEL_OBSERVATION_KEYS)}")
    for key in ("sample_id", "screenshot_path", "goal", "instruction"):
        if not isinstance(observation[key], str):
            raise ValueError(f"{key} must be a string")
    for key in ("screen_width", "screen_height"):
        if type(observation[key]) is not int or observation[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    history = observation["history"]
    if not isinstance(history, list) or any(not isinstance(x, str) for x in history):
        raise ValueError("history must be a list of prior-action strings")
    candidates = observation["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Every observation needs its complete candidate list")
    for candidate in candidates:
        validate_candidate(candidate)
    ids = [c["candidate_id"] for c in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate_id must be unique within an observation")


def validate_record(record: Mapping[str, Any]) -> None:
    if set(record) != {"schema_version", "split", "subsplits", "observation", "supervision"}:
        raise ValueError("Unexpected prepared-record keys")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema version: {record['schema_version']!r}")
    if record["split"] not in OFFICIAL_SPLIT_COUNTS:
        raise ValueError("Unknown dataset split")
    if not isinstance(record["subsplits"], list) or any(not isinstance(x, str) for x in record["subsplits"]):
        raise ValueError("subsplits must be a list of membership names")
    obs = record["observation"]
    required = {"sample_id", "episode_id", "step_index", "screenshot_path", "screen_width",
                "screen_height", "goal", "step_instruction", "history", "candidates"}
    if set(obs) != required:
        raise ValueError("Unexpected prepared-observation keys")
    if type(obs["episode_id"]) is not int or type(obs["step_index"]) is not int or obs["step_index"] < 0:
        raise ValueError("Invalid episode_id or step_index")
    if not isinstance(obs["step_instruction"], str):
        raise ValueError("step_instruction must be a string")
    sanitized = {k: obs[k] for k in MODEL_OBSERVATION_KEYS if k != "instruction"}
    sanitized["instruction"] = obs["goal"]
    validate_model_observation(sanitized)
    sup = record["supervision"]
    if not isinstance(sup, Mapping) or sup.get("type") not in RAW_ACTION_KINDS:
        raise ValueError("Invalid current-step supervision")
    label = sup.get("label")
    if type(label) is not int or (label != IGNORE_INDEX and not 0 <= label < len(obs["candidates"])):
        raise ValueError("Supervision label is not a candidate index or IGNORE_INDEX")
    if sup.get("gt_bbox") is not None:
        validate_box(sup["gt_bbox"], "gt_bbox")
    if type(sup.get("candidate_covered")) is not bool:
        raise ValueError("candidate_covered must be explicit")

