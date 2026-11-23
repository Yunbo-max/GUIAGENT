"""Dynamic action candidates and supervision-only label mapping.

Native action semantics adapted from GUI-JEPA under MIT; see third_party/gui_jepa.
Candidate construction has no action-label argument by design.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .schema import IGNORE_INDEX, RAW_ACTION_KINDS, SCROLL_DIRECTIONS, finite_number, validate_box, validate_candidate


def build_action_candidates(elements: Sequence[Mapping[str, Any]], screen_width: int,
                            screen_height: int) -> list[dict[str, Any]]:
    if screen_width <= 0 or screen_height <= 0:
        raise ValueError("Screen dimensions must be positive")
    result = []
    for index, element in enumerate(elements):
        left, top, right, bottom = validate_box(element["bounds"], "element bounds")
        flags = list(element["flags"])
        if len(flags) != 4 or any(type(flag) is not bool for flag in flags):
            raise ValueError("Element flags must be four bools")
        box = [left / screen_width, top / screen_height, right / screen_width, bottom / screen_height]
        kinds = ["click"] + (["input_text"] if flags[2] else [])
        for kind in kinds:
            result.append({"candidate_id": f"{kind}:{index}", "kind": kind,
                           "element_index": index, "direction": None, "box": list(box),
                           "text": str(element.get("text", "")), "flags": list(flags)})
    for direction in SCROLL_DIRECTIONS:
        result.append({"candidate_id": f"scroll:{direction}", "kind": "scroll",
                       "element_index": None, "direction": direction, "box": None,
                       "text": f"scroll {direction}", "flags": [False] * 4})
    for kind in ("navigate_back", "navigate_home", "wait", "open_app"):
        result.append({"candidate_id": kind, "kind": kind, "element_index": None,
                       "direction": None, "box": None, "text": kind, "flags": [False] * 4})
    for candidate in result:
        validate_candidate(candidate)
    return result


def make_supervision(raw_action: Mapping[str, Any], elements: Sequence[Mapping[str, Any]],
                     candidates: Sequence[Mapping[str, Any]], screen_width: int,
                     screen_height: int) -> dict[str, Any]:
    """Map a recorded label *after* target-independent candidates exist."""
    if screen_width <= 0 or screen_height <= 0:
        raise ValueError("Screen dimensions must be positive")
    kind = raw_action.get("action_type")
    if kind not in RAW_ACTION_KINDS:
        raise ValueError(f"Unsupported recorded action type {kind!r}")
    result: dict[str, Any] = {
        "label": IGNORE_INDEX, "type": kind, "cand_idx": -1, "gt_bbox": None,
        "scroll_dir": None, "text": None, "app_name": None, "candidate_covered": False,
        "raw_action": dict(raw_action),
    }
    if kind in ("click", "long_press"):
        x = finite_number(raw_action.get("x"), "recorded click x")
        y = finite_number(raw_action.get("y"), "recorded click y")
        matches = []
        for index, element in enumerate(elements):
            left, top, right, bottom = validate_box(element["bounds"], "element bounds")
            if left <= x <= right and top <= y <= bottom:
                matches.append(((right - left) * (bottom - top), index))
        if matches:
            _, element_index = min(matches)
            result["cand_idx"] = element_index
            result["gt_bbox"] = list(elements[element_index]["bounds"])
            result["label"] = next((i for i, c in enumerate(candidates)
                                    if c["kind"] == "click" and c["element_index"] == element_index), IGNORE_INDEX)
    elif kind == "input_text":
        if not isinstance(raw_action.get("text"), str):
            raise ValueError("input_text requires a text string")
        result["text"] = raw_action["text"]
        # Native annotations do not identify the editable element for this
        # action. Preserve the upstream first-editable training convention.
        result["label"] = next((i for i, c in enumerate(candidates) if c["kind"] == "input_text"), IGNORE_INDEX)
    elif kind == "scroll":
        direction = raw_action.get("direction")
        if direction not in SCROLL_DIRECTIONS:
            raise ValueError("Recorded scroll requires a valid direction")
        result["scroll_dir"] = direction
        result["label"] = next((i for i, c in enumerate(candidates)
                                if c["kind"] == kind and c["direction"] == direction), IGNORE_INDEX)
    else:
        if kind == "open_app":
            if not isinstance(raw_action.get("app_name"), str):
                raise ValueError("open_app requires an app_name string")
            result["app_name"] = raw_action["app_name"]
        result["label"] = next((i for i, c in enumerate(candidates) if c["kind"] == kind), IGNORE_INDEX)
    result["candidate_covered"] = result["label"] != IGNORE_INDEX
    return result


def history_entry(supervision: Mapping[str, Any], elements: Sequence[Mapping[str, Any]]) -> str:
    """Serialize a *previous* recorded action, following upstream history_text."""
    kind = supervision["type"]
    index = supervision["cand_idx"]
    if kind in ("click", "long_press") and index >= 0:
        return f"{kind} [{elements[index]['text'].split(' | ')[0]}]"
    if kind == "input_text":
        return f"type [{(supervision['text'] or '')[:24]}]"
    if kind == "open_app":
        return f"open [{supervision['app_name']}]"
    return str(kind)


def prediction_from_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    validate_candidate(candidate)
    box = candidate["box"]
    point = None if box is None else [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
    return {"kind": candidate["kind"], "candidate_id": candidate["candidate_id"],
            "element_index": candidate["element_index"], "direction": candidate["direction"],
            "point": point, "text": None, "app_name": None}

