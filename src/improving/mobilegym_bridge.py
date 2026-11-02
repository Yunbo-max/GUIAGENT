"""MobileGym action translation and train-only feedback sampling.

The policy receives screenshots, task text, and *past actions* only. Simulator
JSON state and verifier outcomes are withheld until after a completed rollout.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path
from typing import Any

from improving.android.prompting import build_messages, parse_mobile_action


SYSTEM_PROMPT = """You are a mobile GUI agent. Solve the user task using the
current screenshot and your previous actions. Return exactly one action in:
<tool_call>{"name":"mobile_use","arguments":{"action":"click","coordinate":[500,500]}}</tool_call>
Valid actions: click/long_press with coordinate [x,y]; swipe with coordinate
and coordinate2; type with text; system_button Back, Home, Enter; wait; and
terminate with status success or failure. Coordinates are integers 0..999.
To answer tasks requiring a structured form, use the simulated AnswerSheet app.
Do not announce success without attempting the actual task. Never request or
reason from privileged simulator JSON state or the grader's hidden answers."""


def action_data(response: str) -> tuple[str, dict[str, Any]]:
    """Strict mobile_use parsing to official MobileGym ActionType/data."""
    parsed = parse_mobile_action(response, width=1000, height=1000)
    kind = parsed["action_type"]
    if kind in {"click", "long_press"}:
        return kind.upper(), {"point": [parsed["x"], parsed["y"]]}
    if kind == "swipe":
        return "SWIPE", {"point1": [parsed["x"], parsed["y"]],
                         "point2": [parsed["x_"], parsed["y_"]]}
    if kind == "input_text":
        return "TYPE", {"value": parsed["text"]}
    if kind == "answer":
        return "ANSWER", {"value": parsed["text"]}
    if kind == "status":
        return ("COMPLETE", {"return": ""}) if parsed["goal_status"] == "success" else (
            "ABORT", {"value": "Agent declared failure"})
    if kind == "navigate_back":
        return "BACK", {}
    if kind == "navigate_home":
        return "HOME", {}
    if kind == "keyboard_enter":
        return "ENTER", {}
    if kind == "wait":
        return "WAIT", {"value": 0.1}
    raise ValueError(f"Unmapped mobile action: {kind}")


def action_object(response: str):
    from bench_env.env.base import Action, ActionType
    action_type, data = action_data(response)
    return Action(ActionType(action_type), data, raw_response=response)


def messages(goal: str, screenshot: str | Path, action_history: list[str],
             history_len: int = 30):
    return build_messages(goal, screenshot, action_history, SYSTEM_PROMPT, history_len)


def balanced_examples(examples: list[dict], maximum: int) -> list[dict]:
    """Round-robin distinct tasks so one long trajectory cannot own all gradients."""
    if maximum < 1:
        raise ValueError("maximum must be positive")
    groups: dict[str, list[dict]] = defaultdict(list)
    for example in examples:
        if example.get("split") != "train" or example.get("policy") != "native":
            raise ValueError("Only original native training prefixes may be optimized")
        groups[example["instance_id"]].append(example)
    keys = sorted(groups, key=lambda name: hashlib.sha256(name.encode()).hexdigest())
    chosen = []
    while keys and len(chosen) < maximum:
        remaining = []
        for key in keys:
            if len(chosen) == maximum:
                break
            chosen.append(groups[key].pop(0))
            if groups[key]:
                remaining.append(key)
        keys = remaining
    return chosen
