"""UI-Voyager-compatible screenshot prompts and strict mobile action decoding.

The system prompt is read from the pinned author checkout, not reconstructed.
Both comparison arms use this wrapper. Unlike the author's permissive parser,
invalid output is recorded as an invalid action, never silently replaced by wait.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any


class ActionParseError(ValueError):
    """The model output does not encode one executable mobile_use action."""


def load_system_prompt(upstream_root: str | Path) -> str:
    path = Path(upstream_root) / "androidworld/eval/prompts/qwen3vl_instruct.md"
    return path.read_text(encoding="utf-8")


def action_description(response: str) -> str:
    match = re.search(r"(?:^|\n)Action:\s*([^\n]+)", response)
    return re.sub(r"\s+", " ", match.group(1) if match else response[:200]).strip()


def build_messages(goal: str, screenshot: str | Path, history: list[str],
                   system_prompt: str, history_len: int = 100) -> list[dict[str, Any]]:
    if history_len < 1:
        raise ValueError("history_len must be positive")
    recent = history[-history_len:]
    lines = [f"The user query: {goal}", ""]
    if recent:
        lines.append("Task progress (You have done the following operations on the current device):")
        start = len(history) - len(recent) + 1
        lines.extend(f"Step{start + i}: {text}" for i, text in enumerate(recent))
    lines.extend(["", "Current Screenshot:", "",
                  "Please analyze the current screenshot and history to generate the next step."])
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [
            {"type": "text", "text": "\n".join(lines)},
            {"type": "image", "image": str(Path(screenshot).resolve())},
        ]},
    ]


def _coordinate(value: Any, width: int, height: int) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ActionParseError("coordinate must contain two numbers in [0,999]")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
           not math.isfinite(v) or not 0 <= v <= 999 for v in value):
        raise ActionParseError("coordinate must contain two finite numbers in [0,999]")
    return round(value[0] / 999 * (width - 1)), round(value[1] / 999 * (height - 1))


def parse_mobile_action(response: str, width: int, height: int) -> dict[str, Any]:
    """Return validated JSONAction kwargs; never infer success from terminate.

    Coordinates include both screen edges (0 maps to 0, 999 to size-1).
    The upstream JSONAction has no Menu primitive, so Menu is rejected. Wait
    and long-press use the upstream fixed durations; raw duration requests stay
    in the saved response but cannot change that native action interface.
    """
    if width <= 0 or height <= 0:
        raise ValueError("screen dimensions must be positive")
    blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", response, re.DOTALL)
    if len(blocks) != 1:
        raise ActionParseError("Expected exactly one complete <tool_call> block")
    try:
        payload = json.loads(blocks[0], parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
    except (json.JSONDecodeError, ValueError) as error:
        raise ActionParseError("Tool call must be valid finite JSON") from error
    if not isinstance(payload, dict) or payload.get("name") != "mobile_use":
        raise ActionParseError("Tool name must be mobile_use")
    args = payload.get("arguments")
    if not isinstance(args, dict):
        raise ActionParseError("Tool arguments must be an object")
    action = args.get("action")
    if not isinstance(action, str):
        raise ActionParseError("action must be a string")
    if action in ("click", "long_press"):
        x, y = _coordinate(args.get("coordinate"), width, height)
        return {"action_type": action, "x": x, "y": y}
    if action == "swipe":
        x, y = _coordinate(args.get("coordinate"), width, height)
        x2, y2 = _coordinate(args.get("coordinate2"), width, height)
        # Swipe endpoints are exclusive in the native action interface: the
        # destination coordinate maps to one pixel short of the requested point.
        x2, y2 = max(0, x2 - 1), max(0, y2 - 1)
        return {"action_type": "swipe", "x": x, "y": y, "x_": x2, "y_": y2}
    if action in ("type", "answer"):
        if not isinstance(args.get("text"), str):
            raise ActionParseError("type/answer requires text")
        return {"action_type": "input_text" if action == "type" else "answer", "text": args["text"]}
    if action == "system_button":
        mapping = {"Back": "navigate_back", "Home": "navigate_home", "Enter": "keyboard_enter"}
        if not isinstance(args.get("button"), str) or args["button"] not in mapping:
            raise ActionParseError("Supported system buttons: Back, Home, Enter; Menu is unsupported by native JSONAction")
        return {"action_type": mapping[args["button"]]}
    if action == "terminate":
        if args.get("status") not in ("success", "failure"):
            raise ActionParseError("terminate requires success/failure status")
        return {"action_type": "status", "goal_status": args["status"]}
    if action == "wait":
        return {"action_type": "wait"}
    raise ActionParseError(f"Unsupported mobile action: {action!r}")
