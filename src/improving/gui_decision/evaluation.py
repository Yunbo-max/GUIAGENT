"""Native GUI-JEPA step judge with explicit failure and denominator reporting.

Adapted from scripts/train_policy.py at the commit in third_party/gui_jepa.
Metrics are percentages, and this is offline step accuracy, not task success.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from typing import Any

from .arguments import normalize_text


def failure_result(supervision: Mapping[str, Any], error: str) -> dict[str, Any]:
    gt_kind = supervision["type"]
    is_click = gt_kind in ("click", "long_press")
    covered = bool(supervision.get("gt_bbox")) if is_click else bool(supervision.get("candidate_covered", True))
    return {"correct": False, "type_correct": False, "gt_kind": gt_kind, "pred_kind": None,
            "is_click": is_click, "candidate_covered": covered, "conditional_click_eligible": False,
            "failure": str(error)}


def judge(prediction: Mapping[str, Any] | None, supervision: Mapping[str, Any],
          observation: Mapping[str, Any]) -> dict[str, Any]:
    if prediction is None:
        return failure_result(supervision, "missing_prediction")
    if prediction.get("failure"):
        return failure_result(supervision, str(prediction["failure"]))
    gt_kind, pred_kind = supervision["type"], prediction.get("kind")
    is_click = gt_kind in ("click", "long_press")
    type_correct = pred_kind == gt_kind or (gt_kind == "long_press" and pred_kind == "click")
    covered = bool(supervision.get("gt_bbox")) if is_click else bool(supervision.get("candidate_covered", True))
    result = {"correct": False, "type_correct": bool(type_correct), "gt_kind": gt_kind,
              "pred_kind": pred_kind, "is_click": is_click, "candidate_covered": covered,
              "conditional_click_eligible": bool(is_click and covered and type_correct), "failure": None}
    if not type_correct:
        return result
    if is_click:
        if not covered:
            return result
        # Derive the point from the selected current candidate, as the native
        # scorer does; do not accept arbitrary reported coordinates as a way
        # to bypass the dynamically enumerated action space.
        selected = next((candidate for candidate in observation["candidates"]
                         if candidate["candidate_id"] == prediction.get("candidate_id")
                         and candidate["kind"] == "click"), None)
        if selected is None or selected["box"] is None:
            result["failure"] = "invalid_click_candidate"
            return result
        box = selected["box"]
        x = (box[0] + box[2]) / 2 * observation["screen_width"]
        y = (box[1] + box[3]) / 2 * observation["screen_height"]
        target = supervision["gt_bbox"]
        result["correct"] = bool(math.isfinite(x) and math.isfinite(y)
                                 and target[0] <= x <= target[2] and target[1] <= y <= target[3])
    elif gt_kind == "scroll":
        result["correct"] = prediction.get("direction") == supervision["scroll_dir"]
    elif gt_kind == "input_text":
        value = prediction.get("text")
        result["correct"] = isinstance(value, str) and normalize_text(value) == normalize_text(supervision["text"])
    elif gt_kind == "open_app":
        # Exact case-sensitive string equality in the main GUI-JEPA judge.
        # The separate upstream Qwen baseline normalizes this; do not silently
        # substitute that different rule.
        result["correct"] = prediction.get("app_name") == supervision["app_name"]
    else:
        result["correct"] = True
    return result


class MetricsAccumulator:
    def __init__(self) -> None:
        self.n_steps = self.n_correct = self.n_type_correct = 0
        self.click_total = self.click_covered = self.click_correct = 0
        self.click_conditional_total = self.click_conditional_correct = 0
        self.failure_count = 0
        self.by_type: Counter[str] = Counter()
        self.correct_by_type: Counter[str] = Counter()
        self.type_correct_by_type: Counter[str] = Counter()
        self.predicted_types: Counter[str] = Counter()
        self.failures: Counter[str] = Counter()

    def update(self, result: Mapping[str, Any]) -> None:
        self.n_steps += 1
        correct, type_correct = bool(result["correct"]), bool(result["type_correct"])
        self.n_correct += int(correct)
        self.n_type_correct += int(type_correct)
        kind = str(result["gt_kind"])
        self.by_type[kind] += 1
        self.correct_by_type[kind] += int(correct)
        self.type_correct_by_type[kind] += int(type_correct)
        self.predicted_types[str(result.get("pred_kind"))] += 1
        if result["is_click"]:
            self.click_total += 1
            self.click_covered += int(result["candidate_covered"])
            self.click_correct += int(correct)
        if result["conditional_click_eligible"]:
            self.click_conditional_total += 1
            self.click_conditional_correct += int(correct)
        if result.get("failure"):
            self.failure_count += 1
            self.failures[str(result["failure"])] += 1

    @staticmethod
    def _percent(numerator: int, denominator: int) -> float:
        return 100.0 * numerator / max(denominator, 1)

    def compute(self) -> dict[str, Any]:
        step_accuracy = self._percent(self.n_correct, self.n_steps)
        type_accuracy = self._percent(self.n_type_correct, self.n_steps)
        conditional = self._percent(self.click_conditional_correct, self.click_conditional_total)
        return {
            "n_steps": self.n_steps, "n_correct": self.n_correct,
            "step_accuracy": step_accuracy, "step_acc": step_accuracy,
            "type_accuracy": type_accuracy, "type_acc": type_accuracy,
            "grounding_acc_clicks": conditional, "click_grounding_given_type": conditional,
            "click_grounding_n": self.click_conditional_total,
            "click_success_all_gt_clicks": self._percent(self.click_correct, self.click_total),
            "click_total": self.click_total, "click_correct": self.click_correct,
            "candidate_coverage": self._percent(self.click_covered, self.click_total),
            "candidate_coverage_n": self.click_total, "candidate_covered_clicks": self.click_covered,
            "failed_image_or_inference_count": self.failure_count, "failures": dict(sorted(self.failures.items())),
            "pred_kind_hist": dict(sorted(self.predicted_types.items())),
            "per_type": {kind: {"n": count,
                                "step_acc": self._percent(self.correct_by_type[kind], count),
                                "type_acc": self._percent(self.type_correct_by_type[kind], count)}
                         for kind, count in sorted(self.by_type.items())},
            "units": "percent", "evaluation": "offline_step_prediction",
            "history_source": "recorded_ground_truth",
            "native_compatibility": "GUI-JEPA main train_policy.py judge; long_press accepts click",
        }

