"""Paired comparison checks: no mismatch disguised as performance gain."""
from copy import deepcopy
import pytest

from scripts.compare_mobilegym_evals import compare


def _cases(n=3):
    return {
        f"task{i}": {
            "instance_id": f"inst{i}",
            "success": i == 1,
            "max_steps_effective": 30,
            "judge": {"progress": 0.5 if i == 0 else 1.0 if i == 1 else 0.0},
        }
        for i in range(n)
    }


def test_paired_comparison_and_deltas():
    first = _cases()
    second = deepcopy(first)
    second["task0"]["success"] = True
    second["task0"]["judge"]["progress"] = 1.0
    stats = compare(first, second, require_full=False)
    assert stats["tasks"] == 3
    assert stats["candidate_only_successes"] == 1
    assert stats["sr_delta_pp"] == pytest.approx(100/3)
    assert stats["pr_delta_pp"] == pytest.approx(50/3)


def test_different_task_instances_or_budgets_rejected():
    x, y = _cases(), _cases()
    y["task0"]["instance_id"] = "DIFFERENT"
    with pytest.raises(ValueError, match="parameters"):
        compare(x, y, require_full=False)
    y = _cases()
    y["task0"]["max_steps_effective"] = 10
    with pytest.raises(ValueError, match="step budget"):
        compare(x, y, require_full=False)
    with pytest.raises(ValueError, match="256"):
        compare(x, x, require_full=True)
