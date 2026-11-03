"""CPU contracts for the opt-in MobileGym feedback-only experiment.

These checks do not simulate official MobileGym tasks or validate 4B CUDA.
"""
import json
from types import SimpleNamespace

import pytest
import torch

from improving.mobilegym_bridge import action_data, balanced_examples
from improving.mobilegym_feedback import validate, _settings
from improving.mobilegym_visual_gains import split_by_task


def call(args):
    return '<tool_call>{"name":"mobile_use","arguments":' + json.dumps(args) + '}</tool_call>'


@pytest.mark.parametrize("args,expected", [
    ({"action": "click", "coordinate": [999, 0]},
     ("CLICK", {"point": [999, 0]})),
    ({"action": "swipe", "coordinate": [100, 200],
      "coordinate2": [900, 800]},
     ("SWIPE", {"point1": [100, 200], "point2": [899, 799]})),
    ({"action": "terminate", "status": "success"}, ("COMPLETE", {"return": ""})),
    ({"action": "terminate", "status": "failure"},
     ("ABORT", {"value": "Agent declared failure"})),
    ({"action": "type", "text": "Hello"}, ("TYPE", {"value": "Hello"})),
    ({"action": "system_button", "button": "Back"}, ("BACK", {})),
    ({"action": "wait"}, ("WAIT", {"value": 0.1})),
])
def test_action_conversion(args, expected):
    assert action_data(call(args)) == expected


def test_invalid_actions_are_not_silently_waited():
    with pytest.raises(ValueError):
        action_data("Click at the top of the screen.")
    with pytest.raises(ValueError):
        action_data(call({"action": "click", "coordinate": [1001, 0]}))


def row(task, i):
    return {
        "split": "train", "policy": "native", "instance_id": task,
        "step_index": i, "feedback": "Actual action error.",
    }


def test_balanced_feedback_across_tasks():
    samples = [row("a", i) for i in range(10)] + [row("b", i) for i in range(10)]
    selected = balanced_examples(samples, 4)
    assert len(selected) == 4
    assert sorted(x["instance_id"] for x in selected) == ["a", "a", "b", "b"]
    with pytest.raises(ValueError):
        balanced_examples([dict(row("a", 0), policy="spectral")], 1)


def test_fit_guard_task_disjoint():
    samples = [row(t, i) for t in "abcdef" for i in range(3)]
    fit, guard = split_by_task(samples, fit_limit=8, guard_limit=4)
    assert fit and guard
    assert {r["instance_id"] for r in fit}.isdisjoint(
        {r["instance_id"] for r in guard})
    with pytest.raises(ValueError, match="signal_shortage"):
        split_by_task([row("a", 0)])


def _config():
    return {
        "model": {"checkpoint": "local-model"}, "mobilegym": {"pinned_commit": "sha"},
        "spectral": {"geometry_source": "feedback_only", "tau": 1.0,
                     "alpha": 0.0, "last_n_layers": 2},
        "train": {"max_updates": 2},
        "study": {
            "train_task_limit": 2, "test_task_limit": 2, "round0_repeats": 4,
            "group_size": 4, "rounds": 1, "max_steps": 10, "history_len": 10,
            "arms": ["feedback_sd", "feedback_spectrum_learned"],
        },
        "gain_learning": {"max_steps": 1},
    }


def test_exploration_contract_and_independent_schedules():
    cfg = _config()
    validate(cfg)
    assert _settings(cfg, "feedback_spectrum_learned")["spectral"]["tau"] == 0.0
    assert _settings(cfg, "feedback_spectrum_tau")["spectral"]["tau_decay"] == .6
    assert _settings(cfg, "feedback_spectrum_gated")["spectral"]["tau_mode"] == "adaptive"
    cfg["spectral"]["geometry_source"] = "anchor_feedback"
    with pytest.raises(ValueError, match="feedback_only"):
        validate(cfg)
    cfg["spectral"]["geometry_source"] = "feedback_only"
    cfg["study"]["round0_repeats"] = 2
    with pytest.raises(ValueError, match="Round 0"):
        validate(cfg)


def test_feedback_only_covariance_matches_feedback_moment(tmp_path):
    # Same tiny visual contract fixture used by the existing AndroidWorld tests.
    from tests.test_android_geometry import _Policy, _example
    from improving.android.geometry import collect_operators
    model = _Policy()
    feedback = _example("train", instance_id="train_t1")
    ops, receipt = collect_operators(
        model, [], [feedback],
        {"spectral": {
            "geometry_source": "feedback_only", "last_n_layers": 1,
            "tau": 0.0, "alpha": 0.0,
        }}, tmp_path)
    assert receipt["geometry_source"] == "feedback_only"
    assert receipt["tau_entropy_source"] == "current_checkpoint_training_feedback_kl_moments_only"
    assert not receipt["targets"][next(iter(ops))]["anchor_examples"]
    assert all(torch.allclose(v, torch.eye(len(v), dtype=v.dtype)) for v in ops.values())


def test_no_heldout_feedback_in_gain_fit():
    bad = row("task", 0)
    bad["split"] = "eval"
    with pytest.raises(ValueError, match="native-prefix training feedback"):
        split_by_task([bad, row("other", 1)])
