"""Software contracts for GUI tau variants; not Android benchmark evidence."""
import json
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from improving.android.geometry import normalized_spectral_entropy
from improving.android.tau import resolve_tau, validate_tau_settings


def test_constant_and_decay_have_one_based_rounds():
    assert resolve_tau({"tau": 2.0}, 8)["effective_tau"] == 2.0
    settings = {"tau": 2.0, "tau_decay": 0.6}
    assert resolve_tau(settings, 1)["effective_tau"] == 2.0
    assert resolve_tau(settings, 3)["effective_tau"] == pytest.approx(0.72)
    assert resolve_tau(settings, 3)["mode"] == "decay"


def test_adaptive_binds_round_one_and_can_increase_strength():
    settings = {"tau": 1.0, "tau_mode": "adaptive", "tau_gamma": 5.0}
    first = resolve_tau(settings, 1, entropy=0.8)
    assert first["reference_entropy"] == 0.8
    assert first["effective_tau"] == 1.0
    lower = resolve_tau(settings, 2, entropy=0.4, reference_entropy=first["reference_entropy"])
    assert lower["entropy_ratio"] == 0.5
    assert lower["effective_tau"] == pytest.approx(0.5 ** 5)
    higher = resolve_tau(settings, 3, entropy=1.0, reference_entropy=first["reference_entropy"])
    assert higher["effective_tau"] == pytest.approx(1.25 ** 5)
    assert higher["effective_tau"] > first["tau_base"]  # faithful unclipped gate


def test_zero_current_entropy_is_not_a_new_reference():
    settings = {"tau_mode": "adaptive"}
    result = resolve_tau(settings, 2, entropy=0.0, reference_entropy=0.7)
    assert result["effective_tau"] == 0.0
    assert result["reference_entropy"] == 0.7
    with pytest.raises(ValueError, match="positive round-one"):
        resolve_tau(settings, 1, entropy=0.0)


@pytest.mark.parametrize("kwargs", [
    {"round_index": 1},
    {"round_index": 2, "entropy": 0.7},
    {"round_index": 2, "entropy": 0.7, "reference_entropy": 0.0},
    {"round_index": 2, "entropy": float("nan"), "reference_entropy": 0.7},
    {"round_index": 2, "entropy": 0.7, "reference_entropy": float("inf")},
    {"round_index": 1, "entropy": 0.7, "reference_entropy": 0.8},
])
def test_adaptive_missing_or_changed_reference_fails_instead_of_rebasing(kwargs):
    with pytest.raises(ValueError):
        resolve_tau({"tau_mode": "adaptive"}, **kwargs)


@pytest.mark.parametrize("settings", [
    {"tau": True}, {"tau": -1}, {"tau": float("inf")},
    {"tau_decay": 0}, {"tau_decay": 1.1}, {"tau_decay": True},
    {"tau_mode": "fixed"},
    {"tau_mode": "adaptive", "tau_decay": 1.0},
    {"tau_mode": "adaptive", "tau_gamma": 0},
    {"tau_mode": "adaptive", "tau_gamma": True},
    {"tau_mode": "adaptive", "tau_gamma": float("nan")},
])
def test_invalid_strength_policies_are_rejected(settings):
    with pytest.raises(ValueError):
        validate_tau_settings(settings)


@pytest.mark.parametrize("round_index", [0, -1, True, 1.5])
def test_invalid_round_index_is_rejected(round_index):
    with pytest.raises(ValueError, match="positive integer"):
        resolve_tau({}, round_index)


def test_overflow_is_not_silently_clipped():
    with pytest.raises(ValueError, match="finite|overflow"):
        resolve_tau({"tau_mode": "adaptive", "tau_gamma": 1000}, 2,
                    entropy=1.0, reference_entropy=0.01)


def _record(diagonal):
    return {"covariance": torch.diag(torch.tensor(diagonal, dtype=torch.float64)),
            "example_count": 1, "token_count": 2, "target_token_count": 1}


def test_legacy_entropy_normalizes_positive_rank_not_ambient_width():
    rank_two = normalized_spectral_entropy({"k": _record([1, 1, 0, 0])})
    rank_four = normalized_spectral_entropy({"k": _record([1, 1, 1, 1])})
    assert rank_two == pytest.approx(1)
    assert rank_four == pytest.approx(1)
    assert normalized_spectral_entropy({"k": _record([1, 0, 0, 0])}) is None
    assert normalized_spectral_entropy({"k": _record([1, 1, 0, 0]),
                                        "v": _record([1, 0, 0, 0])}) == pytest.approx(1)


def _mock_moments(monkeypatch, anchor_diagonal, feedback_diagonal):
    import improving.android.geometry as geometry
    model = nn.Module()
    model.projection = nn.Linear(4, 4)
    policy = SimpleNamespace(model=model, checkpoint="unit-test-checkpoint")
    monkeypatch.setattr(geometry, "select_targets", lambda model, settings: ["projection"])
    def collect(policy, examples, names, kind, config):
        return {"projection": _record(anchor_diagonal if kind == "anchor" else feedback_diagonal)}
    monkeypatch.setattr(geometry, "_collect_moments", collect)
    return policy


def test_collect_gate_uses_anchor_entropy_and_actual_mixed_operator(monkeypatch, tmp_path):
    from improving.android.geometry import collect_operators, mix_moments
    from improving.spectral import make_operator
    policy = _mock_moments(monkeypatch, [3, 1, 0, 0], [1, 1, 1, 1])
    settings = {"spectral": {"tau": 1.0, "tau_mode": "adaptive", "tau_gamma": 5.0, "alpha": 0.5}}
    operators, receipt = collect_operators(policy, [{"id": "anchor"}], [{"id": "feedback"}],
                                           settings, tmp_path, round_index=2, reference_entropy=1.0)
    expected_entropy = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25)) / math.log(2)
    assert receipt["tau_policy"]["entropy"] == pytest.approx(expected_entropy)
    assert receipt["tau_policy"]["reference_entropy"] == 1.0
    assert receipt["tau"] == pytest.approx(expected_entropy ** 5)
    mixed, _ = mix_moments(_record([3, 1, 0, 0])["covariance"],
                            _record([1, 1, 1, 1])["covariance"], alpha=0.5)
    assert torch.allclose(operators["projection"], make_operator(mixed, tau=receipt["tau"]))
    assert "fixed_anchor_reference_nll" in receipt["tau_entropy_source"]
    assert json.loads((tmp_path / "calibration.json").read_text())["tau_policy"] == receipt["tau_policy"]


def test_collect_decay_uses_effective_tau_and_keeps_tensor_bundle(monkeypatch, tmp_path):
    from improving.android.geometry import collect_operators, mix_moments
    from improving.spectral import make_operator
    policy = _mock_moments(monkeypatch, [3, 1, 0, 0], [1, 1, 1, 1])
    settings = {"spectral": {"tau": 1.0, "tau_decay": 0.6}}
    operators, receipt = collect_operators(policy, [{"id": "anchor"}], [], settings,
                                           tmp_path, round_index=3)
    assert receipt["tau"] == pytest.approx(0.36)
    assert receipt["tau_policy"]["entropy"] is None
    mixed, _ = mix_moments(_record([3, 1, 0, 0])["covariance"], None)
    assert torch.allclose(operators["projection"], make_operator(mixed, tau=0.36))
    bundle = torch.load(tmp_path / "operators.pt", map_location="cpu", weights_only=True)
    assert set(bundle) == {"operators", "anchor", "feedback"}


def test_unavailable_gate_entropy_creates_no_geometry_artifact(monkeypatch, tmp_path):
    from improving.android.geometry import collect_operators
    policy = _mock_moments(monkeypatch, [1, 0, 0, 0], [1, 1, 1, 1])
    destination = tmp_path / "geometry"
    with pytest.raises(ValueError, match="entropy"):
        collect_operators(policy, [{"id": "anchor"}], [],
                          {"spectral": {"tau_mode": "adaptive"}}, destination)
    assert not destination.exists()
