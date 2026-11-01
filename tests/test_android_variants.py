"""Software orchestration checks; mocks are not native qualification or results."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from improving.android import study
from improving.android.tau import resolve_tau


def config(tmp_path, *, variants=True):
    value = {
        "output_dir": str(tmp_path / "runs"), "manifest": "unused.json",
        "model": {"checkpoint": str(tmp_path / "model")},
        "android": {"upstream_root": "unused"},
        "study": {"warmup_rounds": 1, "rounds": 2, "group_size": 4, "spectral_fraction": .5},
        "spectral": {"tau": 1., "alpha": .5, "max_anchor_examples": 8},
        "train": {"max_updates": 2},
    }
    if variants:
        value["study"]["arms"] = list(study.ARMS)
    return value


def test_both_retains_legacy_selection_and_all_follows_configuration(tmp_path):
    legacy = config(tmp_path, variants=False)
    current = config(tmp_path)
    assert study.configured_arms(legacy) == study.LEGACY_ARMS
    assert study.selected_arms(legacy, "all") == study.LEGACY_ARMS
    assert study.selected_arms(current, "both") == study.LEGACY_ARMS
    assert study.selected_arms(current, "all") == study.ARMS
    with pytest.raises(ValueError, match="not configured"):
        study.selected_arms(legacy, "feedback_spectrum_tau")


def test_arm_settings_are_independent_and_preserve_shared_geometry(tmp_path):
    original = config(tmp_path)
    before = deepcopy(original)
    decay = study.arm_config(original, "feedback_spectrum_tau")
    gate = study.arm_config(original, "feedback_spectrum_gated")
    fixed = study.arm_config(original, "feedback_spectrum")
    assert decay["spectral"]["tau_decay"] == .6
    assert gate["spectral"]["tau_mode"] == "adaptive"
    assert gate["spectral"]["tau_decay"] is None
    assert gate["spectral"]["tau_gamma"] == 5.
    assert "tau_mode" not in fixed["spectral"]
    assert all(value["spectral"]["alpha"] == .5 for value in (decay, gate, fixed))
    decay["train"]["max_updates"] = 99
    assert original == before and gate["train"]["max_updates"] == 2


@pytest.mark.parametrize("override", [
    {"tau_mode": "adaptive", "tau_decay": .6}, {"alpha": .9},
    {"tau": float("nan")}, {"tau": True},
])
def test_invalid_variant_controls_rejected(tmp_path, override):
    value = config(tmp_path)
    value["spectral_variants"] = {"feedback_spectrum_tau": override}
    with pytest.raises(ValueError):
        study.validate_config(value)


@pytest.mark.parametrize("arms", [[], ["unknown"], ["ui_voyager", "ui_voyager"], "all"])
def test_invalid_configured_arm_sets_rejected(tmp_path, arms):
    value = config(tmp_path)
    value["study"]["arms"] = arms
    with pytest.raises(ValueError, match="study.arms"):
        study.validate_config(value)


@pytest.mark.parametrize("arm", ["unknown", "feedback_spectrum_tau", "feedback_spectrum_gated"])
def test_unknown_or_unconfigured_run_stops_before_assets_or_output(tmp_path, monkeypatch, arm):
    value = config(tmp_path, variants=False)
    def unexpected(*args, **kwargs):
        raise AssertionError("Arm validation must precede asset access")
    monkeypatch.setattr(study, "verify_upstream", unexpected)
    monkeypatch.setattr(study, "resolve_checkpoint", unexpected)
    with pytest.raises(ValueError):
        study.run_study(value, arm)
    assert not Path(value["output_dir"]).exists()


def test_unknown_qualification_arm_stops_before_assets(tmp_path, monkeypatch):
    from improving.android import qualification
    value = config(tmp_path, variants=False)
    def unexpected(*args, **kwargs):
        raise AssertionError("Arm validation must precede asset access")
    monkeypatch.setattr(qualification, "verify_upstream", unexpected)
    with pytest.raises(ValueError, match="not configured"):
        qualification.qualify(value, "feedback_spectrum_gated")
    assert not Path(value["output_dir"]).exists()


def test_completed_geometry_rejects_rebased_reference_and_changed_effective_tau(tmp_path):
    spectral = study.arm_config(config(tmp_path), "feedback_spectrum_gated")["spectral"]
    valid = resolve_tau(spectral, 2, entropy=.4, reference_entropy=.8)
    receipt = {"tau": valid["effective_tau"], "tau_policy": valid}
    assert study.validate_geometry_tau(receipt, spectral, 2, .8) == valid
    with pytest.raises(ValueError, match="frozen reference"):
        study.validate_geometry_tau(receipt, spectral, 2, .7)
    receipt["tau"] = 1.
    with pytest.raises(ValueError, match="effective tau"):
        study.validate_geometry_tau(receipt, spectral, 2, .8)


def install_mock_runtime(monkeypatch, calls):
    """Replace every environment/model/training boundary with software records."""
    def module(name, **members):
        result = ModuleType(name)
        result.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, result)

    class Policy:
        def __init__(self, checkpoint, settings):
            pass

        def close(self):
            pass

    def collect(checkpoint, instances, path, settings, repeats, **kwargs):
        calls["collection"].append((path, repeats, checkpoint))
        return [{"source": str(path)}]

    def train(checkpoint, examples, path, settings, kind):
        calls["training"].append((path, kind, checkpoint))
        return str(path / "checkpoint")

    def geometry(policy, anchors, feedback, settings, path, *, round_index=1, reference_entropy=None):
        calls["geometry"].append((path, round_index, reference_entropy, deepcopy(feedback)))
        resolved = resolve_tau(settings["spectral"], round_index, entropy=.8 / round_index,
                               reference_entropy=reference_entropy)
        receipt = {"tau": resolved["effective_tau"], "tau_policy": resolved}
        path.mkdir(parents=True)
        (path / "operators.pt").write_bytes(b"mock operator; never a model artifact")
        study.atomic_json(path / "calibration.json", receipt)
        return {"projection": object()}, receipt

    module("improving.android.model", QwenPolicy=Policy)
    module("improving.android.geometry", collect_operators=geometry)
    module("improving.android.data",
           success_examples=lambda episodes, split="train": [{"source": episodes[0]["source"], "split": split}],
           feedback_examples=lambda episodes, settings: ([{"source": episodes[0]["source"]}], {}))
    module("improving.android.baseline", build_grsd_examples=lambda *args: ([{"source": "grsd"}], {}))
    monkeypatch.setattr(study, "verify_upstream", lambda root: None)
    monkeypatch.setattr(study, "resolve_checkpoint", lambda settings: settings["model"]["checkpoint"])
    monkeypatch.setattr(study, "load_manifest", lambda path: {"seed": 7, "instances": [
        {"split": split} for split in ("train", "calibration", "eval")]})
    monkeypatch.setattr(study, "checkpoint_files", lambda checkpoint: [])
    monkeypatch.setattr(study, "example_images", lambda examples: [])
    monkeypatch.setattr(study, "source_fingerprint", lambda: "mock-source")
    monkeypatch.setattr(study, "runtime_fingerprint", lambda: {})
    monkeypatch.setattr(study, "_collect", collect)
    monkeypatch.setattr(study, "_train", train)
    monkeypatch.setattr(study, "load_operators", lambda path: {"projection": object()})
    monkeypatch.setattr(study, "report", lambda settings: {"status": "mock_orchestration_only"})


def test_all_arms_share_initialization_and_anchors_with_local_feedback_and_frozen_reference(tmp_path, monkeypatch):
    value = config(tmp_path)
    calls = {"collection": [], "training": [], "geometry": []}
    install_mock_runtime(monkeypatch, calls)
    study.run_study(value, "all")
    assert sum(path.name == "anchor_collection" for path, _, _ in calls["collection"]) == 1
    assert sum(kind == "rft" for _, kind, _ in calls["training"]) == 1
    first_inputs = {checkpoint for path, _, checkpoint in calls["training"] if path.parent.name == "round_01"}
    assert len(first_inputs) == 1
    assert [kind for _, kind, _ in calls["training"]].count("feedback") == 6
    for method in study.FEEDBACK_ARMS:
        geometry = [row for row in calls["geometry"] if method in row[0].parts]
        assert geometry[0][3] == []
        assert method in geometry[1][3][0]["source"]
        for name in ("collect_native", "collect_spectral"):
            assert [repeats for path, repeats, _ in calls["collection"] if method in path.parts and path.name == name] == [2, 2]
    gate = [row for row in calls["geometry"] if "feedback_spectrum_gated" in row[0].parts]
    assert [row[2] for row in gate] == [None, .8]

    # Replaying completed stages restores the original H_ref without recalibration.
    study.run_study(value, "all")
    assert len(calls["geometry"]) == 6
    path = Path(value["output_dir"]) / "feedback_spectrum_gated" / "round_02" / "geometry"
    complete = json.loads((path / "complete.json").read_text())
    wrong = resolve_tau(study.arm_config(value, "feedback_spectrum_gated")["spectral"],
                        2, entropy=.4, reference_entropy=.7)
    receipt = {"tau": wrong["effective_tau"], "tau_policy": wrong}
    study.atomic_json(path / "calibration.json", receipt)
    complete["calibration"] = receipt
    complete["artifacts"]["calibration.json"] = study.file_sha256(path / "calibration.json")
    study.atomic_json(path / "complete.json", complete)
    with pytest.raises(ValueError, match="frozen reference"):
        study.run_study(value, "feedback_spectrum_gated")


def test_report_preserves_all_configured_missing_cells_and_tau_history(tmp_path):
    value = config(tmp_path)
    root = Path(value["output_dir"])
    for arm in study.FEEDBACK_ARMS:
        spectral = study.arm_config(value, arm)["spectral"]
        for round_index in (1, 2):
            path = root / arm / f"round_{round_index:02d}" / "geometry"
            path.mkdir(parents=True)
            policy = resolve_tau(spectral, round_index, entropy=.8 / round_index,
                                 reference_entropy=.8 if round_index == 2 else None)
            receipt = {"tau": policy["effective_tau"], "tau_policy": policy}
            study.atomic_json(path / "calibration.json", receipt)
            (path / "operators.pt").write_bytes(b"mock operator; never a model artifact")
            study._finish(path, {"calibration": receipt}, ["operators.pt", "calibration.json"])
    result = study.report(value)
    assert result["arms"] == list(study.ARMS)
    assert len(result["evaluations"]) == 9
    assert all(row["status"] == "not_run" for row in result["evaluations"])
    assert set(study.ARMS) <= set(result["costs"])
    assert len(result["tau_history"]) == 6
    gate = [row for row in result["tau_history"] if row["arm"] == "feedback_spectrum_gated"]
    assert [row["reference_entropy"] for row in gate] == [.8, .8]
    assert gate[1]["effective_tau"] == .5 ** 5


def test_missing_spectral_anchor_does_not_block_preceding_baseline(tmp_path, monkeypatch):
    value = config(tmp_path)
    calls = {"collection": [], "training": [], "geometry": []}
    install_mock_runtime(monkeypatch, calls)
    data = sys.modules["improving.android.data"]
    original = data.success_examples
    monkeypatch.setattr(data, "success_examples",
                        lambda episodes, split="train": [] if split == "calibration" else original(episodes, split))
    with pytest.raises(RuntimeError, match="No native verified calibration success"):
        study.run_study(value, "all")
    assert sum(kind == "grsd" for _, kind, _ in calls["training"]) == 2
    assert calls["geometry"] == []


@pytest.mark.parametrize("arm", ["feedback_spectrum_tau", "feedback_spectrum_gated"])
def test_report_rejects_missing_tau_policy_for_new_variants(tmp_path, arm):
    value = config(tmp_path)
    path = Path(value["output_dir"]) / arm / "round_01" / "geometry"
    path.mkdir(parents=True)
    receipt = {"tau": 1.}
    study.atomic_json(path / "calibration.json", receipt)
    (path / "operators.pt").write_bytes(b"mock operator; never a model artifact")
    study._finish(path, {"calibration": receipt}, ["operators.pt", "calibration.json"])
    with pytest.raises(ValueError, match="no tau_policy"):
        study.report(value)


def test_shared_warmup_accepts_declared_subset_using_all_selection(tmp_path, monkeypatch):
    value = config(tmp_path)
    value["study"]["arms"] = ["feedback_spectrum_gated"]
    calls = {"collection": [], "training": [], "geometry": []}
    install_mock_runtime(monkeypatch, calls)
    result = study.run_study(value, "all", warmup_only=True)
    assert result["status"] == "warmup_complete"
    assert [kind for _, kind, _ in calls["training"]] == ["rft"]
    assert calls["geometry"] == []
