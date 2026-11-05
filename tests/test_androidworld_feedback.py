"""No-emulator tests for the verified AndroidWorld Learned second benchmark."""
import copy
import json
from pathlib import Path
import pytest

from improving.androidworld_feedback import (
    ARMS, PROTOCOL, _fit_and_guard, _verify_episode_population, validate,
)


ROOT=Path(__file__).resolve().parents[1]

def cfg(filename):
    return json.loads((ROOT/"configs"/"android"/filename).read_text())


def test_windows_full_and_pilot_share_true_learned_math():
    main=cfg("androidworld_feedback_learned_windows_96gb.json")
    pilot=cfg("androidworld_feedback_learned_windows_pilot_96gb.json")
    for row in (main,pilot):
        assert validate(row) is row
        assert row["protocol"]==PROTOCOL
        assert row["study"]["arms"]==list(ARMS)
        assert row["spectral"]["geometry_source"]=="feedback_only"
        assert row["model"]["device"]=="cuda:0"
        assert row["android"]["adb_path"]=="auto"
    assert main["study"]["test_task_limit"]==116
    assert pilot["study"]["test_task_limit"]==8
    assert main["output_dir"]!=pilot["output_dir"]


def test_rejects_nonverified_reward_settings_and_mixed_methods():
    base=cfg("androidworld_feedback_learned_windows_96gb.json")
    bad=copy.deepcopy(base)
    bad["spectral"]["geometry_source"]="anchor_feedback"
    with pytest.raises(ValueError, match="calibration"):
        validate(bad)
    bad=copy.deepcopy(base)
    bad["study"]["arms"]=["feedback_spectrum_learned"]
    with pytest.raises(ValueError, match="matched"):
        validate(bad)


def test_holdout_never_provides_training_feedback():
    episodes=[{"split":"eval","instance_id":"e1","status":"complete","success":True}]
    instances=[{"instance_id":"e1"}]
    _verify_episode_population(episodes, instances, 1, "eval")
    with pytest.raises(ValueError, match="split"):
        _verify_episode_population(episodes, instances, 1, "train")


def test_moment_fit_guard_tasks_do_not_overlap():
    rows=[{"split":"train","policy":"native","instance_id":f"task{i}",
           "feedback":"Native action feedback","step_index":0} for i in range(6)]
    fit,guard=_fit_and_guard(rows)
    assert fit and guard
    assert {x["instance_id"] for x in fit}.isdisjoint(
        {x["instance_id"] for x in guard})

def test_windows_config_loader_is_separate_from_legacy_arm_validation(monkeypatch):
    import improving.androidworld_feedback as study
    from improving.android import windows_sdk
    sample = cfg("androidworld_feedback_learned_windows_pilot_96gb.json")
    monkeypatch.setattr(windows_sdk, "resolve_sdk_settings", lambda raw: {
        **raw, **{key: "mock" for key in
        ("adb_path","emulator_path","android_sdk_root","android_avd_home")}
    })
    # The old android.study.load_config rejects the new arm names. This
    # runner must use its own validated opt-in protocol.
    assert study.validate(sample)["study"]["arms"] == list(ARMS)

