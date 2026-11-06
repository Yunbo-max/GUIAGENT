"""Immutable, compute-matched 16-sample Round-0 + Stage-4 study contracts."""
import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]


def load_budget_module():
    spec=importlib.util.spec_from_file_location(
        "gui_samples16",ROOT/"scripts/gui_sample_budget16.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name,train_instances,eval_tasks,rounds", [
    ("mobilegym_full",32,256,3),
    ("mobilegym_pilot",4,8,1),
    ("androidworld_full",32,116,3),
    ("androidworld_pilot",8,8,1),
])
def test_native_round0_and_all_stage4_arms_are_16(name,train_instances,eval_tasks,rounds):
    tool=load_budget_module()
    plan=tool.plan(name)
    assert plan["training_instances"]==train_instances
    assert plan["round0"]["native_per_instance"]==16
    assert plan["round0"]["native_episodes"]==train_instances*16
    assert plan["round0"]["training_updates"]==0
    assert plan["rounds"]==rounds
    assert len(plan["per_round"])==3
    for arm,row in plan["per_round"].items():
        assert row["native_per_instance"]+(row["spectral_per_instance"])==16
        assert row["native_per_instance"]==(16 if arm=="feedback_sd" else 8)
        assert row["spectral_per_instance"]==(0 if arm=="feedback_sd" else 8)
        assert row["native_eval_episodes_per_round"]==eval_tasks
    assert plan["all_training_episodes"]==train_instances*16*(1+3*rounds)
    assert plan["all_eval_episodes"]==eval_tasks*3*rounds
    assert plan["round1_native_repeat_of_round0"] is False
    assert plan["native_checkpoint_per_round"]=="current_post_distillation_checkpoint_theta_r"


def test_full_protocol_changes_sampling_only():
    tool=load_budget_module()
    tool.compare_only_sampling_changes()
    assert tool.plan("mobilegym_full")["all_training_episodes"]==5120
    assert tool.plan("androidworld_full")["all_training_episodes"]==5120
    assert tool.plan("mobilegym_full")["all_eval_episodes"]==2304
    assert tool.plan("androidworld_full")["all_eval_episodes"]==1044


def test_reject_non16_or_wrong_arm_while_preserving_original_configs(tmp_path):
    tool=load_budget_module()
    cfg=tool.config("mobilegym_full")
    cfg["study"]["group_size"]=4
    name=tool.CONFIGS["mobilegym_full"]
    p=tmp_path/name
    p.parent.mkdir(parents=True,exist_ok=True)
    import json
    p.write_text(json.dumps(cfg),encoding="utf-8")
    with pytest.raises(ValueError, match="both be 16"):
        tool.plan("mobilegym_full",root=tmp_path)
