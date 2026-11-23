"""CPU contracts for train-only MobileGym verified rolling distillation.

Tests deliberately use tiny synthetic trajectories, NOT test-split oracle
screenshots, a real GPU, or a completed MobileGym experiment.
"""
import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from improving.mobilegym_rolling import (
    RollingSettings, build_examples, load_episodes, prepare_file,
)
from improving.mobilegym_rolling_train import _load_rows
from improving.mobilegym_step_verify import verify_checkpoint


def episode(split="train", *, good=True, verified=True):
    points = (0.0, 0.0, 0.5, 1.0)
    steps = []
    for i in range(3):
        steps.append({
            "index": i,
            "messages": [
                {"role": "system", "content": "Follow the task"},
                {"role": "user", "content": [
                    {"type": "text", "text": f"step {i}"},
                    {"type": "image", "image": f"/no-test-images/step{i}.jpg"},
                ]},
            ],
            "response": "an executed sampled action",
            "response_token_ids": [25, 26],
            "action": {"action_type": "CLICK", "data": {"point": [10 + i, 20]}},
            "next_screenshot": f"/teacher-only/after{i}.jpg",
            "verification": {
                "status": "verified" if verified else "invalid",
                "progress_before": points[i],
                "progress_after": points[i+1],
                "clean": good,
                "passed": i == 2,
            },
        })
    return {
        "episode_id": "ep1", "instance_id": "instance1",
        "task_name": "test_task", "goal": "achieve test task",
        "split": split, "policy": "native", "steps": steps,
        "success": good, "status": "complete",
    }


def test_rolling_two_horizons_with_real_future_progress():
    rows, receipt = build_examples(
        [episode()], RollingSettings(horizon=3, verify_lookahead=3))
    assert receipt["source_split"] == "train"
    assert receipt["step_reward_claim"] is False
    assert rows
    current = [r for r in rows if r["horizon_offset"] == 0]
    future = [r for r in rows if r["horizon_offset"] > 0]
    assert current and future
    assert all(r["objective"] == "next_action_kl" for r in current)
    assert all(r["objective"] == "future_action_type_kl" for r in future)
    assert all(r["response"] == "CLICK" and r["response_token_ids"] is None for r in future)
    assert all(r["verified_progress_gain"] > 0 for r in rows)
    assert all(r["evidence_provenance"] == "actual_train_episode_interim_verifier" for r in rows)
    assert "Verified progress gain" in future[0]["feedback"]
    assert all("hindsight" not in str(r["messages"]).lower() for r in rows)


def test_strict_no_test_or_unverified_data():
    with pytest.raises(ValueError, match="Unexpected split"):
        build_examples([episode(split="eval")])
    with pytest.raises(RuntimeError, match="signal_shortage"):
        build_examples([episode(verified=False)])
    with pytest.raises(RuntimeError, match="signal_shortage"):
        build_examples([episode(good=False)])
    with pytest.raises(ValueError, match="Require"):
        RollingSettings(horizon=4, verify_lookahead=3)


def test_prepare_and_train_data_validation(tmp_path):
    source = tmp_path / "train" / "task1"
    source.mkdir(parents=True)
    (source / "episode.json").write_text(json.dumps(episode()), encoding="utf-8")
    output = tmp_path / "new" / "verified.jsonl"
    receipt = prepare_file(
        source.parent, output,
        RollingSettings(horizon=3, verify_lookahead=3))
    assert output.is_file()
    assert receipt["source_files"] == 1
    rows, digest = _load_rows(output)
    assert rows and len(digest) == 64
    with pytest.raises(FileExistsError):
        prepare_file(source.parent, output, RollingSettings())

    heldout = tmp_path / "eval"
    heldout.mkdir()
    (heldout / "episode.json").write_text(json.dumps(episode(split="eval")))
    with pytest.raises(ValueError, match="Held-out"):
        load_episodes(heldout)
    raw = json.loads(output.read_text().splitlines()[0])
    raw["split"] = "eval"
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text(json.dumps(raw) + "\n")
    with pytest.raises(ValueError, match="train-only"):
        _load_rows(bad_file)


@dataclass
class _Obs:
    state: dict


class _Env:
    agent_answer = None

    async def get_state(self):
        return {"os": {"active": "profile"}, "apps": {}}


class _Judge:
    judge_error = None
    progress = 0.75
    clean = True
    success = False
    passed = False


class _Evaluator:
    async def evaluate(self, task, initial, current, answer):
        assert current.state["os"]["active"] == "profile"
        assert answer.agent_answer is None
        return _Judge()


def test_interim_verifier_only_uses_real_state():
    result = asyncio.run(verify_checkpoint(
        _Evaluator(), object(), _Obs(state={}), _Obs(state={}),
        _Env(), progress_before=0.25,
    ))
    assert result["status"] == "verified"
    assert result["delta_progress"] == pytest.approx(0.5)

    class BadEnv(_Env):
        async def get_state(self):
            return {}
    invalid = asyncio.run(verify_checkpoint(
        _Evaluator(), object(), _Obs({}), _Obs({}),
        BadEnv(), progress_before=0.25,
    ))
    assert invalid["status"] == "invalid"
    assert invalid["progress_after"] is None


def test_single_gpu_formal_config_matches_160_256():
    from improving.mobilegym_feedback import validate
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "configs/mobilegym/verified_rolling_4b_97gb.json").read_text())
    assert validate(cfg) is cfg
    assert cfg["study"]["train_task_limit"] == 160
    assert cfg["study"]["test_task_limit"] == 256
    assert cfg["study"]["round0_repeats"] == 8
    assert cfg["mobilegym"]["official_rl_step_budgets"] is True
    assert cfg["rolling"]["capture_step_verifier"] is True
    assert cfg["model"]["device"] == "cuda:0"


def test_future_screen_is_teacher_only_and_never_leaks_to_student():
    from improving.mobilegym_rolling_train import _encode_pair

    rows, _ = build_examples([episode()])
    row = next(r for r in rows if r["objective"] == "future_action_type_kl")

    class Tokenizer:
        eos_token_id = 2
        def encode(self, text, add_special_tokens=False):
            assert text == "CLICK"
            return [99]

    class Policy:
        processor = SimpleNamespace(tokenizer=Tokenizer())
        calls = []
        def encode(self, messages, ids, feedback=None):
            self.calls.append((messages, ids, feedback))
            return {"ids": ids}

    policy = Policy()
    _encode_pair(policy, row)
    assert len(policy.calls) == 2
    student_messages, _, student_feedback = policy.calls[0]
    teacher_messages, _, teacher_feedback = policy.calls[1]
    assert student_feedback is None
    assert teacher_feedback
    assert not any(
        block.get("image", "").startswith("/teacher-only")
        for block in student_messages[-1]["content"]
        if isinstance(block, dict)
    )
    assert any(
        block.get("image", "").startswith("/teacher-only")
        for block in teacher_messages[-1]["content"]
        if isinstance(block, dict)
    )
