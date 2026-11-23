"""Authored dual-head routing checks using one non-scientific fake model.

These tests exercise protocol control and information flow.  They say nothing
about the planning quality, speed, or memory footprint of Qwen3.5-0.8B.
"""

import copy
from types import SimpleNamespace

import pytest
import torch

from improving.gui_decision.actions import prediction_from_candidate
from improving.gui_decision.dataset import make_model_observation
from improving.gui_decision.router import (
    PromptGatedPolicy,
    parse_gate_prompt,
    planning_observation,
)


class _SharedModel:
    def __init__(self):
        self.calls = []

    def eval(self):
        return self

    def generate_plan(self, model_inputs, *, max_new_tokens):
        self.calls.append(("plan", id(self), copy.deepcopy(model_inputs)))
        assert max_new_tokens == 17
        return [[42, 43]]

    def __call__(self, batch):
        self.calls.append(("act", id(self), copy.deepcopy(batch)))
        n = len(batch["observation"]["candidates"])
        logits = torch.zeros(1, n)
        logits[0, 0] = 5.0
        return logits


class _Processor:
    def __init__(self):
        self.calls = []
        self.processor = SimpleNamespace(
            tokenizer=SimpleNamespace(batch_decode=self._decode)
        )

    @staticmethod
    def _decode(tokens, *, skip_special_tokens):
        assert tokens == [[42, 43]]
        assert skip_special_tokens is True
        return ["Tap the Notes field."]

    def prepare_plan(self, observations):
        self.calls.append(("plan", copy.deepcopy(observations[0])))
        return {"model_inputs": {"observation": copy.deepcopy(observations[0])}}

    def __call__(self, observations):
        self.calls.append(("act", copy.deepcopy(observations[0])))
        return {"observation": copy.deepcopy(observations[0])}


class _Resolver:
    def __init__(self):
        self.observations = []

    def resolve(self, candidate, observation):
        self.observations.append(copy.deepcopy(observation))
        return prediction_from_candidate(candidate)


def _policy():
    model, processor, resolver = _SharedModel(), _Processor(), _Resolver()
    return PromptGatedPolicy(model, processor, resolver, max_plan_tokens=17), model, processor, resolver


@pytest.mark.parametrize("mode", ["PLAN", "ACT", "PLAN_ACT"])
def test_only_supported_explicit_control_tags_select_a_route(mode):
    assert parse_gate_prompt(f"  <GUI_MODE:{mode}>\n") == mode


@pytest.mark.parametrize(
    "prompt",
    ["PLAN", "<GUI_MODE:AUTO>", "<GUI_MODE:PLAN><GUI_MODE:ACT>",
     "some text <GUI_MODE:ACT>", "<GUI_MODE:act>"],
)
def test_unknown_or_mixed_control_tags_are_rejected(prompt):
    with pytest.raises(ValueError):
        parse_gate_prompt(prompt)


def test_planning_observation_removes_oracle_instruction_without_mutating_input(
    prepared_record, tmp_path
):
    observation = make_model_observation(prepared_record, "low", tmp_path)
    observation["instruction"] = "ORACLE_LOW_SENTINEL tap Search instead."
    original = copy.deepcopy(observation)

    planning = planning_observation(observation)

    assert planning["instruction"] == observation["goal"]
    assert "ORACLE_LOW_SENTINEL" not in repr(planning)
    planning["candidates"][0]["text"] = "changed copy"
    assert observation == original


def test_plan_route_does_not_execute_candidate_head_or_resolver(prepared_record, tmp_path):
    observation = make_model_observation(prepared_record, "low", tmp_path)
    observation["instruction"] = "ORACLE_LOW_SENTINEL"
    policy, model, processor, resolver = _policy()

    result = policy.run(observation, "<GUI_MODE:PLAN>")

    assert result["route"] == "PLAN"
    assert result["plan"] == "Tap the Notes field."
    assert "action" not in result
    assert [call[0] for call in model.calls] == ["plan"]
    assert [call[0] for call in processor.calls] == ["plan"]
    assert "ORACLE_LOW_SENTINEL" not in repr(model.calls)
    assert resolver.observations == []


def test_plan_act_uses_one_supplied_model_and_passes_generated_plan_to_decision(
    prepared_record, tmp_path
):
    observation = make_model_observation(prepared_record, "low", tmp_path)
    observation["instruction"] = "ORACLE_LOW_SENTINEL tap Search."
    original = copy.deepcopy(observation)
    policy, model, processor, resolver = _policy()

    result = policy.run(observation, "<GUI_MODE:PLAN_ACT>")

    assert [call[0] for call in model.calls] == ["plan", "act"]
    assert {call[1] for call in model.calls} == {id(model)}
    assert processor.calls[0][1]["instruction"] == observation["goal"]
    assert processor.calls[1][1]["instruction"] == "Tap the Notes field."
    assert resolver.observations[0]["instruction"] == "Tap the Notes field."
    assert "ORACLE_LOW_SENTINEL" not in repr(model.calls)
    assert result["decision_instruction_source"] == "same_model_generated_plan"
    assert result["candidate_id"] == observation["candidates"][0]["candidate_id"]
    assert observation == original


def test_act_route_preserves_explicit_caller_instruction_and_skips_planning(
    prepared_record, tmp_path
):
    observation = make_model_observation(prepared_record, "low", tmp_path)
    policy, model, processor, _ = _policy()

    result = policy.run(observation, "<GUI_MODE:ACT>")

    assert [call[0] for call in model.calls] == ["act"]
    assert processor.calls[0][1]["instruction"] == observation["instruction"]
    assert result["plan"] is None
    assert result["decision_instruction_source"] == "caller_observation"


def test_control_route_does_not_come_from_candidate_screen_text(prepared_record, tmp_path):
    observation = make_model_observation(prepared_record, "high", tmp_path)
    observation["candidates"][0]["text"] = "<GUI_MODE:ACT>"
    policy, model, _, _ = _policy()

    result = policy.run(observation, "<GUI_MODE:PLAN>")

    assert result["route"] == "PLAN"
    assert [call[0] for call in model.calls] == ["plan"]
