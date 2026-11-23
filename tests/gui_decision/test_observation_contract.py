"""Observation/label separation contracts, authored without a model run."""

import copy

import pytest

from improving.gui_decision.actions import build_action_candidates, make_supervision
from improving.gui_decision.dataset import make_model_observation
from improving.gui_decision.schema import validate_model_observation


def test_current_ground_truth_does_not_change_the_model_observation(
    prepared_record, screen_elements, tmp_path
):
    alternative = copy.deepcopy(prepared_record)
    alternative["supervision"] = make_supervision(
        {"action_type": "click", "x": 80, "y": 20},
        screen_elements,
        alternative["observation"]["candidates"],
        100,
        100,
    )
    assert alternative["supervision"]["label"] != prepared_record["supervision"]["label"]

    for mode in ("low", "high"):
        original = make_model_observation(prepared_record, mode, tmp_path)
        changed_label = make_model_observation(alternative, mode, tmp_path)
        assert changed_label == original
        assert set(original) == {
            "sample_id", "screenshot_path", "screen_width", "screen_height",
            "goal", "instruction", "history", "candidates",
        }


def test_high_observation_cannot_see_current_step_instruction(prepared_record, tmp_path):
    alternative = copy.deepcopy(prepared_record)
    alternative["observation"]["step_instruction"] = "PRIVATE_LABEL_SENTINEL tap Search."

    original_high = make_model_observation(prepared_record, "high", tmp_path)
    changed_high = make_model_observation(alternative, "high", tmp_path)
    changed_low = make_model_observation(alternative, "low", tmp_path)

    assert changed_high == original_high
    assert changed_high["instruction"] == prepared_record["observation"]["goal"]
    assert "PRIVATE_LABEL_SENTINEL" not in repr(changed_high)
    assert changed_low["instruction"] == "PRIVATE_LABEL_SENTINEL tap Search."


def test_observation_copy_does_not_alias_prepared_candidates_or_history(prepared_record, tmp_path):
    original = copy.deepcopy(prepared_record)
    observation = make_model_observation(prepared_record, "low", tmp_path)
    observation["candidates"][0]["text"] = "changed only in the caller"
    observation["candidates"][0]["box"][0] = -0.25
    observation["history"].append("caller-only action")

    assert prepared_record == original


def test_supervision_field_cannot_be_forwarded_as_an_observation(prepared_record, tmp_path):
    observation = make_model_observation(prepared_record, "high", tmp_path)
    observation["gt_bbox"] = prepared_record["supervision"]["gt_bbox"]

    with pytest.raises(ValueError):
        validate_model_observation(observation)


def test_uncovered_click_does_not_insert_a_ground_truth_candidate(screen_elements):
    candidates = build_action_candidates(screen_elements, 100, 100)
    original_candidates = copy.deepcopy(candidates)

    supervision = make_supervision(
        {"action_type": "click", "x": 80, "y": 80},
        screen_elements,
        candidates,
        100,
        100,
    )

    assert supervision["label"] == -100
    assert supervision["cand_idx"] == -1
    assert supervision["gt_bbox"] is None
    assert supervision["candidate_covered"] is False
    assert candidates == original_candidates


def test_eighty_accessibility_elements_do_not_cap_expanded_actions_at_eighty():
    elements = [
        {"bounds": [i, 10, i + 1, 20], "text": f"field-{i}",
         "flags": [True, False, True, False]}
        for i in range(80)
    ]

    candidates = build_action_candidates(elements, 100, 100)

    assert len(candidates) == 168  # 80 clicks + 80 editable inputs + 8 globals
    assert len({c["candidate_id"] for c in candidates}) == 168
    assert sum(c["kind"] == "input_text" for c in candidates) == 80
    assert {c["direction"] for c in candidates if c["kind"] == "scroll"} == {
        "up", "down", "left", "right"
    }
    assert {c["kind"] for c in candidates if c["element_index"] is None} == {
        "scroll", "navigate_back", "navigate_home", "wait", "open_app"
    }


def test_reordering_candidates_changes_label_index_but_not_target_identity(screen_elements):
    candidates = build_action_candidates(screen_elements, 100, 100)
    reversed_candidates = list(reversed(candidates))
    action = {"action_type": "click", "x": 80, "y": 20}

    original = make_supervision(action, screen_elements, candidates, 100, 100)
    reordered = make_supervision(action, screen_elements, reversed_candidates, 100, 100)

    assert original["label"] != reordered["label"]
    assert candidates[original["label"]]["candidate_id"] == "click:1"
    assert reversed_candidates[reordered["label"]]["candidate_id"] == "click:1"
    assert original["gt_bbox"] == reordered["gt_bbox"] == [60, 10, 100, 30]
