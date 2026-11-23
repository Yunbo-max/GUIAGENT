"""Hand-judged edge cases for the retained AndroidControl scoring protocol.

Expected outcomes below are specified directly, rather than obtained from a
second copy of the implementation.  They are step-level fixtures, not agent
success measurements.
"""

import copy

import pytest

from improving.gui_decision.actions import (
    build_action_candidates,
    make_supervision,
    prediction_from_candidate,
)
from improving.gui_decision.dataset import make_model_observation
from improving.gui_decision.evaluation import MetricsAccumulator, judge


def _candidate(observation, candidate_id):
    return next(c for c in observation["candidates"] if c["candidate_id"] == candidate_id)


def test_missing_target_and_inference_failures_stay_in_accuracy_denominator(
    prepared_record, screen_elements, tmp_path
):
    observation = make_model_observation(prepared_record, "high", tmp_path)
    notes = prediction_from_candidate(_candidate(observation, "click:0"))
    missing = make_supervision(
        {"action_type": "click", "x": 80, "y": 80},
        screen_elements, observation["candidates"], 100, 100,
    )
    scroll = make_supervision(
        {"action_type": "scroll", "direction": "up"},
        screen_elements, observation["candidates"], 100, 100,
    )
    wrong_direction = prediction_from_candidate(_candidate(observation, "scroll:down"))

    outcomes = [
        judge(notes, prepared_record["supervision"], observation),
        judge(notes, missing, observation),
        judge(None, prepared_record["supervision"], observation),
        judge(wrong_direction, scroll, observation),
    ]
    accumulator = MetricsAccumulator()
    for result in outcomes:
        accumulator.update(result)
    metrics = accumulator.compute()

    assert [r["correct"] for r in outcomes] == [True, False, False, False]
    assert metrics["n_steps"] == 4
    assert metrics["step_acc"] == pytest.approx(25.0)
    assert metrics["type_acc"] == pytest.approx(75.0)
    assert metrics["click_total"] == 3
    assert metrics["click_success_all_gt_clicks"] == pytest.approx(100.0 / 3)
    assert metrics["candidate_coverage"] == pytest.approx(200.0 / 3)
    # The inherited conditional grounding metric is deliberately not the main
    # metric: it excludes both the unenumerated target and the wrong-type case.
    assert metrics["click_grounding_n"] == 1
    assert metrics["grounding_acc_clicks"] == pytest.approx(100.0)


def test_explicit_inference_error_is_an_incorrect_retained_step(prepared_record, tmp_path):
    observation = make_model_observation(prepared_record, "high", tmp_path)
    outcome = judge(
        {"kind": None, "failure": "inference_error"},
        prepared_record["supervision"],
        observation,
    )
    accumulator = MetricsAccumulator()
    accumulator.update(outcome)

    assert outcome["correct"] is False
    assert outcome["type_correct"] is False
    assert accumulator.compute()["n_steps"] == 1
    assert accumulator.compute()["step_acc"] == 0.0


def test_selected_candidate_center_controls_click_not_self_reported_point(
    prepared_record, tmp_path
):
    observation = make_model_observation(prepared_record, "high", tmp_path)
    prediction = prediction_from_candidate(_candidate(observation, "click:1"))
    # Search is centered at (80, 20), outside Notes. A forged point at (20, 20)
    # cannot turn that selected candidate into a correct Notes click.
    prediction["point"] = [0.2, 0.2]

    outcome = judge(prediction, prepared_record["supervision"], observation)

    assert outcome["type_correct"] is True
    assert outcome["correct"] is False


def test_click_on_ground_truth_box_edge_is_accepted_without_exact_candidate_identity(
    prepared_record, screen_elements, tmp_path
):
    elements = copy.deepcopy(screen_elements)
    elements.append({"bounds": [80, 20, 90, 30], "text": "nested target",
                     "flags": [True, False, False, False]})
    record = copy.deepcopy(prepared_record)
    record["observation"]["candidates"] = build_action_candidates(elements, 100, 100)
    record["supervision"] = make_supervision(
        {"action_type": "click", "x": 85, "y": 25},
        elements, record["observation"]["candidates"], 100, 100,
    )
    observation = make_model_observation(record, "high", tmp_path)
    # The larger Search element has center (80,20), exactly at the target's
    # upper-left corner. Both are legitimate candidates; the native judge uses
    # the inclusive target box, not equality with the supervised candidate ID.
    prediction = prediction_from_candidate(_candidate(observation, "click:1"))

    assert record["supervision"]["gt_bbox"] == [80, 20, 90, 30]
    assert judge(prediction, record["supervision"], observation)["correct"] is True


def test_native_long_press_target_accepts_click_candidate(
    prepared_record, screen_elements, tmp_path
):
    observation = make_model_observation(prepared_record, "low", tmp_path)
    supervision = make_supervision(
        {"action_type": "long_press", "x": 20, "y": 20},
        screen_elements, observation["candidates"], 100, 100,
    )
    prediction = prediction_from_candidate(_candidate(observation, "click:0"))

    outcome = judge(prediction, supervision, observation)

    assert outcome["type_correct"] is True
    assert outcome["correct"] is True


@pytest.mark.parametrize(
    ("predicted_text", "correct"),
    [("  HELLO   world\n", True), ("Hello world!", False), (None, False)],
)
def test_typed_text_keeps_native_normalization_without_semantic_equivalence(
    predicted_text, correct, prepared_record, screen_elements, tmp_path
):
    observation = make_model_observation(prepared_record, "low", tmp_path)
    supervision = make_supervision(
        {"action_type": "input_text", "text": "Hello world"},
        screen_elements, observation["candidates"], 100, 100,
    )
    prediction = prediction_from_candidate(_candidate(observation, "input_text:0"))
    prediction["text"] = predicted_text

    assert judge(prediction, supervision, observation)["correct"] is correct


@pytest.mark.parametrize(("predicted_app", "correct"), [("Notes", True), ("notes", False)])
def test_open_app_retains_exact_app_name_judge(
    predicted_app, correct, prepared_record, screen_elements, tmp_path
):
    observation = make_model_observation(prepared_record, "high", tmp_path)
    supervision = make_supervision(
        {"action_type": "open_app", "app_name": "Notes"},
        screen_elements, observation["candidates"], 100, 100,
    )
    prediction = prediction_from_candidate(_candidate(observation, "open_app"))
    prediction["app_name"] = predicted_app

    assert judge(prediction, supervision, observation)["correct"] is correct
