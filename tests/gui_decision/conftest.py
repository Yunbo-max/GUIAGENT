"""Small, authored AndroidControl-shaped records; never benchmark evidence."""

import pytest

from improving.gui_decision.actions import build_action_candidates, make_supervision


@pytest.fixture
def screen_elements():
    return [
        {"bounds": [10, 10, 30, 30], "text": "Notes", "flags": [True, False, True, False]},
        {"bounds": [60, 10, 100, 30], "text": "Search", "flags": [True, False, False, False]},
    ]


@pytest.fixture
def prepared_record(screen_elements):
    candidates = build_action_candidates(screen_elements, 100, 100)
    supervision = make_supervision(
        {"action_type": "click", "x": 20, "y": 20},
        screen_elements,
        candidates,
        100,
        100,
    )
    return {
        "schema_version": 1,
        "split": "test",
        "subsplits": ["IDD"],
        "observation": {
            "sample_id": "12:2",
            "episode_id": 12,
            "step_index": 2,
            "screenshot_path": "screenshots/12/s2.png",
            "screen_width": 100,
            "screen_height": 100,
            "goal": "Open Notes and enter the text Hello world.",
            "step_instruction": "Tap the Notes text field.",
            "history": ["open [Notes]", "navigate_back"],
            "candidates": candidates,
        },
        "supervision": supervision,
    }
