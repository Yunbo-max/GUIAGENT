"""Authored population/framing checks, using bytes and JSON fixtures only.

AndroidControl's published format has one more screenshot than actions:
https://github.com/google-research/google-research/tree/master/android_control
"""

import copy
import hashlib
import json
import struct

import pytest

from improving.gui_decision.actions import make_supervision
from improving.gui_decision.androidcontrol import episode_steps, iter_tfrecords
from improving.gui_decision.dataset import GuiDecisionDataset


@pytest.mark.parametrize("prefix_length", [3, 10, 14, 18])
def test_truncated_tfrecord_cannot_silently_end_the_population(tmp_path, prefix_length):
    # Known framing: uint64 length, length CRC, three bytes, payload CRC.
    # CRC is explicitly disabled here to isolate truncation from checksum logic.
    frame = struct.pack("<Q", 3) + b"\0" * 4 + b"abc" + b"\0" * 4
    path = tmp_path / "fixture.tfrecord"
    path.write_bytes(frame[:prefix_length])

    with pytest.raises(ValueError):
        list(iter_tfrecords(path, verify_crc=False))


def test_episode_arrays_must_retain_the_terminal_observation():
    example = {
        "episode_id": [1], "goal": [b"Open Notes"],
        "screenshots": [b"pre-action", b"terminal"],
        "accessibility_trees": [b"before", b"after"],
        "screenshot_widths": [100, 100], "screenshot_heights": [200, 200],
        "actions": [b'{"action_type":"open_app","app_name":"Notes"}'],
        "step_instructions": [b"Open Notes"],
    }
    steps = episode_steps(example)
    assert len(steps) == 1
    assert steps[0]["screenshot_png"] == b"pre-action"
    malformed = copy.deepcopy(example)
    malformed["screenshots"].pop()

    with pytest.raises(ValueError):
        episode_steps(malformed)


def _prepared_fixture(root, records):
    raw = "".join(json.dumps(record) + "\n" for record in records).encode()
    (root / "test.jsonl").write_bytes(raw)
    manifest = {
        "schema_version": 1, "status": "complete", "files": {"test": "test.jsonl"},
        "step_counts": {"test": len(records)},
        "prepared_file_hashes": {"test.jsonl": hashlib.sha256(raw).hexdigest()},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))


def test_full_dataset_both_views_keep_uncovered_ground_truth(prepared_record, screen_elements, tmp_path):
    missing = copy.deepcopy(prepared_record)
    missing["observation"]["sample_id"] = "12:3"
    missing["observation"]["step_index"] = 3
    missing["supervision"] = make_supervision(
        {"action_type": "click", "x": 80, "y": 80},
        screen_elements, missing["observation"]["candidates"], 100, 100,
    )
    _prepared_fixture(tmp_path, [prepared_record, missing])

    dataset = GuiDecisionDataset(tmp_path, split="test", mode="both", subset="full")

    assert len(dataset) == 4
    assert [dataset[i]["mode"] for i in range(4)] == ["low", "high", "low", "high"]
    assert [dataset[i]["label"] for i in (2, 3)] == [-100, -100]
    assert dataset[2]["observation"]["instruction"] == missing["observation"]["step_instruction"]
    assert dataset[3]["observation"]["instruction"] == missing["observation"]["goal"]


def test_edited_records_with_unchanged_population_count_do_not_match_the_manifest(
    prepared_record, tmp_path
):
    _prepared_fixture(tmp_path, [prepared_record])
    changed = copy.deepcopy(prepared_record)
    changed["observation"]["candidates"][0]["text"] = "Different content, same sample count"
    (tmp_path / "test.jsonl").write_text(json.dumps(changed) + "\n")

    with pytest.raises(ValueError):
        GuiDecisionDataset(tmp_path, split="test", mode="high")
