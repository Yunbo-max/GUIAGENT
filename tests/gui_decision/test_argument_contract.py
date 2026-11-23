"""Target-free argument routing and lazy asset identity; no weights loaded."""

from pathlib import Path

import pytest

from improving.gui_decision import arguments
from improving.gui_decision.arguments import ArgumentResolver
from improving.gui_decision.dataset import make_model_observation


def _fixture_ranker(root: Path) -> Path:
    directory = root / "ranker"
    (directory / "ranker_lm").mkdir(parents=True)
    # Deliberately not valid weights: any attempt to load a model in these
    # identity/click-only checks is a test error.
    (directory / "ranker_head.pt").write_bytes(b"fixture head bytes")
    (directory / "ranker_lm" / "config.json").write_text("{}")
    (directory / "ranker_lm" / "model.safetensors").write_bytes(b"fixture encoder bytes")
    (directory / "ranker_lm" / "tokenizer.json").write_text("{}")
    return directory


def test_argument_module_branches_on_predicted_kind_without_a_ground_truth_field(
    prepared_record, tmp_path
):
    # The prepared label is a click, but the supplied predicted option is input.
    # Its text must come from the sanitized instruction, not the prepared label.
    observation = make_model_observation(prepared_record, "low", tmp_path)
    observation["instruction"] = 'Type "Alice".'
    resolver = ArgumentResolver(app_vocab=["Notes"], mode="heuristic")
    typed = next(c for c in observation["candidates"] if c["kind"] == "input_text")
    clicked = next(c for c in observation["candidates"] if c["kind"] == "click")

    assert resolver.resolve(typed, observation)["text"] == "Alice"
    assert resolver.resolve(clicked, observation)["text"] is None


def test_click_resolution_does_not_load_the_optional_span_model(
    prepared_record, tmp_path, monkeypatch
):
    directory = _fixture_ranker(tmp_path)

    def forbidden_model_load(*args, **kwargs):
        raise AssertionError("A click must not construct the text span model")

    monkeypatch.setattr(arguments, "_FrozenSpanRanker", forbidden_model_load)
    resolver = ArgumentResolver(app_vocab=["Notes"], span_checkpoint=directory)
    observation = make_model_observation(prepared_record, "high", tmp_path)
    clicked = next(c for c in observation["candidates"] if c["kind"] == "click")

    result = resolver.resolve(clicked, observation)

    assert result["kind"] == "click"
    assert result["candidate_id"] == clicked["candidate_id"]


def test_same_ranker_path_with_changed_weights_has_a_different_identity(tmp_path):
    directory = _fixture_ranker(tmp_path)
    before = ArgumentResolver(app_vocab=["Notes"], span_checkpoint=directory).describe()
    (directory / "ranker_lm" / "model.safetensors").write_bytes(b"different fixture encoder")

    after = ArgumentResolver(app_vocab=["Notes"], span_checkpoint=directory).describe()

    assert before["span_checkpoint"] == after["span_checkpoint"]
    assert before["span_checkpoint_fingerprint"] != after["span_checkpoint_fingerprint"]


def test_missing_span_checkpoint_cannot_silently_select_heuristic_mode(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        ArgumentResolver(app_vocab=["Notes"], span_checkpoint=tmp_path / "missing")
