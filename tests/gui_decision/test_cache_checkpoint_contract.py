"""Authored persistence contracts; these do not qualify real Qwen execution.

Only hand-built features and a small randomly initialized scoring head are used.
No old JEPA cache, model download, benchmark data, or accelerator is required.
"""

import copy
import json

import pytest
import torch

from improving.gui_decision.cache import (
    FeatureCache,
    FeatureCacheWriter,
    collate_features,
    stable_digest,
)
from improving.gui_decision.checkpoint import load_head, save_checkpoint
from improving.gui_decision.model import CandidateScoringHead, HeadConfig


def _features(n=2):
    return {
        "query": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        "candidate_text": torch.arange(n * 4, dtype=torch.float32).reshape(n, 4),
        "candidate_roi": torch.arange(n * 4, dtype=torch.float32).reshape(n, 4) + 1,
        "candidate_aux": torch.zeros(n, 11),
        "candidate_type_ids": torch.arange(n, dtype=torch.long) % 8,
        "candidate_mask": torch.ones(n, dtype=torch.bool),
    }


def _spec():
    # This is deliberately an authored contract fixture, not a claimed model pin.
    return {
        "model": {"name": "fixture/frozen", "revision": "fixture-revision-a"},
        "processor": {"max_image_tokens": 256, "prompt_version": "fixture-v1"},
        "feature_layer": "last_hidden_state",
        "hidden_size": 4,
    }


def _write(writer, key="12:0:high", *, n=2, label=0, digest="observation-a"):
    writer.write(
        key=key,
        features=_features(n),
        label=label,
        weight=1.0,
        record_index=0,
        mode="high",
        input_digest=digest,
        sample_id="12:0",
    )


def _complete_cache(path):
    writer = FeatureCacheWriter(path, _spec())
    _write(writer)
    writer.finish(["12:0:high"])
    return FeatureCache(path, expected_spec=_spec())


def test_cache_cannot_be_consumed_until_requested_population_is_complete(tmp_path):
    writer = FeatureCacheWriter(tmp_path / "features", _spec())
    _write(writer)

    with pytest.raises(ValueError):
        writer.finish(["12:0:high", "12:1:high"])
    with pytest.raises(ValueError):
        FeatureCache(tmp_path / "features")

    writer.finish(["12:0:high"])
    assert len(FeatureCache(tmp_path / "features")) == 1


@pytest.mark.parametrize("change", ["model_revision", "image_budget", "feature_layer"])
def test_cached_features_cannot_be_rebound_to_different_encoder_inputs(tmp_path, change):
    path = tmp_path / "features"
    _complete_cache(path)
    different = copy.deepcopy(_spec())
    if change == "model_revision":
        different["model"]["revision"] = "fixture-revision-b"
    elif change == "image_budget":
        different["processor"]["max_image_tokens"] = 512
    else:
        different["feature_layer"] = "premerge_visual_features"

    with pytest.raises(ValueError):
        FeatureCache(path, expected_spec=different)
    with pytest.raises(ValueError):
        FeatureCacheWriter(path, different, resume=True)


def test_resume_rejects_changed_observation_under_existing_key(tmp_path):
    path = tmp_path / "features"
    _complete_cache(path)
    writer = FeatureCacheWriter(path, _spec(), resume=True)

    assert writer.contains("12:0:high", "observation-a")
    with pytest.raises(ValueError):
        writer.contains("12:0:high", "same-key-different-screenshot-or-text")


def test_resume_rejects_manifest_corruption_instead_of_rewriting_it(tmp_path):
    path = tmp_path / "features"
    _complete_cache(path)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["spec"]["model"]["revision"] = "changed-without-updating-digest"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError):
        FeatureCacheWriter(path, _spec(), resume=True)


def test_resume_cannot_reseal_changed_labels_in_a_completed_cache_index(tmp_path):
    path = tmp_path / "features"
    _complete_cache(path)
    index_path = path / "index.jsonl"
    row = json.loads(index_path.read_text().strip())
    row["label"] = 1  # Still a valid candidate, but not the sealed label.
    index_path.write_text(json.dumps(row) + "\n")

    with pytest.raises(ValueError):
        FeatureCacheWriter(path, _spec(), resume=True)


def test_old_jepa_manifest_is_not_treated_as_a_qwen_feature_cache(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"train": [], "validation": [], "test": []})
    )

    with pytest.raises(ValueError):
        FeatureCache(tmp_path)


def test_missing_candidate_label_is_preserved_in_cache(tmp_path):
    writer = FeatureCacheWriter(tmp_path / "features", _spec())
    _write(writer, label=-100)
    writer.finish(["12:0:high"])
    row = FeatureCache(tmp_path / "features")[0]

    assert row["label"] == -100
    assert row["sample_id"] == "12:0"
    assert row["features"]["candidate_mask"].tolist() == [True, True]


def test_fp16_storage_does_not_introduce_nonfinite_features(tmp_path):
    writer = FeatureCacheWriter(tmp_path / "features", _spec())
    features = _features()
    # Finite in FP32, but overflowing an FP16 feature-cache conversion.
    features["candidate_text"][0, 0] = 1e6

    with pytest.raises(ValueError):
        writer.write(
            "12:0:high", features, 0, 1.0, 0, "high", "observation-a"
        )


def test_corrupted_feature_file_is_not_silently_skipped(tmp_path):
    path = tmp_path / "features"
    cache = _complete_cache(path)
    tensor_path = path / cache.rows[0]["file"]
    tensor_path.write_bytes(tensor_path.read_bytes() + b"unexpected trailing bytes")

    with pytest.raises(ValueError):
        _ = cache[0]


def test_collation_preserves_variable_candidate_counts_and_ignore_labels():
    items = [
        {"features": _features(2), "label": -100, "weight": 1.0},
        {"features": _features(4), "label": 3, "weight": 1.5},
    ]

    batch = collate_features(items)

    assert batch["features"]["candidate_mask"].tolist() == [
        [True, True, False, False], [True, True, True, True]
    ]
    assert batch["labels"].tolist() == [-100, 3]
    assert batch["weights"].tolist() == [1.0, 1.5]
    assert torch.count_nonzero(batch["features"]["candidate_roi"][0, 2:]).item() == 0
    torch.testing.assert_close(
        batch["features"]["candidate_text"][1], items[1]["features"]["candidate_text"]
    )


def test_head_checkpoint_roundtrip_preserves_scores_and_feature_binding(tmp_path):
    head = CandidateScoringHead(
        HeadConfig(hidden_size=4, projection_dim=3, aux_dim=11,
                   num_action_types=8, type_embedding_dim=2, dropout=0.0)
    ).eval()
    features = collate_features([
        {"features": _features(), "label": 0, "weight": 1.0}
    ])["features"]
    scores_before = head(features).detach()
    metadata = {"feature_spec": _spec(), "status": "fixture_only"}

    save_checkpoint(tmp_path / "head", head, metadata)
    reloaded, saved_metadata = load_head(
        tmp_path / "head", expected_feature_spec=_spec(), device="cpu"
    )
    reloaded.eval()

    torch.testing.assert_close(reloaded(features), scores_before)
    assert saved_metadata["feature_spec"] == _spec()
    changed = copy.deepcopy(_spec())
    changed["processor"]["prompt_version"] = "different-prompt"
    with pytest.raises(ValueError):
        load_head(tmp_path / "head", expected_feature_spec=changed)


def test_invalid_checkpoint_metadata_does_not_damage_an_existing_checkpoint(tmp_path):
    head = CandidateScoringHead(
        HeadConfig(hidden_size=4, projection_dim=3, aux_dim=11,
                   num_action_types=8, type_embedding_dim=2, dropout=0.0)
    )
    path = tmp_path / "head"
    save_checkpoint(path, head, {"feature_spec": _spec()})
    original_weights = (path / "head.safetensors").read_bytes()
    original_metadata = (path / "metadata.json").read_bytes()
    with torch.no_grad():
        next(head.parameters()).add_(1.0)

    with pytest.raises(ValueError):
        save_checkpoint(path, head, {"no_feature_spec": True})

    assert (path / "head.safetensors").read_bytes() == original_weights
    assert (path / "metadata.json").read_bytes() == original_metadata
    load_head(path, expected_feature_spec=_spec())


def test_json_order_is_not_part_of_feature_identity_but_actual_content_is():
    one = {"processor": {"budget": 256, "mode": "high"}, "model": "fixture"}
    reordered = {"model": "fixture", "processor": {"mode": "high", "budget": 256}}
    changed = {"model": "fixture", "processor": {"mode": "low", "budget": 256}}

    assert stable_digest(one) == stable_digest(reordered)
    assert stable_digest(one) != stable_digest(changed)
