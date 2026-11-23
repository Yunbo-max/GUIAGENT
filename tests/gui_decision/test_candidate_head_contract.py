"""Fixed-feature scoring contracts, authored but not executed in Web.

Candidate permutation here is deliberately tested *after* frozen feature
extraction.  A causal prompt that serializes candidates need not be invariant
when that prompt itself is reordered.
"""

import torch
from torch.nn import functional as F

from improving.gui_decision.model import CandidateScoringHead, HeadConfig


def _head():
    return CandidateScoringHead(
        HeadConfig(
            hidden_size=4,
            projection_dim=3,
            aux_dim=11,
            num_action_types=8,
            type_embedding_dim=2,
            dropout=0.0,
        )
    ).eval()


def _features():
    return {
        "query": torch.tensor([[1.0, 2.0, 3.0, 4.0], [-2.0, 1.0, 0.0, 3.0]]),
        "candidate_text": torch.arange(32, dtype=torch.float32).reshape(2, 4, 4) / 10,
        "candidate_roi": torch.arange(31, -1, -1, dtype=torch.float32).reshape(2, 4, 4) / 7,
        "candidate_aux": torch.arange(88, dtype=torch.float32).reshape(2, 4, 11) / 50,
        "candidate_type_ids": torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]]),
        "candidate_mask": torch.tensor([[True, True, False, False], [True, True, True, True]]),
    }


def test_padding_is_not_a_selectable_action_even_with_large_features():
    head, features = _head(), _features()
    # Put deliberately hostile values in the first sample's two padding slots.
    for key in ("candidate_text", "candidate_roi", "candidate_aux"):
        features[key][0, 2:] = 1e6

    scores = head(features)
    probabilities = scores.softmax(dim=-1)

    assert torch.isneginf(scores[0, 2:]).all()
    assert probabilities[0, 2:].tolist() == [0.0, 0.0]
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(2))
    assert scores[0].argmax().item() in (0, 1)
    assert torch.isfinite(scores[1]).all()


def test_shared_scorer_and_target_mapping_are_equivariant_for_fixed_features():
    head, features = _head(), _features()
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = {
        key: value if key == "query" else value[:, permutation]
        for key, value in features.items()
    }

    original_scores = head(features)
    reordered_scores = head(permuted)

    torch.testing.assert_close(reordered_scores, original_scores[:, permutation])
    # Original targets [1, 2] move to positions [3, 0].  Failing to remap these
    # labels silently changes the action supervised by a reordered candidate set.
    original_loss = F.cross_entropy(original_scores, torch.tensor([1, 2]))
    reordered_loss = F.cross_entropy(reordered_scores, torch.tensor([3, 0]))
    torch.testing.assert_close(reordered_loss, original_loss)


def test_other_examples_padding_does_not_change_a_samples_scores():
    head, batched = _head(), _features()
    single = {
        key: value[:1] if key == "query" else value[:1, :2]
        for key, value in batched.items()
    }

    single_scores = head(single)
    batched_scores = head(batched)

    torch.testing.assert_close(single_scores[0], batched_scores[0, :2])
