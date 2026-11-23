"""Authored gradient and weight-sharing fixtures, never a real-model run."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from improving.gui_decision.model import FeatureSpec, HeadConfig, QwenDecisionModel


class _TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 4)
        self.calls = []

    def forward(self, input_ids, attention_mask, image_grid_thw, *, use_cache, return_dict):
        self.calls.append({"use_cache": use_cache, "grad_enabled": torch.is_grad_enabled()})
        assert return_dict is True
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class _TinyGenerator(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.model = backbone
        self.lm_head = nn.Linear(4, 16, bias=False)
        self.generate_calls = []

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        suffix = torch.tensor([[7, 8]], device=kwargs["input_ids"].device)
        return torch.cat((kwargs["input_ids"], suffix), dim=1)


def _batch():
    return {
        "model_inputs": {
            "input_ids": torch.tensor([[0, 9, 9, 9, 9, 2, 3, 4]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1]]),
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
        },
        "candidate_end_indices": torch.tensor([[5, 6]]),
        "candidate_bounds": torch.tensor([[[0.0, 0.0, 0.4, 0.4], [0.5, 0.5, 1.0, 1.0]]]),
        "candidate_has_region": torch.tensor([[True, True]]),
        "candidate_aux": torch.tensor([[[0.0] * 11, [1.0] * 11]]),
        "candidate_type_ids": torch.tensor([[0, 4]]),
        "candidate_mask": torch.tensor([[True, True]]),
    }


def _model():
    backbone = _TinyBackbone()
    generator = _TinyGenerator(backbone)
    head_config = HeadConfig(hidden_size=4, projection_dim=3, type_embedding_dim=2)
    model = QwenDecisionModel(
        backbone, head_config, FeatureSpec(hidden_size=4, image_token_id=9),
        conditional_generator=generator,
    )
    return model, backbone, generator


def test_training_mode_cannot_unfreeze_backbone_or_original_lm_head():
    model, backbone, generator = _model()
    model.train()

    assert model.backbone is generator.model is backbone
    assert model.head.training is True
    assert generator.training is False
    assert backbone.training is False
    assert generator.lm_head.training is False
    assert all(not parameter.requires_grad for parameter in generator.parameters())
    assert all(parameter.requires_grad for parameter in model.head.parameters())


def test_only_candidate_head_receives_gradients_from_decision_loss():
    model, backbone, generator = _model()
    frozen_before = {name: value.detach().clone() for name, value in generator.named_parameters()}
    model.train()

    logits = model(_batch())
    F.cross_entropy(logits, torch.tensor([1])).backward()

    assert backbone.calls == [{"use_cache": False, "grad_enabled": False}]
    assert all(parameter.grad is None for parameter in generator.parameters())
    assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
               for parameter in model.head.parameters())
    for name, parameter in generator.named_parameters():
        torch.testing.assert_close(parameter, frozen_before[name])


def test_original_lm_head_generates_only_new_plan_tokens_from_shared_wrapper():
    model, backbone, generator = _model()
    model.train()

    generated = model.generate_plan(_batch()["model_inputs"], max_new_tokens=2)

    assert generated.tolist() == [[7, 8]]
    assert model.backbone is generator.model is backbone
    assert len(generator.generate_calls) == 1
    assert generator.generate_calls[0]["do_sample"] is False
    assert generator.training is False
    assert all(not parameter.requires_grad for parameter in generator.parameters())


def test_second_backbone_cannot_be_attached_to_planning_wrapper():
    one, other = _TinyBackbone(), _TinyBackbone()
    wrapper = _TinyGenerator(one)

    with pytest.raises(ValueError, match="same object"):
        QwenDecisionModel(
            other, HeadConfig(hidden_size=4), FeatureSpec(hidden_size=4, image_token_id=9),
            conditional_generator=wrapper,
        )


def test_candidate_readout_cannot_point_to_padding_before_any_backbone_forward():
    model, backbone, _ = _model()
    batch = _batch()
    batch["candidate_end_indices"][0, 0] = 0

    with pytest.raises(ValueError, match="padding"):
        model.extract_features(batch)

    assert backbone.calls == []
