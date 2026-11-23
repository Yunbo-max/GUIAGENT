"""One frozen Qwen backbone, its original planning head, and an action scorer.

ACT reads ``outer.model.last_hidden_state`` without vocabulary logits. PLAN uses
the same frozen conditional-generation wrapper and its original LM head. Only
the small shared candidate scorer is trainable in version 1.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .processor import (
    ACTION_TYPE_NAMES, AUX_NAMES, DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION,
    MAX_CANDIDATES, PROMPT_VERSION, TRANSFORMERS_VERSION, ProcessorConfig,
    require_transformers_version, validate_model_config_keys,
)
from .spatial import last_valid_token_indices, pool_candidate_regions, restore_merged_grid


FEATURE_KEYS = frozenset({
    "query", "candidate_text", "candidate_roi", "candidate_aux",
    "candidate_type_ids", "candidate_mask",
})


def _strict_dataclass_values(cls: type, value: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(value) - {field.name for field in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return dict(value)


@dataclass(frozen=True)
class HeadConfig:
    hidden_size: int = 1024
    projection_dim: int = 256
    aux_dim: int = 11
    num_action_types: int = 8
    type_embedding_dim: int = 32
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for name in ("hidden_size", "projection_dim", "aux_dim", "num_action_types",
                     "type_embedding_dim"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.dropout, (int, float)) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HeadConfig":
        return cls(**_strict_dataclass_values(cls, value))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FeatureSpec:
    version: str = "qwen35_candidate_features_v1"
    model_id: str = DEFAULT_MODEL_ID
    revision: str = DEFAULT_MODEL_REVISION
    transformers_version: str = TRANSFORMERS_VERSION
    prompt_version: str = PROMPT_VERSION
    hidden_size: int = 1024
    aux_dim: int = 11
    num_action_types: int = 8
    image_token_id: int = 248056
    patch_size: int = 16
    spatial_merge_size: int = 2
    max_candidates: int = MAX_CANDIDATES
    query_source: str = "last_nonpadding_token"
    text_source: str = "last_token_intersecting_candidate_segment"
    image_source: str = "last_hidden_state_image_tokens"
    roi_policy: str = "clip_pool_only_upper_edge_inclusive_global_mean"
    feature_dtype: str = "float32"
    feature_value_policy: str = "fp16_roundtrip_query_text_roi_float32_compute"

    def __post_init__(self) -> None:
        for name in ("hidden_size", "aux_dim", "num_action_types", "patch_size",
                     "spatial_merge_size", "max_candidates"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.image_token_id) is not int or self.image_token_id < 0:
            raise ValueError("image_token_id must be a nonnegative integer")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FeatureSpec":
        return cls(**_strict_dataclass_values(cls, value))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CandidateScoringHead(nn.Module):
    """Ordinary shared query-conditioned scorer over a padded variable-N set.

    All layers and arithmetic use FP32. There is no fixed N-way classification
    matrix: each valid candidate uses the same feature projections and MLP.
    Permuting already-extracted candidate features therefore permutes scores.
    The causal prompt itself is ordered and need not have that invariance.
    """

    def __init__(self, config: HeadConfig):
        super().__init__()
        self.config = config
        dim, projection, small = config.hidden_size, config.projection_dim, config.type_embedding_dim

        def projected():
            return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, projection), nn.GELU())

        self.query_projection = projected()
        self.text_projection = projected()
        self.roi_projection = projected()
        self.aux_projection = nn.Sequential(nn.Linear(config.aux_dim, small), nn.GELU())
        self.type_embedding = nn.Embedding(config.num_action_types, small)
        # Query, text, ROI, plus standard elementwise query/text and query/ROI
        # interactions. The two small blocks encode geometry/flags and type.
        self.scorer = nn.Sequential(
            nn.Linear(5 * projection + 2 * small, projection), nn.GELU(),
            nn.Dropout(float(config.dropout)), nn.Linear(projection, 1),
        )
        self.float()

    def forward(self, features: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if set(features) != FEATURE_KEYS:
            raise ValueError(f"Feature fields must be exactly {sorted(FEATURE_KEYS)}")
        query = features["query"]
        text = features["candidate_text"]
        roi = features["candidate_roi"]
        auxiliary = features["candidate_aux"]
        types = features["candidate_type_ids"]
        mask = features["candidate_mask"]
        if query.ndim != 2 or query.shape[1] != self.config.hidden_size:
            raise ValueError("query must have shape [B, hidden_size]")
        if mask.ndim != 2 or mask.shape[0] != query.shape[0] or mask.shape[1] == 0:
            raise ValueError("candidate_mask must have nonempty shape [B, N]")
        if mask.dtype != torch.bool:
            raise ValueError("candidate_mask must be bool")
        expected = (*mask.shape, self.config.hidden_size)
        if text.shape != expected or roi.shape != expected:
            raise ValueError("Candidate text/ROI shapes do not match [B, N, hidden_size]")
        if auxiliary.shape != (*mask.shape, self.config.aux_dim) or types.shape != mask.shape:
            raise ValueError("Candidate auxiliary/type shapes do not match the mask")
        if types.dtype not in (torch.int32, torch.int64):
            raise ValueError("candidate_type_ids must be integer tensors")
        if not bool(mask.any(dim=-1).all()):
            raise ValueError("Each sample must have at least one selectable candidate")
        if bool(((types[mask] < 0) | (types[mask] >= self.config.num_action_types)).any()):
            raise ValueError("A valid candidate has an unknown action type")
        if any(value.device != query.device for value in features.values()):
            raise ValueError("All features must be on the same device")
        if next(self.parameters()).device != query.device:
            raise ValueError("Move the scoring head and features to the same device")
        if any(parameter.dtype != torch.float32 for parameter in self.parameters()):
            raise ValueError("The candidate scoring head must stay in float32")
        if not bool(torch.isfinite(query).all()):
            raise ValueError("Query features must be finite")
        for name, value in (("text", text), ("ROI", roi), ("auxiliary", auxiliary)):
            if not bool(torch.isfinite(value[mask]).all()):
                raise ValueError(f"Valid candidate {name} features must be finite")

        # Zero padding before projections, including any out-of-range padded
        # type IDs, so padding cannot create invalid gradients in training.
        text = text.float().masked_fill(~mask.unsqueeze(-1), 0)
        roi = roi.float().masked_fill(~mask.unsqueeze(-1), 0)
        auxiliary = auxiliary.float().masked_fill(~mask.unsqueeze(-1), 0)
        safe_types = types.long().masked_fill(~mask, 0)
        with torch.autocast(device_type=query.device.type, enabled=False):
            query_feature = self.query_projection(query.float()).unsqueeze(1).expand(-1, mask.shape[1], -1)
            text_feature = self.text_projection(text)
            roi_feature = self.roi_projection(roi)
            joined = torch.cat((
                query_feature, text_feature, roi_feature,
                query_feature * text_feature, query_feature * roi_feature,
                self.aux_projection(auxiliary), self.type_embedding(safe_types),
            ), dim=-1)
            scores = self.scorer(joined).squeeze(-1)
        return scores.masked_fill(~mask, float("-inf"))


# Backwards-compatible name for small authored fixtures; the checkpoint format
# and public training API use CandidateScoringHead.
CandidateDecisionHead = CandidateScoringHead


class QwenDecisionModel(nn.Module):
    """A shared frozen multimodal backbone with explicit PLAN/ACT entry points.

    ``backbone`` injection permits tiny fake modules in local contract tests.
    Real loading supplies ``conditional_generator`` and registers only that
    wrapper, avoiding a second registered copy of its ``.model`` submodule.
    """

    def __init__(
        self, backbone: nn.Module, head_config: HeadConfig, feature_spec: FeatureSpec,
        *, model_config: Mapping[str, Any] | None = None,
        conditional_generator: nn.Module | None = None,
    ):
        super().__init__()
        if (head_config.hidden_size != feature_spec.hidden_size
                or head_config.aux_dim != feature_spec.aux_dim
                or head_config.num_action_types != feature_spec.num_action_types):
            raise ValueError("Head configuration and feature specification disagree")
        if conditional_generator is not None:
            if getattr(conditional_generator, "model", None) is not backbone:
                raise ValueError("The planning wrapper and ACT backbone must be the same object")
            self.language_model = conditional_generator
            self._standalone_backbone = None
            self.language_model.requires_grad_(False)
            self.language_model.eval()
        else:
            self.language_model = None
            self._standalone_backbone = backbone
            self._standalone_backbone.requires_grad_(False)
            self._standalone_backbone.eval()
        self.head_config = head_config
        self.feature_spec = feature_spec
        self.model_config = dict(model_config or {})
        self.head = CandidateScoringHead(head_config).to(device=self._backbone_device()).float()

    @property
    def backbone(self) -> nn.Module:
        if self.language_model is not None:
            return self.language_model.model
        return self._standalone_backbone

    def _backbone_device(self) -> torch.device:
        for tensor in self.backbone.parameters():
            return tensor.device
        for tensor in self.backbone.buffers():
            return tensor.device
        return torch.device("cpu")

    @classmethod
    def from_pretrained(cls, config: Mapping[str, Any]) -> "QwenDecisionModel":
        validate_model_config_keys(config)
        processor_config = ProcessorConfig.from_dict(config)
        require_transformers_version()
        from transformers import Qwen3_5ForConditionalGeneration

        dtype = config.get("backbone_dtype", "float16")
        if dtype not in ("float16", "fp16"):
            raise ValueError("Version 1 uses an FP16 backbone; bf16 is not required on Turing")
        attention = config.get("attn_implementation", "sdpa")
        if attention not in ("sdpa", "eager"):
            raise ValueError("Use sdpa or eager; this adapter does not require FlashAttention 2")
        local_only = config.get("local_files_only", False)
        if type(local_only) is not bool:
            raise ValueError("local_files_only must be a bool")
        head_config = HeadConfig.from_dict(config.get("head", {}))
        if (head_config.hidden_size != 1024 or head_config.aux_dim != len(AUX_NAMES)
                or head_config.num_action_types != len(ACTION_TYPE_NAMES)):
            raise ValueError("The official 0.8B adapter requires hidden_size=1024, aux_dim=11 and 8 types")
        device = torch.device(config.get("device", "cuda"))
        outer = Qwen3_5ForConditionalGeneration.from_pretrained(
            processor_config.model_id, revision=processor_config.revision,
            dtype=torch.float16, attn_implementation=attention,
            trust_remote_code=False, local_files_only=local_only, use_safetensors=True,
        )
        actual = outer.config
        if (actual.text_config.hidden_size != head_config.hidden_size
                or actual.vision_config.patch_size != 16
                or actual.vision_config.spatial_merge_size != 2):
            raise ValueError("Loaded model dimensions differ from the audited Qwen3.5-0.8B contract")
        outer.to(device=device)
        specification = FeatureSpec(
            hidden_size=actual.text_config.hidden_size,
            image_token_id=actual.image_token_id,
            patch_size=actual.vision_config.patch_size,
            spatial_merge_size=actual.vision_config.spatial_merge_size,
        )
        canonical = {
            **processor_config.to_dict(), "backbone_dtype": "float16",
            "attn_implementation": attention, "head": head_config.to_dict(),
        }
        return cls(outer.model, head_config, specification,
                   model_config=canonical, conditional_generator=outer)

    def train(self, mode: bool = True) -> "QwenDecisionModel":
        super().train(mode)
        if self.language_model is not None:
            self.language_model.eval()
        else:
            self.backbone.eval()
        self.head.train(mode)
        return self

    def fingerprint_metadata(self) -> dict[str, Any]:
        metadata = {
            "feature_spec": self.feature_spec.to_dict(),
            "model_config": self.model_config,
            "head_config": self.head_config.to_dict(),
            "backbone_frozen": True,
            "head_compute_dtype": "float32",
            "act_vocabulary_logits": False,
            "act_use_cache": False,
            "planning_lm_head_retained": self.language_model is not None,
            "shared_backbone": True,
            "implementation_sha256": {
                name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ("model.py", "spatial.py")
            },
        }
        raw = json.dumps(metadata, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
        return {**metadata, "fingerprint": hashlib.sha256(raw).hexdigest()}

    def _move_model_inputs(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        allowed = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids"}
        if set(inputs) - allowed:
            raise ValueError(f"Unexpected model input fields: {sorted(set(inputs) - allowed)}")
        if not {"input_ids", "attention_mask", "image_grid_thw"}.issubset(inputs):
            raise ValueError("input_ids, attention_mask and image_grid_thw are required")
        device = self._backbone_device()
        moved = {}
        for name, value in inputs.items():
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"Model input {name} must be a tensor")
            # The official model casts vision pixels to the visual module dtype.
            # Do not cast token IDs, masks, grid sizes or modality IDs to FP16.
            moved[name] = value.to(device=device)
        return moved

    def extract_features(self, processor_batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        inputs = self._move_model_inputs(processor_batch["model_inputs"])
        ids, attention = inputs["input_ids"], inputs["attention_mask"]
        if ids.ndim != 2 or attention.shape != ids.shape:
            raise ValueError("input_ids and attention_mask must share shape [B, L]")
        batch_size, length = ids.shape
        grid_rows = inputs["image_grid_thw"]
        if grid_rows.shape != (batch_size, 3):
            raise ValueError("One screenshot grid is required for each sample")
        mask = processor_batch["candidate_mask"].to(device=ids.device)
        ends = processor_batch["candidate_end_indices"].to(device=ids.device)
        bounds = processor_batch["candidate_bounds"]
        regions = processor_batch["candidate_has_region"]
        auxiliary = processor_batch["candidate_aux"].to(device=ids.device, dtype=torch.float32)
        types = processor_batch["candidate_type_ids"].to(device=ids.device, dtype=torch.long)
        if mask.ndim != 2 or mask.shape[0] != batch_size or mask.dtype != torch.bool:
            raise ValueError("candidate_mask must be bool [B, N]")
        if not 0 < mask.shape[1] <= self.feature_spec.max_candidates or not bool(mask.any(-1).all()):
            raise ValueError("Each sample requires a supported, nonempty candidate set")
        if ends.shape != mask.shape or ends.dtype not in (torch.int32, torch.int64):
            raise ValueError("candidate_end_indices must be integer [B, N]")
        if bounds.shape != (*mask.shape, 4) or regions.shape != mask.shape:
            raise ValueError("Candidate bounds/region mask have inconsistent shapes")
        if auxiliary.shape != (*mask.shape, self.feature_spec.aux_dim) or types.shape != mask.shape:
            raise ValueError("Candidate auxiliary/type features have inconsistent shapes")
        if bool(((ends[mask] < 0) | (ends[mask] >= length)).any()):
            raise ValueError("A candidate readout index is outside the token sequence")
        safe_ends = ends.long().masked_fill(~mask, 0)
        if not bool(attention.gather(1, safe_ends)[mask].bool().all()):
            raise ValueError("A valid candidate points to padding")
        query_indices = last_valid_token_indices(attention)
        # no_grad (not inference_mode): cached/frozen features can then safely be
        # saved by autograd for gradients of the trainable FP32 scoring head.
        with torch.no_grad():
            self.backbone.eval()
            outputs = self.backbone(**inputs, use_cache=False, return_dict=True)
            hidden = outputs.last_hidden_state
            if hidden.shape != (batch_size, length, self.feature_spec.hidden_size):
                raise ValueError("Backbone last_hidden_state has unexpected dimensions")
            rows = torch.arange(batch_size, device=hidden.device)
            query = hidden[rows, query_indices].float()
            candidate_text = hidden.gather(
                1, safe_ends.unsqueeze(-1).expand(-1, -1, self.feature_spec.hidden_size),
            ).float().masked_fill(~mask.unsqueeze(-1), 0)
            pooled = []
            for row in range(batch_size):
                image_mask = (ids[row] == self.feature_spec.image_token_id) & attention[row].bool()
                grid = restore_merged_grid(
                    hidden[row, image_mask].float(), grid_rows[row],
                    self.feature_spec.spatial_merge_size,
                )
                pooled.append(pool_candidate_regions(grid, bounds[row], regions[row], mask[row]))
            candidate_roi = torch.stack(pooled, dim=0)
            # The persistent cache stores these three frozen tensors in FP16.
            # Apply the same explicit round-trip online so cached and direct
            # scoring see identical feature values, including FP32 ROI means.
            compact_features = []
            for name, value in (("query", query), ("candidate_text", candidate_text),
                                ("candidate_roi", candidate_roi)):
                compact = value.to(dtype=torch.float16)
                if not bool(torch.isfinite(compact).all()):
                    raise ValueError(f"Frozen feature {name} is nonfinite or exceeds FP16 storage range")
                compact_features.append(compact.float())
            query, candidate_text, candidate_roi = compact_features
        return {
            "query": query,
            "candidate_text": candidate_text,
            "candidate_roi": candidate_roi,
            "candidate_aux": auxiliary,
            "candidate_type_ids": types,
            "candidate_mask": mask,
        }

    def forward(self, processor_batch: Mapping[str, Any]) -> torch.Tensor:
        return self.head(self.extract_features(processor_batch))

    @torch.no_grad()
    def generate_plan(
        self, model_inputs: Mapping[str, torch.Tensor], max_new_tokens: int = 64,
    ) -> torch.Tensor:
        """Return only new plan token IDs, using this model's original frozen head.

        Inputs must come from ``DecisionProcessor.prepare_plan``. Routing and
        decoding belong to the caller; this method does not implement a learned
        gate, execute actions, or feed any ground-truth next instruction.
        """
        if self.language_model is None:
            raise RuntimeError("Planning requires the original conditional-generation wrapper")
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        inputs = self._move_model_inputs(model_inputs)
        length = inputs["input_ids"].shape[1]
        if length + max_new_tokens > int(self.model_config.get("max_length", 8192)):
            raise ValueError("Prompt plus requested plan generation exceeds the configured length budget")
        self.language_model.eval()
        result = self.language_model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            use_cache=True, logits_to_keep=1, return_dict_in_generate=False,
        )
        return result[:, length:]
