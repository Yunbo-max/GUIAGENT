"""Leakage-resistant, single-screenshot inputs for the Qwen decision adapter.

Only sanitized model observations are accepted. Candidate descriptions have a
declared prefix cap; the complete option set, task, instruction and history are
retained. The entire token sequence is never truncated. Candidate token indices
refer to the *image-expanded*, padded tokenization returned by Qwen's processor.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .schema import validate_model_observation


DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B"
DEFAULT_MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
TRANSFORMERS_VERSION = "5.3.0"
PROMPT_VERSION = "qwen35_gui_candidates_v1"
PLAN_PROMPT_VERSION = "qwen35_gui_plan_v1"
MAX_CANDIDATES = 168
# Preserve GUI-JEPA's action-type vocabulary, including its unused long-press
# slot. The dataset adapter itself maps a long-press target to a click option.
ACTION_TYPE_NAMES = (
    "click", "long_press", "scroll", "open_app", "input_text",
    "navigate_home", "navigate_back", "wait",
)
ACTION_TYPE_IDS = {name: index for index, name in enumerate(ACTION_TYPE_NAMES)}
AUX_NAMES = (
    "center_x", "center_y", "width", "height", "area", "left", "top",
    "clickable", "long_clickable", "editable", "scrollable",
)
MODEL_CONFIG_KEYS = frozenset({
    "model_id", "revision", "device", "backbone_dtype", "attn_implementation",
    "min_pixels", "max_pixels", "max_length", "candidate_text_max_chars",
    "head", "local_files_only",
})
FINAL_QUERY = "Decision query: select the best available action for the instruction."
PLAN_QUERY = (
    "Planning query: using the goal, prior actions and screenshot, state only the "
    "next short action or subgoal in one sentence. Do not explain or include hidden reasoning."
)


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_transformers_version() -> None:
    """Fail before loading assets when the audited processor/API has changed."""
    actual = importlib.metadata.version("transformers")
    if actual != TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"This adapter was authored for transformers=={TRANSFORMERS_VERSION}; "
            f"found {actual}. Use the independent GUI decision environment."
        )


def validate_model_config_keys(config: Mapping[str, Any]) -> None:
    unknown = set(config) - MODEL_CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown model configuration keys: {sorted(unknown)}")


@dataclass(frozen=True)
class ProcessorConfig:
    model_id: str = DEFAULT_MODEL_ID
    revision: str = DEFAULT_MODEL_REVISION
    min_pixels: int = 4096
    max_pixels: int = 262144
    max_length: int = 8192
    candidate_text_max_chars: int = 128

    def __post_init__(self) -> None:
        if self.model_id != DEFAULT_MODEL_ID or self.revision != DEFAULT_MODEL_REVISION:
            raise ValueError("Version 1 requires the pinned official Qwen3.5-0.8B revision")
        for key in ("min_pixels", "max_pixels", "max_length", "candidate_text_max_chars"):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.max_pixels < self.min_pixels:
            raise ValueError("max_pixels must be at least min_pixels")

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> "ProcessorConfig":
        # The same model section is passed to both the model and processor.
        validate_model_config_keys(config)
        names = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in config.items() if key in names})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PromptParts:
    before_image: str
    after_image: str
    candidate_segments: tuple[str, ...]
    truncated_candidate_count: int
    truncated_character_count: int


def _prompt_json(value: Any) -> str:
    # User/a11y strings must not introduce Qwen special tokens or the processor's
    # temporary <|placeholder|> string. JSON escapes preserve the original text
    # meaning without allowing literal '<' delimiters in these quoted fields.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e")


def build_prompt(
    observation: Mapping[str, Any], candidate_text_max_chars: int = 128,
    *, for_plan: bool = False,
) -> PromptParts:
    """Build observable input only; do not modify records or remove options."""
    validate_model_observation(observation)
    if type(candidate_text_max_chars) is not int or candidate_text_max_chars < 1:
        raise ValueError("candidate_text_max_chars must be a positive integer")
    candidates = observation["candidates"]
    if len(candidates) > MAX_CANDIDATES:
        raise ValueError(f"At most {MAX_CANDIDATES} candidates are supported; none were dropped")
    before = (
        "Choose the next GUI action for a single screenshot.\n"
        "Task and prior actions:\n"
        + _prompt_json({"goal": observation["goal"],
                        # PLAN is autonomous High mode even if this sanitized
                        # observation originated from a Low-mode data loader.
                        "instruction": observation["goal"] if for_plan else observation["instruction"],
                        "history": observation["history"]})
        + "\nScreenshot:\n"
    )
    segments = []
    shortened = 0
    removed = 0
    for index, candidate in enumerate(candidates):
        original = candidate["text"]
        text = original[:candidate_text_max_chars]
        if len(original) > len(text):
            shortened += 1
            removed += len(original) - len(text)
        # Geometry/flags have separate numeric features. Avoid wasting tokens on
        # IDs and repeating each box in the language prompt. The index here maps
        # exactly to the original candidate order, not to a target label.
        description = {"kind": candidate["kind"], "text": text}
        if candidate["direction"] is not None:
            description["direction"] = candidate["direction"]
        segments.append(f"\nOption {index:03d}: {_prompt_json(description)}\n")
    after = (
        "\nAvailable actions in their original order; descriptions may be abbreviated.\n"
        + "".join(segments) + "\n" + (PLAN_QUERY if for_plan else FINAL_QUERY) + "\n"
    )
    return PromptParts(before, after, tuple(segments), shortened, removed)


def _candidate_aux(candidate: Mapping[str, Any]) -> list[float]:
    box = candidate["box"]
    if box is None:
        geometry = [0.0] * 7
    else:
        left, top, right, bottom = (float(value) for value in box)
        width, height = right - left, bottom - top
        geometry = [(left + right) / 2, (top + bottom) / 2,
                    width, height, width * height, left, top]
    return geometry + [float(flag) for flag in candidate["flags"]]


class DecisionProcessor:
    """Render observations, process each screenshot once, and align option ends.

    Injecting an already-created processor is supported for small authored
    fixtures. ``from_pretrained`` is the real asset-loading entry point.
    """

    def __init__(self, processor: Any, config: ProcessorConfig | None = None):
        self.processor = processor
        self.config = config or ProcessorConfig()
        if not getattr(processor.tokenizer, "is_fast", False):
            raise ValueError("A fast tokenizer with character offset mappings is required")
        # Left padding is the runtime default for the hybrid causal backbone.
        # All alignment/readout logic below also handles right-padded fixtures.
        processor.tokenizer.padding_side = "left"
        self.image_token = processor.image_token
        self.image_token_id = int(processor.image_token_id)
        self.merge_size = int(processor.image_processor.merge_size)
        self.patch_size = int(processor.image_processor.patch_size)
        if self.merge_size != 2 or self.patch_size != 16:
            raise ValueError("Unexpected Qwen image patch/merge configuration")

    @classmethod
    def from_pretrained(cls, config: Mapping[str, Any]) -> "DecisionProcessor":
        require_transformers_version()
        from transformers import AutoProcessor

        settings = ProcessorConfig.from_dict(config)
        local_only = config.get("local_files_only", False)
        if type(local_only) is not bool:
            raise ValueError("local_files_only must be a bool")
        processor = AutoProcessor.from_pretrained(
            settings.model_id, revision=settings.revision,
            local_files_only=local_only, trust_remote_code=False,
        )
        return cls(processor, settings)

    def fingerprint_metadata(self) -> dict[str, Any]:
        template = getattr(self.processor, "chat_template", None)
        if template is None:
            template = getattr(self.processor.tokenizer, "chat_template", None)
        metadata = {
            "prompt_version": PROMPT_VERSION,
            "plan_prompt_version": PLAN_PROMPT_VERSION,
            "plan_instruction_policy": "goal_only_ignore_observation_instruction",
            "processor_config": self.config.to_dict(),
            "transformers_version": TRANSFORMERS_VERSION,
            "chat_template_sha256": _digest(template),
            "image_token": self.image_token,
            "image_token_id": self.image_token_id,
            "patch_size": self.patch_size,
            "spatial_merge_size": self.merge_size,
            "padding_side": self.processor.tokenizer.padding_side,
            "text_feature": "last_token_intersecting_candidate_segment",
            "query_feature": "last_nonpadding_token",
            "candidate_text_policy": "candidate_text_prefix",
            "candidate_text_max_chars": self.config.candidate_text_max_chars,
            "sequence_overflow_policy": "error",
            "candidate_overflow_policy": "error",
            "max_candidates": MAX_CANDIDATES,
            "action_type_names": list(ACTION_TYPE_NAMES),
            "aux_names": list(AUX_NAMES),
            "prompt_order": ["goal_instruction_history", "screenshot", "candidates", "fixed_query"],
            "reserved_delimiter_policy": "JSON_escape_angle_brackets",
            "processor_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        return {**metadata, "fingerprint": _digest(metadata)}

    def __call__(self, observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return self._prepare_batch(observations, for_plan=False)

    def prepare_plan(self, observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Prepare an autonomous goal-conditioned PLAN prompt, without Low GT.

        ``instruction`` is ignored and replaced by ``goal`` before rendering.
        The original sanitized observations remain unchanged. The caller decodes
        generated token IDs and may place its own generated subgoal into a new
        ACT observation's instruction field.
        """
        batch = self._prepare_batch(observations, for_plan=True)
        return {
            "model_inputs": batch["model_inputs"],
            "prompt_length": int(batch["model_inputs"]["input_ids"].shape[1]),
            "sample_ids": batch["sample_ids"],
            "metadata": batch["metadata"],
        }

    def _prepare_batch(
        self, observations: Sequence[Mapping[str, Any]], *, for_plan: bool,
    ) -> dict[str, Any]:
        from PIL import Image

        if not observations:
            raise ValueError("Cannot process an empty observation batch")
        parts = [build_prompt(item, self.config.candidate_text_max_chars, for_plan=for_plan)
                 for item in observations]
        prompts = []
        images = []
        try:
            for item, prompt in zip(observations, parts):
                path = Path(item["screenshot_path"])
                if not path.is_absolute():
                    raise ValueError("screenshot_path must be resolved to an absolute local path")
                with Image.open(path) as source:
                    if source.size != (item["screen_width"], item["screen_height"]):
                        raise ValueError(f"Screenshot dimensions disagree with observation {item['sample_id']}")
                    images.append(source.convert("RGB"))
                messages = [{"role": "user", "content": [
                    {"type": "text", "text": prompt.before_image},
                    {"type": "image", "image": str(path)},
                    {"type": "text", "text": prompt.after_image},
                ]}]
                rendered = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=for_plan,
                    enable_thinking=False,
                )
                if not isinstance(rendered, str) or rendered.count(self.image_token) != 1:
                    raise ValueError("The rendered prompt must contain exactly one image placeholder")
                prompts.append(rendered)
            # The processor expands image placeholders BEFORE tokenization.
            # Request offsets on that very call; a separate tokenizer pass over
            # the unexpanded template would give incorrect candidate positions.
            processed = dict(self.processor(
                text=prompts, images=images, padding=True, truncation=False,
                return_tensors="pt", return_offsets_mapping=True,
                return_mm_token_type_ids=True,
                images_kwargs={"min_pixels": self.config.min_pixels,
                               "max_pixels": self.config.max_pixels},
            ))
        finally:
            for image in images:
                image.close()

        if "offset_mapping" not in processed:
            raise ValueError("Qwen processor did not return token offsets")
        offsets = processed.pop("offset_mapping")
        allowed_inputs = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw",
                          "mm_token_type_ids"}
        if set(processed) - allowed_inputs:
            raise ValueError(f"Unexpected processor model inputs: {sorted(set(processed) - allowed_inputs)}")
        required_inputs = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw",
                           "mm_token_type_ids"}
        if not required_inputs.issubset(processed):
            raise ValueError(f"Missing processor inputs: {sorted(required_inputs - set(processed))}")
        if any(not isinstance(value, torch.Tensor) for value in processed.values()):
            raise ValueError("Processor model inputs must be tensors")
        ids, attention = processed["input_ids"], processed["attention_mask"]
        batch_size = len(observations)
        if ids.ndim != 2 or ids.shape[0] != batch_size or attention.shape != ids.shape:
            raise ValueError("Processor input_ids/attention_mask have inconsistent shapes")
        if offsets.shape != (*ids.shape, 2):
            raise ValueError("Token offset mappings do not match padded input_ids")
        modality = processed["mm_token_type_ids"]
        if modality.shape != ids.shape or not torch.equal(modality == 1, ids == self.image_token_id):
            raise ValueError("Modality IDs do not match the image tokens required for multimodal positions")
        token_counts = attention.bool().sum(dim=-1).tolist()
        overflowing = [(observations[index]["sample_id"], int(count))
                       for index, count in enumerate(token_counts)
                       if count > self.config.max_length]
        if overflowing:
            raise ValueError(
                f"Sequence length exceeds max_length={self.config.max_length}: {overflowing}. "
                "No whole-sequence truncation or option removal was applied. Adjust the explicit input config."
            )
        grid_rows = processed["image_grid_thw"]
        if grid_rows.shape != (batch_size, 3):
            raise ValueError("Version 1 requires exactly one image_grid_thw row per sample")
        counts = [len(item["candidates"]) for item in observations]
        width = max(counts)
        ends = torch.zeros((batch_size, width), dtype=torch.long)
        bounds = torch.zeros((batch_size, width, 4), dtype=torch.float32)
        regions = torch.zeros((batch_size, width), dtype=torch.bool)
        auxiliary = torch.zeros((batch_size, width, len(AUX_NAMES)), dtype=torch.float32)
        types = torch.zeros((batch_size, width), dtype=torch.long)
        mask = torch.zeros((batch_size, width), dtype=torch.bool)
        expanded_hashes = []

        for row, (item, prompt_parts, rendered) in enumerate(zip(observations, parts, prompts)):
            temporal, height, image_width = (int(value) for value in grid_rows[row].tolist())
            if temporal != 1 or height <= 0 or image_width <= 0:
                raise ValueError("Image grids must describe a single nonempty screenshot")
            if height % self.merge_size or image_width % self.merge_size:
                raise ValueError("Image grid sides must be divisible by the spatial merge size")
            image_count = temporal * height * image_width // self.merge_size ** 2
            actual_image_count = int(((ids[row] == self.image_token_id) & attention[row].bool()).sum())
            if actual_image_count != image_count:
                raise ValueError("Image-token count disagrees with the expanded image grid")
            image_position = rendered.index(self.image_token)
            expanded = rendered.replace(self.image_token, self.image_token * image_count, 1)
            expanded_hashes.append(hashlib.sha256(expanded.encode("utf-8")).hexdigest())
            shift = (image_count - 1) * len(self.image_token)
            row_offsets = offsets[row]
            for column, (candidate, segment) in enumerate(zip(item["candidates"], prompt_parts.candidate_segments)):
                if rendered.count(segment) != 1:
                    raise ValueError("Candidate segment is not uniquely identifiable in the rendered prompt")
                unexpanded_start = rendered.index(segment)
                if unexpanded_start <= image_position:
                    raise ValueError("Candidate descriptions must occur after the screenshot")
                start = unexpanded_start + shift
                stop = start + len(segment)
                if expanded[start:stop] != segment:
                    raise ValueError("Candidate offsets disagree with image-placeholder expansion")
                intersects = (
                    (row_offsets[:, 0] < stop) & (row_offsets[:, 1] > start)
                    & (row_offsets[:, 1] > row_offsets[:, 0]) & attention[row].bool()
                )
                positions = torch.nonzero(intersects, as_tuple=False).flatten()
                if positions.numel() == 0:
                    raise ValueError(f"Candidate {column} in {item['sample_id']} has no token readout")
                ends[row, column] = positions[-1]
                if candidate["box"] is not None:
                    bounds[row, column] = torch.tensor(candidate["box"], dtype=torch.float32)
                    regions[row, column] = True
                auxiliary[row, column] = torch.tensor(_candidate_aux(candidate), dtype=torch.float32)
                types[row, column] = ACTION_TYPE_IDS[candidate["kind"]]
                mask[row, column] = True

        metadata = {
            **self.fingerprint_metadata(),
            "route": "PLAN" if for_plan else "ACT",
            "active_prompt_version": PLAN_PROMPT_VERSION if for_plan else PROMPT_VERSION,
            "token_counts": [int(value) for value in token_counts],
            "candidate_counts": counts,
            "expanded_prompt_sha256": expanded_hashes,
            "candidate_text_truncation": {
                "policy": "candidate_text_prefix",
                "max_chars": self.config.candidate_text_max_chars,
                "per_sample_candidate_counts": [part.truncated_candidate_count for part in parts],
                "per_sample_removed_characters": [part.truncated_character_count for part in parts],
                "total_candidate_count": sum(part.truncated_candidate_count for part in parts),
                "total_removed_characters": sum(part.truncated_character_count for part in parts),
            },
        }
        return {
            "model_inputs": processed,
            "candidate_end_indices": ends,
            "candidate_bounds": bounds,
            "candidate_has_region": regions,
            "candidate_aux": auxiliary,
            "candidate_type_ids": types,
            "candidate_mask": mask,
            "sample_ids": [item["sample_id"] for item in observations],
            "metadata": metadata,
        }
