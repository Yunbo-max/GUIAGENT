"""Prompt and expanded-offset contracts with a hand-authored fake processor.

The fake tokenizer consumes one ordinary character per token and one image
placeholder per image token. It is intentionally not a Qwen tokenizer replica;
its known offsets make padding/placeholder-alignment mistakes observable.
"""

import copy
import re
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from improving.gui_decision.dataset import make_model_observation
from improving.gui_decision.processor import DecisionProcessor, ProcessorConfig, build_prompt


class _OffsetProcessor:
    image_token = "<|image_pad|>"
    image_token_id = 99

    def __init__(self):
        self.tokenizer = SimpleNamespace(is_fast=True, padding_side="right", chat_template="fixture")
        self.image_processor = SimpleNamespace(merge_size=2, patch_size=16)
        self.expected_candidate_ends = []
        self.expanded_prompts = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False
        assert enable_thinking is False
        contents = messages[0]["content"]
        body = "".join(self.image_token if part["type"] == "image" else part["text"]
                       for part in contents)
        return "<user>" + body + "</user>" + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, *, text, images, padding, truncation, return_tensors,
                 return_offsets_mapping, return_mm_token_type_ids, images_kwargs):
        assert len(text) == len(images)
        assert padding is True and truncation is False
        assert return_tensors == "pt" and return_offsets_mapping is True
        assert return_mm_token_type_ids is True
        assert images_kwargs["max_pixels"] == 262144
        grids = [[1, 4, 6] if index == 0 else [1, 4, 4] for index in range(len(text))]
        rows, offsets, unpadded_ends = [], [], []
        self.expanded_prompts = []
        pattern = re.compile(re.escape(self.image_token) + r"|.", re.DOTALL)
        for prompt, grid in zip(text, grids):
            image_count = 6 if grid == [1, 4, 6] else 4
            expanded = prompt.replace(self.image_token, self.image_token * image_count)
            self.expanded_prompts.append(expanded)
            lexemes = list(pattern.finditer(expanded))
            rows.append([self.image_token_id if match.group() == self.image_token else 3
                         for match in lexemes])
            offsets.append([(match.start(), match.end()) for match in lexemes])
            # Compute expected token positions from the expanded text itself,
            # independently of the adapter's unexpanded-string shift logic.
            token_ending_at = {match.end(): index for index, match in enumerate(lexemes)}
            unpadded_ends.append([
                token_ending_at[match.end()]
                for match in re.finditer(r"\nOption \d{3}: [^\n]*\n", expanded)
            ])
        width = max(map(len, rows))
        ids, masks, padded_offsets = [], [], []
        self.expected_candidate_ends = []
        for row, spans, ends in zip(rows, offsets, unpadded_ends):
            pad = width - len(row)
            assert self.tokenizer.padding_side == "left"
            ids.append([0] * pad + row)
            masks.append([0] * pad + [1] * len(row))
            padded_offsets.append([(0, 0)] * pad + spans)
            self.expected_candidate_ends.append([index + pad for index in ends])
        input_ids = torch.tensor(ids)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.tensor(masks),
            "mm_token_type_ids": (input_ids == self.image_token_id).long(),
            "offset_mapping": torch.tensor(padded_offsets),
            "image_grid_thw": torch.tensor(grids),
            "pixel_values": torch.zeros(sum(grid[1] * grid[2] for grid in grids), 1),
        }


def _image_observation(record, tmp_path):
    observation = make_model_observation(record, "low", tmp_path)
    image_path = tmp_path / "screen.png"
    Image.new("RGB", (100, 100), "white").save(image_path)
    observation["screenshot_path"] = str(image_path)
    return observation


def test_candidate_readouts_follow_expanded_offsets_and_batch_left_padding(prepared_record, tmp_path):
    first = _image_observation(prepared_record, tmp_path)
    second = copy.deepcopy(first)
    second["sample_id"] = "13:0"
    second["goal"] = second["instruction"] = "Go back."
    second["history"] = []
    second["candidates"] = [c for c in second["candidates"] if c["kind"] == "navigate_back"]
    fake = _OffsetProcessor()
    processor = DecisionProcessor(fake, ProcessorConfig())

    batch = processor([first, second])

    for index, expected in enumerate(fake.expected_candidate_ends):
        valid = batch["candidate_mask"][index]
        assert batch["candidate_end_indices"][index, valid].tolist() == expected
    assert batch["candidate_mask"].sum(-1).tolist() == [len(first["candidates"]), 1]
    assert batch["model_inputs"]["attention_mask"][1, 0].item() == 0
    assert (batch["model_inputs"]["input_ids"][0] == fake.image_token_id).sum().item() == 6
    assert (batch["model_inputs"]["input_ids"][1] == fake.image_token_id).sum().item() == 4


def test_candidate_text_cap_preserves_every_option_and_reports_removed_characters(
    prepared_record, tmp_path
):
    observation = make_model_observation(prepared_record, "low", tmp_path)
    observation["candidates"][0]["text"] = "X" * 200
    original = copy.deepcopy(observation)

    prompt = build_prompt(observation, candidate_text_max_chars=128)

    assert len(prompt.candidate_segments) == len(observation["candidates"])
    assert prompt.truncated_candidate_count == 1
    assert prompt.truncated_character_count == 72
    assert "X" * 128 in prompt.candidate_segments[0]
    assert "X" * 129 not in prompt.candidate_segments[0]
    assert observation == original


def test_plan_prompt_ignores_low_instruction_even_when_directly_called(prepared_record, tmp_path):
    observation = make_model_observation(prepared_record, "low", tmp_path)
    observation["instruction"] = "ORACLE_LOW_SENTINEL"

    prompt = build_prompt(observation, for_plan=True)

    assert "ORACLE_LOW_SENTINEL" not in prompt.before_image + prompt.after_image
    assert observation["goal"] in prompt.before_image
    assert len(prompt.candidate_segments) == len(observation["candidates"])


def test_observable_text_cannot_create_additional_image_placeholders(prepared_record, tmp_path):
    observation = make_model_observation(prepared_record, "high", tmp_path)
    observation["candidates"][0]["text"] = "literal <|image_pad|> and <|placeholder|>"

    prompt = build_prompt(observation)

    assert "<|image_pad|>" not in prompt.before_image + prompt.after_image
    assert "<|placeholder|>" not in prompt.before_image + prompt.after_image


def test_sequence_overflow_is_explicit_and_never_deletes_candidate_options(prepared_record, tmp_path):
    observation = _image_observation(prepared_record, tmp_path)
    original = copy.deepcopy(observation)
    processor = DecisionProcessor(_OffsetProcessor(), ProcessorConfig(max_length=16))

    with pytest.raises(ValueError, match="exceeds max_length"):
        processor([observation])

    assert observation == original
