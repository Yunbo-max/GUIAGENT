"""Real visual Qwen batches must preserve individual-example gradients and budgets."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from improving.android.training import language_lora_targets, response_forward, token_kl


def tiny_config():
    from transformers import Qwen3VLConfig
    return Qwen3VLConfig(
        image_token_id=90, vision_start_token_id=91, vision_end_token_id=92,
        video_token_id=93,
        text_config={"vocab_size": 100, "hidden_size": 32, "intermediate_size": 64,
                     "num_hidden_layers": 2, "num_attention_heads": 4,
                     "num_key_value_heads": 2, "head_dim": 8,
                     "max_position_embeddings": 128,
                     "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2],
                                      "mrope_interleaved": True}},
        vision_config={"depth": 2, "hidden_size": 32, "intermediate_size": 64,
                       "num_heads": 4, "patch_size": 2, "spatial_merge_size": 2,
                       "temporal_patch_size": 2, "out_hidden_size": 32,
                       "num_position_embeddings": 16, "deepstack_visual_indexes": [0, 1]},
    )


def encode_fixture(index, feedback=False):
    # Different screenshot sizes, prompt lengths, response lengths, and teacher lengths.
    wide = bool(index % 2)
    prompt = ([1, 2, 3, 91, 90, 90, 92] if wide else [1, 91, 90, 92])
    if feedback:
        prompt += [5, 6, 9, 10] if wide else [5, 6]
    response = [12, 13, 14] if wide else [7, 8]
    ids = torch.tensor([prompt + response])
    patches = 8 if wide else 4
    return {"inputs": {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                       "pixel_values": torch.arange(patches * 24).reshape(patches, 24).float() / 100,
                       "image_grid_thw": torch.tensor([[1, 2, 4 if wide else 2]])},
            "prompt_length": len(prompt), "response_length": len(response)}


@pytest.mark.parametrize("batch", [2, 4])
def test_real_batched_visual_logits_and_lora_gradients_match_individual_examples(batch):
    from transformers import Qwen3VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model
    from improving.android.training import response_batch_forward
    torch.manual_seed(42)
    base = Qwen3VLForConditionalGeneration(tiny_config()).float().eval()
    base.requires_grad_(False)
    model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", r=2, lora_alpha=4,
        lora_dropout=0., target_modules=language_lora_targets(base), bias="none"))
    model.eval()
    students = [encode_fixture(i) for i in range(batch)]
    teachers = [encode_fixture(i, feedback=True) for i in range(batch)]
    before = deepcopy(students)
    with model.disable_adapter(), torch.no_grad():
        target = [response_forward(model, item).detach() for item in teachers]
        batched_target = response_batch_forward(model, teachers, pad_token_id=0)
    for actual, expected in zip(batched_target, target):
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
    individual = [response_forward(model, item) for item in students]
    expected_logits = [item.detach().clone() for item in individual]
    loss = sum(token_kl(item, teacher) for item, teacher in zip(individual, target)) / batch
    loss.backward()
    expected_gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
    model.zero_grad(set_to_none=True)
    seen_batches = []
    hook = model.register_forward_pre_hook(
        lambda module, args, kwargs: seen_batches.append(kwargs["input_ids"].shape[0]), with_kwargs=True)
    batched = response_batch_forward(model, students, pad_token_id=0)
    hook.remove()
    assert seen_batches == [batch], "All examples must share one model forward"
    for actual, expected in zip(batched, expected_logits):
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
    batch_loss = sum(token_kl(item, teacher) for item, teacher in zip(batched, target)) / batch
    torch.testing.assert_close(batch_loss, loss.detach(), rtol=1e-4, atol=1e-6)
    batch_loss.backward()
    for name, parameter in model.named_parameters():
        if name in expected_gradients:
            torch.testing.assert_close(parameter.grad, expected_gradients[name], rtol=2e-4, atol=2e-6)
    for original, current in zip(before, students):
        for key in original["inputs"]:
            assert torch.equal(original["inputs"][key], current["inputs"][key])


@pytest.mark.parametrize("mode", ["feedback", "sft"])
@pytest.mark.parametrize("budget", ["updates", "epoch"])
@pytest.mark.parametrize("candidate", [2, 4])
def test_training_batch_preserves_effective_batch_and_saved_update(tmp_path, monkeypatch, mode, budget, candidate):
    from transformers import Qwen3VLForConditionalGeneration
    import improving.android.training as training
    observations = {}

    class Processor:
        tokenizer = SimpleNamespace(pad_token_id=0)
        def save_pretrained(self, path):
            (path / "fixture_processor.txt").write_text("tiny model test")

    class Policy:
        def __init__(self, checkpoint, config):
            self.model = Qwen3VLForConditionalGeneration(tiny_config()).float().eval()
            self.model.requires_grad_(False)
            self.device = torch.device("cpu")
            self.processor = Processor()
            key = config["train"]["batch_size"]
            observations[key] = []
            self.model.model.register_forward_pre_hook(lambda module, args, kwargs:
                observations[key].append(kwargs["input_ids"].shape[0]), with_kwargs=True)
        def encode_example(self, example, feedback=None):
            return encode_fixture(example["index"], feedback is not None)
        def close(self):
            self.model = None

    monkeypatch.setattr(training, "QwenPolicy", Policy)
    examples = [{"index": i, "split": "train", "policy": "native",
                 "messages": [{"role": "user", "content": "fixture"}],
                 "response_token_ids": [12, 13, 14] if i % 2 else [7, 8],
                 "feedback": "Actual action parser error in this fixture"} for i in range(5)]
    receipts = []
    for batch in (1, candidate):
        cfg = {"runtime_seed": 77, "train": {"max_updates": 2,
               "batch_size": batch, "gradient_accumulation_steps": 4 // batch,
               "lora_rank": 2, "lora_alpha": 4, "learning_rate": .001,
               "gradient_checkpointing": True}}
        if budget == "epoch":
            cfg["train"].pop("max_updates")
            cfg["train"]["epochs"] = 1
        train = training.train_feedback if mode == "feedback" else training.train_sft
        receipts.append(train("fixture", examples, tmp_path / str(batch), cfg))
    assert set(observations[1]) == {1}
    assert set(observations[candidate]) == ({1, candidate} if budget == "epoch" else {candidate})
    for field in ("optimizer_steps", "example_presentations", "response_token_presentations",
                  "student_prompt_token_presentations", "teacher_prompt_token_presentations",
                  "example_exposure_counts", "exposure_order_sha256"):
        assert receipts[0][field] == receipts[1][field]
    assert receipts[1]["example_presentations"] == (5 if budget == "epoch" else 8)
    assert receipts[1]["microbatch_size"] == candidate
    assert receipts[1]["effective_batch_size"] == 4
    assert receipts[1]["forward_microbatches"] == ((5 + candidate - 1) // candidate if budget == "epoch" else 8 // candidate)
    one = Qwen3VLForConditionalGeneration.from_pretrained(tmp_path / "1").state_dict()
    two = Qwen3VLForConditionalGeneration.from_pretrained(tmp_path / str(candidate)).state_dict()
    for name in one:
        torch.testing.assert_close(one[name], two[name], rtol=1e-4, atol=2e-6)


def test_batched_forward_rejects_invalid_metadata_and_unknown_fields():
    from improving.android.training import response_batch_forward
    rows = [encode_fixture(0), encode_fixture(1)]
    rows[1]["response_length"] += 1
    with pytest.raises(ValueError, match="length"):
        response_batch_forward(None, rows, pad_token_id=0)
    rows = [encode_fixture(0), encode_fixture(1)]
    for row in rows:
        row["inputs"]["unsupported_field"] = torch.tensor([1])
    with pytest.raises(ValueError, match="Unsupported"):
        response_batch_forward(None, rows, pad_token_id=0)
