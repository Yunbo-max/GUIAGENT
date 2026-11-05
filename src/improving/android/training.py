"""Response-only LoRA SFT and feedback self-distillation for Android.

The frozen base checkpoint supplies the teacher through PEFT.disable_adapter().
Only fresh LoRA parameters are optimized; after training they are merged into
one native student. No GRSD or reward loss is added to feedback training.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile
import time
from typing import Any

from .model import QwenPolicy


def response_logits(logits, prompt_length: int, response_length: int):
    """Align full-sequence logits to the same recorded response token positions."""
    if prompt_length < 1 or response_length < 1:
        raise ValueError("Prompt and response must each contain at least one token")
    start = prompt_length - 1
    stop = start + response_length
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] < stop:
        raise ValueError("Full logits do not cover the requested response positions")
    return logits[:, start:stop, :]


def response_forward(model, encoded: dict):
    """Compute only logits predicting response tokens, retaining native vision.

    Qwen3VLForConditionalGeneration supports tensor ``logits_to_keep`` from
    Transformers 4.57.1. This avoids allocating vocabulary logits for image and
    history positions. Do not pass labels: loss is evaluated explicitly below.
    """
    import torch
    prompt = int(encoded["prompt_length"])
    count = int(encoded["response_length"])
    if prompt < 1 or count < 1:
        raise ValueError("Empty prompt/continuation is not trainable")
    inputs = encoded["inputs"]
    if inputs["input_ids"].shape[1] != prompt + count:
        raise ValueError("Encoded sequence length disagrees with continuation metadata")
    positions = torch.arange(prompt - 1, prompt + count - 1,
                             device=inputs["input_ids"].device)
    logits = model(**inputs, logits_to_keep=positions, use_cache=False).logits
    if logits.shape[:2] != (1, count):
        raise RuntimeError("Model did not honor native response-logit selection")
    return logits


def response_batch_forward(model, encoded: list[dict], *, pad_token_id: int):
    """One visual model forward for several recorded prompt/response pairs.

    Right padding preserves each example's native token positions. Qwen accepts
    one shared logits_to_keep vector, so select the union of response positions
    and then recover each row's exact continuation. Padding contributes no loss.
    Vision patches/grids retain example order for native multimodal RoPE.
    """
    import torch
    import torch.nn.functional as F

    if not encoded:
        raise ValueError("An empty visual microbatch cannot be forwarded")
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError("A nonnegative pad_token_id is required")
    if len(encoded) == 1:
        return [response_forward(model, encoded[0])]
    token_fields = {"input_ids", "attention_mask", "token_type_ids", "mm_token_type_ids"}
    visual_fields = {"pixel_values", "image_grid_thw"}
    keys = set(encoded[0]["inputs"])
    if keys - token_fields - visual_fields:
        raise ValueError(f"Unsupported batched input fields: {sorted(keys - token_fields - visual_fields)}")
    if not {"input_ids", "attention_mask", *visual_fields}.issubset(keys):
        raise ValueError("Batched inputs require native screenshot tensors and an attention mask")
    lengths, positions = [], []
    for row in encoded:
        inputs = row["inputs"]
        prompt, count = int(row["prompt_length"]), int(row["response_length"])
        length = prompt + count
        if set(inputs) != keys:
            raise ValueError("Every example must have the same native input fields")
        if prompt < 1 or count < 1 or inputs["input_ids"].shape != (1, length):
            raise ValueError("Encoded sequence length disagrees with continuation metadata")
        if any(inputs[key].shape != (1, length) for key in keys & token_fields):
            raise ValueError("Token field length differs from input_ids")
        if not bool((inputs["attention_mask"] == 1).all()):
            raise ValueError("Microbatch inputs must be unpadded individual encodings")
        lengths.append(length)
        positions.append(torch.arange(prompt - 1, length - 1, device=inputs["input_ids"].device))
    width = max(lengths)
    inputs = {}
    for key in keys:
        if key in token_fields:
            pad = pad_token_id if key == "input_ids" else 0
            inputs[key] = torch.cat([F.pad(row["inputs"][key], (0, width - length), value=pad)
                                     for row, length in zip(encoded, lengths)], dim=0)
        else:
            inputs[key] = torch.cat([row["inputs"][key] for row in encoded], dim=0)
    selected = torch.unique(torch.cat(positions), sorted=True)
    logits = model(**inputs, logits_to_keep=selected, use_cache=False).logits
    if logits.shape[:2] != (len(encoded), len(selected)):
        raise RuntimeError("Model did not honor batched response-logit selection")
    return [logits[index:index + 1].index_select(1, torch.searchsorted(selected, positions[index]))
            for index in range(len(encoded))]


def token_kl(student_logits, teacher_logits, top_k: int = 0, chunk_size: int = 32):
    """Mean response reverse KL, with a detached teacher and bounded temporaries.

    top_k=0 is exact vocabulary KL. Positive top_k uses the union of student and
    teacher top-k tokens plus ONE exact complement-mass bin per token. This is a
    coarse-grained KL, not a renormalized top-k approximation. Duplicate union
    indices are counted once. Teacher logits may reside on CPU.
    """
    import torch
    import torch.nn.functional as F

    if student_logits.shape != teacher_logits.shape or student_logits.ndim not in {2, 3}:
        raise ValueError("Student and teacher logits must have identical token/vocabulary shapes")
    if student_logits.ndim == 3 and student_logits.shape[0] != 1:
        raise ValueError("KL runtime supports one example at a time")
    if type(top_k) is not int or top_k < 0 or type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("Invalid KL top-k or chunk size")
    student = student_logits.reshape(-1, student_logits.shape[-1])
    teacher = teacher_logits.detach().reshape(-1, teacher_logits.shape[-1])
    if student.shape[0] == 0 or student.shape[1] < 2:
        raise ValueError("KL needs response tokens and at least two vocabulary entries")
    total = student.new_zeros((), dtype=torch.float32)
    vocab = student.shape[-1]
    for offset in range(0, len(student), chunk_size):
        s = student[offset:offset + chunk_size].float()
        t = teacher[offset:offset + chunk_size].to(device=s.device, dtype=torch.float32)
        if not bool(torch.isfinite(s).all()) or not bool(torch.isfinite(t).all()):
            raise FloatingPointError("Nonfinite logits in feedback distillation")
        logp = F.log_softmax(s, dim=-1)
        logq = F.log_softmax(t, dim=-1)
        if top_k == 0 or top_k >= vocab:
            values = (logp.exp() * (logp - logq)).sum(dim=-1)
        else:
            selected = torch.cat((s.detach().topk(top_k, dim=-1).indices,
                                  t.topk(top_k, dim=-1).indices), dim=-1).sort(dim=-1).values
            valid = torch.ones_like(selected, dtype=torch.bool)
            valid[:, 1:] = selected[:, 1:] != selected[:, :-1]
            selected_p = logp.gather(-1, selected)
            selected_q = logq.gather(-1, selected)
            values = (selected_p.exp() * (selected_p - selected_q) * valid).sum(dim=-1)
            mask = torch.zeros_like(s, dtype=torch.bool).scatter_(1, selected, True)
            # Complement mass via masked logsumexp avoids subtractive cancellation.
            tail_p = torch.logsumexp(logp.masked_fill(mask, -torch.inf), dim=-1)
            tail_q = torch.logsumexp(logq.masked_fill(mask, -torch.inf), dim=-1)
            # A full union has no complement: its mass and contribution are zero.
            tail_exists = (~mask).any(dim=-1)
            safe_p = torch.where(tail_exists, tail_p, torch.zeros_like(tail_p))
            safe_q = torch.where(tail_exists, tail_q, torch.zeros_like(tail_q))
            values = values + torch.where(tail_exists, safe_p.exp() * (safe_p - safe_q),
                                           torch.zeros_like(safe_p))
        total = total + values.sum()
    return total / len(student)


def response_cross_entropy(logits, target_ids, chunk_size: int = 32):
    """Mean response-token CE without prompt-token supervision or giant FP32 logits."""
    import torch
    import torch.nn.functional as F
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError("Expected single-example response logits")
    targets = target_ids.reshape(-1).to(logits.device)
    values = logits.reshape(-1, logits.shape[-1])
    if len(targets) != len(values) or len(values) == 0:
        raise ValueError("Response labels and logits must align exactly")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    loss = values.new_zeros((), dtype=torch.float32)
    for offset in range(0, len(values), chunk_size):
        loss = loss + F.cross_entropy(values[offset:offset + chunk_size].float(),
                                     targets[offset:offset + chunk_size], reduction="sum")
    return loss / len(values)


def language_lora_targets(model) -> list[str]:
    """Only native language decoder attention linears; never visual projections."""
    import torch
    pattern = re.compile(r"^model\.language_model\.layers\.[0-9]+\.self_attn\.(q|k|v|o)_proj$")
    targets = []
    for name, module in model.named_modules():
        if pattern.fullmatch(name):
            if type(module) is not torch.nn.Linear:
                raise TypeError(f"Non-native attention projection: {name}")
            targets.append(name)
    expected = int(model.config.text_config.num_hidden_layers) * 4
    if len(targets) != expected:
        raise ValueError(f"Expected {expected} native language Q/K/V/O projections, found {len(targets)}")
    return targets


def validate_training_examples(examples: list[dict], feedback: bool) -> None:
    if not isinstance(examples, list) or not examples:
        raise ValueError("No usable training examples; collect reliable training experience first")
    for example in examples:
        if example.get("split") != "train":
            raise ValueError("Only train split examples can update a checkpoint")
        if not example.get("messages"):
            raise ValueError("Missing recorded student prompt")
        ids = example.get("response_token_ids")
        if ids is not None and (not isinstance(ids, list) or not ids):
            raise ValueError("Empty or malformed sampled response_token_ids")
        if ids is None and not example.get("response"):
            raise ValueError("Missing response")
        if feedback:
            if example.get("policy") != "native":
                raise ValueError("Feedback KL must use native student prefixes, never spectral prefixes")
            if ids is None:
                raise ValueError("Feedback KL requires exact sampled response token IDs")
            if not isinstance(example.get("feedback"), str) or not example["feedback"].strip():
                raise ValueError("Feedback KL requires informative evidence from training trajectories")


def plan_training_batches(num_examples: int, accumulation: int, *, epochs: int = 1,
                          max_updates: int | None = None, seed: int = 0):
    """Create exact update groups, with deterministic full-pass sampling.

    A fixed update budget exposes exactly max_updates*accumulation examples.
    The budget may expose fewer than all records, and may cycle the corpus. Both
    cases are explicitly accounted for; this is not a matched token/FLOP budget.
    Without max_updates the historical epoch behavior (including a partial
    accumulation group at each epoch boundary) is preserved.
    """
    if any(type(x) is not int or x < 1 for x in (num_examples, accumulation, epochs)):
        raise ValueError("Example count, accumulation, and epochs must be positive integers")
    if type(seed) is not int or seed < 0 or seed >= 2**63:
        raise ValueError("runtime_seed must be an integer in [0, 2**63)")
    if max_updates is not None and (type(max_updates) is not int or max_updates < 1):
        raise ValueError("max_updates must be a positive integer or null")
    rng = random.Random(seed)
    groups = []
    if max_updates is not None:
        needed = max_updates * accumulation
        order = []
        while len(order) < needed:
            permutation = list(range(num_examples))
            rng.shuffle(permutation)
            order.extend(permutation[:needed - len(order)])
        groups = [order[i:i + accumulation] for i in range(0, needed, accumulation)]
    else:
        for _ in range(epochs):
            groups.extend([list(range(start, min(start + accumulation, num_examples)))
                           for start in range(0, num_examples, accumulation)])
    exposures = [0] * num_examples
    for group in groups:
        for index in group:
            exposures[index] += 1
    presentations = sum(exposures)
    return groups, {
        "budget_mode": "fixed_optimizer_updates" if max_updates is not None else "fixed_epochs",
        "configured_max_updates": max_updates,
        "configured_epochs": epochs,
        "epochs_applied": epochs if max_updates is None else None,
        "total_examples": num_examples,
        "unique_examples_used": sum(x > 0 for x in exposures),
        "example_presentations": presentations,
        "example_exposure_counts": exposures,
        "completed_corpus_passes": presentations // num_examples,
        "effective_corpus_passes": presentations / num_examples,
        "sampling_order": "seeded_full_corpus_permutations" if max_updates is not None else "recorded_order_each_epoch",
        "exposure_order_sha256": hashlib.sha256(json.dumps(groups, separators=(",", ":")).encode()).hexdigest(),
        "randomization_sha256": hashlib.sha256(str(seed).encode()).hexdigest(),
        "budget_matching_contract": "optimizer_updates_and_example_presentations; not equal tokens or FLOPs",
    }


def _train(checkpoint: str, examples: list[dict], output_dir: str | Path,
           config: dict[str, Any], *, feedback: bool) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model

    validate_training_examples(examples, feedback)
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to replace existing checkpoint: {destination}")
    train = dict(config.get("train", {}))
    mode = "feedback" if feedback else "sft"
    train.update(train.get(mode, {}))
    epochs = int(train.get("epochs", 1))
    accumulation = int(train.get("gradient_accumulation_steps", 8))
    rank = int(train.get("lora_rank", 16))
    chunk = int(train.get("logit_chunk_size", 32))
    top_k = int(train.get("kl_top_k", 0))
    if min(epochs, accumulation, rank, chunk) < 1 or top_k < 0:
        raise ValueError("Invalid training hyperparameters")
    batch_size = train.get("batch_size", 1)
    if type(batch_size) is not int or batch_size not in (1, 2, 4):
        raise ValueError("Visual training batch_size must be 1, 2 or 4")
    learning_rate = float(train.get("learning_rate", 1e-5))
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    runtime_seed = config.get("runtime_seed")
    groups, exposure = plan_training_batches(
        len(examples), accumulation * batch_size, epochs=epochs,
        max_updates=train.get("max_updates"), seed=runtime_seed)
    torch.manual_seed(runtime_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(runtime_seed)
    start = time.monotonic()
    policy = QwenPolicy(checkpoint, config)
    stage = None
    try:
        targets = language_lora_targets(policy.model)
        model = get_peft_model(policy.model, LoraConfig(
            task_type="CAUSAL_LM", r=rank, lora_alpha=int(train.get("lora_alpha", 32)),
            lora_dropout=float(train.get("lora_dropout", 0.0)),
            target_modules=targets, bias="none"))
        policy.model = model
        if train.get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                     lr=learning_rate, weight_decay=float(train.get("weight_decay", 0.0)))
        total_updates = len(groups)
        losses: list[float] = []
        optimizer_steps = 0
        forward_microbatches = 0
        pad_token_id = int(policy.processor.tokenizer.pad_token_id) if batch_size > 1 else 0
        response_tokens = 0
        teacher_prompt_tokens = 0
        student_prompt_tokens = 0
        optimizer.zero_grad(set_to_none=True)
        if policy.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(policy.device)
        for group in groups:
            for start_index in range(0, len(group), batch_size):
                microbatch = [examples[index] for index in group[start_index:start_index + batch_size]]
                student = [policy.encode_example(example) for example in microbatch]
                response_tokens += sum(int(row["response_length"]) for row in student)
                student_prompt_tokens += sum(int(row["prompt_length"]) for row in student)
                if feedback:
                    teacher = [policy.encode_example(example, feedback=example["feedback"])
                               for example in microbatch]
                    teacher_prompt_tokens += sum(int(row["prompt_length"]) for row in teacher)
                    model.eval()
                    # Frozen base == pre-update checkpoint for this whole training call.
                    # No separate model copy and no gradient through the teacher.
                    with model.disable_adapter(), torch.no_grad():
                        teacher_logits = [value.detach().cpu() for value in
                            response_batch_forward(model, teacher, pad_token_id=pad_token_id)]
                    del teacher
                    model.train()
                    student_logits = response_batch_forward(model, student, pad_token_id=pad_token_id)
                    example_losses = [token_kl(s, t, top_k=top_k, chunk_size=chunk)
                                      for s, t in zip(student_logits, teacher_logits)]
                    del teacher_logits, student_logits
                else:
                    model.train()
                    logits = response_batch_forward(model, student, pad_token_id=pad_token_id)
                    example_losses = [response_cross_entropy(value,
                        row["inputs"]["input_ids"][:, row["prompt_length"]:], chunk_size=chunk)
                        for value, row in zip(logits, student)]
                    del logits
                # Equal example weighting matches the original accumulated objective,
                # including variable response lengths and a partial final microbatch.
                loss = torch.stack(example_losses).sum()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite Android loss; no completed checkpoint published")
                (loss / len(group)).backward()
                losses.extend(float(value.detach().cpu()) for value in example_losses)
                forward_microbatches += 1
                del loss, example_losses, student
            norm = torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                float(train.get("max_grad_norm", 1.0)))
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("Nonfinite Android gradient; checkpoint not saved")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        if optimizer_steps != total_updates:
            raise RuntimeError("Incomplete optimizer schedule")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        peak_memory = (torch.cuda.max_memory_allocated(policy.device)
                       if policy.device.type == "cuda" else None)
        model.gradient_checkpointing_disable()
        merged = model.merge_and_unload(safe_merge=True)
        policy.model = merged
        merged.eval()
        merged.requires_grad_(False)
        merged.config.text_config.use_cache = True
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=destination.name + ".partial-", dir=destination.parent))
        merged.save_pretrained(stage, safe_serialization=True, max_shard_size="2GB")
        policy.processor.save_pretrained(stage)
        receipt = {
            "status": "completed", "mode": mode, "checkpoint_in": str(checkpoint),
            "checkpoint_out": str(destination), "examples": len(examples),
            "epochs": epochs if train.get("max_updates") is None else None,
            "optimizer_steps": optimizer_steps, **exposure,
            "microbatch_size": batch_size, "gradient_accumulation_steps": accumulation,
            "effective_batch_size": batch_size * accumulation,
            "forward_microbatches": forward_microbatches,
            "response_token_presentations": response_tokens,
            "student_prompt_token_presentations": student_prompt_tokens,
            "teacher_prompt_token_presentations": teacher_prompt_tokens,
            "mean_example_loss": sum(losses) / len(losses), "trainable_parameters": trainable,
            "lora_targets": targets, "lora_rank": rank,
            "learning_rate": learning_rate,
            "teacher": "frozen_input_checkpoint_adapter_disabled" if feedback else None,
            "loss": ("reverse_kl_exact" if top_k == 0 else "reverse_kl_topk_union_plus_tail") if feedback else "response_only_cross_entropy",
            "kl_top_k": top_k if feedback else None,
            "prefix_source": "native_recorded_rollout" if feedback else "supervised_response",
            "sampling_contract": "fixed_collected_batch; later optimizer steps reuse recorded prefixes",
            "peak_cuda_allocated_bytes": peak_memory,
            "elapsed_seconds": time.monotonic() - start,
            "training_config": train,
        }
        (stage / "android_training_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        os.replace(stage, destination)
        stage = None
        return receipt
    finally:
        # Partial save directories remain visible for diagnosis; never masquerade
        # as a completed checkpoint or overwrite an earlier completed student.
        policy.close()


def train_sft(checkpoint: str, examples: list[dict], output_dir: str | Path,
              config: dict[str, Any]) -> dict:
    return _train(checkpoint, examples, output_dir, config, feedback=False)


def train_feedback(checkpoint: str, examples: list[dict], output_dir: str | Path,
                   config: dict[str, Any]) -> dict:
    return _train(checkpoint, examples, output_dir, config, feedback=True)
