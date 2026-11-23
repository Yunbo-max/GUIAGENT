"""Single-GPU Qwen3-VL-4B verified rolling horizon-conditioned soft-KL trainer.

Teacher = frozen input checkpoint with training-only verified hindsight.
Student = LoRA adapter conditioned on current screenshot / history only.
No GRPO, no future screenshot generation, no test state in training.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time


def _load_rows(path: str | Path) -> tuple[list[dict], str]:
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    rows = []
    with source.open("rb") as f:
        for raw in f:
            digest.update(raw)
            if raw.strip():
                rows.append(json.loads(raw))
    if not rows:
        raise ValueError("Empty rolling training dataset")
    for row in rows:
        if (row.get("split") != "train"
                or row.get("policy") != "native"
                or row.get("evidence_provenance") != "actual_train_episode_interim_verifier"):
            raise ValueError("Only native train-only verified rolling rows are allowed")
        if row.get("objective") not in {"next_action_kl", "future_action_type_kl"}:
            raise ValueError("Unknown rolling objective")
        if not isinstance(row.get("messages"), list) or not row["messages"]:
            raise ValueError("Missing recorded GUI screenshot messages")
        if not isinstance(row.get("feedback"), str) or not row["feedback"]:
            raise ValueError("Missing verified teacher hindsight")
        if row.get("verified_progress_gain", 0) <= 0:
            raise ValueError("Verified progress gain must be positive")
        weight = row.get("loss_weight")
        if type(weight) not in (int, float) or not 0 <= weight <= 1:
            raise ValueError("Invalid example loss weight")
        if (row["objective"] == "next_action_kl"
                and not isinstance(row.get("response_token_ids"), list)):
            raise ValueError("Current action requires original sampled token IDs")
        if (row["objective"] == "future_action_type_kl"
                and row.get("response_token_ids") is not None):
            raise ValueError("Future action-type labels must be tokenised locally")
    return rows, digest.hexdigest()


def _encode_pair(policy, row: dict) -> tuple[dict, dict, list[int]]:
    ids = row.get("response_token_ids")
    if ids is None:
        # These are verified future ACTION-TYPE labels, not unseen future
        # screen coordinates or an on-policy action response.
        kind = row["response"]
        if kind not in {
            "CLICK", "LONG_PRESS", "SWIPE", "TYPE", "ANSWER",
            "BACK", "HOME", "ENTER", "WAIT", "COMPLETE", "ABORT",
        }:
            raise ValueError(f"Invalid semantic future action type: {kind!r}")
        ids = policy.processor.tokenizer.encode(kind, add_special_tokens=False)
        eos = policy.processor.tokenizer.eos_token_id
        if eos is not None and (not ids or ids[-1] != eos):
            ids = ids + [eos]
    if not ids:
        raise ValueError("Empty response token IDs")
    student = policy.encode(row["messages"], ids)
    teacher = policy.encode(row["messages"], ids, feedback=row["feedback"])
    return student, teacher, ids


def train_rolling(config: dict, dataset_path: str | Path) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from improving.android.model import QwenPolicy
    from improving.android.training import (
        language_lora_targets, response_forward, response_cross_entropy,
        token_kl, plan_training_batches,
    )

    dataset, digest = _load_rows(dataset_path)
    train = config["train"]
    checkpoint = Path(config["model"]["checkpoint"]).expanduser().resolve()
    if not (checkpoint / "config.json").exists():
        raise FileNotFoundError(f"Missing Qwen native checkpoint: {checkpoint}")
    out = Path(config["output_dir"]).expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {out}")
    rank = int(train.get("lora_rank", 16))
    accum = int(train.get("gradient_accumulation_steps", 8))
    steps = int(train.get("max_updates", 32))
    lr = float(train.get("learning_rate", 1e-5))
    seed = int(config.get("runtime_seed", 11))
    chunk = int(train.get("logit_chunk_size", 16))
    ce_future = float(config.get("rolling", {}).get("future_label_ce_weight", 0.1))
    if (min(rank, accum, steps, chunk) < 1 or not math.isfinite(lr)
            or not 0 < lr < 1 or not 0 <= ce_future <= 1
            or type(train.get("batch_size", 1)) is not int
            or train.get("batch_size", 1) != 1):
        raise ValueError("Rolling trainer currently requires valid LoRA settings, microbatch=1")
    groups, exposure = plan_training_batches(
        len(dataset), accum, max_updates=steps, seed=seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    started = time.monotonic()
    policy = QwenPolicy(str(checkpoint), config)
    try:
        targets = language_lora_targets(policy.model)
        model = get_peft_model(policy.model, LoraConfig(
            task_type="CAUSAL_LM", r=rank,
            lora_alpha=int(train.get("lora_alpha", 32)),
            lora_dropout=0.0, target_modules=targets, bias="none",
        ))
        policy.model = model
        if train.get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=lr, weight_decay=float(train.get("weight_decay", 0.0)),
        )
        measured: list[float] = []
        objective_counts: Counter = Counter()
        optimizer.zero_grad(set_to_none=True)
        if policy.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(policy.device)

        for group in groups:
            for row_index in group:
                row = dataset[row_index]
                student, teacher, token_ids = _encode_pair(policy, row)
                model.eval()
                with model.disable_adapter(), torch.no_grad():
                    fixed_logits = response_forward(model, teacher).detach().cpu()
                model.train()
                logits = response_forward(model, student)
                loss = token_kl(logits, fixed_logits, top_k=0, chunk_size=chunk)
                if row["objective"] == "future_action_type_kl" and ce_future:
                    labels = torch.tensor(token_ids, device=logits.device)
                    loss = loss + ce_future * response_cross_entropy(
                        logits, labels, chunk_size=chunk)
                weighted = loss * float(row["loss_weight"])
                if not bool(torch.isfinite(weighted)):
                    raise FloatingPointError("Non-finite verified rolling loss")
                (weighted / len(group)).backward()
                measured.append(float(weighted.detach().cpu()))
                objective_counts[row["objective"]] += 1
                del logits, fixed_logits, loss, weighted, student, teacher

            norm = torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                float(train.get("max_grad_norm", 1.0)))
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("Non-finite rolling gradient norm")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        peak = (
            int(torch.cuda.max_memory_allocated(policy.device))
            if policy.device.type == "cuda" else None
        )
        trainable_parameters = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        model.gradient_checkpointing_disable()
        merged = model.merge_and_unload(safe_merge=True)
        policy.model = merged
        merged.eval().requires_grad_(False)
        merged.config.text_config.use_cache = True

        out.parent.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix=out.name + ".partial-", dir=out.parent))
        # Atomic directory publication. Incomplete files remain clearly marked.
        merged.save_pretrained(temp, safe_serialization=True, max_shard_size="2GB")
        policy.processor.save_pretrained(temp)
        receipt = {
            "status": "completed",
            "method": "verified_rolling_horizon_conditioned_soft_kl_v1",
            "world_model_used": False,
            "uses_grpo": False,
            "native_multihead_mtp": False,
            "training_split": "train",
            "checkpoint_in": str(checkpoint),
            "checkpoint_out": str(out),
            "dataset_sha256": digest,
            "records": len(dataset),
            "optimizer_updates": len(groups),
            "example_exposure_counts": dict(objective_counts),
            "mean_weighted_example_loss": sum(measured) / len(measured),
            "future_label_ce_weight": ce_future,
            "peak_cuda_allocated_bytes": peak,
            "elapsed_seconds": time.monotonic() - started,
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad),
            "lora_target_modules": targets,
            "sampling": exposure,
        }
        (temp / "rolling_training_receipt.json").write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.rename(temp, out)
        return receipt
    finally:
        policy.close()
