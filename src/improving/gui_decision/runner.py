"""Local feature extraction, head training and native single-step evaluation.

No model or dataset work runs when this module is imported. Native workloads are
explicit CLI operations, to be executed on the user's qualified execution host.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import secrets
import subprocess
import time

from .cache import (FeatureCache, FeatureCacheWriter, collate_features,
                    file_digest, read_json, stable_digest, write_json)


def runtime_identity():
    """Record source identity at actual Local execution, without model loading."""
    root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                capture_output=True, text=True, check=False)
        code_sha = commit.stdout.strip() if commit.returncode == 0 else None
    except OSError:
        code_sha = None  # Staged GPU directories may have no Git executable.
    return {"git_commit": code_sha,
            "implementation_sha256": {p.name: file_digest(p)
                                      for p in sorted(Path(__file__).parent.glob("*.py"))}}


def verify_feature_code(feature_spec):
    root = Path(__file__).parent
    for name, digest in feature_spec["model"]["implementation_sha256"].items():
        if file_digest(root / name) != digest:
            raise ValueError(f"Feature implementation changed since extraction: {name}")
    if file_digest(root / "processor.py") != feature_spec["processor"]["processor_source_sha256"]:
        raise ValueError("Processor source changed since feature extraction")


def observation_digest(observation: dict) -> str:
    value = copy.deepcopy(observation)
    image_path = value.pop("screenshot_path")
    value["screenshot_sha256"] = file_digest(image_path)
    return stable_digest(value)


def feature_identity(model, processor):
    return {"model": model.fingerprint_metadata(),
            "processor": processor.fingerprint_metadata()}


def extract(config: dict, split="train", mode="both", subset="full",
            output_dir=None, resume=False):
    import torch
    from .dataset import GuiDecisionDataset
    from .model import QwenDecisionModel
    from .processor import DecisionProcessor

    data_root = Path(config["data"]["prepared_dir"])
    dataset = GuiDecisionDataset(data_root, split=split, mode=mode, subset=subset)
    model = QwenDecisionModel.from_pretrained(config["model"])
    processor = DecisionProcessor.from_pretrained(config["model"])
    model.eval()
    spec = {"feature_spec": feature_identity(model, processor),
            "model_request": copy.deepcopy(config["model"]),
            "dataset_manifest_sha256": file_digest(data_root / "manifest.json"),
            "records_sha256": file_digest(data_root / f"{split}.jsonl"),
            "split": split, "mode": mode, "subset": subset,
            "nonclick_weight": config["training"]["nonclick_weight"],
            "history_source": "recorded_ground_truth",
            "evaluation_mode": "offline_step_prediction"}
    root = Path(output_dir or Path(config["data"]["cache_root"]) / split)
    writer = FeatureCacheWriter(root, spec, resume=resume)
    expected = []
    start = time.monotonic()
    max_tokens = 0
    shortened_candidates = shortened_characters = 0
    extracted = skipped = unlabelled = 0
    device = torch.device(config["model"]["device"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for index in range(len(dataset)):
        item = dataset[index]
        observation = item["observation"]
        key = observation["sample_id"] + "::" + item["mode"]
        expected.append(key)
        digest = observation_digest(observation)
        unlabelled += int(item["label"] == -100)
        if writer.contains(key, digest):
            skipped += 1
            continue
        batch = processor([observation])
        max_tokens = max(max_tokens, int(batch["model_inputs"]["input_ids"].shape[-1]))
        shortening = batch["metadata"]["candidate_text_truncation"]
        shortened_candidates += shortening["total_candidate_count"]
        shortened_characters += shortening["total_removed_characters"]
        with torch.no_grad():
            features = model.extract_features(batch)
        features = {name: value[0] for name, value in features.items()}
        label = item["label"]
        # Upstream weights recorded long_press as non-click even though the
        # candidate action space maps long_press supervision onto click.
        nonclick = item["supervision"]["type"] != "click"
        weight = config["training"]["nonclick_weight"] if nonclick else 1.0
        writer.write(key, features, label, weight, index, item["mode"], digest,
                     sample_id=observation["sample_id"])
        extracted += 1
        if extracted % 100 == 0:
            print(f"features {split}/{mode}: {index+1}/{len(dataset)}", flush=True)
    writer.finish(expected)
    receipt = {"status": "completed", "count": len(dataset),
               "newly_extracted": extracted, "resumed": skipped,
               "no_training_label": unlabelled, "max_new_tokens": max_tokens,
               "newly_extracted_shortened_candidates": shortened_candidates,
               "newly_extracted_removed_characters": shortened_characters,
               "elapsed_seconds": time.monotonic() - start,
               "torch": torch.__version__, "device": str(device),
               "cache_spec_digest": writer.spec_digest, "execution": runtime_identity()}
    if device.type == "cuda":
        receipt.update(device_name=torch.cuda.get_device_name(device),
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    write_json(root / "extraction_receipt.json", receipt)
    return receipt


def _dataset_for_cache(cache, data_root):
    from .dataset import GuiDecisionDataset
    root = Path(data_root)
    if file_digest(root / "manifest.json") != cache.spec["dataset_manifest_sha256"]:
        raise ValueError("Prepared data manifest changed since feature extraction")
    if file_digest(root / f"{cache.spec['split']}.jsonl") != cache.spec["records_sha256"]:
        raise ValueError("Prepared records changed since feature extraction")
    dataset = GuiDecisionDataset(root, split=cache.spec["split"],
                                 mode=cache.spec["mode"], subset=cache.spec["subset"])
    if len(dataset) != len(cache):
        raise ValueError("Feature cache does not cover the full requested dataset")
    indices = [row["record_index"] for row in cache.rows]
    if sorted(indices) != list(range(len(dataset))):
        raise ValueError("Feature cache record indices do not cover each view exactly once")
    return dataset


def make_resolver(config):
    from .arguments import ArgumentResolver
    return ArgumentResolver(**config["arguments"])


def _to_device(features, device):
    return {key: value.to(device=device) for key, value in features.items()}


def evaluate_head(head, cache, data_root, resolver, device="cpu", batch_size=32,
                  output_dir=None):
    import torch
    from torch.utils.data import DataLoader
    from .evaluation import MetricsAccumulator, judge

    dataset = _dataset_for_cache(cache, data_root)
    totals = MetricsAccumulator()
    by_mode = {"low": MetricsAccumulator(), "high": MetricsAccumulator()}
    loader = DataLoader(cache, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_features, num_workers=0)
    head.eval()
    start = time.monotonic()
    output = None
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        if (Path(output_dir) / "predictions.jsonl").exists():
            raise FileExistsError("Evaluation output exists; use a new output directory")
        output = (Path(output_dir) / "predictions.jsonl").open("w", encoding="utf-8")
    try:
        with torch.no_grad():
            for batch in loader:
                logits = head(_to_device(batch["features"], device))
                choices = logits.argmax(-1).cpu().tolist()
                for row, choice in zip(batch["items"], choices):
                    item = dataset[row["record_index"]]
                    obs = item["observation"]
                    if obs["sample_id"] != row["sample_id"] or item["mode"] != row["mode"]:
                        raise ValueError("Feature record does not match its observation")
                    prediction = None
                    error = None
                    try:
                        prediction = resolver.resolve(obs["candidates"][choice], obs)
                    except Exception as exc:
                        # A failed action parameter is an incorrect prediction;
                        # preserve the native sample in the denominator and log it.
                        error = f"{type(exc).__name__}: {exc}"
                    result = judge(prediction, item["supervision"], obs)
                    totals.update(result)
                    by_mode[item["mode"]].update(result)
                    if output is not None:
                        output.write(json.dumps({"sample_id": row["sample_id"],
                            "mode": item["mode"], "choice": choice,
                            "candidate_id": obs["candidates"][choice]["candidate_id"],
                            "prediction": prediction, "result": result, "error": error},
                            ensure_ascii=False, allow_nan=False) + "\n")
        metrics = {"overall": totals.compute(),
                   "by_mode": {mode: accumulator.compute() for mode, accumulator in by_mode.items()},
                   "count": len(cache), "split": cache.spec["split"],
                   "subset": cache.spec["subset"],
                   "history_source": "recorded_ground_truth",
                   "evaluation_mode": "offline_step_prediction",
                   "cache_spec_digest": cache.manifest["spec_digest"],
                   "arguments": resolver.describe(),
                   "elapsed_seconds": time.monotonic() - start}
        if output is not None:
            output.flush()
            write_json(Path(output_dir) / "metrics.json", metrics)
        return metrics
    finally:
        if output is not None:
            output.close()


def _private_seed(path):
    path = Path(path)
    if not path.exists():
        write_json(path, {"seed": secrets.randbelow(2**31)})
    return int(read_json(path)["seed"]), file_digest(path)


def train(config, output_dir=None, resume=False):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset
    from .checkpoint import load_head, save_checkpoint
    from .model import CandidateScoringHead, HeadConfig

    cfg = config["training"]
    root = Path(output_dir or cfg["output_dir"])
    train_cache = FeatureCache(Path(config["data"]["cache_root"]) / "train")
    validation_cache = FeatureCache(Path(config["data"]["cache_root"]) / "validation")
    if train_cache.spec["split"] != "train" or validation_cache.spec["split"] != "validation":
        raise ValueError("Training/checkpoint selection must use official train/validation")
    feature_spec = train_cache.spec["feature_spec"]
    verify_feature_code(feature_spec)
    if train_cache.spec.get("model_request") != config["model"] or validation_cache.spec.get("model_request") != config["model"]:
        raise ValueError("Requested model/processor config differs from the prepared feature caches")
    if stable_digest(validation_cache.spec["feature_spec"]) != stable_digest(feature_spec):
        raise ValueError("Train and validation feature definitions differ")
    if train_cache.spec["mode"] != validation_cache.spec["mode"]:
        raise ValueError("Train and validation modes differ")
    if train_cache.spec["nonclick_weight"] != cfg["nonclick_weight"]:
        raise ValueError("Cache training weights differ from the requested config")
    _dataset_for_cache(train_cache, config["data"]["prepared_dir"])
    _dataset_for_cache(validation_cache, config["data"]["prepared_dir"])
    eligible = [i for i, row in enumerate(train_cache.rows) if row["label"] >= 0]
    if not eligible:
        raise ValueError("No supervised candidate labels in training cache")
    seed, seed_digest = _private_seed(cfg["seed_file"])
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    device = torch.device(cfg["device"])
    head = CandidateScoringHead(HeadConfig.from_dict(config["model"]["head"])).to(device).float()
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg["learning_rate"],
                                 weight_decay=cfg["weight_decay"])
    resolver = make_resolver(config)
    run_identity = {"config": config, "train_cache": train_cache.manifest["spec_digest"],
                    "validation_cache": validation_cache.manifest["spec_digest"],
                    "seed_file_sha256": seed_digest, "arguments": resolver.describe(),
                    "execution": runtime_identity()}
    run_digest = stable_digest(run_identity)
    first_epoch, step, best = 0, 0, -1.0
    if resume:
        head, previous = load_head(root / "last", feature_spec, device)
        if previous["run_digest"] != run_digest:
            raise ValueError("Resume config/data/argument/seed identity differs")
        state_path = root / "last" / "training.pt"
        if file_digest(state_path) != previous["training_sha256"]:
            raise ValueError("Optimizer state integrity failed")
        optimizer = torch.optim.AdamW(head.parameters(), lr=cfg["learning_rate"],
                                     weight_decay=cfg["weight_decay"])
        saved = torch.load(state_path, map_location="cpu", weights_only=True)
        optimizer.load_state_dict(saved["optimizer"])
        state = saved["training_state"]
        first_epoch, step, best = state["completed_epochs"], state["step"], state["best_score"]
        torch.set_rng_state(state["torch_rng_state"])
        generator.set_state(state["loader_rng_state"])
        if device.type == "cuda" and state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    elif root.exists() and any(root.iterdir()):
        raise FileExistsError("Training output exists; use explicit --resume or a fresh directory")
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "run_identity.json", {"run_digest": run_digest, **run_identity})
    loader = DataLoader(Subset(train_cache, eligible), batch_size=cfg["batch_size"],
                        shuffle=True, generator=generator, collate_fn=collate_features,
                        num_workers=0)
    for epoch in range(first_epoch, cfg["epochs"]):
        head.train()
        total_loss = 0.0
        examples = 0
        for batch in loader:
            logits = head(_to_device(batch["features"], device))
            labels = batch["labels"].to(device)
            weights = batch["weights"].to(device)
            loss = (F.cross_entropy(logits, labels, reduction="none") * weights).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite head loss; failing batch retained in cache")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), cfg["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * labels.numel()
            examples += labels.numel()
            step += 1
        metrics = evaluate_head(head, validation_cache, config["data"]["prepared_dir"],
                                resolver, device, batch_size=cfg["batch_size"])
        active_modes = ["low", "high"] if validation_cache.spec["mode"] == "both" else [validation_cache.spec["mode"]]
        selection = sum(metrics["by_mode"][mode]["step_accuracy"] for mode in active_modes) / len(active_modes)
        improved = selection > best
        best = max(best, selection)
        metadata = {"feature_spec": feature_spec, "config": config,
                    "run_digest": run_digest, "epoch": epoch + 1, "step": step,
                    "selection_metric": "validation_mean_low_high_native_step_accuracy",
                    "selection_score": selection, "best_score": best,
                    "arguments": resolver.describe(),
                    "train_cache_spec_digest": train_cache.manifest["spec_digest"],
                    "validation_cache_spec_digest": validation_cache.manifest["spec_digest"],
                    "training_labelled_views": len(eligible),
                    "training_unlabelled_views": len(train_cache) - len(eligible),
                    "seed_file_sha256": seed_digest}
        state = {"completed_epochs": epoch + 1, "step": step, "best_score": best,
                 "torch_rng_state": torch.get_rng_state(),
                 "loader_rng_state": generator.get_state(),
                 "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
        save_checkpoint(root / "last", head, metadata, optimizer, state)
        if improved:
            save_checkpoint(root / "best", head, metadata)
        report = {"epoch": epoch + 1, "mean_weighted_ce": total_loss / examples,
                  "validation": metrics, "selection_score": selection,
                  "best_score": best, "improved": improved}
        write_json(root / f"epoch_{epoch+1:03d}.json", report)
        print(json.dumps({"epoch": epoch+1, "loss": report["mean_weighted_ce"],
                          "validation_selection": selection, "best": best}), flush=True)
    return {"output_dir": str(root), "best_score": best, "epochs_completed": cfg["epochs"]}


def evaluate(config, checkpoint, split="test", output_dir=None):
    from .checkpoint import load_head
    cache = FeatureCache(Path(config["data"]["cache_root"]) / split)
    verify_feature_code(cache.spec["feature_spec"])
    device = config["training"]["device"]
    head, metadata = load_head(checkpoint, cache.spec["feature_spec"], device)
    resolver = make_resolver(config)
    if resolver.describe() != metadata["arguments"]:
        raise ValueError("Action argument resolver differs from the checkpoint's validation setup")
    root = Path(output_dir or Path(config["training"]["output_dir"]) / ("evaluation_" + split))
    metrics = evaluate_head(head, cache, config["data"]["prepared_dir"], resolver,
                            device, config["training"]["batch_size"], root)
    write_json(root / "evaluation_identity.json", {"checkpoint_metadata_sha256": file_digest(Path(checkpoint) / "metadata.json"),
               "head_sha256": metadata["head_sha256"], "feature_spec": cache.spec["feature_spec"],
               "execution": runtime_identity()})
    return metrics


def load_policy(config, checkpoint):
    from .checkpoint import load_head
    from .model import QwenDecisionModel
    from .processor import DecisionProcessor
    from .router import PromptGatedPolicy
    model = QwenDecisionModel.from_pretrained(config["model"])
    processor = DecisionProcessor.from_pretrained(config["model"])
    head, metadata = load_head(checkpoint, feature_identity(model, processor), config["model"]["device"])
    model.head = head
    model.eval()
    resolver = make_resolver(config)
    if resolver.describe() != metadata["arguments"]:
        raise ValueError("Action argument resolver differs from checkpoint metadata")
    return PromptGatedPolicy(model, processor, resolver,
                             max_plan_tokens=config["dual_head"]["max_plan_tokens"])


def predict(config, checkpoint, observation: dict, gate_prompt="<GUI_MODE:PLAN_ACT>"):
    """Public inference entry: observation + explicit route, with no labels."""
    return load_policy(config, checkpoint).run(observation, gate_prompt)


def evaluate_online(config, checkpoint, split="test", mode="high", route="PLAN_ACT",
                    subset="full", output_dir=None):
    """Full offline native step evaluation including actual planner generation.

    This is not an Android emulator rollout. Every step starts from its released
    observation/history. PLAN_ACT always uses High inputs and its own plan.
    """
    import torch
    from .dataset import GuiDecisionDataset
    from .evaluation import MetricsAccumulator, failure_result, judge
    from .router import parse_gate_prompt
    gate_prompt = f"<GUI_MODE:{route}>"
    parse_gate_prompt(gate_prompt)
    if route == "PLAN":
        raise ValueError("PLAN alone has no executable action and cannot receive step accuracy")
    if route == "PLAN_ACT" and mode != "high":
        raise ValueError("Autonomous PLAN_ACT evaluation must use mode=high; no oracle step instruction")
    dataset = GuiDecisionDataset(config["data"]["prepared_dir"], split=split,
                                 mode=mode, subset=subset)
    root = Path(output_dir or Path(config["training"]["output_dir"]) / f"online_{split}_{mode}_{route.lower()}")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Online evaluation output exists; use a new output directory")
    root.mkdir(parents=True, exist_ok=True)
    policy = load_policy(config, checkpoint)
    totals = MetricsAccumulator()
    start = time.monotonic()
    status = "incomplete"
    with (root / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        try:
            for index in range(len(dataset)):
                item = dataset[index]
                prediction, error = None, None
                try:
                    prediction = policy.run(item["observation"], gate_prompt)
                    result = judge(prediction["action"], item["supervision"], item["observation"])
                except torch.cuda.OutOfMemoryError:
                    # A capacity failure aborts qualification; it must not be
                    # presented as a completed benchmark or zero remaining score.
                    raise
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    result = failure_result(item["supervision"], error)
                totals.update(result)
                handle.write(json.dumps({"sample_id": item["observation"]["sample_id"],
                    "mode": item["mode"], "prediction": prediction,
                    "result": result, "error": error}, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                if (index + 1) % 100 == 0:
                    print(f"online {route}: {index+1}/{len(dataset)}", flush=True)
            status = "completed"
        finally:
            metrics = {"status": status, "expected_steps": len(dataset),
                       "metrics": totals.compute(), "route": route, "mode": mode,
                       "split": split, "subset": subset,
                       "elapsed_seconds": time.monotonic() - start,
                       "gate_kind": "explicit_prompt", "planner_trained_in_this_stage": False,
                       "execution": runtime_identity(),
                       "arguments": policy.resolver.describe(),
                       "checkpoint_metadata_sha256": file_digest(Path(checkpoint) / "metadata.json"),
                       "data_manifest_sha256": file_digest(Path(config["data"]["prepared_dir"]) / "manifest.json")}
            write_json(root / "metrics.json", metrics)
    return metrics
