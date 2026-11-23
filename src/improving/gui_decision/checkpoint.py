"""Head-only checkpoints with explicit backbone/input identity."""
from __future__ import annotations

from pathlib import Path
from .cache import file_digest, read_json, stable_digest, write_json

CHECKPOINT_SCHEMA = "qwen_gui_decision_checkpoint_v1"


def save_checkpoint(output_dir, head, metadata: dict, optimizer=None,
                    training_state=None):
    from safetensors.torch import save_file
    import torch
    if "feature_spec" not in metadata:
        raise ValueError("Checkpoint must bind its feature specification")
    feature_digest = stable_digest(metadata["feature_spec"])
    stable_digest(metadata)  # Reject non-JSON/nonfinite metadata before mutations.
    config = head.config.to_dict()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu().contiguous()
             for name, value in head.state_dict().items()}
    temporary = root / "head.safetensors.tmp"
    save_file(state, str(temporary))
    temporary.replace(root / "head.safetensors")
    payload = {**metadata, "schema": CHECKPOINT_SCHEMA, "head_config": config,
               "head_sha256": file_digest(root / "head.safetensors")}
    payload["feature_spec_digest"] = feature_digest
    if optimizer is not None:
        torch.save({"optimizer": optimizer.state_dict(), "training_state": training_state},
                   root / "training.pt.tmp")
        (root / "training.pt.tmp").replace(root / "training.pt")
        payload["training_sha256"] = file_digest(root / "training.pt")
    write_json(root / "metadata.json", payload)


def load_head(output_dir, expected_feature_spec=None, device="cpu"):
    from safetensors.torch import load_file
    from .model import CandidateScoringHead, HeadConfig
    root = Path(output_dir)
    metadata = read_json(root / "metadata.json")
    if metadata.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("Unsupported GUI decision checkpoint schema")
    if stable_digest(metadata["feature_spec"]) != metadata["feature_spec_digest"]:
        raise ValueError("Checkpoint feature metadata is corrupt")
    if expected_feature_spec is not None and stable_digest(expected_feature_spec) != metadata["feature_spec_digest"]:
        raise ValueError("Checkpoint feature specification mismatch")
    if file_digest(root / "head.safetensors") != metadata["head_sha256"]:
        raise ValueError("Checkpoint head integrity failed")
    head = CandidateScoringHead(HeadConfig.from_dict(metadata["head_config"]))
    head.load_state_dict(load_file(str(root / "head.safetensors"), device="cpu"), strict=True)
    head.to(device=device).float()
    return head, metadata
