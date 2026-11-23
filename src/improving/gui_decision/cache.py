"""Versioned, restartable caches of *new Qwen* features (never JEPA ViT caches)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

FEATURE_KEYS = (
    "query", "candidate_text", "candidate_roi", "candidate_aux",
    "candidate_type_ids", "candidate_mask",
)
CACHE_SCHEMA = "qwen_gui_decision_features_v1"


def stable_digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class FeatureCacheWriter:
    """A single writer; explicit resume preserves completed feature records.

    ``features`` are unbatched tensors, with candidate dimension N. An unfinished
    cache cannot be consumed by training. Each resume rechecks written identities.
    """

    def __init__(self, cache_dir, spec: dict, resume: bool = False):
        self.root = Path(cache_dir)
        self.spec = spec
        self.spec_digest = stable_digest(spec)
        self.index_path = self.root / "index.jsonl"
        self.rows: dict[str, dict] = {}
        manifest_path = self.root / "manifest.json"
        if manifest_path.exists():
            if not resume:
                raise FileExistsError(f"Cache exists; use explicit --resume: {self.root}")
            manifest = read_json(manifest_path)
            if (manifest.get("schema") != CACHE_SCHEMA or
                    manifest.get("spec_digest") != self.spec_digest or
                    stable_digest(manifest.get("spec")) != self.spec_digest):
                raise ValueError("Cache fingerprint mismatch; use a different output directory")
            if manifest.get("complete") and (not self.index_path.is_file() or
                    file_digest(self.index_path) != manifest.get("index_sha256")):
                raise ValueError("Complete cache index integrity failed during resume")
            if self.index_path.exists():
                for line in self.index_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row["key"] in self.rows:
                        raise ValueError("Duplicate cache key in index")
                    tensor_path = self.root / row["file"]
                    if file_digest(tensor_path) != row["sha256"]:
                        raise ValueError(f"Changed/incomplete cached feature: {row['key']}")
                    self.rows[row["key"]] = row
        elif self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(f"Nonempty cache directory has no manifest: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "features").mkdir(exist_ok=True)
        self._manifest(False)

    def _manifest(self, complete: bool):
        manifest = {"schema": CACHE_SCHEMA, "spec": self.spec,
                    "spec_digest": self.spec_digest, "complete": complete,
                    "count": len(self.rows), "index": "index.jsonl"}
        if complete:
            manifest["index_sha256"] = file_digest(self.index_path)
        write_json(self.root / "manifest.json", manifest)

    def contains(self, key: str, input_digest: str) -> bool:
        if key not in self.rows:
            return False
        if self.rows[key]["input_digest"] != input_digest:
            raise ValueError(f"Observation changed under existing cache key: {key}")
        return True

    def write(self, key, features, label, weight, record_index, mode,
              input_digest, sample_id=None):
        from safetensors.torch import save_file
        import torch

        if self.contains(key, input_digest):
            raise ValueError(f"Duplicate write: {key}")
        if set(features) != set(FEATURE_KEYS):
            raise ValueError(f"Wrong feature fields: {sorted(features)}")
        tensors = {}
        for name in FEATURE_KEYS:
            value = features[name].detach().cpu().contiguous()
            if value.is_floating_point():
                if not torch.isfinite(value).all():
                    raise ValueError(f"Nonfinite feature {name} for {key}")
                # Compact storage; the scoring head always computes in FP32.
                value = value.to(torch.float16) if name != "candidate_aux" else value.float()
                if not torch.isfinite(value).all():
                    raise ValueError(f"Feature {name} overflows its cache dtype for {key}")
            tensors[name] = value
        n = tensors["candidate_mask"].numel()
        if n == 0 or not tensors["candidate_mask"].any():
            raise ValueError("Feature record must have a valid candidate")
        if label != -100 and (not 0 <= label < n or not tensors["candidate_mask"][label]):
            raise ValueError("Label does not refer to a valid candidate")
        relative = Path("features") / (stable_digest(key) + ".safetensors")
        target = self.root / relative
        temporary = target.with_suffix(".tmp")
        save_file(tensors, str(temporary))
        temporary.replace(target)
        row = {"key": key, "sample_id": sample_id or key.rsplit(":", 1)[0],
               "mode": mode, "record_index": int(record_index), "label": int(label),
               "weight": float(weight), "input_digest": input_digest,
               "file": relative.as_posix(), "sha256": file_digest(target)}
        with self.index_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.rows[key] = row

    def finish(self, expected_keys: list[str]):
        if len(expected_keys) != len(set(expected_keys)):
            raise ValueError("Requested cache keys contain duplicates")
        if set(expected_keys) != set(self.rows):
            raise ValueError("Cache coverage differs from the complete requested dataset")
        if not self.index_path.exists():
            self.index_path.touch()
        self._manifest(True)


class FeatureCache:
    def __init__(self, cache_dir, expected_spec: dict | None = None):
        self.root = Path(cache_dir)
        self.manifest = read_json(self.root / "manifest.json")
        if self.manifest.get("schema") != CACHE_SCHEMA or not self.manifest.get("complete"):
            raise ValueError("Not a complete Qwen GUI feature cache")
        self.spec = self.manifest["spec"]
        if stable_digest(self.spec) != self.manifest["spec_digest"]:
            raise ValueError("Cache manifest fingerprint is corrupt")
        if expected_spec is not None and stable_digest(expected_spec) != self.manifest["spec_digest"]:
            raise ValueError("Cache fingerprint mismatch")
        index_path = self.root / self.manifest["index"]
        if file_digest(index_path) != self.manifest.get("index_sha256"):
            raise ValueError("Cache index has changed")
        self.rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()
                     if line.strip()]
        if len(self.rows) != self.manifest["count"] or len({r["key"] for r in self.rows}) != len(self.rows):
            raise ValueError("Cache count/duplicate mismatch")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from safetensors.torch import load_file
        row = self.rows[index]
        target = self.root / row["file"]
        if file_digest(target) != row["sha256"]:
            raise ValueError(f"Cached tensor integrity failed: {row['key']}")
        features = load_file(str(target), device="cpu")
        if set(features) != set(FEATURE_KEYS):
            raise ValueError("Cached feature schema mismatch")
        return {**row, "features": features}


def collate_features(items):
    import torch
    if not items:
        raise ValueError("Empty feature batch")
    sizes = [item["features"]["candidate_mask"].numel() for item in items]
    width = max(sizes)
    features = {}
    for name in FEATURE_KEYS:
        values = [item["features"][name] for item in items]
        if name == "query":
            features[name] = torch.stack(values)
            continue
        shape = (len(items), width, *values[0].shape[1:])
        padded = torch.zeros(shape, dtype=values[0].dtype)
        for i, (value, size) in enumerate(zip(values, sizes)):
            if value.shape[0] != size:
                raise ValueError("Candidate feature dimensions are inconsistent")
            padded[i, :size] = value
        features[name] = padded
    return {"features": features,
            "labels": torch.tensor([item["label"] for item in items], dtype=torch.long),
            "weights": torch.tensor([item["weight"] for item in items], dtype=torch.float32),
            "items": items}
