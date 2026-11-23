"""Offset-indexed JSONL data access with an explicit observation allowlist."""
from __future__ import annotations

import copy
import hashlib
import json
from array import array
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from .schema import SCHEMA_VERSION, validate_model_observation, validate_record

TEST_SUBSETS = ("full", "paper_idd", "official_idd", "ood_union", "app_unseen", "task_unseen", "category_unseen")


def make_model_observation(record: Mapping[str, Any], mode: str,
                           data_root: str | Path) -> dict[str, Any]:
    """Return no current target, label, bbox target, raw action or Low-only key."""
    if mode not in ("low", "high"):
        raise ValueError("A model observation must be low or high")
    raw = record["observation"]
    screenshot = Path(raw["screenshot_path"])
    if not screenshot.is_absolute():
        screenshot = Path(data_root) / screenshot
    # Deliberate allowlist: do not copy then blacklist a handful of GT names.
    result = {
        "sample_id": raw["sample_id"], "screenshot_path": str(screenshot.resolve()),
        "screen_width": raw["screen_width"], "screen_height": raw["screen_height"],
        "goal": raw["goal"],
        "instruction": (raw["step_instruction"] or raw["goal"]) if mode == "low" else raw["goal"],
        "history": list(raw["history"]), "candidates": copy.deepcopy(raw["candidates"]),
    }
    validate_model_observation(result)
    return result


def in_subset(record: Mapping[str, Any], subset: str = "full") -> bool:
    if subset not in TEST_SUBSETS:
        raise ValueError(f"Unknown subset {subset!r}")
    if subset == "full":
        return True
    memberships = set(record["subsplits"])
    is_ood = bool(memberships & {"app_unseen", "task_unseen", "category_unseen"})
    if subset == "official_idd":
        return "IDD" in memberships
    if subset == "paper_idd":
        return "IDD" in memberships and not is_ood
    if subset == "ood_union":
        return is_ood
    return subset in memberships


class GuiDecisionDataset(Sequence[dict[str, Any]]):
    """All records are retained, including label=-100; a trainer may skip them.

    mode='both' is a virtual pair of views: even index Low, odd index High.
    record_at(index) uses exactly the same mapping as __getitem__.
    This class is a map-style dataset without a Torch import or model loading.
    """

    def __init__(self, data_root: str | Path, split: str = "train", mode: str = "low",
                 subset: str = "full") -> None:
        if split not in ("train", "validation", "test"):
            raise ValueError("split must be train, validation or test")
        if mode not in ("low", "high", "both"):
            raise ValueError("mode must be low, high or both")
        if subset not in TEST_SUBSETS or (split != "test" and subset != "full"):
            raise ValueError("Named test subsets are only valid on the test split")
        self.data_root = Path(data_root).resolve()
        self.split, self.mode, self.subset = split, mode, subset
        self.manifest = json.loads((self.data_root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("status") != "complete" or self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Dataset preparation is incomplete or its schema is incompatible")
        self.records_path = self.data_root / self.manifest["files"][split]
        self._offsets = array("Q")
        digest = hashlib.sha256()
        observed_steps = 0
        sample_ids: set[str] = set()
        with self.records_path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                digest.update(line)
                if not line.strip():
                    raise ValueError(f"Blank record in {self.records_path} at byte {offset}")
                record = json.loads(line)
                validate_record(record)
                if record["split"] != split:
                    raise ValueError("Record split disagrees with its file")
                sample_id = record["observation"]["sample_id"]
                if sample_id in sample_ids:
                    raise ValueError(f"Duplicate prepared sample {sample_id}")
                sample_ids.add(sample_id)
                observed_steps += 1
                if in_subset(record, subset):
                    self._offsets.append(offset)
        expected = self.manifest["step_counts"][split]
        if observed_steps != expected:
            raise ValueError(f"Prepared split count mismatch: file={observed_steps}, manifest={expected}")
        self.records_sha256 = digest.hexdigest()
        declared_hash = self.manifest.get("prepared_file_hashes", {}).get(self.records_path.name)
        self.file_hash_verified = declared_hash is not None
        if declared_hash is not None and declared_hash != self.records_sha256:
            raise ValueError("Prepared JSONL checksum disagrees with its manifest")
        if not self._offsets:
            raise ValueError(f"No records in {split}/{subset}")

    @property
    def base_record_count(self) -> int:
        return len(self._offsets)

    def __len__(self) -> int:
        return len(self._offsets) * (2 if self.mode == "both" else 1)

    def _base_index(self, index: int) -> int:
        if not isinstance(index, int):
            raise TypeError("Dataset indices must be integers")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return index // 2 if self.mode == "both" else index

    def record_at(self, index: int) -> dict[str, Any]:
        base_index = self._base_index(index)
        with self.records_path.open("rb") as stream:
            stream.seek(self._offsets[base_index])
            return json.loads(stream.readline())

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.record_at(index)
        if index < 0:
            index += len(self)
        mode = ("low" if index % 2 == 0 else "high") if self.mode == "both" else self.mode
        return {"observation": make_model_observation(record, mode, self.data_root),
                "label": record["supervision"]["label"], "supervision": record["supervision"], "mode": mode}


def iter_records(data_root: str | Path, split: str = "test", subset: str = "full") -> Iterator[dict[str, Any]]:
    dataset = GuiDecisionDataset(data_root, split=split, mode="low", subset=subset)
    for index in range(len(dataset)):
        yield dataset.record_at(index)

