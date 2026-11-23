"""Prepare all AndroidControl splits as raw PNGs and auditable JSONL records.

This command does not run an encoder or download data. Default failures are
fatal. The explicit quarantine_train option may omit malformed *training*
episodes with a receipt; validation/test populations can never be shrunk.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .actions import build_action_candidates, history_entry, make_supervision
from .androidcontrol import episode_steps, extract_elements, iter_episodes
from .schema import GUI_JEPA_COMMIT, IGNORE_INDEX, OFFICIAL_SPLIT_COUNTS, SCHEMA_VERSION, validate_record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_id_sets(path: Path) -> dict[str, set[int]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected split-name to episode-list mapping")
    result = {}
    for name, values in raw.items():
        if not isinstance(name, str) or not isinstance(values, list) or any(type(x) is not int for x in values):
            raise ValueError(f"{path}: invalid membership list {name!r}")
        if len(values) != len(set(values)):
            raise ValueError(f"{path}: duplicate episode ids within {name}")
        result[name] = set(values)
    return result


def _validate_png(png: bytes, width: int, height: int, sample_id: str) -> None:
    from PIL import Image

    try:
        with Image.open(io.BytesIO(png)) as image:
            if image.format != "PNG" or image.size != (width, height):
                raise ValueError(f"{sample_id}: PNG size/format does not match TFRecord dimensions")
            image.verify()
    except (OSError, SyntaxError) as exc:
        raise ValueError(f"{sample_id}: unreadable screenshot: {exc}") from exc


def prepare_episode(example: dict[str, list[Any]], split: str, subsplits: list[str],
                    *, max_candidates: int = 80) -> tuple[list[dict[str, Any]], list[tuple[str, bytes]]]:
    """Pure episode transformation except PNG decoding; no model or cache."""
    records, screenshots, history = [], [], []
    for step in episode_steps(example):
        eid, si = step["episode_id"], step["step_index"]
        sample_id = f"{eid}:{si}"
        width, height = step["screen_width"], step["screen_height"]
        _validate_png(step["screenshot_png"], width, height, sample_id)
        elements = extract_elements(step["forest_bytes"], width, height, max_candidates=max_candidates)
        # No target/action is available to either enumeration function.
        candidates = build_action_candidates(elements, width, height)
        screenshot_path = f"screenshots/{eid}/s{si}.png"
        observation = {
            "sample_id": sample_id, "episode_id": eid, "step_index": si,
            "screenshot_path": screenshot_path, "screen_width": width, "screen_height": height,
            "goal": step["goal"], "step_instruction": step["step_instruction"],
            "history": list(history[-4:]), "candidates": candidates,
        }
        supervision = make_supervision(step["raw_action"], elements, candidates, width, height)
        record = {"schema_version": SCHEMA_VERSION, "split": split,
                  "subsplits": list(subsplits), "observation": observation, "supervision": supervision}
        validate_record(record)
        records.append(record)
        screenshots.append((screenshot_path, step["screenshot_png"]))
        # Append only after constructing the current observation.
        history.append(history_entry(supervision, elements))
    return records, screenshots


def prepare(raw_dir: str | Path, output_dir: str | Path, *, splits: str | Path | None = None,
            test_subsplits: str | Path | None = None, max_candidates: int = 80,
            on_error: str = "error", verify_crc: bool = True,
            hash_shards: bool = False) -> dict[str, Any]:
    if on_error not in ("error", "quarantine_train"):
        raise ValueError("on_error must be error or quarantine_train")
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    raw_dir, output_dir = Path(raw_dir).resolve(), Path(output_dir).resolve()
    split_path = Path(splits).resolve() if splits else raw_dir / "splits.json"
    subsplit_path = Path(test_subsplits).resolve() if test_subsplits else raw_dir / "test_subsplits.json"
    split_ids, subsplit_ids = _load_id_sets(split_path), _load_id_sets(subsplit_path)
    if set(split_ids) != set(OFFICIAL_SPLIT_COUNTS):
        raise ValueError("splits.json must contain exactly train, validation and test")
    names = list(OFFICIAL_SPLIT_COUNTS)
    for i, name in enumerate(names):
        for other in names[i + 1:]:
            if split_ids[name] & split_ids[other]:
                raise ValueError(f"Episode leakage between {name} and {other}")
    required_subsplits = {"IDD", "app_unseen", "task_unseen", "category_unseen"}
    if set(subsplit_ids) != required_subsplits:
        raise ValueError(f"test_subsplits.json must contain {sorted(required_subsplits)}")
    subsplit_union = set().union(*subsplit_ids.values())
    if subsplit_union != split_ids["test"]:
        raise ValueError("Test-subsplits union must equal the declared full test set")
    split_of = {eid: name for name, ids in split_ids.items() for eid in ids}
    shards = sorted(path for path in raw_dir.glob("android_control-*-of-*") if path.is_file())
    if not shards:
        raise FileNotFoundError(f"No AndroidControl TFRecord shards under {raw_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use a new empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "status": "preparing",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {"dataset": "google-research/android_control",
                   "url": "gs://gresearch/android_control", "raw_dir": str(raw_dir),
                   "gui_jepa_commit": GUI_JEPA_COMMIT,
                   "splits_sha256": _sha256(split_path), "test_subsplits_sha256": _sha256(subsplit_path)},
        "official_reference_episode_counts": dict(OFFICIAL_SPLIT_COUNTS),
        "declared_episode_counts": {name: len(ids) for name, ids in split_ids.items()},
        "declared_subsplit_episode_counts": {name: len(ids) for name, ids in subsplit_ids.items()},
        "official_population_shape": all(len(split_ids[name]) == count for name, count in OFFICIAL_SPLIT_COUNTS.items()),
        "configuration": {"max_candidates": max_candidates, "on_error": on_error,
                          "verify_crc": verify_crc, "hash_shards": hash_shards,
                          "image_export": "original_png_bytes", "history_steps": 4},
        "protocol": {"evaluation": "offline_step_prediction", "history_source": "recorded_ground_truth",
                     "default_test_subset": "full", "paper_idd": "IDD minus union of OOD memberships",
                     "candidate_generation": "current_a11y_only", "label_ignore_index": IGNORE_INDEX,
                     "long_press_target": "click", "input_text_target": "first_editable_candidate",
                     "test_failures_may_be_dropped": False},
        "shards": [{"name": p.name, "bytes": p.stat().st_size} for p in shards],
        "files": {name: f"{name}.jsonl" for name in names}, "app_vocab": "app_vocab.json",
        "actual_episode_counts": {name: 0 for name in names}, "step_counts": {name: 0 for name in names},
        "uncovered_label_counts": {name: 0 for name in names}, "subset_counts": {},
        "quarantined_train_episodes": [], "missing_episode_ids": [],
    }
    def write_manifest() -> None:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_manifest()
    seen: set[int] = set()
    app_counts: Counter[str] = Counter()
    subset_episodes: dict[str, set[int]] = {name: set() for name in (*required_subsplits, "paper_idd", "ood_union", "full")}
    subset_steps: Counter[str] = Counter()
    try:
        from google.protobuf.message import DecodeError

        with ExitStack() as stack:
            outputs = {name: stack.enter_context((output_dir / f"{name}.jsonl").open("w", encoding="utf-8")) for name in names}
            quarantine = stack.enter_context((output_dir / "quarantine.jsonl").open("w", encoding="utf-8"))
            for shard in shards:
                for example in iter_episodes(shard, verify_crc=verify_crc):
                    eid = int(example["episode_id"][0])
                    if eid in seen:
                        raise ValueError(f"Duplicate raw episode_id {eid}")
                    if eid not in split_of:
                        raise ValueError(f"Raw episode {eid} is absent from the declared splits")
                    seen.add(eid)
                    split = split_of[eid]
                    memberships = [name for name, ids in subsplit_ids.items() if eid in ids]
                    try:
                        records, screenshots = prepare_episode(example, split, memberships, max_candidates=max_candidates)
                    except (ValueError, TypeError, KeyError, DecodeError) as exc:
                        if on_error != "quarantine_train" or split != "train":
                            raise
                        receipt = {"episode_id": eid, "split": split, "shard": shard.name,
                                   "error_type": type(exc).__name__, "error": str(exc)}
                        quarantine.write(json.dumps(receipt, ensure_ascii=False) + "\n")
                        manifest["quarantined_train_episodes"].append(receipt)
                        continue
                    # Filesystem failures remain fatal, even in quarantine mode.
                    for relative_path, png in screenshots:
                        target = output_dir / relative_path
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(png)
                    for record in records:
                        outputs[split].write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                        sup = record["supervision"]
                        manifest["uncovered_label_counts"][split] += int(sup["label"] == IGNORE_INDEX)
                        if split == "train" and sup["type"] == "open_app":
                            app_counts[sup["app_name"]] += 1
                    manifest["actual_episode_counts"][split] += 1
                    manifest["step_counts"][split] += len(records)
                    if split == "test":
                        ood = any(name != "IDD" for name in memberships)
                        selected = memberships + ["full"]
                        if "IDD" in memberships and not ood:
                            selected.append("paper_idd")
                        if ood:
                            selected.append("ood_union")
                        for subset in selected:
                            subset_episodes[subset].add(eid)
                            subset_steps[subset] += len(records)
            missing = sorted(set(split_of) - seen)
            manifest["missing_episode_ids"] = missing
            if missing:
                raise ValueError(f"Raw shards are incomplete: {len(missing)} declared episodes missing")
        vocab = [name for name, _ in app_counts.most_common()]
        (output_dir / "app_vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output_dir / "app_vocab_counts.json").write_text(json.dumps(dict(app_counts.most_common()), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # Persist input memberships verbatim, including overlapping test subsets.
        (output_dir / "splits.json").write_bytes(split_path.read_bytes())
        (output_dir / "test_subsplits.json").write_bytes(subsplit_path.read_bytes())
        if hash_shards:
            for entry, shard in zip(manifest["shards"], shards):
                entry["sha256"] = _sha256(shard)
        for name in names:
            path = output_dir / f"{name}.jsonl"
            manifest.setdefault("prepared_file_hashes", {})[path.name] = _sha256(path)
        manifest["prepared_file_hashes"]["app_vocab.json"] = _sha256(output_dir / "app_vocab.json")
        manifest["subset_counts"] = {name: {"episodes": len(ids), "steps": subset_steps[name]}
                                     for name, ids in sorted(subset_episodes.items())}
        manifest["status"] = "complete"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        write_manifest()
        return manifest
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        manifest["missing_episode_ids"] = sorted(set(split_of) - seen)
        write_manifest()
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", help="Defaults to RAW_DIR/splits.json")
    parser.add_argument("--test-subsplits", help="Defaults to RAW_DIR/test_subsplits.json")
    parser.add_argument("--max-candidates", type=int, default=80)
    parser.add_argument("--on-error", choices=("error", "quarantine_train"), default="error")
    parser.add_argument("--skip-crc-checks", action="store_true", help="Explicitly disable CRC checks; framing remains strict")
    parser.add_argument("--hash-shards", action="store_true", help="Also SHA256 all raw shards (one extra read pass)")
    args = parser.parse_args(argv)
    manifest = prepare(args.raw_dir, args.output_dir, splits=args.splits, test_subsplits=args.test_subsplits,
                       max_candidates=args.max_candidates, on_error=args.on_error,
                       verify_crc=not args.skip_crc_checks, hash_shards=args.hash_shards)
    print(json.dumps({"output_dir": str(Path(args.output_dir).resolve()), "status": manifest["status"],
                      "episodes": manifest["actual_episode_counts"], "steps": manifest["step_counts"],
                      "quarantined_train": len(manifest["quarantined_train_episodes"])}, indent=2))


if __name__ == "__main__":
    main()

