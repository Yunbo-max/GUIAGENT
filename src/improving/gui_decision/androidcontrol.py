"""Strict raw AndroidControl reader and native a11y candidate enumeration.

Adapted from GUI-JEPA (MIT); see third_party/gui_jepa for the pinned source
and copyright. No GUI-JEPA package, Torch, encoder or feature cache is used.
"""
from __future__ import annotations

import gzip
import json
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def _read_exact(stream: Any, count: int, context: str) -> bytes:
    data = stream.read(count)
    if len(data) != count:
        raise ValueError(f"Truncated TFRecord {context}: expected {count} bytes, got {len(data)}")
    return data


def _masked_crc(data: bytes) -> int:
    try:
        import google_crc32c
    except ImportError as exc:
        raise ImportError("CRC verification requires google-crc32c; install the GUI data requirements") from exc
    crc = google_crc32c.value(data)
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def iter_tfrecords(path: str | Path, *, verify_crc: bool = True) -> Iterator[bytes]:
    """Read gzip or plain TFRecords; truncated framing always raises."""
    path = Path(path)
    with path.open("rb") as raw:
        is_gzip = raw.read(2) == b"\x1f\x8b"
    opener = gzip.open if is_gzip else open
    with opener(path, "rb") as stream:
        record_index = 0
        while True:
            header = stream.read(8)
            if not header:
                return
            if len(header) != 8:
                raise ValueError(f"{path}: truncated header at record {record_index}")
            length_crc = _read_exact(stream, 4, "length checksum")
            if verify_crc and struct.unpack("<I", length_crc)[0] != _masked_crc(header):
                raise ValueError(f"{path}: bad length CRC at record {record_index}")
            length = struct.unpack("<Q", header)[0]
            payload = _read_exact(stream, length, "payload")
            payload_crc = _read_exact(stream, 4, "payload checksum")
            if verify_crc and struct.unpack("<I", payload_crc)[0] != _masked_crc(payload):
                raise ValueError(f"{path}: bad payload CRC at record {record_index}")
            yield payload
            record_index += 1


def parse_example(payload: bytes) -> dict[str, list[Any]]:
    from tfrecord import example_pb2

    example = example_pb2.Example()
    example.ParseFromString(payload)
    result: dict[str, list[Any]] = {}
    for key, feature in example.features.feature.items():
        kind = feature.WhichOneof("kind")
        if kind in ("bytes_list", "int64_list", "float_list"):
            result[key] = list(getattr(feature, kind).value)
        else:
            raise ValueError(f"Feature {key!r} has no supported value type")
    return result


def iter_episodes(path: str | Path, *, verify_crc: bool = True) -> Iterator[dict[str, list[Any]]]:
    for payload in iter_tfrecords(path, verify_crc=verify_crc):
        example = parse_example(payload)
        if len(example.get("episode_id", [])) != 1:
            raise ValueError(f"{path}: each record must have exactly one episode_id")
        yield example


def episode_steps(example: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Validate aligned episode arrays, then expose pre-action observations."""
    required = {"episode_id", "goal", "screenshots", "accessibility_trees", "screenshot_widths",
                "screenshot_heights", "actions", "step_instructions"}
    missing = required - set(example)
    if missing:
        raise ValueError(f"Episode missing required fields: {sorted(missing)}")
    if len(example["episode_id"]) != 1 or len(example["goal"]) != 1:
        raise ValueError("Episode needs one id and one goal")
    eid = int(example["episode_id"][0])
    goal = example["goal"][0].decode("utf-8")
    actions = [json.loads(value.decode("utf-8")) for value in example["actions"]]
    if not actions:
        raise ValueError(f"Episode {eid} has no actions")
    n_frames = len(example["screenshots"])
    if n_frames != len(actions) + 1:
        raise ValueError(f"Episode {eid}: expected actions+1 screenshots, got {n_frames}/{len(actions)}")
    for key in ("accessibility_trees", "screenshot_widths", "screenshot_heights"):
        if len(example[key]) != n_frames:
            raise ValueError(f"Episode {eid}: {key} does not align with screenshots")
    if len(example["step_instructions"]) != len(actions):
        raise ValueError(f"Episode {eid}: instructions do not align with actions")
    steps = []
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            raise ValueError(f"Episode {eid} step {index}: action must be an object")
        width, height = int(example["screenshot_widths"][index]), int(example["screenshot_heights"][index])
        if width <= 0 or height <= 0:
            raise ValueError(f"Episode {eid} step {index}: invalid dimensions")
        steps.append({
            "episode_id": eid, "step_index": index, "goal": goal,
            "step_instruction": example["step_instructions"][index].decode("utf-8"),
            "raw_action": action, "screenshot_png": example["screenshots"][index],
            "forest_bytes": example["accessibility_trees"][index],
            "screen_width": width, "screen_height": height,
        })
    return steps


def _label(element: dict[str, Any]) -> str:
    return (element["own_text"] or element["content_description"]
            or element["resource_id"].split("/")[-1] or element["class_name"].split(".")[-1])


def extract_elements(forest_bytes: bytes, screen_width: int, screen_height: int,
                     *, max_candidates: int = 80) -> list[dict[str, Any]]:
    """Current-a11y-only enumeration, matching GUI-JEPA filtering and order."""
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    from android_env.proto.a11y import android_accessibility_forest_pb2

    forest = android_accessibility_forest_pb2.AndroidAccessibilityForest()
    forest.ParseFromString(forest_bytes)
    elements, text_nodes = [], []
    for window in forest.windows:
        for node in window.tree.nodes:
            bounds = node.bounds_in_screen
            box = [bounds.left, bounds.top, bounds.right, bounds.bottom]
            own_text = (node.text or "").strip() or (node.content_description or "").strip()
            if own_text and box[2] > box[0] and box[3] > box[1]:
                text_nodes.append((box, own_text[:40]))
            flags = [bool(getattr(node, key, False)) for key in
                     ("is_clickable", "is_long_clickable", "is_editable", "is_scrollable")]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            if box[2] < 0 or box[3] < 0 or box[0] > screen_width or box[1] > screen_height:
                continue
            if not getattr(node, "is_visible_to_user", True) or not any(flags):
                continue
            elements.append({
                "bounds": box, "own_text": (node.text or "")[:128],
                "content_description": (node.content_description or "")[:128],
                "resource_id": (node.view_id_resource_name or "")[:128],
                "class_name": node.class_name or "", "flags": flags,
            })
    unique: dict[tuple[int, ...], dict[str, Any]] = {}
    for element in elements:
        key = tuple(element["bounds"])
        if key not in unique or (not _label(unique[key]) and _label(element)):
            unique[key] = element
    elements = sorted(unique.values(), key=lambda e: (e["bounds"][2] - e["bounds"][0])
                      * (e["bounds"][3] - e["bounds"][1]))[:max_candidates]
    for element in elements:
        left, top, right, bottom = element["bounds"]
        own = _label(element)[:40]
        inside = [text for box, text in text_nodes
                  if box[0] >= left - 2 and box[1] >= top - 2 and box[2] <= right + 2
                  and box[3] <= bottom + 2 and text.lower() != own.lower()][:4]
        cx, cy = (left + right) / 2, (top + bottom) / 2
        vertical = "top" if cy < screen_height / 3 else ("mid" if cy < 2 * screen_height / 3 else "bottom")
        horizontal = "left" if cx < screen_width / 3 else ("center" if cx < 2 * screen_width / 3 else "right")
        parts = ([own] if own else []) + inside + [element["class_name"].split(".")[-1],
                                                f"{vertical}-{horizontal}"]
        if element["flags"][2]:
            parts.append("editable")
        if not own and not inside:
            parts.insert(0, "icon")
        element["text"] = " | ".join(part for part in parts if part)
    return elements

