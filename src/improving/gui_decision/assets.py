"""Explicit acquisition of pinned model and AndroidControl inputs on Local."""
from __future__ import annotations

import base64
import hashlib
from pathlib import Path
import urllib.parse
import urllib.request

from .cache import file_digest, read_json, write_json


def _verify_gcs(path: Path, item: dict):
    if path.stat().st_size != int(item["size"]):
        raise ValueError(f"Wrong byte count: {path}")
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    if base64.b64encode(digest.digest()).decode("ascii") != item["md5Hash"]:
        raise ValueError(f"Official GCS MD5 mismatch: {path}; preserve file for diagnosis")


def download_gcs(item: dict, root: Path, verify_only: bool = False):
    target = root / Path(item["name"]).name
    if target.exists():
        _verify_gcs(target, item)
        return target
    if verify_only:
        raise FileNotFoundError(target)
    root.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".partial")
    offset = part.stat().st_size if part.exists() else 0
    if offset > int(item["size"]):
        raise ValueError(f"Partial download exceeds expected size: {part}")
    if offset < int(item["size"]):
        name = urllib.parse.quote(item["name"], safe="")
        url = (f"https://storage.googleapis.com/download/storage/v1/b/gresearch/o/{name}"
               f"?alt=media&generation={item['generation']}")
        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            if offset and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                raise ValueError("Server did not honor resume range; partial file retained")
            with part.open("ab" if offset else "wb") as handle:
                for chunk in iter(lambda: response.read(4 * 1024 * 1024), b""):
                    handle.write(chunk)
    _verify_gcs(part, item)
    part.replace(target)
    return target


def acquire(config: dict, pin_file, asset="all", verify_only=False):
    from huggingface_hub import snapshot_download
    pins = read_json(pin_file)
    receipts = []
    root = Path(config["assets"]["root"])
    root.mkdir(parents=True, exist_ok=True)
    for name in ("qwen", "arguments"):
        if asset not in ("all", name):
            continue
        pin = pins[name]
        kwargs = {"repo_id": pin["repo_id"], "revision": pin["revision"],
                  "local_files_only": verify_only}
        # Qwen stays in the standard revision-aware HF cache; AutoProcessor and
        # from_pretrained load the same pinned snapshot offline.
        if name == "arguments":
            kwargs["local_dir"] = config["arguments"]["span_checkpoint"]
            kwargs["allow_patterns"] = pin["allow_patterns"]
        snapshot = Path(snapshot_download(**kwargs))
        required = pin["required_files"]
        for filename in required:
            path = snapshot / filename
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing required {name} file: {path}")
            with path.open("rb") as handle:
                if handle.read(128).startswith(b"version https://git-lfs.github.com/spec"):
                    raise ValueError(f"Unresolved LFS pointer: {path}")
        # Resolve every shard referenced by the released index.
        if name == "qwen":
            index = read_json(snapshot / "model.safetensors.index.json")
            required = sorted(set(required) | set(index["weight_map"].values()))
        files = [{"file": filename, "bytes": (snapshot / filename).stat().st_size,
                  "sha256": file_digest(snapshot / filename)} for filename in required]
        receipts.append({"asset": name, "repo_id": pin["repo_id"],
                         "revision": pin["revision"], "snapshot": str(snapshot), "files": files})
    if asset in ("all", "data"):
        paths = []
        for item in pins["androidcontrol"]["objects"]:
            path = download_gcs(item, Path(config["data"]["raw_dir"]), verify_only)
            paths.append({**item, "local_path": str(path), "sha256": file_digest(path)})
            print(f"Verified {path.name} ({path.stat().st_size} bytes)", flush=True)
        receipts.append({"asset": "data", "objects": paths})
    write_json(root / f"receipt_{asset}.json", {"pin_file_sha256": file_digest(pin_file),
                                               "verify_only": verify_only, "assets": receipts})
    return receipts
