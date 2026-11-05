#!/usr/bin/env python
"""Download one pinned local Qwen3-VL-4B checkpoint shared by both GUI benchmarks.

No model API key, remote inference, quantization or WSL. The official revision
is pinned so AndroidWorld and MobileGym do not silently compare two backbones.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

MODEL="Qwen/Qwen3-VL-4B-Instruct"
REVISION="ebb281ec70b05090aa6165b016eac8ec08e71b17"


def verify(path: Path) -> dict:
    model=path/"config.json"
    if not model.is_file():
        raise FileNotFoundError(f"Checkpoint config.json not found at {path}")
    index=path/"model.safetensors.index.json"
    if index.is_file():
        shards=set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
        missing=sorted(name for name in shards if not (path/name).is_file())
        if missing:
            raise ValueError(f"Missing model weight shards: {missing}")
    else:
        shards={p.name for p in path.glob("*.safetensors")}
        if not shards:
            raise FileNotFoundError("No safetensors shards in model directory")
    return {"repo_id":MODEL,"revision_requested":REVISION,"directory":str(path),
            "shards":len(shards),
            "shard_bytes":sum((path/p).stat().st_size for p in shards),
            "processor_config_found":(path/"preprocessor_config.json").is_file(),
            "tokenizer_config_found":(path/"tokenizer_config.json").is_file()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",default="models/qwen3-vl-4b-instruct")
    p.add_argument("--verify-only",action="store_true")
    args=p.parse_args()
    target=Path(args.output).expanduser().resolve()
    if not args.verify_only:
        from huggingface_hub import snapshot_download
        target=Path(snapshot_download(
            repo_id=MODEL,revision=REVISION,local_dir=target,
            allow_patterns=["*.json","*.safetensors","*.txt","*.jinja","*.model"]))
    print(json.dumps(verify(target),indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
