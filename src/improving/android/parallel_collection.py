"""Native Windows AndroidWorld parallel AVD collection.

Each worker owns a distinct real AVD (and its task state / process RNG) and
one local inference-only Qwen policy. The main round driver still computes
feedback, geometry, learned gains and student weights in its original order.
A fixed task/attempt budget and the official native verifier are preserved.

This intentionally uses *processes*, not threads: upstream AndroidWorld calls
numpy/python global RNG setters, and concurrent threads would leak task seeds.
Experimental rollout policy changes (independent GPU sampling RNG streams)
are isolated to a separately sealed output root.
"""
from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Any


def parse_worker_devices(android: dict) -> list[dict]:
    """Require a separate emulator, ports and explicitly named AVD per worker."""
    count = android.get("environment_workers", 1)
    if type(count) is not int or count not in (1, 2, 3, 4):
        raise ValueError("android.environment_workers must be 1, 2, 3 or 4")
    if count == 1:
        return [{
            "avd_name": android["avd_name"],
            "console_port": android.get("console_port", 5554),
            "grpc_port": android.get("grpc_port", 8554),
        }]
    devices = android.get("worker_devices")
    if not isinstance(devices, list) or len(devices) != count:
        raise ValueError("Parallel AndroidWorld needs exactly one worker_devices entry per independent AVD")
    seen_avd, seen_console, seen_grpc = set(), set(), set()
    for device in devices:
        if not isinstance(device, dict) or set(device) != {"avd_name","console_port","grpc_port"}:
            raise ValueError("Each worker device must have avd_name, console_port and grpc_port")
        avd, port, grpc = (device[k] for k in ("avd_name","console_port","grpc_port"))
        if not isinstance(avd,str) or not avd.strip():
            raise ValueError("AVD name must be nonempty")
        if any(type(v) is not int or v < 1024 or v > 65000 for v in (port, grpc)):
            raise ValueError("AVD ports must be numeric and in 1024..65000")
        if port%2 or (port+1) in seen_console:
            raise ValueError("Android console ports must be even and independent; ADB uses console+1")
        if avd in seen_avd or port in seen_console or grpc in seen_grpc:
            raise ValueError("Every worker must own a unique AVD, console port and gRPC port")
        seen_avd.add(avd);seen_console.add(port);seen_grpc.add(grpc)
    # Also reject any cross-type collision of gRPC with console or adb ports.
    occupied=set(seen_console) | {p+1 for p in seen_console} | {android.get("adb_server_port",5037)}
    if seen_grpc & occupied:
        raise ValueError("gRPC ports collide with ADB server or emulator console/device ports")
    return [dict(item) for item in devices]


def _worker(payload: dict[str,Any]) -> dict:
    """Top-level spawn target: independently load the real emulator and model."""
    from improving.android.environment import collect_episodes
    from improving.android.model import QwenPolicy
    from improving.spectral import folded_operators
    from contextlib import nullcontext
    import torch
    import random
    import numpy as np

    config=dict(payload["config"])
    android=dict(config["android"])
    android.update(payload["device"])
    android.pop("environment_workers",None)
    android.pop("worker_devices",None)
    android.pop("inference_cuda_fraction_gib",None)
    android.pop("inference_cpu_threads",None)
    config["android"]=android
    config["collection"]={"repeats":payload["repeats"]}
    cpu_threads=payload["cpu_threads"]
    torch.set_num_threads(cpu_threads)
    seed=payload["seed"]
    random.seed(seed)
    np.random.seed(seed%2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        limit_gib=payload.get("cuda_limit_gib")
        if limit_gib is not None:
            total=torch.cuda.get_device_properties(0).total_memory
            fraction=float(limit_gib)*1024**3/total
            if not 0<fraction<1:
                raise ValueError("Configured per-AVD GPU inference limit exceeds device")
            torch.cuda.set_per_process_memory_fraction(fraction,0)
    policy=QwenPolicy(payload["checkpoint"],config)
    try:
        operators=payload.get("operators")
        with (folded_operators(policy.model,operators) if operators is not None else nullcontext()):
            episodes=collect_episodes(
                payload["instances"],policy.generate,payload["output"],
                config,policy=payload["policy_name"],checkpoint=payload["checkpoint"],
                round_index=payload["round_index"])
        return {"index":payload["index"],"count":len(episodes),
                "path":str(payload["output"]),
                "gpu_peak_bytes":(torch.cuda.max_memory_reserved(0)
                                  if torch.cuda.is_available() else None)}
    finally:
        policy.close()


def collect_parallel(cfg: dict, *, checkpoint: str, instances: list[dict],
                     path: str|Path, repeats: int, policy_name: str,
                     round_index: int, operators: dict|None, seed: int) -> list[dict]:
    from improving.android.data import validate_episodes
    from improving.androidworld_feedback import _verify_episode_population
    from improving.android.assets import file_sha256
    from improving.androidworld_feedback import _write_json

    devices=parse_worker_devices(cfg["android"])
    if len(devices)==1:
        raise ValueError("Parallel dispatcher needs at least 2 devices")
    if not instances or policy_name not in ("native","spectral"):
        raise ValueError("Invalid parallel collection population")
    path=Path(path)
    if path.exists():
        raise FileExistsError(f"Collection output already exists: {path}")
    path.mkdir(parents=True)
    # Assignment by frozen instance index keeps all trials of the same
    # instance on the same independent emulator, retaining task identities.
    groups=[instances[i::len(devices)] for i in range(len(devices))]
    config = dict(cfg)
    from improving.androidworld_feedback import _hash
    frozen = {
        "protocol":"androidworld-multi-avd-collection-v1",
        "checkpoint":checkpoint,"policy":policy_name,"round":round_index,
        "repeats":repeats,"source_config_sha256":_hash(cfg),
        "devices":devices,"instance_ids":[i["instance_id"] for i in instances],
        "algorithm_changes":"none; independent AVD/sampling streams, same prompts/rewards"
    }
    _write_json(path/"dispatch.json",frozen)
    cores=int(cfg["android"].get("inference_cpu_threads",2))
    if type(cores) is not int or not 1<=cores<=8:
        raise ValueError("inference_cpu_threads must be an integer in 1..8")
    cuda_limit=cfg["android"].get("inference_cuda_fraction_gib",20)
    if cuda_limit is not None and (type(cuda_limit) not in (int,float) or
                                    not 1<=cuda_limit<=80):
        raise ValueError("inference_cuda_fraction_gib must be 1..80 or null")
    payloads=[]
    for i,(device,group) in enumerate(zip(devices,groups)):
        payloads.append({"index":i,"device":device,"config":config,
            "checkpoint":checkpoint,"instances":group,
            "output":str(path/f"worker_{i:02d}"),"policy_name":policy_name,
            "round_index":round_index,"repeats":repeats,
            "operators":operators,"cpu_threads":cores,
            "cuda_limit_gib":cuda_limit,
            "seed":int(_hash([seed,round_index,policy_name,i])[:15],16)})
    # Windows spawn prevents inherited CUDA context and global task RNG.
    reports=[]
    with ProcessPoolExecutor(max_workers=len(devices),
                             mp_context=multiprocessing.get_context("spawn")) as executor:
        futures=[executor.submit(_worker,p) for p in payloads]
        for future in futures:
            reports.append(future.result())
    if {v["index"] for v in reports}!=set(range(len(devices))):
        raise RuntimeError("Missing AVD worker output")
    observations=[]
    for report in sorted(reports,key=lambda x:x["index"]):
        sub=Path(report["path"])
        logfile=sub/"episodes.jsonl"
        if not logfile.is_file():
            raise RuntimeError(f"Worker missing complete native result file: {logfile}")
        items=[json.loads(line) for line in logfile.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(items)!=report["count"] or len(items)!=len(groups[report["index"]])*repeats:
            raise ValueError("AVD worker finished incomplete task/attempt population")
        observations.append(items)
    # Preserve the original task-major order rather than worker completion order.
    grouped={}
    for worker_index,items in enumerate(observations):
        for row in items:
            key=(row["instance_id"],row["repeat_id"])
            if key in grouped:
                raise ValueError("Duplicate native episode across AVD workers")
            grouped[key]=row
    ordered=[]
    for instance in instances:
        for repeat in range(repeats):
            key=(instance["instance_id"],repeat)
            if key not in grouped:
                raise ValueError(f"Missing frozen task/attempt: {key}")
            ordered.append(grouped[key])
    _verify_episode_population(ordered,instances,repeats,instances[0]["split"])
    # All steps preserve exact screenshot provenance in per-worker subfolders.
    with (path/"episodes.jsonl").open("x",encoding="utf-8") as f:
        for row in ordered:
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False,sort_keys=True)+"\n")
    _write_json(path/"complete.json",{
        "status":"complete","episodes":len(ordered),
        "episodes_sha256":file_sha256(path/"episodes.jsonl"),
        "workers":reports,"dispatch_sha256":file_sha256(path/"dispatch.json"),
        "score":"verified_native_AndroidWorld_not_model_self_report",
        "worker_source_scripts_unchanged_during_stage":True})
    return ordered
