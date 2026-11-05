#!/usr/bin/env python3
"""Read-only native AndroidWorld progress and per-step wall-clock profile.

Safe while an ORIGINAL slow collector is still running: reads JSONL but never
modifies models, emulators, partial files or sealed study metadata. Output is
descriptive; unfinished stages or early short tasks do not justify a fixed ETA.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def read_jsonl(path):
    if not path.is_file():
        return []
    records=[]
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    record=json.loads(line)
                except json.JSONDecodeError:
                    # The active writer may be in the middle of its final line.
                    # Never count or discard that partial line as a failure.
                    continue
                if record.get("status")=="complete" and type(record.get("success")) is bool:
                    records.append(record)
    return records


def inspect(root: Path):
    root=Path(root).expanduser().resolve()
    study=root/"study.json"
    seal=json.loads(study.read_text(encoding="utf-8")) if study.is_file() else None
    stages={"round0_native":root/"round0_native"}
    for arm in ("feedback_sd","feedback_spectrum_fixed","feedback_spectrum_learned"):
        for round_index in (1,2,3):
            prefix=root/arm/f"round_{round_index:02d}"
            for name in ("train_native","train_spectral","evaluation"):
                stages[f"{arm}/round_{round_index:02d}/{name}"]=prefix/name
    summary={}
    for name,folder in stages.items():
        stage_file=folder/"episodes.jsonl"
        if stage_file.is_file():
            rows=read_jsonl(stage_file)
        else:
            rows=[]
            for shard in sorted(folder.glob("worker_*/episodes.jsonl")):
                rows.extend(read_jsonl(shard))
        if not rows and not folder.exists():
            continue
        finished=sum(bool(x["success"]) for x in rows)
        timing=defaultdict(float)
        for x in rows:
            for key,value in x.get("timing_totals_seconds",{}).items():
                timing[key]+=float(value)
        elapsed=sum(float(x.get("elapsed_seconds",0)) for x in rows)
        step_count=sum(len(x.get("steps",[])) for x in rows)
        completed_receipt=folder/"complete.json"
        stage_wall=None
        if completed_receipt.exists():
            receipt=json.loads(completed_receipt.read_text(encoding="utf-8"))
            stage_wall=receipt.get("elapsed_seconds")
        summary[name]={
            "episodes_completed":len(rows),
            "native_verified_successes":finished,
            "native_verified_failures":len(rows)-finished,
            "actions_attempted":step_count,
            "mean_episode_seconds":elapsed/len(rows) if rows else None,
            "mean_steps_per_episode":step_count/len(rows) if rows else None,
            "sum_per_episode_wall_seconds":elapsed,
            "stage_elapsed_wall_seconds":stage_wall,
            "stage_complete_receipt":completed_receipt.is_file(),
            "per_step_wall_seconds":dict(timing) if timing else None,
        }
    return {"status":"read_only","run_root":str(root),
            "study_manifest_present":seal is not None,
            "benchmark_protocol":(seal.get("protocol") if seal else None),
            "stages":summary,
            "caveats":[
              "In-progress file may not contain the final JSONL line.",
              "Summed episode times are NOT wall-clock times under parallel collection.",
              "Early task lengths and sampling vary; do not turn this into a promised ETA.",
              "A missing stage is not a zero success rate; it is not yet observed."
            ]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",required=True)
    args=p.parse_args()
    print(json.dumps(inspect(Path(args.run_dir)),indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
