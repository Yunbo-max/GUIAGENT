#!/usr/bin/env python3
"""Read-only-science AndroidWorld A/B: 1 AVD versus 2 AVDs on fixed train tasks.

NO SFT, no model updates, no calibration, no benchmark evaluation.
Only real training-side emulator episodes with native task.is_successful
are produced. Both modes use the same frozen task instances and action
budget but distinct model sampling streams; report lengths and elapsed
time, never interpret score differences as controlled model efficacy.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))


def summarize(episodes, seconds):
    counts=defaultdict(float)
    for ep in episodes:
        if ep.get("status")!="complete" or type(ep.get("success")) is not bool:
            raise ValueError("Only real completed native verifier outcomes admitted")
        counts["steps"]+=len(ep.get("steps",[]))
        counts["verified_successes"]+=int(ep["success"])
        for key,value in ep.get("timing_totals_seconds",{}).items():
            counts["t_"+key]+=float(value)
        counts["generated_action_tokens"]+=sum(
            len(step.get("response_token_ids",[])) for step in ep["steps"])
    n=len(episodes)
    if not n or seconds<=0:
        raise ValueError("No completed episodes or invalid wall-clock measurement")
    return {
        "episodes":n,"real_actions":int(counts["steps"]),
        "real_verified_successes":int(counts["verified_successes"]),
        "generated_action_tokens":int(counts["generated_action_tokens"]),
        "wall_seconds":seconds,"episodes_per_minute":60*n/seconds,
        "actions_per_minute":60*counts["steps"]/seconds,
        "per_step_component_seconds":{k[2:]:v for k,v in counts.items()
                                      if k.startswith("t_")},
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/android/androidworld_feedback_learned_windows_pilot_parallel2.json")
    parser.add_argument("--output-dir",default="runs/androidworld_windows/avd_ab_probe_v1")
    parser.add_argument("--mode",choices=("one","two","both"),default="both")
    parser.add_argument("--instances",type=int,default=2)
    parser.add_argument("--repeats",type=int,default=2)
    args=parser.parse_args()
    from improving.androidworld_feedback import _config,_inputs,_collection
    from improving.android.parallel_collection import parse_worker_devices
    from improving.android.model import QwenPolicy
    from improving.androidworld_feedback import _write_json
    from improving.android.assets import verify_upstream

    config=_config(args.config)
    if len(parse_worker_devices(config["android"]))!=2:
        raise ValueError("This probe needs the configured two independent AVDs")
    if not 1<=args.instances<=8 or not 1<=args.repeats<=4:
        raise ValueError("Keep the rapid pilot to 1..8 instances, 1..4 repeats")
    _,train,_=_inputs(config)
    instances=train[:args.instances]
    checkpoint=config["model"]["checkpoint"]
    output=Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError("Refuse to overwrite a previous timing probe")
    output.mkdir(parents=True)
    report={
        "schema":1,"purpose":"throughput_only_no_training",
        "fixed_train_instances":[i["instance_id"] for i in instances],
        "repeats":args.repeats,"modes":{},
        "gpu_config":config["model"],"native_verifier":"AndroidWorld",
        "policy_changes":"different independent RNG streams; identical task parameters",
    }
    _write_json(output/"plan.json",report)
    modes=("one","two") if args.mode=="both" else (args.mode,)
    for mode in modes:
        work=deepcopy(config)
        work["android"]["collect_timing"]=True
        if mode=="one":
            work["android"]["environment_workers"]=1
            work["android"].pop("worker_devices",None)
        run_dir=output/mode
        policy=None if mode=="two" else QwenPolicy(checkpoint,work)
        start=time.perf_counter()
        try:
            episodes=_collection(
                work,checkpoint,instances,run_dir,
                repeats=args.repeats,policy_name="native",
                policy=policy,operators=None,round_index=0)
        finally:
            if policy is not None:
                policy.close()
        data=summarize(episodes,time.perf_counter()-start)
        report["modes"][mode]=data
        _write_json(output/f"{mode}.json",data)
    if set(report["modes"])=={"one","two"}:
        one,two=report["modes"]["one"],report["modes"]["two"]
        report["two_vs_one_actions_per_minute_ratio"]=(
            two["actions_per_minute"]/one["actions_per_minute"]
            if one["actions_per_minute"] else None)
        report["two_vs_one_episodes_per_minute_ratio"]=(
            two["episodes_per_minute"]/one["episodes_per_minute"])
    report["training_started"]=False
    report["caveats"]=(
        "Short fixed-task comparisons are not a full 30-step evaluation; "
        "sampling streams and generated action counts can differ; "
        "measure full run before extrapolating total time.")
    _write_json(output/"report.json",report)
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
