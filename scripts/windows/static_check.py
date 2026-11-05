#!/usr/bin/env python
"""No-dependency AST and JSON contract preflight for native Windows GUIAGENT."""
from __future__ import annotations
import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PYFILES=(
 "src/improving/android/windows_sdk.py",
 "src/improving/android/portable_lock.py",
 "src/improving/androidworld_feedback.py",
 "src/improving/android/study.py",
 "src/improving/android/environment.py",
 "src/improving/android/parallel_collection.py",
 "scripts/run_androidworld_feedback.py",
 "scripts/windows/download_qwen3vl.py",
 "scripts/windows/preflight_guiagent.py",
 "tests/test_androidworld_feedback.py",
 "tests/test_android_windows_native.py",
 "tests/test_android_parallel_collection.py",
)
FILES=(
 "configs/android/androidworld_feedback_learned_windows_96gb.json",
 "configs/android/androidworld_feedback_learned_windows_pilot_96gb.json",
 "configs/android/androidworld_feedback_learned_windows_pilot_parallel2.json",
 "configs/android/androidworld_feedback_learned_windows_full_parallel2.json",
 "configs/mobilegym/feedback_only_qwen4b_learned.json",
 "configs/mobilegym/feedback_only_qwen4b_pilot.json",
)

def check():
    for name in PYFILES:
        ast.parse((ROOT/name).read_text(encoding="utf-8"),filename=name)
    full,pilot,fast_pilot,fast_full,mobile,mobile_pilot=[
        json.loads((ROOT/name).read_text(encoding="utf-8")) for name in FILES]
    arms=["feedback_sd","feedback_spectrum_fixed","feedback_spectrum_learned"]
    for cfg in (full,pilot):
        if cfg["protocol"]!="androidworld-native-feedback-learned-v1":
            raise ValueError("Unexpected AndroidWorld protocol")
        if cfg["study"]["arms"]!=arms:
            raise ValueError("New AndroidWorld controls differ")
        if cfg["spectral"]["geometry_source"]!="feedback_only":
            raise ValueError("No fixed references in the new protocol")
        if cfg["model"]["checkpoint"]!=mobile["model"]["checkpoint"]:
            raise ValueError("Both benchmarks must share the pinned model path")
        for sdk in ("adb_path","emulator_path","android_sdk_root","android_avd_home"):
            if cfg["android"][sdk]!="auto":
                raise ValueError("Win32 SDK paths must be discovered locally")
    if full["study"]["test_task_limit"]!=116 or pilot["study"]["test_task_limit"]!=8:
        raise ValueError("Full/pilot populations changed")
    if fast_full["study"]["test_task_limit"]!=116 or fast_pilot["study"]["test_task_limit"]!=8:
        raise ValueError("Parallel full/pilot task populations changed")
    if fast_full["manifest"]!=full["manifest"] or fast_pilot["manifest"]!=pilot["manifest"]:
        raise ValueError("Parallel variant must reuse the frozen original task identities")
    if {fast_full["output_dir"],fast_pilot["output_dir"]}&{full["output_dir"],pilot["output_dir"]}:
        raise ValueError("Parallel output must not overwrite active original run")
    if fast_full["output_dir"]==fast_pilot["output_dir"]:
        raise ValueError("Parallel pilot/full need independent results")
    if fast_full["android"]["environment_workers"]!=2 or fast_pilot["android"]["environment_workers"]!=2:
        raise ValueError("Parallel variant must declare exactly 2 AVDs")
    for fast in (fast_full,fast_pilot):
        if fast["study"]["arms"]!=arms or fast["model"]!=full["model"]:
            raise ValueError("Parallel version must keep model/method identity")
        if fast["android"]["wait_after_action_seconds"]!=full["android"]["wait_after_action_seconds"]:
            raise ValueError("Parallel speedup must not silently change action wait semantics")
        if fast["android"]["max_steps"]!=30 or fast["android"]["history_len"]!=30:
            raise ValueError("Parallel speedup must not reduce benchmark episode length")
        if fast["spectral"]!=full["spectral"] or fast["gain_learning"]!=full["gain_learning"]:
            raise ValueError("Parallel variants must preserve gain learning")
    if full["output_dir"]==pilot["output_dir"] or full["manifest"]==pilot["manifest"]:
        raise ValueError("Full/pilot roots must be disjoint")
    if mobile["output_dir"]==mobile_pilot["output_dir"]:
        raise ValueError("MobileGym full/pilot roots must be disjoint")
    return {"status":"static_checks_passed","python_files":len(PYFILES),
            "json_configs":len(FILES),"windows_emulator_tested":False,
            "gpu_trained":False}

if __name__=="__main__":
    print(json.dumps(check(),indent=2))
