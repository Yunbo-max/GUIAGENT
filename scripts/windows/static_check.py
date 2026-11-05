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
 "scripts/run_androidworld_feedback.py",
 "scripts/windows/download_qwen3vl.py",
 "scripts/windows/preflight_guiagent.py",
 "tests/test_androidworld_feedback.py",
 "tests/test_android_windows_native.py",
)
FILES=(
 "configs/android/androidworld_feedback_learned_windows_96gb.json",
 "configs/android/androidworld_feedback_learned_windows_pilot_96gb.json",
 "configs/mobilegym/feedback_only_qwen4b_learned.json",
 "configs/mobilegym/feedback_only_qwen4b_pilot.json",
)

def check():
    for name in PYFILES:
        ast.parse((ROOT/name).read_text(encoding="utf-8"),filename=name)
    full,pilot,mobile,mobile_pilot=[
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
    if full["output_dir"]==pilot["output_dir"] or full["manifest"]==pilot["manifest"]:
        raise ValueError("Full/pilot roots must be disjoint")
    if mobile["output_dir"]==mobile_pilot["output_dir"]:
        raise ValueError("MobileGym full/pilot roots must be disjoint")
    return {"status":"static_checks_passed","python_files":len(PYFILES),
            "json_configs":len(FILES),"windows_emulator_tested":False,
            "gpu_trained":False}

if __name__=="__main__":
    print(json.dumps(check(),indent=2))
