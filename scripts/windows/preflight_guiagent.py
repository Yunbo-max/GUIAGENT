#!/usr/bin/env python
"""Native Windows GUIAGENT preflight: SDK/WHPX/ADB/CUDA/96GB class/Node/space.

Prints observed facts, not a claim of full native emulator or training success.
Use --strict to enforce available prerequisites for the selected benchmark.
Does not launch an emulator, download files, modify system settings or train.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))


def command(argv,timeout=30):
    try:
        cp=subprocess.run(argv,capture_output=True,text=True,
                          encoding="utf-8",errors="replace",timeout=timeout)
        return {"returncode":cp.returncode,"stdout":cp.stdout.strip()[-3000:],
                "stderr":cp.stderr.strip()[-1200:]}
    except (OSError,subprocess.TimeoutExpired) as exc:
        return {"returncode":None,"error":f"{type(exc).__name__}: {exc}"}


def check(kind="both",strict=False):
    if kind not in {"both","androidworld","mobilegym"}:
        raise ValueError("benchmark must be androidworld, mobilegym or both")
    windows=os.name=="nt"
    report={"host_system":platform.system(),"windows_native":windows,
            "windows_not_wsl_required":True,"python":sys.version.split()[0],
            "checks":{},"warnings":[],"gpu":{}}
    failures=[]
    if not windows:
        failures.append("Run this preflight on native Windows, not Linux/WSL")
    if sys.version_info < (3,11) or sys.version_info >= (3,12):
        failures.append("This pinned GUI experiment expects Python 3.11")
    smi=command(["nvidia-smi","--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader"],timeout=15)
    report["gpu"]["nvidia_smi"]=smi
    try:
        import torch
        gpu={"torch":torch.__version__,"cuda_runtime":torch.version.cuda,
             "cuda_available":bool(torch.cuda.is_available())}
        if gpu["cuda_available"]:
            gpu["device_name"]=torch.cuda.get_device_name(0)
            gib=torch.cuda.get_device_properties(0).total_memory/(1024**3)
            gpu["vram_gib"]=round(gib,2)
            gpu["bfloat16_supported"]=bool(torch.cuda.is_bf16_supported())
            gpu["meets_recommended_80GiB_class"]=gib>=80
            if not gpu["bfloat16_supported"]:
                failures.append("NVIDIA device lacks BF16 support")
            if gib<80:
                report["warnings"].append("VRAM below expected 96GB class; measure pilot peak")
        else:
            failures.append("PyTorch cannot access NVIDIA CUDA")
        report["gpu"].update(gpu)
    except ImportError:
        failures.append("PyTorch not installed in the active Python 3.11 venv")
    disk=shutil.disk_usage(ROOT)
    report["disk_free_gib"]=round(disk.free/(1024**3),1)
    if disk.free<200*1024**3:
        report["warnings"].append("Below 200GiB free; full checkpoints/screenshots can exhaust disk")

    for executable in ("git","ffmpeg"):
        result=command([executable,"-version" if executable=="ffmpeg" else "--version"],timeout=20)
        report["checks"][executable]=result
        if result["returncode"]!=0:
            failures.append(f"Missing {executable}")
    if kind in ("both","mobilegym"):
        n=command(["node","--version"])
        report["checks"]["node"]=n
        version=n.get("stdout","")
        try:
            node_major=int(version.lstrip("v").split(".")[0])
        except (ValueError,IndexError):
            node_major=0
        if node_major<22:
            failures.append("MobileGym requires Node.js 22 or newer")
        for relative in ("external/mobilegym/bench_env/task/registry.py",
                         "external/mobilegym/dist/index.html"):
            exists=(ROOT/relative).is_file()
            report["checks"][relative]=exists
            if not exists:
                failures.append(f"MobileGym source/build not found: {relative}")
        try:
            import playwright  # noqa: F401
            report["checks"]["playwright_import"]=True
        except ImportError:
            failures.append("Install Playwright in GUIAGENT venv")
    if kind in ("both","androidworld"):
        from improving.android.windows_sdk import sdk_defaults
        try:
            paths=sdk_defaults()
            for key in ("adb_path","emulator_path"):
                path=Path(paths[key])
                report["checks"][key]={"path":str(path),"exists":path.is_file()}
                if not path.is_file():
                    failures.append(f"Android SDK missing {key}: {path}")
            emulator=paths["emulator_path"]
            adb=paths["adb_path"]
            if Path(emulator).is_file():
                accel=command([emulator,"-accel-check"],timeout=30)
                report["checks"]["android_emulator_hypervisor"]=accel
                if accel.get("returncode")!=0:
                    failures.append("Android Emulator hardware acceleration unavailable (WHPX/nested virtualization)")
                avds=command([emulator,"-list-avds"],timeout=30)
                report["checks"]["android_avds"]=avds
                if "AndroidWorldAvd" not in avds.get("stdout","").splitlines():
                    failures.append("Create Pixel 6 API33 AndroidWorldAvd before native benchmark")
            if Path(adb).is_file():
                adb_devices=command([adb,"devices"],timeout=30)
                report["checks"]["adb_devices"]=adb_devices
                report["checks"]["androidworld_device_ready"]=(
                    "emulator-5554\tdevice" in adb_devices.get("stdout",""))
            upstream=ROOT/"data"/"android"/"UI-Voyager"/"androidworld"/"android_world"/"registry.py"
            report["checks"]["androidworld_pinned_source_exists"]=upstream.is_file()
            if not upstream.is_file():
                failures.append("Run AndroidWorld fetch to clone pinned UI-Voyager source")
        except (EnvironmentError,ValueError) as exc:
            failures.append(str(exc))
        try:
            import android_env  # noqa: F401
            report["checks"]["android_env_import"]=True
        except ImportError:
            failures.append("android_env dependency is absent from venv")
    report["failures"]=failures
    report["status"]="prerequisites_present" if not failures else "prerequisites_missing"
    report["caveat"]=("Only environment/static checks. Does not prove the "
        "AndroidWorld Python launcher, full GPU training or simulator reward work.")
    if strict and failures:
        raise RuntimeError("; ".join(failures))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark",choices=("both","androidworld","mobilegym"),default="both")
    p.add_argument("--strict",action="store_true")
    p.add_argument("--output")
    args=p.parse_args()
    try:
        result=check(args.benchmark,args.strict)
    except RuntimeError as exc:
        print(json.dumps({"status":"blocked","reason":str(exc)},indent=2))
        sys.exit(2)
    if args.output:
        path=Path(args.output).resolve()
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("x",encoding="utf-8") as f:
            json.dump(result,f,indent=2,ensure_ascii=False)
            f.write("\n")
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
