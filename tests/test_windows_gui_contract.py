"""Static second-benchmark protocol audit with no CUDA/emulator requirement."""
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_windows_guiagent_two_benchmark_contract():
    spec=importlib.util.spec_from_file_location(
        "windows_static", ROOT/"scripts/windows/static_check.py")
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    report=obj.check()
    assert report["status"]=="static_checks_passed"
    assert not report["gpu_trained"]
