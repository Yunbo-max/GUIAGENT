"""No-emulator contracts for native Windows paths and exclusive study locks."""
from pathlib import Path
import multiprocessing
import pytest

from improving.android.windows_sdk import resolve_sdk_settings, sdk_defaults
from improving.android.portable_lock import exclusive_run_lock


def test_auto_windows_path_resolution_does_not_require_linux():
    env = {"LOCALAPPDATA": r"C:\Users\student\AppData\Local",
           "USERPROFILE": r"C:\Users\student"}
    data = sdk_defaults(env)
    assert data["adb_path"].lower().endswith(r"android\sdk\platform-tools\adb.exe")
    assert data["emulator_path"].lower().endswith(r"android\sdk\emulator\emulator.exe")
    assert data["android_avd_home"].lower().endswith(r"student\.android\avd")
    settings = resolve_sdk_settings(
        {"adb_path": "auto", "emulator_path": "auto",
         "android_sdk_root": "auto", "android_avd_home": "auto"},
        platform="win32", environ=env)
    assert settings == data


def test_auto_paths_reject_linux_instead_of_treating_auto_as_binary():
    with pytest.raises(ValueError, match="native Windows"):
        resolve_sdk_settings({"adb_path": "auto"}, platform="linux")


def _hold_lock(path, ready, release):
    from improving.android.portable_lock import exclusive_run_lock
    with exclusive_run_lock(path):
        ready.set()
        release.wait(10)


def test_second_process_cannot_acquire_same_lock(tmp_path):
    path = tmp_path / ".run.lock"
    ready = multiprocessing.Event()
    release = multiprocessing.Event()
    process = multiprocessing.Process(target=_hold_lock, args=(str(path), ready, release))
    process.start()
    try:
        assert ready.wait(5)
        with pytest.raises(RuntimeError, match="Another process"):
            with exclusive_run_lock(path):
                pass
    finally:
        release.set()
        process.join(10)
    assert process.exitcode == 0
    with exclusive_run_lock(path):
        assert path.exists()
