"""Path expansion for native (not WSL) AndroidWorld on Windows.

The opt-in Windows benchmark config uses "auto" for SDK tools. Resolve SDK
paths from ANDROID_SDK_ROOT / ANDROID_HOME / LOCALAPPDATA. This does not
probe or install tools and does not change historical Linux configurations.
"""
from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath
import sys


SDK_KEYS = frozenset(("adb_path", "emulator_path", "android_sdk_root", "android_avd_home"))


def sdk_defaults(environ=None) -> dict[str, str]:
    env = dict(os.environ if environ is None else environ)
    if not (env.get("ANDROID_SDK_ROOT") or env.get("ANDROID_HOME") or env.get("LOCALAPPDATA")):
        raise EnvironmentError(
            "Android SDK not located. Install Android Studio and define "
            "ANDROID_SDK_ROOT or ANDROID_HOME (native Windows path).")
    sdk = env.get("ANDROID_SDK_ROOT") or env.get("ANDROID_HOME")
    if not sdk:
        sdk = str(PureWindowsPath(env["LOCALAPPDATA"]) / "Android" / "Sdk")
    sdk = PureWindowsPath(sdk)
    user = env.get("USERPROFILE")
    if not user:
        raise EnvironmentError("USERPROFILE is required to resolve the Windows AVD directory")
    avd = PureWindowsPath(env.get("ANDROID_AVD_HOME") or
                          str(PureWindowsPath(user) / ".android" / "avd"))
    return {
        "adb_path": str(sdk / "platform-tools" / "adb.exe"),
        "emulator_path": str(sdk / "emulator" / "emulator.exe"),
        "android_sdk_root": str(sdk),
        "android_avd_home": str(avd),
    }


def resolve_sdk_settings(settings: dict, *, platform: str | None = None,
                         environ=None) -> dict:
    """Return a copy with explicit Windows SDK paths; reject auto on Linux."""
    p = sys.platform if platform is None else platform
    values = dict(settings)
    names = [key for key in SDK_KEYS if values.get(key) == "auto"]
    if names and p != "win32":
        raise ValueError("android.*=auto SDK paths require native Windows, not Linux/WSL")
    if names:
        detected = sdk_defaults(environ=environ)
        for name in names:
            values[name] = detected[name]
    for name in SDK_KEYS:
        if values.get(name) == "auto":
            raise ValueError(f"Unresolved Android SDK setting: {name}")
    return values


def validate_sdk_files(settings: dict) -> dict:
    """Verify configured Windows SDK executables; never start an emulator."""
    checked = {}
    for key in ("adb_path", "emulator_path"):
        raw = settings.get(key)
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"Missing android.{key}")
        p = Path(os.path.expandvars(raw)).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"Missing {key}: {p}")
        checked[key] = str(p.resolve())
    return checked
