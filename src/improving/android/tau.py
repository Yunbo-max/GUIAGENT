"""Generation-only spectral strength policies; no model or GPU imports.

The fixed schedule and entropy gate follow SPECTRUM main at b8bcc68f.
The gate is not clipped: entropy above its round-one reference can increase tau.
Neither policy is a task-success guarantee or a learning-rate schedule.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Integral
from typing import Any


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def validate_tau_settings(spectral: Mapping[str, Any]) -> None:
    """Validate the historical constant/decay/adaptive alternatives."""
    if not isinstance(spectral, Mapping):
        raise ValueError("spectral configuration must be a mapping")
    tau = _finite_number(spectral.get("tau", 1.0), "spectral.tau")
    if tau < 0:
        raise ValueError("spectral.tau must be nonnegative")
    mode = spectral.get("tau_mode")
    if mode is not None and mode != "adaptive":
        raise ValueError("spectral.tau_mode must be 'adaptive' or null")
    decay = spectral.get("tau_decay")
    if decay is not None:
        decay = _finite_number(decay, "spectral.tau_decay")
        if not 0 < decay <= 1:
            raise ValueError("spectral.tau_decay must lie in (0, 1]")
    if mode == "adaptive":
        if decay is not None:
            raise ValueError("adaptive tau and tau_decay are mutually exclusive")
        gamma = _finite_number(spectral.get("tau_gamma", 5.0), "spectral.tau_gamma")
        if gamma <= 0:
            raise ValueError("spectral.tau_gamma must be positive")


def resolve_tau(
    spectral: Mapping[str, Any],
    round_index: int,
    *,
    entropy: float | None = None,
    reference_entropy: float | None = None,
) -> dict[str, Any]:
    """Resolve one round without rebasing a later adaptive reference.

    Undefined or zero round-one entropy cannot supply a ratio and fails
    explicitly. A later current entropy of zero produces zero tau. A degenerate
    entropy calculation returning None must not silently fall back to constant
    strength. Reference persistence and artifact binding belong to the study.
    """
    validate_tau_settings(spectral)
    if isinstance(round_index, bool) or not isinstance(round_index, Integral) or round_index < 1:
        raise ValueError("round_index must be a positive integer")
    tau = float(spectral.get("tau", 1.0))
    decay = spectral.get("tau_decay")
    mode = "adaptive" if spectral.get("tau_mode") == "adaptive" else (
        "decay" if decay is not None else "constant")
    gamma = None
    ratio = None
    current = None
    reference = None
    if mode == "adaptive":
        current = _finite_number(entropy, "current spectral entropy")
        if current < 0:
            raise ValueError("current spectral entropy must be nonnegative")
        gamma = float(spectral.get("tau_gamma", 5.0))
        if round_index == 1:
            reference = current
            if reference_entropy is not None:
                supplied = _finite_number(reference_entropy, "reference spectral entropy")
                if not math.isclose(supplied, current, rel_tol=1e-12, abs_tol=0.0):
                    raise ValueError("round-one reference entropy must equal current entropy")
        else:
            reference = _finite_number(reference_entropy, "reference spectral entropy")
        if reference <= 0:
            raise ValueError("adaptive tau needs a positive round-one reference entropy")
        ratio = current / reference
        try:
            effective_tau = tau * ratio ** gamma
        except OverflowError as error:
            raise ValueError("adaptive tau overflowed; no clipping or fallback was applied") from error
    elif mode == "decay":
        effective_tau = tau * float(decay) ** (int(round_index) - 1)
    else:
        effective_tau = tau
    if not math.isfinite(effective_tau) or effective_tau < 0:
        raise ValueError("resolved tau must be finite and nonnegative; no clipping was applied")
    return {
        "mode": mode,
        "round_index": int(round_index),
        "tau_base": tau,
        "effective_tau": effective_tau,
        "entropy": current,
        "reference_entropy": reference,
        "entropy_ratio": ratio,
        "tau_decay": None if decay is None else float(decay),
        "tau_gamma": gamma,
    }
