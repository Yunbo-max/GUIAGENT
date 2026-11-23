"""Opt-in, train-only MobileGym intermediate state verification.

The online policy NEVER receives the simulator JSON or judge outputs. This
module records goal-check progress after actual environment transitions for
offline teacher construction. A partial-progress delta is not causal credit.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace


async def verify_checkpoint(evaluator, task, initial_obs, current_obs, env,
                            progress_before: float | None = None) -> dict:
    """Return a fail-closed verdict for a real current simulator state.

    Missing apps, failed state retrieval and judge errors are NOT treated as
    zero reward. Call this only on the training split; official held-out
    evaluation retains the original terminal-only scoring protocol.
    """
    try:
        current_state = await env.get_state()
        if not isinstance(current_state, dict) or "os" not in current_state:
            raise ValueError("No complete simulator state")
        scored_obs = replace(current_obs, state=current_state)
        verdict = await evaluator.evaluate(
            task, initial_obs, scored_obs,
            SimpleNamespace(agent_answer=env.agent_answer),
        )
        if verdict.judge_error:
            raise ValueError("Native judge error: " + verdict.judge_error)
        progress = float(verdict.progress)
        if not 0 <= progress <= 1:
            raise ValueError("Progress is outside [0,1]")
        if progress_before is None:
            # Used only for the initial pre-action snapshot.
            progress_before = progress
        if not 0 <= progress_before <= 1:
            raise ValueError("Previous progress is invalid")
        return {
            "status": "verified",
            "progress_before": float(progress_before),
            "progress_after": progress,
            "delta_progress": progress - float(progress_before),
            "clean": bool(verdict.clean),
            "goal_success": bool(verdict.success),
            "passed": bool(verdict.passed),
            "judge_version": "mobilegym_state_grounded_interim",
        }
    except Exception as exc:
        # An unknown state is NOT an observed failure or a safe transition.
        return {
            "status": "invalid",
            "reason": f"{type(exc).__name__}: {str(exc)[:300]}",
            "progress_before": progress_before,
            "progress_after": None,
        }
