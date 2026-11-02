"""Full-spectrum GUI gains learned from execution-conditioned token KL.

Uses the EXACT main-branch learned_gains.SpectralGainLinear and eigenvalue
grouping, but replaces answer CE by same-checkpoint feedback-conditioned KL
at native sampled action prefixes. A held-out *training-task* guard and native
full-vocabulary KL probe prevent unconstrained fitting. No reward-gradient
claim or long-horizon credit theorem is made.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import math

import torch

from improving.android.training import response_forward, token_kl
from improving.learned_gains import (
    SpectralGainLinear, _installed, _enabled, _frozen_native,
    forward_kl_sum, spectral_groups, validate_gain_settings,
)
from improving.spectral import _covariance_eigh, _native_linear, folded_operators


def split_by_task(examples: list[dict], fit_limit=8, guard_limit=4):
    """Fit/guard from different training task instances; never test references."""
    if type(fit_limit) is not int or type(guard_limit) is not int or min(fit_limit, guard_limit) < 1:
        raise ValueError("fit/guard limits must be positive")
    groups = defaultdict(list)
    for item in examples:
        if item.get("split") != "train" or item.get("policy") != "native" or not item.get("feedback"):
            raise ValueError("Learned gains need native-prefix training feedback")
        groups[item["instance_id"]].append(item)
    keys = sorted(groups, key=lambda key: hashlib.sha256(key.encode()).hexdigest())
    if len(keys) < 2:
        raise ValueError("signal_shortage: need feedback from >=2 independent train task instances")
    n_guard = max(1, len(keys) // 5)
    if n_guard >= len(keys):
        n_guard = 1
    fit_keys, guard_keys = keys[n_guard:], keys[:n_guard]
    def sample(names, count):
        from improving.mobilegym_bridge import balanced_examples
        return balanced_examples([x for k in names for x in groups[k]], count)
    fit, guard = sample(fit_keys, fit_limit), sample(guard_keys, guard_limit)
    if not fit or not guard:
        raise ValueError("signal_shortage: empty fit/guard")
    if {x["instance_id"] for x in fit} & {x["instance_id"] for x in guard}:
        raise AssertionError("Fit/guard task leakage")
    return fit, guard


def _logits(policy, example):
    return response_forward(policy.model, policy.encode_example(example))


def _evaluate_loss(policy, examples, fixed_teacher, wrappers=None):
    value = 0.0
    with torch.no_grad():
        for example, teacher in zip(examples, fixed_teacher):
            logits = _logits(policy, example)
            if wrappers and any(wrapper.calls != 1 for wrapper in wrappers.values()):
                raise ValueError("Each selected spectral module must execute once")
            value += float(token_kl(logits, teacher, top_k=0, chunk_size=16))
            if not math.isfinite(value):
                raise FloatingPointError("Non-finite GUI feedback fit loss")
            if wrappers:
                for wrapper in wrappers.values():
                    wrapper.calls = 0
    return value / len(examples)


def _gradient(policy, examples, fixed_teacher, wrappers):
    parameters = [w.gains for w in wrappers.values()]
    accumulator = [torch.zeros_like(p) for p in parameters]
    for example, teacher in zip(examples, fixed_teacher):
        for wrapper in wrappers.values():
            wrapper.calls = 0
        with torch.enable_grad():
            logits = _logits(policy, example)
            if any(w.calls != 1 for w in wrappers.values()):
                raise ValueError("Spectral wrapper has unexpected call count")
            loss = token_kl(logits, teacher, top_k=0, chunk_size=16) / len(examples)
            gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        for dest, grad in zip(accumulator, gradients):
            if grad is not None:
                if not torch.isfinite(grad).all():
                    raise FloatingPointError("Nonfinite learned gain gradient")
                dest.add_(grad.detach())
    return accumulator


def _probe_kl(policy, examples, wrappers=None, operators=None, chunk=16):
    sum_kl, total, negative = 0.0, 0, 0
    with torch.no_grad():
        for example in examples:
            if wrappers is not None:
                with _enabled(wrappers, False):
                    native = _logits(policy, example).detach().cpu()
                with _enabled(wrappers, True):
                    altered = _logits(policy, example)
            else:
                native = _logits(policy, example).detach().cpu()
                with folded_operators(policy.model, operators):
                    altered = _logits(policy, example)
            mask = torch.ones(altered.shape[:-1], dtype=torch.bool, device=altered.device)
            val, count, _, neg = forward_kl_sum(native, altered, mask, chunk)
            sum_kl += val
            total += count
            negative += neg
    if total < 1:
        raise ValueError("No native action token probe")
    return {"mean": sum_kl / total, "tokens": total, "clamped_negative": negative}


def _set_gains(wrappers, values):
    with torch.no_grad():
        for wrapper, value in zip(wrappers.values(), values):
            wrapper.gains.copy_(value)


def fit_visual_feedback_gains(policy, feedback_rows, moment_records, settings=None):
    """Return deployment operators and complete acceptance/fallback report.

    Teacher logits are computed once from the unsteered frozen current model
    with actual same-instance verified hindsight; student logits use the same
    native sampled response IDs but omit hindsight. No extra teacher network.
    """
    cfg = validate_gain_settings(settings or {})
    fit, guard = split_by_task(
        feedback_rows, fit_limit=int((settings or {}).get("fit_examples", 8))
        if "fit_examples" in (settings or {}) else 8,
        guard_limit=int((settings or {}).get("guard_examples", 4))
        if "guard_examples" in (settings or {}) else 4)
    if not isinstance(moment_records, dict) or not moment_records:
        raise ValueError("Missing feedback-only gradient covariance")
    model = policy.model
    bases, eigenvalues, groups, original = {}, {}, {}, {}
    for name, record in moment_records.items():
        original[name] = _native_linear(model, name)
        _, values, bases[name] = _covariance_eigh(record["covariance"])
        eigenvalues[name] = values
        groups[name] = spectral_groups(values, cfg["degeneracy_rtol"], cfg["degeneracy_atol"])
        if bases[name].shape[0] != original[name].out_features:
            raise ValueError(f"Spectral covariance incompatible with {name}")
    wrappers = {name: SpectralGainLinear(original[name], bases[name], groups[name])
                for name in original}
    identities = {name: torch.eye(v.shape[0], dtype=torch.float64)
                  for name, v in eigenvalues.items()}
    # Probe is the first several native training action prefixes, without feedback.
    probe = fit[:max(1, min(4, len(fit)))]
    audit = {
        "objective": "feedback_conditioned_reverse_kl_on_native_action_tokens",
        "geometry": "feedback_only_gradient_second_moment",
        "kl_guard": "native_to_steered_full_vocabulary_on_native_action_prefixes",
        "guard_policy": "independent_train_tasks_final_candidate_only",
        "selection": "identity", "fit_tasks": sorted({r["instance_id"] for r in fit}),
        "guard_tasks": sorted({r["instance_id"] for r in guard}),
        "probe_examples": len(probe), "settings": cfg, "history": [],
        "accepted_steps": 0, "fallback_reason": None,
        "verified_reward_gradient": False, "native_gui_success_claim": False,
    }
    with _frozen_native(model):
        teacher_fit, teacher_guard = [], []
        for rows, sink in ((fit, teacher_fit), (guard, teacher_guard)):
            with torch.no_grad():
                for row in rows:
                    sink.append(response_forward(
                        model, policy.encode_example(row, feedback=row["feedback"])
                    ).detach().cpu())
        native_fit = _evaluate_loss(policy, fit, teacher_fit)
        native_guard = _evaluate_loss(policy, guard, teacher_guard)
        audit["native"] = {"fit_kl": native_fit, "guard_kl": native_guard}
        accepted, current_fit = 0, native_fit
        with _installed(model, wrappers):
            for iteration in range(cfg["max_steps"]):
                gradients = _gradient(policy, fit, teacher_fit, wrappers)
                largest = max(float(grad.abs().max()) for grad in gradients)
                if not math.isfinite(largest) or largest == 0:
                    audit["stop_reason"] = "zero_or_nonfinite_gradient"
                    break
                previous = [w.gains.detach().clone() for w in wrappers.values()]
                selected = False
                for backtrack in range(cfg["max_backtracks"] + 1):
                    eta = cfg["learning_rate"] * cfg["backtrack_factor"] ** backtrack
                    proposed = [(old - eta * grad / largest).clamp(cfg["min_gain"], 1.0)
                                for old, grad in zip(previous, gradients)]
                    item = {"iteration": iteration + 1, "backtrack": backtrack, "eta": eta}
                    if all(torch.equal(a, b) for a, b in zip(previous, proposed)):
                        item["reason"] = "no_projected_step"
                        audit["history"].append(item)
                        break
                    _set_gains(wrappers, proposed)
                    try:
                        for wrapper in wrappers.values():
                            wrapper.calls = 0
                        fit_loss = _evaluate_loss(policy, fit, teacher_fit, wrappers)
                        item["fit_kl"] = fit_loss
                        if fit_loss >= current_fit - cfg["min_improvement"]:
                            item["reason"] = "no_fit_improvement"
                        else:
                            probe_kl = _probe_kl(policy, probe, wrappers=wrappers,
                                                 chunk=cfg["kl_chunk_tokens"])
                            item["probe_kl"] = probe_kl
                            if probe_kl["mean"] > cfg["kl_budget"] + 1e-6:
                                item["reason"] = "native_probe_budget"
                            else:
                                selected = True
                                current_fit = fit_loss
                                accepted += 1
                                item["reason"] = "accepted"
                    except FloatingPointError as e:
                        item["reason"] = f"nonfinite:{e}"
                    audit["history"].append(item)
                    if selected:
                        break
                if not selected:
                    _set_gains(wrappers, previous)
                    audit["stop_reason"] = "backtracking_exhausted"
                    break
            audit["accepted_steps"] = accepted
            for w in wrappers.values():
                w.calls = 0
            wrapper_guard = _evaluate_loss(policy, guard, teacher_guard, wrappers)
            operators = {}
            for name, wrapper in wrappers.items():
                value = wrapper.gains.detach().cpu().double()[groups[name]]
                basis = bases[name]
                operators[name] = identities[name] + (basis * (value - 1).unsqueeze(0)) @ basis.T
                audit.setdefault("modules", {})[name] = {
                    "gain_min": float(value.min()), "gain_max": float(value.max()),
                    "gain_mean": float(value.mean()), "group_count": len(wrapper.gains),
                    "output_dimension": len(value)
                }
        audit["wrapper_guard_kl"] = wrapper_guard
        if accepted == 0:
            audit["fallback_reason"] = "no_accepted_update"
        elif wrapper_guard > native_guard + cfg["guard_tolerance"] + 1e-6:
            audit["fallback_reason"] = "independent_guard_regressed"
        else:
            # The BF16 folded operator is what will actually generate trajectories.
            with folded_operators(model, operators):
                folded_fit = _evaluate_loss(policy, fit, teacher_fit)
                folded_guard = _evaluate_loss(policy, guard, teacher_guard)
            folded_kl = _probe_kl(policy, probe, operators=operators,
                                  chunk=cfg["kl_chunk_tokens"])
            audit["folded"] = {"fit_kl": folded_fit, "guard_kl": folded_guard,
                               "probe_kl": folded_kl}
            if folded_fit >= native_fit - cfg["min_improvement"]:
                audit["fallback_reason"] = "folded_fit_not_improved"
            elif folded_guard > native_guard + cfg["guard_tolerance"] + 1e-6:
                audit["fallback_reason"] = "folded_guard_regressed"
            elif folded_kl["mean"] > cfg["kl_budget"] + 1e-6:
                audit["fallback_reason"] = "folded_native_probe_budget"
        if audit["fallback_reason"]:
            operators = identities
        else:
            audit["selection"] = "learned"
    return operators, audit
