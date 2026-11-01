"""Visual, feedback-calibrated K/V geometry for Android trajectory generation.

The matrices are uncentered activation-gradient second moments, not a Fisher
matrix, a Hessian, or a reward-ascent direction. The resulting intervention is
used only while generating trajectories, through ``folded_operators``. Neither
this module nor the operator guarantees improved exploration or task success.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from improving.spectral import (
    _covariance_eigh,
    _native_linear,
    decoder_attention_layout,
    folded_operators,
    make_operator,
    target_modules,
)
from .tau import resolve_tau, validate_tau_settings


def normalized_spectral_entropy(anchor_records: Mapping[str, Mapping[str, Any]]) -> float | None:
    """Historical mean entropy, normalized by strictly positive spectral rank.

    This intentionally follows the main-branch gate: rank-one projections are
    skipped, and entropy is normalized by log(positive rank), not log(width).
    Thus uniform rank-two and uniform full-rank spectra both have entropy one;
    this is not an effective-rank or correct-trajectory diversity measure.
    No usable rank-two-or-higher projection returns None and blocks the gate.
    Zero or invalid moments are rejected earlier by the covariance validator.
    """
    entropies = []
    for record in anchor_records.values():
        _, eigenvalues, _ = _covariance_eigh(record["covariance"])
        positive = eigenvalues[eigenvalues > 0]
        if len(positive) < 2:
            continue
        probabilities = positive / positive.sum()
        probabilities = probabilities[probabilities > 0]
        entropy = -float((probabilities * probabilities.log()).sum())
        entropies.append(entropy / math.log(len(positive)))
    return sum(entropies) / len(entropies) if entropies else None


def mix_moments(
    anchor: Tensor,
    feedback: Tensor | None,
    *,
    alpha: float = 0.5,
    trace_floor: float = 1e-20,
) -> tuple[Tensor, dict[str, Any]]:
    """Trace-normalize two PSD moments, falling back to the fixed anchor.

    A negligible feedback trace is not normalized into an arbitrary noise
    direction. A zero/negligible anchor fails: there is no admitted reference
    geometry. No cross-checkpoint moments are accumulated by this module.
    """
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("spectral alpha must be finite and in [0, 1]")
    if not math.isfinite(trace_floor) or trace_floor < 0:
        raise ValueError("moment_trace_floor must be finite and nonnegative")

    def checked(value: Tensor, name: str) -> tuple[Tensor, float]:
        if not isinstance(value, Tensor) or value.ndim != 2 or value.shape[0] != value.shape[1] or not value.numel():
            raise ValueError(f"{name} moment must be a nonempty square tensor")
        if value.is_complex():
            raise ValueError(f"{name} moment must be real")
        matrix = value.detach().to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(matrix).all():
            raise ValueError(f"{name} moment must be finite")
        scale = max(float(matrix.abs().max()), trace_floor)
        if not torch.allclose(matrix, matrix.T, rtol=1e-7, atol=scale * 1e-9):
            raise ValueError(f"{name} moment must be symmetric")
        matrix = (matrix + matrix.T) * 0.5
        if float(torch.linalg.eigvalsh(matrix)[0]) < -scale * 1e-7:
            raise ValueError(f"{name} moment must be positive semidefinite")
        trace = float(matrix.trace())
        return matrix, trace

    a, a_trace = checked(anchor, "anchor")
    if a_trace <= trace_floor:
        raise ValueError("fixed anchor has no usable gradient signal")
    a = a / a_trace
    f_trace = None
    reason = "no_feedback_examples"
    effective_alpha = 0.0
    result = a
    if feedback is not None:
        f, f_trace = checked(feedback, "feedback")
        if f.shape != a.shape:
            raise ValueError("anchor and feedback moment dimensions differ")
        if f_trace <= trace_floor:
            reason = "feedback_trace_below_floor"
        elif alpha == 0:
            reason = "feedback_disabled"
        else:
            effective_alpha = float(alpha)
            result = (1.0 - alpha) * a + alpha * (f / f_trace)
            reason = None
    return result, {
        "anchor_trace": a_trace,
        "feedback_trace": f_trace,
        "requested_alpha": float(alpha),
        "effective_alpha": effective_alpha,
        "fallback_reason": reason,
        "trace_floor": float(trace_floor),
    }


def select_targets(model: nn.Module, config: Mapping[str, Any]) -> list[str]:
    """Resolve full-attention language K/V projections, excluding the vision tower."""
    _, _, full_layers = decoder_attention_layout(model)
    layers = config.get("layers")
    if layers is None:
        last_n = config.get("last_n_layers", 2)
        if isinstance(last_n, bool) or not isinstance(last_n, Integral) or not 1 <= last_n <= len(full_layers):
            raise ValueError("last_n_layers must select at least one available full-attention layer")
        layers = full_layers[-int(last_n):]
    return target_modules(model, layers=layers, projections=("k_proj", "v_proj"))


@contextmanager
def _captured_native_outputs(model: nn.Module, names: list[str]):
    """Expose real native visual-conditioned activations with exact state cleanup.

    Freezing model parameters avoids parameter-gradient memory. If an output has
    no gradient graph, make that output a leaf requiring gradients. Later target
    outputs keep their existing graph, preserving dependence between layers.
    Teacher passes explicitly disable the capture flag and remain no-grad.
    """
    modules = {name: _native_linear(model, name) for name in names}
    parameters = [(p, p.requires_grad) for p in model.parameters()]
    training = [(m, m.training) for m in model.modules()]
    capture: dict[str, Any] = {"enabled": False, "outputs": {}, "calls": {}}
    handles = []

    def hook(name: str):
        def save_output(_module: nn.Module, _inputs: tuple[Any, ...], output: Tensor):
            if not capture["enabled"]:
                return None
            if not isinstance(output, Tensor) or output.ndim != 3:
                raise ValueError(f"{name} must produce [batch, sequence, output] activations")
            if not output.requires_grad:
                output.requires_grad_(True)
            capture["outputs"][name] = output
            capture["calls"][name] = capture["calls"].get(name, 0) + 1
            return output
        return save_output

    try:
        model.eval()
        # Native Qwen checkpointing is training-only; eval mode avoids
        # recomputation without changing the caller's checkpoint configuration.
        for parameter, _ in parameters:
            parameter.requires_grad_(False)
        for name, module in modules.items():
            handles.append(module.register_forward_hook(hook(name)))
        yield capture
    finally:
        capture["enabled"] = False
        capture["outputs"].clear()
        for handle in handles:
            handle.remove()
        for parameter, original in parameters:
            parameter.requires_grad_(original)
        for module, original in training:
            module.training = original


def _check_encoding(encoded: Mapping[str, Any]) -> tuple[dict[str, Tensor], int, int]:
    inputs = encoded.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("policy.encode_example must return an inputs dictionary")
    for field in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"):
        if field not in inputs or not isinstance(inputs[field], Tensor):
            raise ValueError(f"visual calibration requires tensor {field}")
    ids, mask = inputs["input_ids"], inputs["attention_mask"]
    if ids.ndim != 2 or ids.shape[0] != 1 or mask.shape != ids.shape:
        raise ValueError("calibration processes one image-conditioned example at a time")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("attention_mask must be binary")
    prompt, response = encoded.get("prompt_length"), encoded.get("response_length")
    if (isinstance(prompt, bool) or not isinstance(prompt, Integral) or prompt < 1
            or isinstance(response, bool) or not isinstance(response, Integral) or response < 1
            or prompt + response != ids.shape[1]):
        raise ValueError("invalid prompt/response token lengths")
    if not mask[:, prompt - 1:prompt + response].all():
        raise ValueError("all selected continuation targets and predecessors must be nonpadding")
    return inputs, int(prompt), int(response)


def _validate_example(example: Mapping[str, Any], kind: str) -> None:
    split = example.get("split")
    if kind == "anchor":
        if split != "calibration":
            raise ValueError("fixed anchors must come from the calibration split")
    else:
        if split != "train":
            raise ValueError("feedback geometry must come from the training split")
        if example.get("policy") != "native":
            raise ValueError("feedback geometry requires native student prefixes")
        if not isinstance(example.get("feedback"), str) or not example["feedback"].strip():
            raise ValueError("feedback geometry needs informative admitted feedback")
        if not example.get("response_token_ids"):
            raise ValueError("feedback geometry requires original sampled response_token_ids")
    if not isinstance(example.get("messages"), list) or not example["messages"]:
        raise ValueError("calibration example has no visual messages")


def _collect_moments(
    policy: Any,
    examples: list[Mapping[str, Any]],
    names: list[str],
    kind: str,
    config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    # Lazy dependency allows plan/config tooling without importing the GPU stack.
    from improving.android.training import response_forward, token_kl

    records = {
        name: {
            "second_moment_sum": torch.zeros(
                _native_linear(policy.model, name).out_features,
                _native_linear(policy.model, name).out_features,
                dtype=torch.float64,
            ),
            "token_count": 0,
            "example_count": 0,
            "target_token_count": 0,
        }
        for name in names
    }
    with _captured_native_outputs(policy.model, names) as capture:
        for example in examples:
            _validate_example(example, kind)
            encoded = policy.encode_example(example)
            inputs, prompt, response = _check_encoding(encoded)
            teacher_logits = None
            if kind == "feedback":
                teacher_encoded = policy.encode_example(example, feedback=example["feedback"])
                teacher_inputs, teacher_prompt, teacher_response = _check_encoding(teacher_encoded)
                if teacher_response != response or not torch.equal(
                    inputs["input_ids"][:, prompt:],
                    teacher_inputs["input_ids"][:, teacher_prompt:].to(inputs["input_ids"].device),
                ):
                    raise ValueError("teacher and student must use identical continuation token IDs")
                with torch.no_grad():
                    teacher_logits = response_forward(policy.model, teacher_encoded).detach().to("cpu", copy=True)
                del teacher_encoded, teacher_inputs
            capture["outputs"].clear()
            capture["calls"].clear()
            capture["enabled"] = True
            try:
                with torch.enable_grad():
                    student_logits = response_forward(policy.model, encoded)
                    if kind == "anchor":
                        labels = inputs["input_ids"][:, prompt:prompt + response]
                        loss = F.cross_entropy(
                            student_logits.float().reshape(-1, student_logits.shape[-1]),
                            labels.reshape(-1), reduction="mean",
                        )
                    else:
                        loss = token_kl(
                            student_logits, teacher_logits,
                            top_k=int(config.get("kl_top_k", 0)),
                            chunk_size=int(config.get("kl_chunk_size", 32)),
                        )
                    if not torch.isfinite(loss):
                        raise ValueError(f"nonfinite {kind} calibration loss")
                    if any(capture["calls"].get(name) != 1 for name in names):
                        raise ValueError("each target projection must execute once per calibration example")
                    outputs = [capture["outputs"][name] for name in names]
                    gradients = torch.autograd.grad(loss, outputs, allow_unused=True)
                mask = inputs["attention_mask"].bool()
                for name, gradient in zip(names, gradients):
                    if gradient is None or gradient.shape[:2] != mask.shape:
                        raise ValueError(f"{name}: target is not connected to the calibration loss")
                    selected = gradient.detach()[mask.to(gradient.device)].to(device="cpu", dtype=torch.float64)
                    if not torch.isfinite(selected).all():
                        raise ValueError(f"{name}: nonfinite calibration gradients")
                    record = records[name]
                    for chunk in selected.split(256):
                        record["second_moment_sum"].add_(chunk.T @ chunk)
                    record["token_count"] += int(selected.shape[0])
                    record["example_count"] += 1
                    record["target_token_count"] += response
            finally:
                capture["enabled"] = False
                capture["outputs"].clear()
                capture["calls"].clear()
            del teacher_logits, student_logits, loss, gradients, outputs, encoded, inputs
    for record in records.values():
        count = record["token_count"]
        record["covariance"] = record["second_moment_sum"] / count if count else record["second_moment_sum"].clone()
    return records


def _limited(examples: Iterable[Mapping[str, Any]], maximum: int, name: str) -> list[Mapping[str, Any]]:
    if isinstance(maximum, bool) or not isinstance(maximum, Integral) or maximum < 1:
        raise ValueError(f"{name} must be a positive integer")
    result = []
    for example in examples:
        if len(result) == maximum:
            break
        result.append(example)
    return result


def _fingerprint(examples: list[Mapping[str, Any]]) -> str:
    payload = json.dumps(examples, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def collect_operators(
    policy: Any,
    anchor_examples: Iterable[Mapping[str, Any]],
    feedback_examples: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
    output_dir: str | Path,
    *,
    round_index: int = 1,
    reference_entropy: float | None = None,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Recompute visual anchor and feedback moments at one native checkpoint.

    ``config`` accepts either the full study configuration with a ``spectral``
    subsection or that subsection itself. Raw reference content is not written
    into the public receipt; only identities/hashes and scalar counts are saved.
    The caller owns fixed-anchor selection and train/evaluation separation.
    """
    spectral = config.get("spectral", config)
    if not isinstance(spectral, Mapping):
        raise ValueError("spectral configuration must be a mapping")
    validate_tau_settings(spectral)
    destination = Path(output_dir)
    tensor_path = destination / "operators.pt"
    receipt_path = destination / "calibration.json"
    if tensor_path.exists() or receipt_path.exists():
        raise FileExistsError("calibration output already exists; use a new round/output directory")
    names = select_targets(policy.model, spectral)
    anchors = _limited(anchor_examples, spectral.get("max_anchor_examples", 16), "max_anchor_examples")
    feedback = _limited(feedback_examples, spectral.get("max_feedback_examples", 16), "max_feedback_examples")
    if not anchors:
        raise ValueError("spectral collection needs real successful fixed calibration references")
    anchor_records = _collect_moments(policy, anchors, names, "anchor", spectral)
    # Match the older code gate's reference-NLL signal. Feedback changes the
    # operator geometry below, but not the population used to measure entropy.
    entropy = (normalized_spectral_entropy(anchor_records)
               if spectral.get("tau_mode") == "adaptive" else None)
    tau_policy = resolve_tau(spectral, round_index, entropy=entropy,
                             reference_entropy=reference_entropy)
    alpha = float(spectral.get("alpha", 0.5))
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("spectral alpha must be finite and in [0, 1]")
    feedback_records = _collect_moments(policy, feedback, names, "feedback", spectral) if feedback and alpha else {}
    operators = {}
    details = {}
    for name in names:
        mixed, detail = mix_moments(
            anchor_records[name]["covariance"],
            feedback_records.get(name, {}).get("covariance"),
            alpha=alpha,
            trace_floor=float(spectral.get("moment_trace_floor", 1e-20)),
        )
        operator = make_operator(mixed, kind="spectral_soft", tau=tau_policy["effective_tau"])
        operators[name] = operator
        detail.update({
            "anchor_examples": anchor_records[name]["example_count"],
            "anchor_nonpadding_positions": anchor_records[name]["token_count"],
            "anchor_target_tokens": anchor_records[name]["target_token_count"],
            "feedback_examples": feedback_records.get(name, {}).get("example_count", 0),
            "feedback_nonpadding_positions": feedback_records.get(name, {}).get("token_count", 0),
            "feedback_target_tokens": feedback_records.get(name, {}).get("target_token_count", 0),
            "output_dimension": int(operator.shape[0]),
            "operator_sha256": hashlib.sha256(operator.numpy().tobytes()).hexdigest(),
        })
        details[name] = detail
    receipt = {
        "schema_version": 1,
        "method": "feedback_calibrated_spectrum",
        "checkpoint": str(getattr(policy, "checkpoint", "unknown")),
        "moment_definition": "per-example mean response loss; uncentered K/V activation-gradient second moment over all nonpadding positions",
        "visual_inputs": "native pixel_values and image_grid_thw; no text-only embedding bypass",
        "operator_lifetime": "generation_only; restore before training and evaluation",
        "anchor_examples_sha256": _fingerprint(anchors),
        "feedback_examples_sha256": _fingerprint(feedback if feedback_records else []),
        "tau": tau_policy["effective_tau"],
        "tau_policy": tau_policy,
        "tau_entropy_source": "current_checkpoint_fixed_anchor_reference_nll_moments_only",
        "tau_entropy_definition": "mean projection entropy -sum(p*log(p))/log(strictly_positive_eigenvalue_count); rank<2 projections skipped; not effective rank",
        "targets": details,
    }
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / "operators.pt.tmp"
    torch.save({"operators": operators, "anchor": anchor_records, "feedback": feedback_records}, temporary)
    temporary.replace(tensor_path)
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return operators, receipt


__all__ = ["collect_operators", "mix_moments", "select_targets", "folded_operators",
           "normalized_spectral_entropy"]
