"""Finite, reference-calibrated full-spectrum gains for native text models.

This is a bounded numerical solver, not a global optimizer or a guarantee about
unseen tasks. The frozen model and current spectral basis are never trained here.
Only gains are fitted; the held-out reference guard is used once, at the end.
The final operator is rechecked through the actual weight-folding implementation.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import contextmanager
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .spectral import _covariance_eigh, _native_linear, folded_operators


CORE_DEFAULTS = {
    'max_steps': 8,
    'learning_rate': 0.1,
    'min_gain': 0.1,
    'kl_budget': 0.02,
    'max_backtracks': 8,
    'backtrack_factor': 0.5,
    'min_improvement': 1e-6,
    'guard_tolerance': 0.0,
    'degeneracy_rtol': 1e-7,
    'degeneracy_atol': 1e-12,
    'kl_chunk_tokens': 32,
}
NUMERICAL_TOLERANCE = 1e-6


def validate_gain_settings(settings: Mapping[str, Any] | None = None) -> dict:
    """Validate solver settings only; probe generation belongs to the pipeline."""
    if settings is None:
        settings = {}
    if not isinstance(settings, Mapping):
        raise ValueError('gain_learning solver settings must be a mapping')
    unknown = set(settings) - set(CORE_DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown gain_learning solver settings: {sorted(unknown)}')
    result = {**CORE_DEFAULTS, **settings}
    for name, minimum in (('max_steps', 1), ('max_backtracks', 0), ('kl_chunk_tokens', 1)):
        if type(result[name]) is not int or result[name] < minimum:
            raise ValueError(f'gain_learning.{name} must be an integer >= {minimum}')
    for name in set(CORE_DEFAULTS) - {'max_steps', 'max_backtracks', 'kl_chunk_tokens'}:
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'gain_learning.{name} must be finite and numeric')
        result[name] = float(value)
    if result['learning_rate'] <= 0:
        raise ValueError('gain_learning.learning_rate must be positive')
    if not 0 < result['min_gain'] <= 1:
        raise ValueError('gain_learning.min_gain must be in (0, 1]')
    if not 0 < result['backtrack_factor'] < 1:
        raise ValueError('gain_learning.backtrack_factor must be in (0, 1)')
    for name in ('kl_budget', 'min_improvement', 'guard_tolerance'):
        if result[name] < 0:
            raise ValueError(f'gain_learning.{name} must be nonnegative')
    for name in ('degeneracy_rtol', 'degeneracy_atol'):
        if not 0 <= result[name] < 1:
            raise ValueError(f'gain_learning.{name} must be in [0, 1)')
    return result


def spectral_groups(eigenvalues: Tensor, rtol: float = 1e-7, atol: float = 1e-12) -> Tensor:
    """Tie only numerically equal eigenvalues, not manually chosen spectral bands.

    Tolerances apply to eigenvalues divided by their maximum. Each group is
    compared to its FIRST value, so a chain of small gaps cannot merge a wide
    interval. Exact repeated eigenvalues, including the nullspace, always tie.
    """
    if (not isinstance(eigenvalues, Tensor) or eigenvalues.ndim != 1
            or not eigenvalues.numel() or eigenvalues.is_complex()):
        raise ValueError('eigenvalues must be a nonempty real vector')
    for value in (rtol, atol):
        if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError('spectral equality tolerances must be finite and in [0, 1)')
    values = eigenvalues.detach().to(device='cpu', dtype=torch.float64)
    if (not torch.isfinite(values).all() or bool((values < 0).any())
            or float(values[-1]) <= 0 or bool((values[1:] < values[:-1]).any())):
        raise ValueError('eigenvalues must be finite, sorted, nonnegative, with positive maximum')
    values = (values / values[-1]).tolist()
    groups = [0]
    anchor = values[0]
    group = 0
    for value in values[1:]:
        if abs(value - anchor) > atol + rtol * max(abs(value), abs(anchor)):
            group += 1
            anchor = value
        groups.append(group)
    return torch.tensor(groups, dtype=torch.long)


class SpectralGainLinear(nn.Module):
    """Differentiable row-activation intervention with an exact identity start.

    The residual form h + (h U diag(a-1)) U.T avoids a numerical U U.T
    roundtrip at identity. Native parameters are not modified. Gain coordinates
    are shared only for the supplied eigenvalue-equality groups.
    """

    def __init__(self, linear: nn.Linear, basis: Tensor, group_ids: Tensor):
        super().__init__()
        if type(linear) is not nn.Linear:
            raise TypeError('SpectralGainLinear requires a native torch.nn.Linear')
        d = linear.out_features
        if (not isinstance(basis, Tensor) or basis.is_complex()
                or basis.shape != (d, d) or not torch.isfinite(basis).all()):
            raise ValueError('basis must be a finite square output-dimension matrix')
        if (group_ids.shape != (d,) or group_ids.dtype != torch.long
                or bool((group_ids < 0).any())):
            raise ValueError('group_ids must be nonnegative long indices per direction')
        unique = group_ids.detach().cpu().unique(sorted=True)
        if not torch.equal(unique, torch.arange(len(unique))):
            raise ValueError('group_ids must be contiguous and start at zero')
        self.linear = linear
        dtype = torch.float64 if linear.weight.dtype == torch.float64 else torch.float32
        self.register_buffer('basis', basis.detach().to(device=linear.weight.device, dtype=dtype))
        self.register_buffer('group_ids', group_ids.detach().to(device=linear.weight.device))
        self.gains = nn.Parameter(torch.ones(len(unique), dtype=dtype, device=linear.weight.device))
        self.enabled = True
        self.calls = 0

    def forward(self, inputs: Tensor) -> Tensor:
        self.calls += 1
        native = self.linear(inputs)
        if not self.enabled:
            return native
        displacement = ((native.to(self.basis.dtype) @ self.basis)
                        * (self.gains[self.group_ids] - 1)) @ self.basis.T
        return native + displacement.to(native.dtype)


def forward_kl_sum(native_logits: Tensor, candidate_logits: Tensor, target_mask: Tensor,
                   chunk_tokens: int = 32) -> tuple[float, int, float, int]:
    """Full-vocabulary KL(native || candidate) on exactly the selected positions.

    Native logits may reside on CPU. At most one token chunk is converted to
    working precision on the candidate device. Returns sum, count, minimum raw
    token KL, and number of tiny negative roundoff values clamped to zero.
    """
    if (not isinstance(native_logits, Tensor) or not isinstance(candidate_logits, Tensor)
            or not isinstance(target_mask, Tensor)
            or native_logits.ndim < 2 or native_logits.shape != candidate_logits.shape
            or not native_logits.is_floating_point() or not candidate_logits.is_floating_point()
            or native_logits.shape[-1] < 1
            or target_mask.shape != native_logits.shape[:-1] or target_mask.dtype != torch.bool):
        raise ValueError('KL logits must match and target_mask must match their leading dimensions')
    if type(chunk_tokens) is not int or chunk_tokens < 1:
        raise ValueError('chunk_tokens must be a positive integer')
    positions = target_mask.detach().reshape(-1).nonzero(as_tuple=False).flatten().cpu()
    if not positions.numel():
        raise ValueError('KL probe has no selected next-token positions')
    vocab = native_logits.shape[-1]
    native = native_logits.detach().reshape(-1, vocab)
    candidate = candidate_logits.detach().reshape(-1, vocab)
    dtype = torch.float64 if candidate.dtype == torch.float64 else torch.float32
    total, minimum, clamped = 0.0, math.inf, 0
    for start in range(0, positions.numel(), chunk_tokens):
        ids = positions[start:start + chunk_tokens]
        left = native.index_select(0, ids.to(native.device)).to(device=candidate.device, dtype=dtype)
        right = candidate.index_select(0, ids.to(candidate.device)).to(dtype=dtype)
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise FloatingPointError('nonfinite native or intervened probe logits')
        lp, lq = F.log_softmax(left, dim=-1), F.log_softmax(right, dim=-1)
        token_kl = (lp.exp() * (lp - lq)).sum(dim=-1)
        if not torch.isfinite(token_kl).all():
            raise FloatingPointError('nonfinite full-vocabulary probe KL')
        minimum = min(minimum, float(token_kl.min()))
        if float(token_kl.min()) < -NUMERICAL_TOLERANCE:
            raise ValueError('negative probe KL exceeds the numerical tolerance')
        clamped += int((token_kl < 0).sum())
        total += float(token_kl.clamp_min(0).sum())
    return total, positions.numel(), minimum, clamped


def _examples(batches: Iterable[Mapping[str, Tensor]], role: str) -> list[dict]:
    result = []
    for batch in batches:
        if not isinstance(batch, Mapping) or not all(k in batch for k in ('input_ids', 'attention_mask', 'labels')):
            raise ValueError(f'{role} batches require input_ids, attention_mask, labels')
        ids, mask, labels = (batch[k] for k in ('input_ids', 'attention_mask', 'labels'))
        if (not all(isinstance(t, Tensor) and t.ndim == 2 for t in (ids, mask, labels))
                or ids.shape != mask.shape or ids.shape != labels.shape or ids.shape[1] < 2
                or ids.dtype not in (torch.int32, torch.int64)
                or labels.dtype not in (torch.int32, torch.int64)):
            raise ValueError(f'{role} tensors must have matching [batch, sequence>=2] shapes and integer IDs/labels')
        if (not bool(((mask == 0) | (mask == 1)).all()) or bool((ids < 0).any())
                or bool(((labels < 0) & (labels != -100)).any())):
            raise ValueError(f'{role} contains invalid masks or token IDs')
        for index in range(ids.shape[0]):
            one = {k: batch[k][index:index + 1].detach().to(device='cpu', dtype=torch.long).clone()
                   for k in ('input_ids', 'attention_mask', 'labels')}
            valid = one['attention_mask'].bool()
            target_mask = valid[:, :-1] & valid[:, 1:] & (one['labels'][:, 1:] != -100)
            count = int(target_mask.sum())
            if not count:
                raise ValueError(f'each {role} example must have a selected nonpadding next-token target')
            one['target_mask'] = target_mask
            one['target_count'] = count
            result.append(one)
    if not result:
        raise ValueError(f'{role} must contain at least one example')
    return result


@contextmanager
def _frozen_native(model):
    if getattr(model, 'is_gradient_checkpointing', False):
        raise ValueError('Disable gradient checkpointing before learned-gain calibration; it is a separate frozen-model stage')
    parameters = [(p, p.requires_grad) for p in model.parameters()]
    states = [(m, m.training) for m in model.modules()]
    configs = []
    config = getattr(model, 'config', None)
    for item in (config, getattr(config, 'text_config', None)):
        if item is not None and hasattr(item, 'use_cache') and not any(item is c for c, _ in configs):
            configs.append((item, item.use_cache))
    try:
        model.eval()
        for p, _ in parameters:
            p.requires_grad_(False)
        for item, _ in configs:
            item.use_cache = False
        yield
    finally:
        for p, requires_grad in parameters:
            p.requires_grad_(requires_grad)
        for module, training in states:
            module.training = training
        for item, use_cache in configs:
            item.use_cache = use_cache


@contextmanager
def _installed(model, wrappers):
    replaced = []
    try:
        for name, wrapper in wrappers.items():
            parent_name, _, leaf = name.rpartition('.')
            parent = model.get_submodule(parent_name) if parent_name else model
            original = getattr(parent, leaf)
            replaced.append((parent, leaf, original))
            setattr(parent, leaf, wrapper)
        yield
    finally:
        for parent, leaf, original in reversed(replaced):
            setattr(parent, leaf, original)


@contextmanager
def _enabled(wrappers, value):
    previous = [(wrapper, wrapper.enabled) for wrapper in wrappers.values()]
    try:
        for wrapper, _ in previous:
            wrapper.enabled = value
        yield
    finally:
        for wrapper, enabled in previous:
            wrapper.enabled = enabled


def _logits(model, example, wrappers=None):
    device = model.get_input_embeddings().weight.device
    if wrappers is not None:
        for wrapper in wrappers.values():
            wrapper.calls = 0
    output = model(input_ids=example['input_ids'].to(device),
                   attention_mask=example['attention_mask'].to(device), use_cache=False)
    logits = output.logits if hasattr(output, 'logits') else output['logits']
    if logits.ndim != 3 or logits.shape[:2] != example['input_ids'].shape:
        raise ValueError('model logits must preserve [batch, sequence] dimensions')
    if wrappers is not None and any(wrapper.calls != 1 for wrapper in wrappers.values()):
        raise ValueError('each learned-gain target must execute exactly once per example')
    return logits


def _nll(logits, example):
    mask = example['target_mask'].to(logits.device)
    selected = logits[:, :-1][mask]
    labels = example['labels'][:, 1:].to(logits.device)[mask]
    if not torch.isfinite(selected).all():
        raise FloatingPointError('nonfinite selected reference logits')
    dtype = torch.float64 if selected.dtype == torch.float64 else torch.float32
    loss = F.cross_entropy(selected.to(dtype), labels, reduction='mean')
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite reference NLL')
    return loss


def _mean_nll(model, examples, wrappers=None):
    with torch.no_grad():
        return sum(float(_nll(_logits(model, example, wrappers), example))
                   for example in examples) / len(examples)


def _fit_gradient(model, examples, wrappers):
    parameters = [wrapper.gains for wrapper in wrappers.values()]
    total = [torch.zeros_like(p) for p in parameters]
    with torch.enable_grad():
        for example in examples:
            loss = _nll(_logits(model, example, wrappers), example) / len(examples)
            if not loss.requires_grad:
                # Every selected gain may be disconnected from this example's
                # target. Its derivative is then exactly zero, not an error.
                continue
            grads = torch.autograd.grad(loss, parameters, allow_unused=True)
            for accumulator, gradient in zip(total, grads):
                if gradient is not None:
                    if not torch.isfinite(gradient).all():
                        raise FloatingPointError('nonfinite gain gradient')
                    accumulator.add_(gradient.detach())
    return total


def _probe_kl(model, examples, chunk_tokens, *, wrappers=None, operators=None):
    total, count, minimum, clamped = 0.0, 0, math.inf, 0
    with torch.no_grad():
        for example in examples:
            if wrappers is not None:
                with _enabled(wrappers, False):
                    native = _logits(model, example, wrappers)[:, :-1].detach().to('cpu')
                with _enabled(wrappers, True):
                    candidate = _logits(model, example, wrappers)[:, :-1]
            else:
                native = _logits(model, example)[:, :-1].detach().to('cpu')
                with folded_operators(model, operators):
                    candidate = _logits(model, example)[:, :-1]
            value, n, low, negative = forward_kl_sum(native, candidate, example['target_mask'], chunk_tokens)
            total += value
            count += n
            minimum = min(minimum, low)
            clamped += negative
            del native, candidate
    return {'mean': total / count, 'sum': total, 'tokens': count,
            'minimum_raw_token_kl': minimum, 'clamped_negative_tokens': clamped}


def _set_gains(wrappers, values):
    with torch.no_grad():
        for wrapper, value in zip(wrappers.values(), values):
            wrapper.gains.copy_(value)


def fit_learned_operators(model, fit_batches, guard_batches, probe_batches,
                          covariances, settings=None):
    """Fit full-spectrum gains and return deployed operators plus a JSON audit.

    ``fit`` and ``guard`` require disjoint reference examples (the pipeline binds
    IDs). Probe ``labels`` select rollout target positions only; their values
    are never used as supervised CE targets. All losses average token CE inside
    each example, then average examples. Probe KL averages all selected tokens.

    Gain fitting uses projected, globally max-absolute-gradient-normalized
    descent and finite backtracking. Guard data never chooses an iterate; the
    last accepted fit/probe iterate is checked on the guard, then through the
    actual folded policy. Failure of its fit/guard/KL checks returns identity. Runtime errors
    and invalid native data propagate after state restoration.
    """
    cfg = validate_gain_settings(settings)
    fit = _examples(fit_batches, 'fit')
    guard = _examples(guard_batches, 'guard')
    probe = _examples(probe_batches, 'probe')
    if not isinstance(covariances, Mapping) or not covariances:
        raise ValueError('learned-gain calibration requires nonempty module covariances')
    bases, eigs, ids, originals = {}, {}, {}, {}
    for name, record in covariances.items():
        originals[name] = _native_linear(model, name)
        _, eigs[name], bases[name] = _covariance_eigh(record['covariance'])
        if bases[name].shape[0] != originals[name].out_features:
            raise ValueError(f'covariance dimension disagrees with target {name}')
        ids[name] = spectral_groups(eigs[name], cfg['degeneracy_rtol'], cfg['degeneracy_atol'])
    if len({id(m) for m in originals.values()}) != len(originals):
        raise ValueError('multiple learned-gain names refer to the same module')
    wrappers = {name: SpectralGainLinear(originals[name], bases[name], ids[name]) for name in originals}
    identities = {name: torch.eye(len(eigs[name]), dtype=torch.float64) for name in originals}
    zero_kl = {'mean': 0.0, 'sum': 0.0,
               'tokens': sum(x['target_count'] for x in probe),
               'minimum_raw_token_kl': 0.0, 'clamped_negative_tokens': 0}
    audit = {'format_version': 1, 'solver': 'projected_gradient_finite_backtracking',
             'settings': cfg, 'numerical_tolerance': NUMERICAL_TOLERANCE,
             'scope': 'generation_only_uniform_sft_unchanged',
             'kl_direction': 'native_to_intervened_full_vocabulary',
             'kl_position_policy': 'selected_causally_shifted_nonpadding_probe_targets',
             'nll_reduction': 'mean_of_per_example_mean_selected_token_nll',
             'guard_policy': 'final_candidate_then_folded_veto_only',
             'examples': {'fit': len(fit), 'guard': len(guard), 'probe': len(probe)},
             'target_tokens': {name: sum(x['target_count'] for x in examples)
                               for name, examples in [('fit', fit), ('guard', guard), ('probe', probe)]},
             'history': [], 'modules': {}, 'selected_identity': True,
             'fallback_reason': None, 'global_optimum_claimed': False}
    with _frozen_native(model):
        native_fit = _mean_nll(model, fit)
        native_guard = _mean_nll(model, guard)
        # Probe native logits must be valid even when fitting returns identity.
        native_probe = _probe_kl(model, probe, cfg['kl_chunk_tokens'], operators={})
        audit['native'] = {'fit_nll': native_fit, 'guard_nll': native_guard, 'probe_kl': native_probe}
        current_fit, current_kl = native_fit, dict(zero_kl)
        accepted_steps, stop_reason = 0, 'maximum_steps_reached'
        with _installed(model, wrappers):
            for step in range(cfg['max_steps']):
                previous = [wrapper.gains.detach().clone() for wrapper in wrappers.values()]
                gradients = _fit_gradient(model, fit, wrappers)
                magnitude = max(float(g.abs().max()) for g in gradients)
                if step == 0:
                    for name, gradient in zip(wrappers, gradients):
                        audit['modules'][name] = {'initial_group_gradient': gradient.cpu().double().tolist()}
                if magnitude == 0:
                    stop_reason = 'zero_gain_gradient'
                    break
                accepted = False
                for backtrack in range(cfg['max_backtracks'] + 1):
                    rate = cfg['learning_rate'] * cfg['backtrack_factor'] ** backtrack
                    candidate = [(old - rate * grad / magnitude).clamp(cfg['min_gain'], 1)
                                 for old, grad in zip(previous, gradients)]
                    item = {'step': step + 1, 'backtrack': backtrack, 'step_size': rate,
                            'gradient_max_abs': magnitude, 'accepted': False}
                    if all(torch.equal(old, new) for old, new in zip(previous, candidate)):
                        item['reason'] = 'projected_step_is_zero'
                        audit['history'].append(item)
                        break
                    _set_gains(wrappers, candidate)
                    try:
                        candidate_fit = _mean_nll(model, fit, wrappers)
                        item['fit_nll'] = candidate_fit
                        if candidate_fit >= current_fit - cfg['min_improvement']:
                            item['reason'] = 'insufficient_fit_improvement'
                        else:
                            kl = _probe_kl(model, probe, cfg['kl_chunk_tokens'], wrappers=wrappers)
                            item['probe_kl'] = kl
                            if kl['mean'] > cfg['kl_budget'] + NUMERICAL_TOLERANCE:
                                item['reason'] = 'probe_kl_budget_exceeded'
                            else:
                                accepted = True
                                accepted_steps += 1
                                current_fit, current_kl = candidate_fit, kl
                                item.update(accepted=True, reason='fit_improved_and_probe_feasible')
                    except FloatingPointError as error:
                        item['reason'] = 'nonfinite_candidate'
                        item['error'] = str(error)
                    audit['history'].append(item)
                    if accepted:
                        break
                if not accepted:
                    _set_gains(wrappers, previous)
                    stop_reason = 'no_accepted_backtracking_step'
                    break
            audit['accepted_steps'] = accepted_steps
            audit['stop_reason'] = stop_reason
            wrapper_guard_error = None
            try:
                wrapper_guard = _mean_nll(model, guard, wrappers)
            except FloatingPointError as error:
                wrapper_guard = None
                wrapper_guard_error = str(error)
            audit['wrapper'] = {'fit_nll': current_fit, 'guard_nll': wrapper_guard,
                                'guard_error': wrapper_guard_error, 'probe_kl': current_kl}
            operators = {}
            for name, wrapper in wrappers.items():
                group_gains = wrapper.gains.detach().cpu().double()
                gains = group_gains[ids[name]]
                basis = bases[name]
                operators[name] = identities[name] + (basis * (gains - 1).unsqueeze(0)) @ basis.T
                positive = eigs[name][eigs[name] > 0]
                probabilities = positive / positive.sum()
                entropy = (float(-(probabilities * probabilities.log()).sum()) / math.log(len(positive))
                           if len(positive) > 1 else 0.0)
                audit['modules'].setdefault(name, {}).update({
                    'dimension': len(gains), 'gain_parameters': len(group_gains),
                    'eigenvalues': eigs[name].tolist(), 'group_ids': ids[name].tolist(),
                    'candidate_group_gains': group_gains.tolist(), 'candidate_gains': gains.tolist(),
                    'covariance_trace': float(eigs[name].sum()),
                    'covariance_lambda_max': float(eigs[name][-1]),
                    'normalized_positive_spectrum_entropy': entropy,
                    'candidate_gain_min': float(gains.min()), 'candidate_gain_max': float(gains.max()),
                    'candidate_gain_mean': float(gains.mean()),
                })
        if wrapper_guard_error is not None:
            audit['fallback_reason'] = 'nonfinite_wrapper_guard'
            audit['folded_candidate'] = None
        elif wrapper_guard > native_guard + cfg['guard_tolerance'] + NUMERICAL_TOLERANCE:
            audit['fallback_reason'] = 'wrapper_guard_deteriorated'
            audit['folded_candidate'] = None
        elif not accepted_steps:
            audit['fallback_reason'] = 'no_feasible_improving_update'
            audit['folded_candidate'] = None
        else:
            try:
                with folded_operators(model, operators):
                    actual_fit = _mean_nll(model, fit)
                    actual_guard = _mean_nll(model, guard)
                actual_kl = _probe_kl(model, probe, cfg['kl_chunk_tokens'], operators=operators)
                audit['folded_candidate'] = {'fit_nll': actual_fit, 'guard_nll': actual_guard,
                                             'probe_kl': actual_kl}
                failed = []
                if actual_fit >= native_fit - cfg['min_improvement']:
                    failed.append('folded_fit_not_improved')
                if actual_guard > native_guard + cfg['guard_tolerance'] + NUMERICAL_TOLERANCE:
                    failed.append('folded_guard_deteriorated')
                if actual_kl['mean'] > cfg['kl_budget'] + NUMERICAL_TOLERANCE:
                    failed.append('folded_probe_kl_budget_exceeded')
                audit['selected_identity'] = bool(failed)
                audit['fallback_reason'] = ','.join(failed) if failed else None
            except FloatingPointError as error:
                audit['folded_candidate'] = {'error': str(error)}
                audit['fallback_reason'] = 'nonfinite_folded_candidate'
        if audit['selected_identity']:
            operators = identities
            audit['selected'] = dict(audit['native'])
        else:
            audit['selected'] = dict(audit['folded_candidate'])
        for info in audit['modules'].values():
            info['selected_gains'] = ([1.0] * info['dimension'] if audit['selected_identity']
                                      else info['candidate_gains'])
    return operators, audit
