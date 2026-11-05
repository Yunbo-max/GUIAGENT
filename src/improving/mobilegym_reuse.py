"""Explicit, provenance-checked reuse of completed native MobileGym episodes."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    default=str, separators=(',', ':')).encode()).hexdigest()


def verify_reuse_source(cfg):
    request = cfg.get('reuse_completed_episodes')
    if request is None:
        return None
    path = Path(request['config_path']).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != request['config_sha256']:
        raise ValueError('Reuse source configuration hash mismatch')
    original = json.loads(path.read_text(encoding='utf-8'))
    if original.get('reuse_completed_episodes') is not None:
        raise ValueError('Chained episode reuse is not supported; select the original sealed source')
    root = Path(original['output_dir']).resolve()
    if root == Path(cfg['output_dir']).resolve():
        raise ValueError('Reuse must write a new experiment root')
    seal = json.loads((root / 'study.json').read_text(encoding='utf-8'))
    if seal['configuration'] != original or seal['config_hash'] != _digest(original):
        raise ValueError('Reuse source study configuration seal mismatch')
    def scientific(value):
        result = deepcopy(value)
        result.pop('output_dir', None)
        result.pop('reuse_completed_episodes', None)
        result['mobilegym'].pop('environment_workers', None)
        return result
    if scientific(cfg) != scientific(original):
        raise ValueError('Reuse requires identical scientific configuration')
    return root


def reusable_episode(collection, episode_id, task_name, checkpoint, policy_name, round_index):
    path = Path(collection) / episode_id / 'episode.json'
    if not path.is_file():
        return None  # Partial attempts remain in the source and are never scored.
    row = json.loads(path.read_text(encoding='utf-8'))
    expected = {'episode_id': episode_id, 'task_name': task_name, 'round': round_index,
                'policy': 'native' if policy_name == 'eval' else policy_name,
                'split': 'eval' if policy_name == 'eval' else 'train', 'status': 'complete'}
    if (any(row.get(k) != v for k, v in expected.items())
            or Path(row.get('checkpoint', '')).resolve() != Path(checkpoint).resolve()
            or type(row.get('success')) is not bool
            or row.get('native_score') != float(row['success'])
            or not isinstance(row.get('judge'), dict)
            or not isinstance(row.get('steps'), list) or not row['steps']):
        raise ValueError('Completed reuse episode identity/verifier mismatch')
    for step in row['steps']:
        screenshot = Path(step['screenshot']).resolve()
        if not screenshot.is_relative_to(Path(collection).resolve()) or not screenshot.is_file():
            raise ValueError('Reuse screenshot missing or outside the source collection')
        for message in step.get('messages', []):
            if isinstance(message.get('content'), list):
                for block in message['content']:
                    if block.get('type') == 'image' and Path(block['image']).resolve() != screenshot:
                        raise ValueError('Reuse prompt screenshot does not match the recorded observation')
    return row


def collection_reuse(cfg, out, tasks, repeats, checkpoint, policy_name, round_index):
    """Called after run-level scientific config verification; hash again on reads."""
    request = cfg.get('reuse_completed_episodes')
    if request is None:
        return {}, None
    config = Path(request['config_path'])
    if hashlib.sha256(config.read_bytes()).hexdigest() != request['config_sha256']:
        raise ValueError('Reuse source configuration hash changed')
    old = json.loads(config.read_text(encoding='utf-8'))
    if Path(checkpoint).resolve() != Path(old['model']['checkpoint']).resolve():
        return {}, {'reused_completed': 0, 'planned': len(tasks) * repeats,
                    'skipped_reason': 'different_checkpoint_lineage'}
    relative = Path(out).resolve().relative_to(Path(cfg['output_dir']).resolve())
    source = Path(old['output_dir']).resolve() / relative
    found, receipts = {}, []
    for slot, (task, attempt) in enumerate((t, a) for t in tasks for a in range(repeats)):
        key = _digest([task.id, attempt, policy_name, round_index, str(source)])[:24]
        row = reusable_episode(source, key, task.id, checkpoint, policy_name, round_index)
        if row is not None:
            found[slot] = row
            path = source / key / 'episode.json'
            receipts.append({'slot': slot, 'episode_id': key, 'source': str(path),
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return found, {'source_collection': str(source), 'episodes': receipts,
                   'reused_completed': len(found), 'planned': len(tasks) * repeats,
                   'partial_attempts_reused': False,
                   'sampling_note': 'New pending episodes use the new worker schedule; prior completed prefixes remain unchanged.'}
