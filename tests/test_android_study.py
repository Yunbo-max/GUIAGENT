import json
from pathlib import Path
import sys
import pytest
from improving.android.study import validate_config, seal_study, summarize_episodes, read_complete_stage


def frozen_manifest():
    from improving.android.environment import UPSTREAM_COMMIT, _digest
    splits = ('train', 'calibration', 'eval')
    rows = []
    for value, split in enumerate(splits):
        params = {'value': value, 'seed': value + 10}
        rows.append({'task_name': 'Task', 'params': params, 'goal': f'Find {value}',
                     'initialization_seed': value + 20, 'split': split,
                     'instance_id': _digest({'task_name': 'Task', 'params': {'value': value}})})
    value = {'schema_version': 1, 'upstream_commit': UPSTREAM_COMMIT,
             'protocol': 'parameter_disjoint_same_templates', 'seed': 91,
             'registered_task_count': 1,
             'selected_tasks': {split: ['Task'] for split in splits}, 'instances': rows}
    value['manifest_hash'] = _digest(value)
    return value


def save_manifest(path, manifest, *, reseal=False):
    if reseal:
        from improving.android.environment import _digest
        manifest['manifest_hash'] = _digest({k: v for k, v in manifest.items() if k != 'manifest_hash'})
    path.write_text(json.dumps(manifest))
    return path


def test_frozen_manifest_loader_accepts_valid_instance_disjoint_data(tmp_path):
    from improving.android.study import load_manifest
    manifest = frozen_manifest()
    assert load_manifest(save_manifest(tmp_path/'manifest.json', manifest)) == manifest


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows default text encoding regression')
def test_frozen_manifest_loader_reads_utf8_goals_on_windows(tmp_path):
    from improving.android.environment import _digest
    from improving.android.study import load_manifest

    manifest = frozen_manifest()
    manifest['instances'][0]['goal'] = '创建文件夹'
    manifest['manifest_hash'] = _digest({k: v for k, v in manifest.items() if k != 'manifest_hash'})
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding='utf-8')

    assert load_manifest(path) == manifest


def test_frozen_manifest_loader_rejects_modified_content(tmp_path):
    from improving.android.study import load_manifest
    manifest = frozen_manifest()
    manifest['instances'][0]['goal'] = 'changed goal'
    with pytest.raises(ValueError, match='hash'):
        load_manifest(save_manifest(tmp_path/'manifest.json', manifest))


def test_frozen_manifest_loader_rejects_rehashed_cross_split_collision(tmp_path):
    from copy import deepcopy
    from improving.android.study import load_manifest
    manifest = frozen_manifest()
    duplicate = deepcopy(manifest['instances'][0])
    duplicate['split'] = 'eval'
    duplicate['params']['seed'] = 999
    duplicate['initialization_seed'] = 999
    manifest['instances'][2] = duplicate
    with pytest.raises(ValueError, match='Duplicate semantic instance'):
        load_manifest(save_manifest(tmp_path/'manifest.json', manifest, reseal=True))


@pytest.mark.parametrize('case,match', [
    ('schema', 'schema'), ('upstream', 'upstream'), ('identity', 'fingerprint'),
    ('missing_split', 'nonempty'), ('selection', 'selected_tasks'),
])
def test_frozen_manifest_loader_rejects_invalid_rehashed_metadata(tmp_path, case, match):
    from improving.android.study import load_manifest
    manifest = frozen_manifest()
    if case == 'schema':
        manifest['schema_version'] = 2
    elif case == 'upstream':
        manifest['upstream_commit'] = 'wrong'
    elif case == 'identity':
        manifest['instances'][0]['instance_id'] = 'wrong'
    elif case == 'missing_split':
        manifest['instances'] = manifest['instances'][:2]
    else:
        manifest['selected_tasks']['eval'] = ['OtherTask']
    with pytest.raises(ValueError, match=match):
        load_manifest(save_manifest(tmp_path/'manifest.json', manifest, reseal=True))


@pytest.mark.parametrize('entrypoint', ['study', 'qualification'])
def test_invalid_manifest_stops_before_checkpoint_resolution_or_study_seal(tmp_path, monkeypatch, entrypoint):
    import importlib
    module = importlib.import_module(f'improving.android.{entrypoint}')
    manifest = frozen_manifest()
    manifest['seed'] += 1
    settings = config(tmp_path)
    settings['manifest'] = str(save_manifest(tmp_path/'manifest.json', manifest))
    settings['android'] = {'upstream_root': 'unused'}
    monkeypatch.setattr(module, 'verify_upstream', lambda root: None)
    def unexpected_checkpoint(config):
        raise AssertionError('Checkpoint resolution preceded manifest validation')
    monkeypatch.setattr(module, 'resolve_checkpoint', unexpected_checkpoint)
    function = module.run_study if entrypoint == 'study' else module.qualify
    with pytest.raises(ValueError, match='hash'):
        function(settings)
    assert not Path(settings['output_dir']).exists()


def config(tmp_path):
    return {'output_dir':str(tmp_path/'runs'),'model':{'checkpoint':str(tmp_path/'model')},
            'study':{'warmup_rounds':3,'rounds':3,'group_size':4,'spectral_fraction':0.5},
            'generation':{'temperature':0.7,'top_p':0.8,'max_new_tokens':256},
            'train':{'max_updates':16},'spectral':{'alpha':0.5,'tau':1.0}}


def test_invalid_collection_budget_is_rejected(tmp_path):
    c=config(tmp_path); c['study']['group_size']=3
    with pytest.raises(ValueError,match='even'):
        validate_config(c)


def test_sealed_study_refuses_changed_method(tmp_path):
    c=config(tmp_path); root=tmp_path/'run'
    seal_study(root,c,{'instances':[]},'source-a')
    seal_study(root,c,{'instances':[]},'source-a')
    c['spectral']['tau']=2
    with pytest.raises(ValueError,match='identity'):
        seal_study(root,c,{'instances':[]},'source-a')


def test_missing_episode_never_becomes_failed_score():
    with pytest.raises(ValueError,match='expected'):
        summarize_episodes([{'episode_id':'a','success':True,'steps':[]}],expected=2)


def test_eval_summary_retains_native_outcomes():
    rows=[{'episode_id':'a','instance_id':'x','success':True,'policy':'native','steps':[{'response_token_ids':[1,2],'action':{'action':'click'}}]},
          {'episode_id':'b','instance_id':'y','success':False,'policy':'native','steps':[]}]
    r=summarize_episodes(rows,expected=2)
    assert r['success_rate']==0.5 and r['episodes']==2 and r['response_tokens']==2


def test_incomplete_stage_is_not_resumed_as_success(tmp_path):
    stage=tmp_path/'stage'; stage.mkdir()
    (stage/'episodes.jsonl').write_text('')
    with pytest.raises(RuntimeError,match='Incomplete'):
        read_complete_stage(stage)


def test_resume_loads_only_named_operators(tmp_path):
    import torch
    from improving.android.study import load_operators
    path=tmp_path/'operators.pt'
    torch.save({'operators':{'layer.k_proj':torch.eye(2)},'anchor':{'extra':1},'feedback':{}},path)
    assert list(load_operators(path))==['layer.k_proj']


def test_empty_supervision_records_no_update_and_keeps_checkpoint(tmp_path):
    from improving.android.study import _train
    model=tmp_path/'model'; model.mkdir(); (model/'config.json').write_text('{}')
    stage=tmp_path/'stage'
    result=_train(str(model),[],stage,config(tmp_path),'grsd')
    assert result==str(model)
    receipt=json.loads((stage/'complete.json').read_text())
    assert receipt['training']['status']=='no_update'
    assert receipt['training']['optimizer_steps']==0


def test_changed_referenced_image_is_rejected(tmp_path):
    from improving.android.study import _finish
    stage=tmp_path/'stage'; stage.mkdir()
    image=tmp_path/'image.png'; image.write_bytes(b'original')
    _finish(stage,{},external_paths=[image])
    image.write_bytes(b'changed')
    with pytest.raises(ValueError,match='External'):
        read_complete_stage(stage)
