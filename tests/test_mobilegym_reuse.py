import hashlib
import json
from pathlib import Path

import pytest

from improving.mobilegym_reuse import verify_reuse_source, reusable_episode, collection_reuse


def sealed(tmp_path):
    root = tmp_path / 'old'
    root.mkdir()
    cfg = {'output_dir': str(root), 'model': {'checkpoint': 'base'},
           'mobilegym': {'url': 'local'}, 'study': {'rounds': 3}, 'train': {'batch_size': 4}}
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(cfg), encoding='utf-8')
    digest = lambda v: hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,default=str,separators=(',',':')).encode()).hexdigest()
    (root/'study.json').write_text(json.dumps({'configuration':cfg,'config_hash':digest(cfg)}))
    new = json.loads(json.dumps(cfg))
    new['output_dir'] = str(tmp_path/'new')
    new['mobilegym']['environment_workers'] = 2
    new['reuse_completed_episodes'] = {'config_path':str(config),'config_sha256':hashlib.sha256(config.read_bytes()).hexdigest()}
    return cfg, new, root


def test_reuse_requires_same_scientific_config_and_sealed_source(tmp_path):
    old, cfg, root = sealed(tmp_path)
    assert verify_reuse_source(cfg) == root
    cfg['train']['batch_size'] = 8
    with pytest.raises(ValueError, match='configuration'):
        verify_reuse_source(cfg)
    cfg['train']['batch_size'] = 4
    cfg['reuse_completed_episodes']['config_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='hash'):
        verify_reuse_source(cfg)


def test_only_completed_exact_slot_native_episode_reused(tmp_path):
    _, cfg, root = sealed(tmp_path)
    collection = root/'round0_native'
    key = 'episode'
    directory = collection/key
    directory.mkdir(parents=True)
    image = directory/'step.jpg'
    image.write_bytes(b'fixture')
    row = {'episode_id':key,'task_name':'task','checkpoint':'base','round':0,
           'policy':'native','split':'train','status':'complete','success':False,
           'native_score':0.,'judge':{},'steps':[{'screenshot':str(image),'messages':[{'content':[{'type':'image','image':str(image)}]}]}]}
    path = directory/'episode.json'
    path.write_text(json.dumps(row))
    assert reusable_episode(collection,key,'task','base','native',0)['episode_id']==key
    row['split']='eval'
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='identity'):
        reusable_episode(collection,key,'task','base','native',0)
    path.unlink()
    (directory/'episode.partial.json').write_text(json.dumps(row))
    assert reusable_episode(collection,key,'task','base','native',0) is None


def test_reuse_fails_if_saved_screenshot_missing(tmp_path):
    directory=tmp_path/'ep'
    directory.mkdir()
    row={'episode_id':'ep','task_name':'task','checkpoint':'base','round':0,'policy':'native','split':'train',
         'status':'complete','success':False,'native_score':0.,'judge':{},
         'steps':[{'screenshot':str(directory/'missing.jpg'),'messages':[]}]}
    (directory/'episode.json').write_text(json.dumps(row))
    with pytest.raises(ValueError, match='screenshot'):
        reusable_episode(tmp_path,'ep','task','base','native',0)


def test_new_checkpoint_collection_is_not_reused(tmp_path):
    _, cfg, root = sealed(tmp_path)
    found, receipt = collection_reuse(cfg,Path(cfg['output_dir'])/'feedback_sd/round_01/eval',
                                     [],1,'new-checkpoint','eval',1)
    assert found == {}
    assert receipt['skipped_reason'] == 'different_checkpoint_lineage'


def test_chained_reuse_is_rejected_before_running(tmp_path):
    _, cfg, _ = sealed(tmp_path)
    source = Path(cfg['reuse_completed_episodes']['config_path'])
    old = json.loads(source.read_text())
    old['reuse_completed_episodes'] = {'config_path':'ancestor'}
    source.write_text(json.dumps(old))
    digest = hashlib.sha256(json.dumps(old,sort_keys=True,ensure_ascii=False,default=str,separators=(',',':')).encode()).hexdigest()
    (Path(old['output_dir'])/'study.json').write_text(json.dumps({'configuration':old,'config_hash':digest}))
    cfg['reuse_completed_episodes']['config_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='Chained'):
        verify_reuse_source(cfg)
