"""Read-only status counts native outcomes, never partial episodes."""
import json
from pathlib import Path
import importlib.util

ROOT=Path(__file__).resolve().parents[1]

def module():
    spec=importlib.util.spec_from_file_location(
        "androidstatus",ROOT/"scripts/windows/inspect_androidworld_progress.py")
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def test_read_only_partial_and_complete_stage(tmp_path):
    source=tmp_path/"round0_native"
    source.mkdir()
    valid={"episode_id":"a","status":"complete","success":False,
           "steps":[{"index":0}],"elapsed_seconds":20.0}
    bad={"status":"aborted","success":None,"steps":[]}
    (source/"episodes.jsonl").write_text(
        json.dumps(valid)+"\n"+json.dumps(bad)+"\n")
    value=module().inspect(tmp_path)
    stage=value["stages"]["round0_native"]
    assert stage["episodes_completed"]==1
    assert stage["native_verified_failures"]==1
    assert stage["mean_episode_seconds"]==20.0
    assert not stage["stage_complete_receipt"]

def test_read_only_reports_parallel_workers_when_no_root_summary(tmp_path):
    folder=tmp_path/"round0_native"
    for number in (0,1):
        p=folder/f"worker_{number:02d}"
        p.mkdir(parents=True)
        x={"status":"complete","success":True,"steps":[{},{}],
           "elapsed_seconds":40.0}
        (p/"episodes.jsonl").write_text(json.dumps(x)+"\n")
    stats=module().inspect(tmp_path)["stages"]["round0_native"]
    assert stats["episodes_completed"]==2
    assert stats["actions_attempted"]==4
    assert stats["stage_elapsed_wall_seconds"] is None
