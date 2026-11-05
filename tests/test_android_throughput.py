"""No GPU needed to unit-test actual native benchmark throughput accounting."""
import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]

def module():
    spec=importlib.util.spec_from_file_location(
        "androidperf",ROOT/"scripts/windows/bench_androidworld_throughput.py")
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def test_trajectory_stage_speed_uses_true_wall_and_counts():
    good=[
      {"status":"complete","success":True,"steps":[{"response_token_ids":[1,2]},{"response_token_ids":[3]}],
       "timing_totals_seconds":{"model_generation":2.0,"explicit_wait":3.0}},
      {"status":"complete","success":False,"steps":[{"response_token_ids":[7]}],
       "timing_totals_seconds":{"model_generation":1.0,"explicit_wait":1.5}},
    ]
    output=module().summarize(good,30.0)
    assert output["episodes"]==2
    assert output["real_actions"]==3
    assert output["real_verified_successes"]==1
    assert output["episodes_per_minute"]==4
    assert output["actions_per_minute"]==6
    assert output["per_step_component_seconds"]["explicit_wait"]==4.5

def test_does_not_treat_partial_as_failed():
    invalid=[{"status":"aborted","success":None,"steps":[]}]
    with pytest.raises(ValueError,match="native"):
        module().summarize(invalid,10.0)
