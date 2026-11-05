"""No-AVD tests: independent emulator constraints and deterministic pairing."""
import pytest
from improving.android.parallel_collection import parse_worker_devices

def configuration():
    return {"avd_name":"AndroidWorldAvd","console_port":5554,"grpc_port":8554,
            "adb_server_port":5037,
            "environment_workers":2,
            "worker_devices":[
                {"avd_name":"AndroidWorldAvd","console_port":5554,"grpc_port":8554},
                {"avd_name":"AndroidWorldAvd2","console_port":5556,"grpc_port":8556}]}

def test_distinct_real_avds():
    devices=parse_worker_devices(configuration())
    assert len(devices)==2
    assert devices[0]["console_port"]+1!=devices[1]["console_port"]+1

def test_duplicate_emu_and_ports_fails():
    cfg=configuration()
    cfg["worker_devices"][1]["avd_name"]="AndroidWorldAvd"
    with pytest.raises(ValueError,match="unique"):
        parse_worker_devices(cfg)
    cfg=configuration()
    cfg["worker_devices"][1]["console_port"]=5554
    with pytest.raises(ValueError):
        parse_worker_devices(cfg)
    cfg=configuration()
    cfg["worker_devices"][1]["grpc_port"]=5554
    with pytest.raises(ValueError):
        parse_worker_devices(cfg)

def test_no_parallel_reuse_of_one_avd():
    cfg=configuration()
    cfg["worker_devices"]=cfg["worker_devices"][:1]
    with pytest.raises(ValueError,match="exactly one"):
        parse_worker_devices(cfg)
    cfg=configuration()
    cfg["environment_workers"]=1
    assert len(parse_worker_devices(cfg))==1
    cfg["environment_workers"]=3
    with pytest.raises(ValueError,match="exactly one"):
        parse_worker_devices(cfg)

def test_worker_count_rejects_bool_or_24_browser_mistake():
    cfg=configuration()
    for value in (True,24,0):
        cfg["environment_workers"]=value
        with pytest.raises(ValueError,match="1, 2, 3 or 4"):
            parse_worker_devices(cfg)


def test_parallel_aggregation_orders_frozen_task_and_repeat_not_finish_order():
    from improving.android.parallel_collection import assemble_frozen_episodes
    cases=[{"instance_id":"task-A","split":"train"},{"instance_id":"task-B","split":"train"}]
    def row(task,repeat):
        return {"instance_id":task,"repeat_id":repeat,"split":"train",
                "success":repeat==0,"status":"complete"}
    results=[
        [row("task-B",0),row("task-B",1)],
        [row("task-A",0),row("task-A",1)]]
    actual=assemble_frozen_episodes(cases,2,results)
    assert [(r["instance_id"],r["repeat_id"]) for r in actual]==[
        ("task-A",0),("task-A",1),("task-B",0),("task-B",1)]
    with pytest.raises(ValueError,match="Duplicate"):
        assemble_frozen_episodes(cases,2,[results[0],results[0],results[1]])
    with pytest.raises(ValueError,match="Missing"):
        assemble_frozen_episodes(cases,2,[results[0],results[1][:1]])
    wrong=row("task-A",0)
    wrong["success"]=None
    with pytest.raises(ValueError,match="native-verifier"):
        assemble_frozen_episodes(cases,2,[[wrong,row("task-A",1)],results[0]])
