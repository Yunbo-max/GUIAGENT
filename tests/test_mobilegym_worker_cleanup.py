import asyncio
import sys
from types import ModuleType, SimpleNamespace

import pytest


def test_cancellation_during_browser_warmup_closes_environment(tmp_path, monkeypatch):
    from improving.mobilegym_feedback import _collect_async
    async def check():
        started = asyncio.Event()
        closed = []
        class Env:
            def __init__(self, **kwargs): pass
            @staticmethod
            def get_launch_args(*args): return {}
            async def get_state(self, **kwargs): return {}
            async def open_app(self, *args, **kwargs): pass
            async def start(self): pass
            async def wait_ready(self): pass
            async def warm_apps(self, apps):
                started.set()
                await asyncio.sleep(10)
            async def close(self): closed.append(self)
        envmod = ModuleType('bench_env.env.mobile_gym')
        envmod.MobileGymEnv = Env
        basemod = ModuleType('bench_env.runner.base')
        basemod.Evaluator = lambda **kwargs: None
        monkeypatch.setitem(sys.modules, envmod.__name__, envmod)
        monkeypatch.setitem(sys.modules, basemod.__name__, basemod)
        original_launch = Env.get_launch_args
        job = asyncio.create_task(_collect_async({'mobilegym':{'url':'fixture','environment_workers':2}},
            'base',[SimpleNamespace(id='task',apps=['x'])],tmp_path/'collection',1,'native',
            SimpleNamespace(model=None,generate=lambda m: None),None,0))
        await started.wait()
        job.cancel()
        with pytest.raises(asyncio.CancelledError): await job
        assert len(closed) == 1
        assert Env.get_launch_args is original_launch
    asyncio.run(check())
