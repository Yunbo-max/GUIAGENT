import asyncio
import threading
import time

import pytest

from improving.rollout_workers import ordered_workers, SerialInference


def test_workers_overlap_but_return_original_order():
    async def check():
        active = 0
        peak = 0
        async def worker(index, items):
            nonlocal active, peak
            output = []
            for slot, value in items:
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(.01 * (4 - value))
                active -= 1
                output.append((slot, value * 10))
            return output
        assert await ordered_workers([0, 1, 2, 3], 2, worker) == [0, 10, 20, 30]
        assert peak == 2
    asyncio.run(check())


def test_failure_cancels_other_workers_and_runs_cleanup():
    async def check():
        cleaned = []
        started = asyncio.Event()
        async def worker(index, items):
            try:
                if index == 0:
                    await started.wait()
                    raise RuntimeError('native verifier failed')
                started.set()
                await asyncio.sleep(10)
            finally:
                cleaned.append(index)
        with pytest.raises(RuntimeError, match='native verifier'):
            await ordered_workers([0, 1], 2, worker)
        assert sorted(cleaned) == [0, 1]
    asyncio.run(check())


def test_external_cancellation_finishes_slow_worker_cleanup():
    async def check():
        started = asyncio.Event()
        closed = []
        async def worker(index, items):
            try:
                if index == 1:
                    started.set()
                await asyncio.sleep(10)
            finally:
                await asyncio.sleep(.01 if index == 0 else .08)
                closed.append(index)
        job = asyncio.create_task(ordered_workers([0, 1], 2, worker))
        await started.wait()
        job.cancel()
        with pytest.raises(asyncio.CancelledError):
            await job
        assert sorted(closed) == [0, 1]
    asyncio.run(check())


def test_inference_has_one_model_entry_and_leaves_event_loop_available():
    active = 0
    peak = 0
    threads = set()
    def generate(value):
        nonlocal active, peak
        threads.add(threading.get_ident())
        active += 1
        peak = max(peak, active)
        time.sleep(.03)
        active -= 1
        return value + 1
    async def check():
        ticks = []
        async def ticker():
            await asyncio.sleep(.01)
            ticks.append(True)
        async with SerialInference(generate) as inference:
            results = await asyncio.gather(inference(1), inference(2), ticker())
        assert results[:2] == [2, 3] and ticks
    asyncio.run(check())
    assert peak == 1 and len(threads) == 1


@pytest.mark.parametrize('workers', [0, -1, True, 3])
def test_invalid_workers_fail_before_running(workers):
    with pytest.raises(ValueError):
        asyncio.run(ordered_workers([1], workers, None))


def test_twenty_four_workers_cover_each_slot_once():
    async def check():
        entered = set()
        async def worker(index, items):
            entered.add(index)
            await asyncio.sleep(.001)
            return [(slot, value) for slot, value in items]
        assert await ordered_workers(list(range(48)), 24, worker) == list(range(48))
        assert entered == set(range(24))
    asyncio.run(check())
