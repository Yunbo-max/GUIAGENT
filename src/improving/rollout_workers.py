"""Overlap independent environments while retaining one owner of model inference."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor


async def ordered_workers(items, workers: int, worker):
    if type(workers) is not int or workers not in (1, 2, 4, 8, 16, 24):
        raise ValueError('environment_workers must be 1, 2, 4, 8, 16 or 24')
    if not items:
        return []
    slots = list(enumerate(items))
    jobs = [asyncio.create_task(worker(i, slots[i::workers]))
            for i in range(min(workers, len(items)))]
    try:
        results = await asyncio.gather(*jobs)
    except BaseException:
        for job in jobs:
            if not job.done() and not job.cancelling():
                job.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
        raise
    flattened = sorted((row for result in results for row in result), key=lambda row: row[0])
    if [row[0] for row in flattened] != list(range(len(items))):
        raise RuntimeError('Worker outputs have missing or duplicate collection slots')
    return [row[1] for row in flattened]


class SerialInference:
    """A single model thread lets browser I/O proceed without concurrent forwards."""
    def __init__(self, generate):
        self.generate = generate
        self.executor = None

    async def __aenter__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='policy')
        return self

    async def __call__(self, messages):
        if self.executor is None:
            raise RuntimeError('Inference service is not open')
        return await asyncio.get_running_loop().run_in_executor(self.executor, self.generate, messages)

    async def __aexit__(self, *exc):
        # A running CUDA forward cannot be cancelled. Finish it before restoring
        # spectral weights or freeing the model; queued requests can be discarded.
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.executor = None
