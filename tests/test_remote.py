import asyncio
import json
import threading
import time

import pytest
from conftest import CORRECT_RELU, ModeBackend, answer, make_context

from specgen.control.loop import IterationLoop
from specgen.evaluation.executor import GpuSlot, LocalEvaluator
from specgen.evaluation.jobs import EvalJob
from specgen.remote.client import EvaluatorRouter, RemoteEvaluator, WorkerError, fetch_worker_info
from specgen.remote.server import WorkerState, make_server
from specgen.resources.pool import make_devices
from specgen.tasks.base import KernelBenchTask

SLOW_KERNEL = "import time\ntime.sleep(60)\n" + CORRECT_RELU


@pytest.fixture
def worker(tmp_path):
    state = WorkerState(tmp_path / "worker", gpus=[], cuda_arch={}, gpu_type="H100", timeout_s=120, token="t0k")
    server = make_server(state, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", state
    finally:
        server.shutdown()
        server.server_close()


def kernel_dir(tmp_path, code, name="k1"):
    work = tmp_path / "local" / "run" / "task" / "iter_000" / "kernels" / name
    work.mkdir(parents=True)
    (work / "kernel.py").write_text(code)
    return work


def test_info_and_auth(worker):
    url, _ = worker
    info = fetch_worker_info(url, "t0k")
    assert info["gpus"] == [] and info["gpu_type"] == "H100"
    with pytest.raises(WorkerError, match="401"):
        fetch_worker_info(url, "wrong")
    with pytest.raises(WorkerError, match="unreachable"):
        fetch_worker_info("http://127.0.0.1:9", None, timeout_s=2)


def test_remote_validation_round_trip(worker, tmp_path, relu_task):
    url, state = worker
    work = kernel_dir(tmp_path, CORRECT_RELU)
    job = EvalJob("validate", "kernelbench", str(relu_task), str(work / "kernel.py"), str(work),
                  {"seed": 0, "trials": 2, "ref_cache": str(tmp_path / "local_ref")})

    async def go():
        remote = RemoteEvaluator(url, token="t0k", timeout_s=120)
        try:
            return await remote.run(job, GpuSlot(None, "validator", 1, url))
        finally:
            await remote.aclose()

    result = asyncio.run(go())
    assert result["ok"] and result["worker"] == url and result["slot"] == f"validator:cpu#1@{url}"
    assert (state.root / "work" / "run" / "task" / "iter_000" / "kernels" / "k1" / "kernel.py").is_file()
    assert list((state.root / "ref_cache").iterdir())  # the cache path is the worker's own
    assert not (tmp_path / "local_ref").exists()


def test_remote_errors_are_results(worker, tmp_path, relu_task):
    url, _ = worker
    work = kernel_dir(tmp_path, CORRECT_RELU)
    job = EvalJob("validate", "kernelbench", str(relu_task), str(work / "kernel.py"), str(work), {})

    async def go(token, gpu):
        remote = RemoteEvaluator(url, token=token, timeout_s=60)
        try:
            return await remote.run(job, GpuSlot(gpu, "validator", 0, url))
        finally:
            await remote.aclose()

    unauthorized = asyncio.run(go("bad", None))
    assert not unauthorized["ok"] and unauthorized["status"] == "worker_error" and "401" in unauthorized["error"]
    wrong_gpu = asyncio.run(go("t0k", "7"))
    assert wrong_gpu["status"] == "worker_error" and "not served" in wrong_gpu["error"]


def test_cancelling_a_remote_job_kills_it(worker, tmp_path, relu_task):
    url, state = worker
    work = kernel_dir(tmp_path, SLOW_KERNEL)
    job = EvalJob("validate", "kernelbench", str(relu_task), str(work / "kernel.py"), str(work), {})

    async def go():
        remote = RemoteEvaluator(url, token="t0k", timeout_s=120)
        task = asyncio.create_task(remote.run(job, GpuSlot(None, "validator", 0, url)))
        for _ in range(100):
            await asyncio.sleep(0.05)
            if state._running:
                break
        t0 = time.monotonic()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        for _ in range(100):
            if not state._running:
                break
            await asyncio.sleep(0.05)
        await remote.aclose()
        return time.monotonic() - t0, dict(state._running)

    elapsed, running = asyncio.run(go())
    assert running == {} and elapsed < 10


def test_iteration_on_remote_devices(worker, tmp_path, relu_task):
    url, _ = worker
    remote = RemoteEvaluator(url, token="t0k", timeout_s=120)
    router = EvaluatorRouter(LocalEvaluator(timeout_s=120), {url: remote})
    ctx = make_context(tmp_path / "out", ModeBackend(main=[answer(CORRECT_RELU)]), evaluator=router,
                       devices=make_devices([], host=url), tasks=[relu_task])

    async def go():
        await ctx.pool.start()
        try:
            return await IterationLoop(KernelBenchTask.from_path(relu_task), ctx).run()
        finally:
            await ctx.pool.close()
            await router.aclose()
            ctx.recorder.close()

    history = asyncio.run(go())
    assert history.outcomes[0].status == "success"
    record = json.loads((ctx.layout.run_dir / "kernels.jsonl").read_text().splitlines()[0])
    assert record["validation"]["slot"].endswith(f"@{url}")
