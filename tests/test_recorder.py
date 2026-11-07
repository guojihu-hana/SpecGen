import json

import pytest

from specgen.telemetry.recorder import PerfRecorder


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_records_and_statistics(tmp_path):
    rec = PerfRecorder(tmp_path)
    rec.event("custom", a=1)
    rec.kernel("t", "k", iteration=0, source="main")
    rec.kernel("t", "k", status="success")
    rec.observe_job("validate", {"ok": True, "exec_s": 2.0, "wait_s": 0.5})
    rec.observe_job("validate", {"ok": False, "exec_s": 4.0})
    rec.finish_kernel("t", "k")
    rec.finish_iteration({"task": "t", "iteration": 0, "end_reason": "main_finished", "selected": "k"})
    rec.close()

    kernels = lines(tmp_path / "kernels.jsonl")
    assert kernels == [{"task_id": "t", "kernel_id": "k", "iteration": 0, "source": "main", "status": "success"}]
    assert [e["type"] for e in lines(tmp_path / "events.jsonl")] == ["custom", "kernel_finished", "iteration_finished"]
    assert lines(tmp_path / "iterations.jsonl")[0]["selected"] == "k"
    stats = rec.stats()["validate"]
    assert stats["count"] == 2 and stats["pass_rate"] == 0.5
    assert stats["exec_s"] == pytest.approx(0.3 * 4.0 + 0.7 * 2.0)
    assert rec.kernels == {}
