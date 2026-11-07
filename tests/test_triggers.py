import re
from pathlib import Path

import pytest

from specgen.control.triggers import (
    FALLBACK, TriggerConfig, TriggerConfigError, TriggerGroup, TriggerMonitor, load_trigger_config, parse_trigger_config,
)

REPO_TRIGGERS = Path(__file__).resolve().parents[1] / "configs" / "triggers.yaml"


def config(after=100, fallback=1000, gap=200, cap=20, **groups):
    groups = groups or {"smem": ([r"\bshared memory\b"], None), "code": (["__global__"], 10)}
    return TriggerConfig(
        groups=tuple(TriggerGroup(name, tuple(re.compile(p, re.IGNORECASE) for p in pats), after_tokens)
                     for name, (pats, after_tokens) in groups.items()),
        after_tokens=after, fallback_tokens=fallback, min_gap_tokens=gap, max_triggers=cap,
    )


def feed(monitor, steps):
    """steps: (text, total_tokens) pairs; returns (step index, trigger) for every trigger."""
    return [(i, t) for i, (text, tokens) in enumerate(steps) for t in monitor.feed(text, tokens)]


def test_keyword_fires_after_tokens():
    fired = feed(TriggerMonitor(config()), [
        ("we could use Shared Memory here", 50), ("more", 120), ("and more", 151), ("x", 160),
    ])
    assert [(i, t.reason, t.keyword, t.matched_tokens, t.fired_tokens) for i, t in fired] == [
        (2, "smem", "Shared Memory", 50, 151)]
    assert fired[0][1].index == 1 and fired[0][1].prefix_chars == len("we could use Shared Memory here" "more" "and more")


def test_armed_trigger_ignores_new_matches_and_respects_gap():
    monitor = TriggerMonitor(config(after=100, gap=300))
    fired = feed(monitor, [
        ("shared memory", 10), ("__global__ shared memory", 50), ("x", 110),  # fires (armed at 10)
        ("__global__", 120),  # armed again: after=10 but the gap pushes it to 110 + 300
        ("y", 200), ("z", 409), ("w", 410),
    ])
    assert [(i, t.reason, t.fired_tokens) for i, t in fired] == [(2, "smem", 110), (6, "code", 410)]


def test_keyword_split_across_deltas():
    fired = feed(TriggerMonitor(config(after=0)), [("use shared me", 5), ("mory tiles", 9)])
    assert [(i, t.keyword) for i, t in fired] == [(1, "shared memory")]


def test_fallback_and_cap():
    monitor = TriggerMonitor(config(fallback=100, cap=2))
    fired = feed(monitor, [("a", 50), ("b", 100), ("c", 150), ("d", 200), ("e", 300), ("f", 400)])
    assert [(i, t.reason, t.fired_tokens) for i, t in fired] == [(1, FALLBACK, 100), (3, FALLBACK, 200)]
    assert monitor.fired == 2


def test_refunded_triggers_do_not_count_against_the_cap():
    monitor = TriggerMonitor(config(fallback=100, cap=1))
    assert [t.index for t in monitor.feed("a", 100)] == [1]
    monitor.refund()  # launched nothing
    assert [t.index for t in monitor.feed("b", 200)] == [2]
    assert monitor.feed("c", 400) == [] and monitor.fired == 2 and monitor.counted == 1


def test_stop_ends_triggering():
    monitor = TriggerMonitor(config(after=0))
    monitor.stop()
    assert monitor.feed("shared memory", 5000) == []


def test_old_matches_do_not_rearm():
    monitor = TriggerMonitor(config(after=0, gap=0))
    assert len(monitor.feed("shared memory", 10)) == 1
    assert monitor.feed(" and then", 20) == []  # the earlier match is still in the window


def test_repository_config_loads():
    cfg = load_trigger_config(REPO_TRIGGERS)
    assert cfg.after_tokens == 500 and cfg.fallback_tokens == 2000 and cfg.min_gap_tokens == 500
    names = {g.name for g in cfg.groups}
    assert {"memory_hierarchy", "tiling", "tensor_core", "code_start"} <= names
    code = next(g for g in cfg.groups if g.name == "code_start")
    assert code.after_tokens == 300
    monitor = TriggerMonitor(cfg)
    assert monitor.feed("Let me think about the Tile Size and bank conflicts", 10) == []
    assert monitor._armed.reason == "tiling"  # earliest match wins
    overridden = cfg.with_overrides(after_tokens=50, fallback_tokens=900)
    assert overridden.after_tokens == 50 and overridden.fallback_tokens == 900
    assert all(g.after_tokens is None for g in overridden.groups)


@pytest.mark.parametrize("raw", [
    {"groups": [{"name": "x"}]},
    {"groups": [{"name": "x", "patterns": ["("]}]},
    {"fallback_tokens": 0},
    {"groups": [{"name": "x", "patterns": ["a"], "after_tokens": -1}]},
])
def test_bad_configs(raw):
    with pytest.raises(TriggerConfigError):
        parse_trigger_config(raw)
