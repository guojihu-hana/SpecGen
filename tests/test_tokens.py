import sys

import pytest

from specgen.llm import tokens
from specgen.llm.tokens import TokenCounter


def test_generic_counter():
    counter = TokenCounter("generic")
    assert counter.name in ("tiktoken:o200k_base", "chars/4")
    assert counter.count("") == 0
    assert counter.count("tile size 32 in shared memory") > 0


CODE = '''
__global__ void k(const float* x, float* y, int n) {
    __shared__ float tile[32][32];  // tile size 32; coalesced loads
    int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {   y[i] = x[i] > 0.f ? x[i] : 0.f; }
}
'''


@pytest.mark.parametrize("step", [1, 3, 7, 64])
def test_stream_count_equals_full_count(step):
    counter = TokenCounter("generic")
    text = CODE * 10 + "Reasoning: use warp shuffle for the reduction, then write back.\n" * 5
    stream = counter.stream()
    for i in range(0, len(text), step):
        stream.add(text[i:i + step])
    assert stream.total == counter.count(text)


def test_fallback_without_tiktoken(monkeypatch):
    tokens._load.cache_clear()
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    try:
        counter = TokenCounter("generic")
        assert counter.name == "chars/4"
        assert counter.count("abcd" * 4) == 4
    finally:
        tokens._load.cache_clear()
