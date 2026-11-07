import json

import pytest
from torch.utils import cpp_extension

from specgen.evaluation import fastbuild


def test_strip_heavy_includes():
    source = '#include <torch/extension.h>\n  # include "torch/torch.h"\n#include <torch/types.h>\n__global__ void k() {}\n'
    assert fastbuild.strip_heavy_includes(source) == '\n\n#include <torch/types.h>\n__global__ void k() {}\n'
    assert fastbuild.strip_heavy_includes(["#include <torch/all.h>\nx", "y"]) == ["\nx", "y"]
    assert fastbuild.strip_heavy_includes(None) is None


def test_tiers():
    tiers = fastbuild.tier_overrides({"cpp_sources": "decl;", "cuda_sources": "#include <torch/extension.h>\n"
                                                                          "#include <ATen/ATen.h>\nk"})
    assert [t for t, _ in tiers] == ["slim", "trimmed", "original"]
    slim = tiers[0][1]
    assert slim["no_implicit_headers"] is True and slim["cpp_sources"] == ["#include <torch/extension.h>", "decl;"]
    assert slim["cuda_sources"] == [fastbuild.SLIM_PRELUDE, "\n\nk"]
    assert tiers[1][1] == {"cuda_sources": "\n#include <ATen/ATen.h>\nk"}
    assert [t for t, _ in fastbuild.tier_overrides({"cuda_sources": "k"})] == ["bare", "slim", "original"]
    assert [t for t, _ in fastbuild.tier_overrides({"cuda_sources": "k", "no_implicit_headers": True})] == ["original"]


def test_torch_free_sources_build_bare_first():
    source = '__global__ void k(float* x) { printf("at::x"); }  // torch::Tensor is in the cpp sources\n/* at:: */'
    assert fastbuild.torch_free(source) and fastbuild.torch_free([source, "void launch(float* x, cudaStream_t s);"])
    assert fastbuild.torch_free("auto v = flat::x;")
    for uses_torch in ("void f(torch::Tensor x);", "at::cuda::getCurrentCUDAStream()", "TORCH_CHECK(x);",
                       "AT_DISPATCH_FLOATING_TYPES(t, \"k\", [&] {});", "#include <ATen/cuda/CUDAContext.h>",
                       "void f(Tensor x);", "c10::Half h;"):
        assert not fastbuild.torch_free(["__global__ void k() {}", uses_torch]), uses_torch
    assert not fastbuild.torch_free("") and not fastbuild.torch_free(None)
    tiers = fastbuild.tier_overrides({"cpp_sources": "wrappers;", "cuda_sources": source})
    assert [t for t, _ in tiers] == ["bare", "slim", "original"]
    bare = tiers[0][1]
    assert bare == {"cuda_sources": [fastbuild.BARE_PRELUDE, source], "no_implicit_headers": True,
                    "cpp_sources": ["#include <torch/extension.h>", "wrappers;"]}
    assert "torch" not in fastbuild.BARE_PRELUDE and "ATen" not in fastbuild.BARE_PRELUDE


class FakeLoadInline:
    """Stands in for load_inline: the slim prelude lacks torch::nn, torch::nn needs <torch/extension.h>, and
    NEEDS_ATEN only builds with ATen's headers."""

    def __init__(self):
        self.calls = []

    def __call__(self, name, cpp_sources, cuda_sources=None, functions=None, extra_cflags=None, build_directory=None,
                 with_cuda=None, no_implicit_headers=False):
        text = "".join(cuda_sources) if isinstance(cuda_sources, list) else (cuda_sources or "")
        bare = text.startswith(fastbuild.BARE_PRELUDE)
        self.calls.append(("bare" if bare else "slim" if no_implicit_headers else "implicit", text))
        if "BROKEN" in text or ("torch::nn" in text and "torch/extension.h" not in text) or (bare and "NEEDS_ATEN" in text):
            raise RuntimeError(f"Error building extension '{name}'")
        return f"module:{name}"


@pytest.fixture
def fake(monkeypatch, tmp_path):
    fake = FakeLoadInline()
    monkeypatch.setattr(cpp_extension, "load_inline", fake)
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path))
    assert fastbuild.install(pch=False) and not fastbuild.install(pch=False)  # idempotent
    return fake, tmp_path


def marker(root):
    return json.loads((root / fastbuild.MARKER).read_text())


def test_slim_build_is_used_and_recorded(fake):
    fake_load, root = fake
    assert cpp_extension.load_inline("ext", "decl;", cuda_sources="#include <torch/extension.h>\nk",
                                     functions=["f"]) == "module:ext"
    assert [kind for kind, _ in fake_load.calls] == ["slim"]
    assert fake_load.calls[0][1].startswith(fastbuild.SLIM_PRELUDE) and "torch/extension.h" not in fake_load.calls[0][1]
    assert marker(root) == {"ext": "slim"}


def test_fallback_through_the_tiers_and_later_loads_reuse_the_choice(fake):
    fake_load, root = fake
    source = "#include <torch/extension.h>\ntorch::nn::functional::relu(x);"
    cpp_extension.JIT_EXTENSION_VERSIONER.entries["ext2"] = "stale"
    assert cpp_extension.load_inline(name="ext2", cpp_sources="decl;", cuda_sources=source) == "module:ext2"
    assert [kind for kind, _ in fake_load.calls] == ["slim", "implicit", "implicit"]
    assert fake_load.calls[1][1] == "\ntorch::nn::functional::relu(x);" and fake_load.calls[2][1] == source
    assert "ext2" not in cpp_extension.JIT_EXTENSION_VERSIONER.entries  # same name and build dir on the retries
    assert marker(root) == {"ext2": "original"}
    cpp_extension.load_inline(name="ext2", cpp_sources="decl;", cuda_sources=source)  # e.g. the validation process
    assert fake_load.calls[-1] == ("implicit", source) and len(fake_load.calls) == 4


def test_bare_build_is_used_and_recorded_and_falls_back_to_slim(fake):
    fake_load, root = fake
    assert cpp_extension.load_inline("ext7", "wrappers;", cuda_sources="__global__ void k() {}") == "module:ext7"
    assert [kind for kind, _ in fake_load.calls] == ["bare"] and marker(root) == {"ext7": "bare"}
    assert cpp_extension.load_inline("ext8", "wrappers;", cuda_sources="NEEDS_ATEN k") == "module:ext8"
    assert [kind for kind, _ in fake_load.calls] == ["bare", "bare", "slim"]
    assert marker(root) == {"ext7": "bare", "ext8": "slim"}
    cpp_extension.load_inline("ext7", "wrappers;", cuda_sources="__global__ void k() {}")  # e.g. the validation process
    assert fake_load.calls[-1][0] == "bare" and len(fake_load.calls) == 4


def test_broken_code_raises_the_error_of_the_source_as_written(fake):
    fake_load, root = fake
    with pytest.raises(RuntimeError, match="Error building extension 'bad'"):
        cpp_extension.load_inline("bad", "decl;", cuda_sources="BROKEN")
    assert [kind for kind, _ in fake_load.calls] == ["bare", "slim", "implicit"]
    assert not (root / fastbuild.MARKER).exists()


def test_untouched_calls_go_straight_through(fake):
    fake_load, _ = fake
    cpp_extension.load_inline("ext4", "only cpp")
    cpp_extension.load_inline("ext5", "decl;", cuda_sources="k", no_implicit_headers=True)
    assert fake_load.calls == [("implicit", ""), ("slim", "k")]


def test_other_errors_are_not_retried(fake, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("CUDA driver error")

    monkeypatch.setattr(cpp_extension, "load_inline", broken)
    fastbuild.install(pch=False)
    with pytest.raises(RuntimeError, match="driver"):
        cpp_extension.load_inline("ext6", "decl;", cuda_sources="#include <torch/extension.h>\nk")


def test_disabled_by_environment(monkeypatch):
    monkeypatch.setenv(fastbuild.ENV_FLAG, "0")
    monkeypatch.setenv(fastbuild.PCH_FLAG, "0")
    assert fastbuild.install_if_enabled() is False


def test_main_cpp_flags_match_torch_without_the_extension_name():
    compiler, flags = fastbuild.main_cpp_flags(["-O3"], [], True)
    assert compiler and "-fPIC" in flags and "-DTORCH_API_INCLUDE_EXTENSION_H" in flags
    assert any(f.startswith("-std=c++") for f in flags)  # torch's standard: c++17 before 2.13, c++20 since
    assert flags[-1] == "-O3" and not any(f.startswith("-DTORCH_EXTENSION_NAME") for f in flags)
    assert "-isystem" in flags


class FakeCompiler:
    def __init__(self, uses_pch):
        self.uses_pch, self.commands = uses_pch, []

    def __call__(self, command):
        import subprocess

        self.commands.append(command)
        if "-x" in command:  # building the PCH
            open(command[command.index("-o") + 1], "w").close()
            return subprocess.CompletedProcess(command, 0, "", "")
        gch = command[command.index("-include") + 1] + ".gch"
        return subprocess.CompletedProcess(command, 0, "", f"! {gch}\n" if self.uses_pch else ". other.h\n")


@pytest.mark.parametrize("uses_pch", [True, False])
def test_pch_is_built_once_and_checked(monkeypatch, tmp_path, uses_pch):
    monkeypatch.setenv(fastbuild.PCH_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(fastbuild, "main_cpp_flags", lambda *a: ("c++", ["-std=c++17", "-fPIC"]))
    monkeypatch.setattr(fastbuild, "_compiler_version", lambda compiler: "test-cc 1")
    compiler = FakeCompiler(uses_pch)
    monkeypatch.setattr(fastbuild, "_run", compiler)
    first = fastbuild.pch_include([], [], True)
    second = fastbuild.pch_include([], [], True)
    directory = next(tmp_path.iterdir())
    if uses_pch:
        assert first == second == ["-include", str(directory / "specgen_torch_ext.h")]
        assert (directory / "ok").read_text() == "-std=c++17 -fPIC"
    else:  # the compiler ignored the PCH: never use it, never rebuild it
        assert first == second == [] and (directory / "failed").is_file()
    assert len(compiler.commands) == 2  # one build and one check, the second call reused the result


def test_pch_failures_never_break_a_build(monkeypatch, tmp_path):
    monkeypatch.setenv(fastbuild.PCH_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(fastbuild, "main_cpp_flags", lambda *a: (_ for _ in ()).throw(RuntimeError("torch changed")))
    assert fastbuild.pch_include([], [], True) == []
