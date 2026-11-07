import pytest

from specgen.llm.profiles import ProfileError, load_profiles


def test_repository_models_file():
    profiles = load_profiles()
    assert set(profiles) == {"GLM-5.1-FP8", "DeepSeek-V4-Pro", "DeepSeek-V4-Flash", "MiniMax-M3"}
    for profile in profiles.values():
        assert profile.tokenizer == "generic"
        assert set(profile.endpoints) == {"selfhost", "api"}
        assert profile.endpoints["selfhost"].backend == "vllm"


def test_deepseek_flash_api_modes():
    profile = load_profiles()["DeepSeek-V4-Flash"]
    assert profile.reasoning == "field" and profile.reasoning_field == "reasoning_content"
    resolved = profile.endpoint("api").resolve({"SPECGEN_DEEPSEEK_API_KEY": "k"})
    assert (resolved.base_url, resolved.model, resolved.api_key) == ("https://api.deepseek.com", "deepseek-flash", "k")
    assert profile.mode_params("main", resolved) == {"thinking": {"type": "enabled"}, "reasoning_effort": "max"}
    assert profile.mode_params("spec", resolved) == {"thinking": {"type": "disabled"}}
    assert (resolved.prefill_style, resolved.prefill_base_url) == ("deepseek", "https://api.deepseek.com/beta")
    assert profile.endpoint("selfhost").prefill_style is None


def test_endpoint_modes_override_profile_modes(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(
        "m:\n"
        "  modes: {main: {a: 1, b: 1}, spec: {c: 1}}\n"
        "  endpoints:\n"
        "    api: {backend: api, base_url: 'http://x/v1', model: m, modes: {main: {b: 2}}}\n"
    )
    profile = load_profiles(path)["m"]
    resolved = profile.endpoint("api").resolve({})
    assert profile.mode_params("main", resolved) == {"a": 1, "b": 2}
    assert profile.mode_params("spec", resolved) == {"c": 1}


def test_placeholders_refuse_to_resolve():
    endpoint = load_profiles()["GLM-5.1-FP8"].endpoint("selfhost")
    with pytest.raises(ProfileError, match="SPECGEN_GLM_SELFHOST_URL"):
        endpoint.resolve({})
    with pytest.raises(ProfileError, match="placeholder"):
        endpoint.resolve({"SPECGEN_GLM_SELFHOST_URL": "http://127.0.0.1:8000/v1"})


def test_resolve_from_environment(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(
        "m:\n"
        "  reasoning: inline\n"
        "  inline_starts_in_reasoning: true\n"
        "  sampling: {temperature: 0.2}\n"
        "  endpoints:\n"
        "    api: {backend: api, base_url: '${URL}', model: real-model, api_key_env: KEY}\n"
    )
    profile = load_profiles(path)["m"]
    assert profile.reasoning == "inline" and profile.inline_starts_in_reasoning
    assert profile.sampling.temperature == 0.2 and profile.sampling.max_tokens == 32768
    resolved = profile.endpoint("api").resolve({"URL": "https://api.example/v1", "KEY": "secret"})
    assert (resolved.base_url, resolved.model, resolved.api_key) == ("https://api.example/v1", "real-model", "secret")
    assert profile.endpoint("api").resolve({"URL": "https://x/v1"}).api_key == "EMPTY"
    with pytest.raises(ProfileError, match="no 'selfhost' endpoint"):
        profile.endpoint("selfhost")


@pytest.mark.parametrize("body", [
    "m: {reasoning: sometimes}\n",
    "m: {think_tags: ['<think>']}\n",
    "m: {sampling: {temp: 1}}\n",
    "m: {endpoints: {x: {backend: grpc}}}\n",
    "m: {modes: {draft: {}}}\n",
    "m: {endpoints: {x: {modes: {main: 1}}}}\n",
    "m: {endpoints: {x: {prefill: {style: openai}}}}\n",
    "m: {endpoints: {x: {options: [1]}}}\n",
])
def test_invalid_profiles(tmp_path, body):
    path = tmp_path / "models.yaml"
    path.write_text(body)
    with pytest.raises(ProfileError):
        load_profiles(path)
