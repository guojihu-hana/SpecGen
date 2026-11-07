from specgen.generation.extract import extract_python_module


def test_last_python_fence_wins():
    text = "draft:\n```python\nx = 1\n```\nfinal:\n```py\nimport torch\nclass Model: pass\n```\n"
    assert extract_python_module(text) == "import torch\nclass Model: pass"


def test_generic_fence_needs_module_shape():
    assert extract_python_module("```\nimport torch\nclass Model: pass\n```") == "import torch\nclass Model: pass"
    assert extract_python_module("```\nls -la\n```") is None


def test_nothing_to_extract():
    assert extract_python_module("") is None
    assert extract_python_module("no code here") is None
