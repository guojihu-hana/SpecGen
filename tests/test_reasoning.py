from specgen.llm.reasoning import CONTENT, REASONING, InlineThinkSplitter


def split(pieces, **kwargs):
    splitter = InlineThinkSplitter(**kwargs)
    out = []
    for piece in pieces:
        out += splitter.feed(piece)
    out += splitter.flush()
    merged = []
    for kind, text in out:
        if merged and merged[-1][0] == kind:
            merged[-1] = (kind, merged[-1][1] + text)
        else:
            merged.append((kind, text))
    return merged


def test_tags_split_across_chunks():
    assert split(["<thi", "nk>plan", " tiles</th", "ink>\ncode"]) == [(REASONING, "plan tiles"), (CONTENT, "\ncode")]


def test_no_tags_is_content():
    assert split(["hello ", "world"]) == [(CONTENT, "hello world")]


def test_starts_in_reasoning():
    assert split(["plan", "</think>", "code"], starts_in_reasoning=True) == [(REASONING, "plan"), (CONTENT, "code")]


def test_partial_tag_at_end_is_flushed():
    assert split(["text <th"]) == [(CONTENT, "text <th")]


def test_only_the_first_block_is_reasoning():
    assert split(["<think>a</think>b<think>c"]) == [(REASONING, "a"), (CONTENT, "b<think>c")]


def test_unclosed_reasoning():
    assert split(["<think>abc"]) == [(REASONING, "abc")]
