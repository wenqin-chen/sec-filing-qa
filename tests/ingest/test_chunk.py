"""Tests for secqa.ingest.chunk: bounds, overlap, page attribution, ids, sections, fallback."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from secqa.core.contracts import Chunk, Page
from secqa.core.ids import chunk_id
from secqa.ingest import chunk_pages, count_tokens, detect_section, get_tokenizer
from secqa.ingest.chunk import APPROX_ENCODING, ApproxTokenizer, TiktokenTokenizer

DOC = "FIXTURE_2023_10K"
ENCODING = "cl100k_base"


def _page(num: int, text: str, doc: str = DOC) -> Page:
    return Page(doc_name=doc, page_num=num, text=text)


def _long_text(prefix: str, n: int = 300) -> str:
    return " ".join(
        f"{prefix} sentence number {i} reports revenue of {i} million dollars." for i in range(n)
    )


@pytest.fixture(autouse=True)
def _fresh_tokenizer_cache() -> Iterator[None]:
    get_tokenizer.cache_clear()
    yield
    get_tokenizer.cache_clear()


# ---------------------------------------------------------------------------------------------
# basic shape
# ---------------------------------------------------------------------------------------------


def test_short_pages_become_one_chunk_each() -> None:
    pages = [_page(1, "Item 7. Net sales were $1,577 million."), _page(2, "Item 8. Net income.")]
    chunks = chunk_pages(pages, max_tokens=512, overlap_tokens=64, encoding=ENCODING)
    assert len(chunks) == 2
    for chunk, page in zip(chunks, pages, strict=True):
        assert isinstance(chunk, Chunk)
        assert chunk.doc_name == DOC
        assert chunk.page_num == page.page_num
        assert chunk.chunk_idx == 0
        assert chunk.text == page.text
        assert chunk.n_tokens == count_tokens(page.text, ENCODING)


def test_empty_pages_are_skipped() -> None:
    chunks = chunk_pages([_page(1, ""), _page(2, "   \n "), _page(3, "text")], encoding=ENCODING)
    assert [(c.page_num, c.text) for c in chunks] == [(3, "text")]


def test_long_page_splits_within_token_bounds() -> None:
    text = _long_text("ALPHA")
    chunks = chunk_pages([_page(1, text)], max_tokens=50, overlap_tokens=10, encoding=ENCODING)
    assert len(chunks) > 10
    assert [c.chunk_idx for c in chunks] == list(range(len(chunks)))
    vocabulary = set(text.split())
    for chunk in chunks:
        assert chunk.page_num == 1
        assert 0 < chunk.n_tokens <= 50
        assert chunk.n_tokens == count_tokens(chunk.text, ENCODING)
        assert chunk.text == chunk.text.strip()
        # windows are snapped to whitespace: no word is ever cut in half
        assert set(chunk.text.split()) <= vocabulary


def test_every_sentence_survives_chunking() -> None:
    # A span is guaranteed intact in some window only when the overlap is at least as long as
    # the span (each sentence here is ~13 tokens); 64/32 leaves >= 15 tokens after snapping.
    text = _long_text("ALPHA", n=200)
    chunks = chunk_pages([_page(1, text)], max_tokens=64, overlap_tokens=32, encoding=ENCODING)
    joined = "\n".join(c.text for c in chunks)
    for i in range(200):
        assert f"sentence number {i} reports revenue of {i} million dollars." in joined


def test_consecutive_chunks_overlap() -> None:
    text = _long_text("ALPHA")
    chunks = chunk_pages([_page(1, text)], max_tokens=60, overlap_tokens=20, encoding=ENCODING)
    for previous, current in zip(chunks, chunks[1:], strict=False):
        tail = " ".join(previous.text.split()[-3:])
        assert tail in current.text, (previous.text[-80:], current.text[:80])


def test_zero_overlap_partitions_the_page_exactly() -> None:
    text = _long_text("ALPHA", n=80)
    chunks = chunk_pages([_page(1, text)], max_tokens=40, overlap_tokens=0, encoding=ENCODING)
    assert len(chunks) > 1
    assert " ".join(c.text for c in chunks) == text


def test_chunks_never_cross_pages() -> None:
    pages = [_page(1, _long_text("ALPHA", 120)), _page(2, _long_text("BETA", 120))]
    chunks = chunk_pages(pages, max_tokens=64, overlap_tokens=16, encoding=ENCODING)
    assert {c.page_num for c in chunks} == {1, 2}
    for chunk in chunks:
        marker = "ALPHA" if chunk.page_num == 1 else "BETA"
        other = "BETA" if chunk.page_num == 1 else "ALPHA"
        assert marker in chunk.text
        assert other not in chunk.text
    # chunk_idx restarts on every page
    assert [c.chunk_idx for c in chunks if c.page_num == 2][0] == 0


def test_pages_are_processed_in_page_order_regardless_of_input_order() -> None:
    pages = [_page(3, "Item 8. third"), _page(1, "Item 7. first"), _page(2, "second")]
    chunks = chunk_pages(pages, encoding=ENCODING)
    assert [c.page_num for c in chunks] == [1, 2, 3]
    assert [c.section for c in chunks] == ["Item 7", "Item 7", "Item 8"]


# ---------------------------------------------------------------------------------------------
# ids
# ---------------------------------------------------------------------------------------------


def test_chunk_ids_are_deterministic_and_content_addressed() -> None:
    pages = [_page(1, _long_text("ALPHA", 60))]
    first = chunk_pages(pages, max_tokens=40, overlap_tokens=5, encoding=ENCODING)
    second = chunk_pages(pages, max_tokens=40, overlap_tokens=5, encoding=ENCODING)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.chunk_id for c in first}) == len(first)
    for chunk in first:
        assert chunk.chunk_id == chunk_id(DOC, chunk.page_num, chunk.chunk_idx, chunk.text)
    changed = chunk_pages([_page(1, "changed text")], encoding=ENCODING)
    assert changed[0].chunk_id != first[0].chunk_id


# ---------------------------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "previous", "expected"),
    [
        ("Item 7. Management's Discussion", None, "Item 7"),
        ("ITEM 1A. RISK FACTORS", None, "Item 1A"),
        ("Item 9A: Controls and Procedures", None, "Item 9A"),
        ("Item 1A.Risk Factors", None, "Item 1A"),
        ("Part II, Item 7 Management's Discussion", "Item 6", "Item 7"),
        ("plain prose with no heading", "Item 7", "Item 7"),
        ("plain prose with no heading", None, None),
        ("", "Item 7", "Item 7"),
        ("see Item 1A. Risk Factors for details", "Item 7", "Item 7"),  # cross-reference
        ("as discussed in Item 8. of this report", "Item 7", "Item 7"),
        ("Items 1 and 2. Business and Properties", None, None),  # plural, not a heading
        ("Item 7.5 of the plan", None, None),  # sub-numbering
        ("item 8 of this filing", "Item 7", "Item 7"),  # lowercase cross-reference
        ("Item 1. Business 3 Item 1A. Risk Factors 10 Item 16. Summary 90", None, "Item 16"),
        ("Item 7. MD&A ... Item 7A. Market Risk", None, "Item 7A"),
    ],
)
def test_detect_section(text: str, previous: str | None, expected: str | None) -> None:
    assert detect_section(text, previous) == expected


def test_section_is_carried_forward_and_reset_per_document() -> None:
    pages = [
        _page(1, "Item 7. Management's discussion of results."),
        _page(2, "Continued discussion without any heading."),
        _page(3, "Item 8. Financial Statements and Supplementary Data."),
        _page(1, "Opening page of another filing.", doc="OTHER_2022_10K"),
    ]
    chunks = chunk_pages(pages, encoding=ENCODING)
    by_doc = {(c.doc_name, c.page_num): c.section for c in chunks}
    assert by_doc[(DOC, 1)] == "Item 7"
    assert by_doc[(DOC, 2)] == "Item 7"
    assert by_doc[(DOC, 3)] == "Item 8"
    assert by_doc[("OTHER_2022_10K", 1)] is None


# ---------------------------------------------------------------------------------------------
# parameters and tokenizers
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("max_tokens", "overlap"), [(0, 0), (-1, 0), (10, 10), (10, 11), (10, -1)])
def test_invalid_parameters_are_rejected(max_tokens: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_pages([_page(1, "x")], max_tokens=max_tokens, overlap_tokens=overlap)


def test_unknown_encoding_is_a_config_error() -> None:
    with pytest.raises(ValueError, match="unknown tiktoken encoding"):
        get_tokenizer("not-an-encoding")


def test_tiktoken_tokenizer_when_available() -> None:
    tok = get_tokenizer(ENCODING)
    if not isinstance(tok, TiktokenTokenizer):
        pytest.skip("tiktoken BPE file unavailable offline; fallback path covered separately")
    assert tok.name == ENCODING
    assert tok.decode(tok.encode("Net sales were $1,577 million.")) == (
        "Net sales were $1,577 million."
    )
    assert tok.starts_with_space(tok.encode("hello world")[-1])
    assert not tok.starts_with_space(tok.encode("hello world")[0])
    assert tok.encode("<|endoftext|>")  # special-token text must not raise


@pytest.mark.parametrize(
    "text",
    ["", "hello", "hello world", "  leading and trailing  ", "Net sales were $1,577 million.", "ﬁ"],
)
def test_approx_tokenizer_roundtrips(text: str) -> None:
    tok = ApproxTokenizer()
    tokens = tok.encode(text)
    assert tok.decode(tokens) == text
    assert all(len(str(t).strip()) <= 4 for t in tokens)


def test_falls_back_to_approx_tokenizer_when_bpe_cannot_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tiktoken

    def _boom(name: str) -> object:
        raise OSError("simulated: no network and no cache")

    monkeypatch.setattr(tiktoken, "get_encoding", _boom)
    tok = get_tokenizer(ENCODING)
    assert isinstance(tok, ApproxTokenizer)
    chunks = chunk_pages([_page(1, _long_text("ALPHA", 50))], max_tokens=30, overlap_tokens=5)
    assert len(chunks) > 1
    assert all(c.n_tokens <= 30 for c in chunks)


def test_approx_encoding_can_be_selected_explicitly() -> None:
    assert isinstance(get_tokenizer(APPROX_ENCODING), ApproxTokenizer)
    text = _long_text("ALPHA", 40)
    chunks = chunk_pages(
        [_page(1, text)], max_tokens=25, overlap_tokens=5, encoding=APPROX_ENCODING
    )
    assert all(0 < c.n_tokens <= 25 for c in chunks)
    assert all(set(c.text.split()) <= set(text.split()) for c in chunks)
    assert count_tokens("abcdefgh", APPROX_ENCODING) == 2
