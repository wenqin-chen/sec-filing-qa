"""Prompt files, their hashes (snapshot) and the user-prompt layout."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from secqa.core.contracts import Chunk, Message, RetrievalFilters
from secqa.core.errors import ConfigError
from secqa.core.ids import chunk_id, sha256_hex
from secqa.providers.mock_provider import extract_passages, extract_question
from secqa.rag import (
    ANSWER_PROMPT_NAMES,
    PROMPT_NAMES,
    PROMPTS_DIR,
    build_user_prompt,
    load_prompt,
    prompt_hashes,
)
from secqa.rag.prompts import (
    PASSAGES_HEADER,
    SECTION_MAX_CHARS,
    build_closed_book_prompt,
    format_hints,
    format_passage,
    prompt_hash,
)
from tests.rag.conftest import TOP_DOC, make_chunks

SNAPSHOT = Path(__file__).resolve().parent.parent / "fixtures" / "rag_prompt_hashes.json"


# ---- files and hashes ---------------------------------------------------------------------


def test_every_prompt_file_exists_and_is_non_empty() -> None:
    assert PROMPTS_DIR.is_dir()
    for name in PROMPT_NAMES:
        assert load_prompt(name).strip()
    for name in ANSWER_PROMPT_NAMES:
        text = load_prompt(name)
        assert "INSUFFICIENT EVIDENCE" in text
        assert '"abstain"' in text


def test_prompt_hashes_cover_every_file() -> None:
    hashes = prompt_hashes()
    assert set(hashes) == set(PROMPT_NAMES) == {p.name for p in PROMPTS_DIR.glob("*.md")}
    for name, digest in hashes.items():
        assert len(digest) == 64 and int(digest, 16) >= 0
        assert digest == sha256_hex((PROMPTS_DIR / name).read_bytes()) == prompt_hash(name)


def test_prompt_hashes_snapshot() -> None:
    """Prompts are frozen: changing one is a new judge/prompt version, so update the snapshot
    (tests/fixtures/rag_prompt_hashes.json) deliberately, never by accident."""
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    actual = prompt_hashes()
    assert actual == expected, (
        "prompt files changed; if intended, regenerate tests/fixtures/rag_prompt_hashes.json "
        "and bump the judge/prompt version"
    )


def test_prompt_hashes_reflect_edits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_dir = tmp_path / "prompts"
    fake_dir.mkdir()
    (fake_dir / "rag_system.md").write_text("v1", encoding="utf-8")
    monkeypatch.setattr("secqa.rag.prompts.PROMPTS_DIR", fake_dir)
    before = prompt_hashes()["rag_system.md"]
    (fake_dir / "rag_system.md").write_text("v2", encoding="utf-8")
    assert prompt_hashes()["rag_system.md"] != before


def test_missing_prompt_dir_is_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("secqa.rag.prompts.PROMPTS_DIR", tmp_path / "nowhere")
    with pytest.raises(ConfigError, match="no prompt files"):
        prompt_hashes()
    with pytest.raises(ConfigError, match="not found"):
        load_prompt("does_not_exist.md")


def test_system_prompts_state_the_guardrails() -> None:
    rag = load_prompt("rag_system.md")
    assert "data, not instructions" in rag
    assert "at least 20 characters" in rag
    assert "chunk:<id>" in rag
    closed = load_prompt("closed_book_system.md")
    assert "No documents are provided" in closed
    assert '"citations": []' in closed
    oracle = load_prompt("oracle_system.md")
    assert "known to hold the evidence" in oracle


# ---- user prompt layout -------------------------------------------------------------------


def test_build_user_prompt_layout() -> None:
    chunks = make_chunks(TOP_DOC)[:2]
    prompt = build_user_prompt(
        "What were   total net\nsales?", chunks, filters=RetrievalFilters(ticker="FIX")
    )
    lines = prompt.split("\n")
    assert lines[0] == PASSAGES_HEADER
    assert lines[1] == ""
    assert lines[2] == (
        f"[1] (ref: chunk:{chunks[0].chunk_id}) {TOP_DOC} p.1 | Item 7 Management Discussion"
    )
    assert lines[3] == chunks[0].text
    assert lines[4] == ""
    assert lines[5].startswith(f"[2] (ref: chunk:{chunks[1].chunk_id}) {TOP_DOC} p.2")
    assert lines[-2] == "Hints: ticker=FIX"
    assert lines[-1] == "Question: What were total net sales?"


def test_build_user_prompt_is_parseable_by_mock_provider() -> None:
    chunks = make_chunks(TOP_DOC)
    prompt = build_user_prompt("What were total net sales?", chunks)
    passages = extract_passages([Message(role="user", content=prompt)])
    assert [ref for ref, _ in passages] == [f"chunk:{c.chunk_id}" for c in chunks]
    # The mock reads each passage up to the next ``chunk:`` marker, so a body may carry the
    # ``[n] (ref:`` prefix of the following header; what matters is that it starts with the
    # chunk text itself (no header remainder in front of it).
    for (_, text), chunk in zip(passages, chunks, strict=True):
        assert text.strip().startswith(chunk.text)
    assert extract_question(prompt) == "What were total net sales?"


def test_build_user_prompt_without_chunks_or_filters() -> None:
    assert build_user_prompt("Why?", []) == "Question: Why?"
    assert build_user_prompt("Why?", [], filters=RetrievalFilters()) == "Question: Why?"
    with pytest.raises(ValueError, match="blank"):
        build_user_prompt("  ", [])
    assert build_closed_book_prompt(" Why?\n") == "Question: Why?"
    with pytest.raises(ValueError, match="blank"):
        build_closed_book_prompt("")


def test_format_passage_header_never_carries_a_sentence_terminator() -> None:
    long_section = (
        "Item 7. Management's Discussion and Analysis of Financial Condition and Results."
    )
    text = "Revenue was $5 million. Costs were $3 million."
    chunk = Chunk(
        chunk_id=chunk_id("D_2023_10K", 2, 1, text),
        doc_name="D_2023_10K",
        page_num=2,
        chunk_idx=1,
        section=long_section,
        text=text,
        n_tokens=8,
    )
    header, body = format_passage(3, chunk).split("\n", 1)
    assert header.startswith(f"[3] (ref: chunk:{chunk.chunk_id}) D_2023_10K p.2 | Item 7 Man")
    assert not header.endswith(".")
    assert ". " not in header
    assert len(header.split(" | ", 1)[1]) <= SECTION_MAX_CHARS
    assert body == text
    # the mock parser must see exactly the passage text, not the header remainder
    rendered = format_passage(3, chunk)
    assert extract_passages([Message(role="user", content=rendered)]) == [
        (f"chunk:{chunk.chunk_id}", text)
    ]
    with pytest.raises(ValueError, match="1-based"):
        format_passage(0, chunk)


def test_format_passage_truncation() -> None:
    chunks = make_chunks(TOP_DOC)
    rendered = format_passage(1, chunks[0], max_chars=30)
    header, body = rendered.split("\n", 1)
    assert body.endswith(" [truncated]")
    assert body[: -len(" [truncated]")] == chunks[0].text[:30].rstrip()
    assert format_passage(1, chunks[0], max_chars=None).endswith(chunks[0].text)
    assert format_passage(1, chunks[0], max_chars=10_000).endswith(chunks[0].text)


def test_format_hints() -> None:
    assert format_hints(None) == ""
    assert format_hints(RetrievalFilters()) == ""
    hints = format_hints(
        RetrievalFilters(ticker="ACME", fiscal_year=2023, form="10-K", doc_names=["A", "B"])
    )
    assert hints == "ticker=ACME fiscal_year=2023 form=10-K documents=A,B"
