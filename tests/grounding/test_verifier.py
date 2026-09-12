"""Tests for secqa.grounding.verifier (offline, deterministic, synthetic data only)."""

from __future__ import annotations

from datetime import date

import pytest

from secqa.core.contracts import Chunk, Citation, CitationRef, FactRow
from secqa.core.ids import chunk_id
from secqa.grounding import CitationVerifier, parse_ref, quote_in_chunk
from secqa.grounding.verifier import SNIPPET_MAX_CHARS, _mask_years

PAGE_TEXT = (
    "Item 7. Management Discussion. Total net sales were $1,577 million in fiscal 2023, "
    "an increase of 12% over 2022. Long-term debt was 1,200 million at year end."
)


def make_chunk(text: str = PAGE_TEXT, page_num: int = 3, idx: int = 0) -> Chunk:
    return Chunk(
        chunk_id=chunk_id("FIXTURE_2023_10K", page_num, idx, text),
        doc_name="FIXTURE_2023_10K",
        page_num=page_num,
        chunk_idx=idx,
        section="Item 7",
        text=text,
        n_tokens=len(text.split()),
    )


def make_fact(tag: str = "Revenues", fy: int = 2023, val: float = 1.577e9) -> FactRow:
    return FactRow(
        cik="0000000001",
        ticker="FIX",
        taxonomy="us-gaap",
        tag=tag,
        unit="USD",
        fy=fy,
        fp="FY",
        form="10-K",
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
        val=val,
        accn="0000000001-24-000001",
        filed=date(2024, 2, 15),
        frame=f"CY{fy}",
    )


@pytest.fixture
def chunk() -> Chunk:
    return make_chunk()


@pytest.fixture
def chunks(chunk: Chunk) -> dict[str, Chunk]:
    return {chunk.chunk_id: chunk}


@pytest.fixture
def verifier() -> CitationVerifier:
    return CitationVerifier()


# ---------------------------------------------------------------- parse_ref


def test_parse_ref_chunk() -> None:
    assert parse_ref("chunk:abc123") == ("chunk", "abc123")
    assert parse_ref("  chunk:abc123 \n") == ("chunk", "abc123")


def test_parse_ref_xbrl() -> None:
    kind, payload = parse_ref("xbrl:Revenues|FY2023|0000000001-24-000001")
    assert kind == "xbrl"
    assert payload == "Revenues|FY2023|0000000001-24-000001"


def test_parse_ref_matches_factrow_ref_property() -> None:
    fact = make_fact()
    kind, payload = parse_ref(fact.ref)
    assert kind == "xbrl"
    assert payload == f"{fact.tag}|FY{fact.fy}|{fact.accn}"


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "page:3",
        "chunk:",
        "xbrl:Revenues",
        "xbrl:Revenues|2023|accn",
        "xbrl:Revenues||accn",
        "abc123",
    ],
)
def test_parse_ref_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_ref(bad)


# ---------------------------------------------------------------- quote_in_chunk


def test_quote_in_chunk_exact_and_normalised() -> None:
    assert quote_in_chunk("Total net sales were $1,577 million", PAGE_TEXT)
    # Curly quotes, line breaks, casing and trailing punctuation must not matter.
    assert quote_in_chunk("total NET sales\nwere “$1,577 million”.", PAGE_TEXT)


def test_quote_in_chunk_rejects_short_and_paraphrased() -> None:
    assert not quote_in_chunk("net sales", PAGE_TEXT)  # too short after normalisation
    assert not quote_in_chunk("Net sales rose to $1,577 million in FY2023", PAGE_TEXT)
    assert not quote_in_chunk("", PAGE_TEXT)


def test_quote_in_chunk_min_chars_is_configurable() -> None:
    assert not quote_in_chunk("net sales were", PAGE_TEXT, min_chars=20)
    assert quote_in_chunk("net sales were", PAGE_TEXT, min_chars=10)
    with pytest.raises(ValueError):
        quote_in_chunk("net sales were", PAGE_TEXT, min_chars=0)


# ---------------------------------------------------------------- verify: chunk citations


def test_verified_quote(verifier: CitationVerifier, chunk: Chunk, chunks: dict[str, Chunk]) -> None:
    refs = [CitationRef(ref=f"chunk:{chunk.chunk_id}", quote="Total net sales were $1,577 million")]
    citations, grounded = verifier.verify("Net sales were $1,577 million.", refs, chunks, {}, [])
    assert len(citations) == 1
    citation = citations[0]
    assert isinstance(citation, Citation)
    assert citation.kind == "chunk"
    assert citation.valid is True
    assert citation.verified is True
    assert citation.doc_name == "FIXTURE_2023_10K"
    assert citation.page_num == 3
    assert citation.chunk_id == chunk.chunk_id
    assert citation.quote == "Total net sales were $1,577 million"
    assert citation.snippet == PAGE_TEXT[:SNIPPET_MAX_CHARS]
    assert grounded is True


def test_paraphrased_quote_kept_but_unverified(
    verifier: CitationVerifier, chunk: Chunk, chunks: dict[str, Chunk]
) -> None:
    refs = [
        CitationRef(
            ref=f"chunk:{chunk.chunk_id}", quote="Net sales climbed to $1,577 million in 2023"
        )
    ]
    citations, grounded = verifier.verify("Net sales were $1,577 million.", refs, chunks, {}, [])
    assert len(citations) == 1
    assert citations[0].valid is True
    assert citations[0].verified is False
    assert citations[0].quote == "Net sales climbed to $1,577 million in 2023"
    assert citations[0].snippet.startswith("Item 7.")  # still from the store
    assert grounded is False  # the number is only in an unverified quote


def test_unknown_ref_invalid_but_kept(verifier: CitationVerifier, chunks: dict[str, Chunk]) -> None:
    refs = [CitationRef(ref="chunk:deadbeef", quote="Total net sales were $1,577 million")]
    citations, grounded = verifier.verify("Net sales were $1,577 million.", refs, chunks, {}, [])
    assert len(citations) == 1
    citation = citations[0]
    assert citation.ref == "chunk:deadbeef"
    assert citation.kind == "chunk"
    assert citation.chunk_id == "deadbeef"
    assert citation.valid is False
    assert citation.verified is False
    assert citation.snippet == ""  # nothing in the store to show
    assert citation.quote == "Total net sales were $1,577 million"  # model text kept, flagged
    assert grounded is False


def test_malformed_ref_invalid_but_kept(
    verifier: CitationVerifier, chunks: dict[str, Chunk]
) -> None:
    refs = [CitationRef(ref="page:3", quote="Total net sales were $1,577 million")]
    citations, grounded = verifier.verify("No numbers here.", refs, chunks, {}, [])
    assert len(citations) == 1
    assert citations[0].valid is False
    assert citations[0].verified is False
    assert citations[0].ref == "page:3"
    assert grounded is True  # vacuous: nothing numeric to ground


def test_snippet_is_store_sourced_and_capped(verifier: CitationVerifier) -> None:
    long_text = " ".join(f"word{i}" for i in range(200))
    chunk = make_chunk(text=long_text, page_num=9)
    refs = [CitationRef(ref=f"chunk:{chunk.chunk_id}", quote="FABRICATED TEXT THAT IS NOT THERE")]
    citations, _ = verifier.verify("text", refs, {chunk.chunk_id: chunk}, {}, [])
    assert citations[0].snippet == long_text[:SNIPPET_MAX_CHARS]
    assert len(citations[0].snippet) <= SNIPPET_MAX_CHARS
    assert "FABRICATED" not in citations[0].snippet


def test_full_ref_accepted_as_chunk_key(verifier: CitationVerifier, chunk: Chunk) -> None:
    refs = [CitationRef(ref=f"chunk:{chunk.chunk_id}", quote="Total net sales were $1,577 million")]
    citations, _ = verifier.verify("text", refs, {f"chunk:{chunk.chunk_id}": chunk}, {}, [])
    assert citations[0].valid is True and citations[0].verified is True


def test_duplicate_refs_collapsed_and_order_kept(
    verifier: CitationVerifier, chunk: Chunk, chunks: dict[str, Chunk]
) -> None:
    ref = f"chunk:{chunk.chunk_id}"
    refs = [
        CitationRef(ref=ref, quote="Total net sales were $1,577 million"),
        CitationRef(ref="chunk:unknown", quote=""),
        CitationRef(ref=ref, quote="Total net sales were $1,577 million"),
        CitationRef(ref=ref, quote="Long-term debt was 1,200 million"),
    ]
    citations, _ = verifier.verify("text", refs, chunks, {}, [])
    assert [c.ref for c in citations] == [ref, "chunk:unknown", ref]
    assert [c.verified for c in citations] == [True, False, True]


def test_min_quote_chars_respected(chunk: Chunk, chunks: dict[str, Chunk]) -> None:
    refs = [CitationRef(ref=f"chunk:{chunk.chunk_id}", quote="net sales were")]
    strict, _ = CitationVerifier(min_quote_chars=20).verify("text", refs, chunks, {}, [])
    lenient, _ = CitationVerifier(min_quote_chars=10).verify("text", refs, chunks, {}, [])
    assert strict[0].verified is False
    assert lenient[0].verified is True
    with pytest.raises(ValueError):
        CitationVerifier(min_quote_chars=0)


# ---------------------------------------------------------------- verify: grounded flag


def test_grounded_true_across_surface_forms(
    verifier: CitationVerifier, chunk: Chunk, chunks: dict[str, Chunk]
) -> None:
    refs = [CitationRef(ref=f"chunk:{chunk.chunk_id}", quote="Long-term debt was 1,200 million")]
    _, grounded = verifier.verify(
        "Long-term debt was $1.2 billion at year end.", refs, chunks, {}, []
    )
    assert grounded is True


def test_grounded_false_for_number_appearing_nowhere(
    verifier: CitationVerifier, chunk: Chunk, chunks: dict[str, Chunk]
) -> None:
    refs = [CitationRef(ref=f"chunk:{chunk.chunk_id}", quote="Total net sales were $1,577 million")]
    _, grounded = verifier.verify(
        "Net sales were $1,577 million and operating income was $245 million.",
        refs,
        chunks,
        {},
        [],
    )
    assert grounded is False


def test_grounded_false_without_any_citation(verifier: CitationVerifier) -> None:
    _, grounded = verifier.verify("Net sales were $1,577 million.", [], {}, {}, [])
    assert grounded is False


def test_grounded_vacuous_when_answer_has_no_numbers(verifier: CitationVerifier) -> None:
    _, grounded = verifier.verify(
        "The filing does not disclose this; I cannot answer.", [], {}, {}, []
    )
    assert grounded is True


def test_grounded_uses_calculator_results(
    verifier: CitationVerifier, chunk: Chunk, chunks: dict[str, Chunk]
) -> None:
    refs = [CitationRef(ref=f"chunk:{chunk.chunk_id}", quote="Total net sales were $1,577 million")]
    text = "Net sales were $1,577 million, so the margin was 15.5%."
    _, without_calc = verifier.verify(text, refs, chunks, {}, [])
    _, with_calc = verifier.verify(text, refs, chunks, {}, [0.155])
    assert without_calc is False
    assert with_calc is True


def test_grounded_accepts_percent_ratio_equivalence(verifier: CitationVerifier) -> None:
    _, grounded = verifier.verify("The margin was 15.5%.", [], {}, {}, [15.5])
    assert grounded is True


def test_grounded_ignores_bare_years(
    verifier: CitationVerifier, chunk: Chunk, chunks: dict[str, Chunk]
) -> None:
    refs = [CitationRef(ref=f"chunk:{chunk.chunk_id}", quote="Total net sales were $1,577 million")]
    _, grounded = verifier.verify(
        "In fiscal 2023 (versus 2022), net sales were $1,577 million.", refs, chunks, {}, []
    )
    assert grounded is True


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("fiscal 2023", "fiscal YEAR"),
        ("(2022)", "(YEAR)"),
        ("from 2021.", "from YEAR."),
        ("$2,022", "$2,022"),
        ("2022 million", "2022 million"),
        ("2022.5", "2022.5"),
        ("2022%", "2022%"),
        ("12022", "12022"),
        ("FY2022", "FY2022"),
    ],
)
def test_mask_years(text: str, expected: str) -> None:
    assert _mask_years(text) == expected


# ---------------------------------------------------------------- verify: xbrl citations


def test_xbrl_ref_matched_to_returned_rows_only(verifier: CitationVerifier) -> None:
    fact = make_fact()
    facts = {fact.ref: fact}
    other = make_fact(tag="NetIncomeLoss", val=1.9e8)  # never returned by a tool
    refs = [CitationRef(ref=fact.ref), CitationRef(ref=other.ref)]
    citations, grounded = verifier.verify(
        "Revenue was $1,577 million and net income was $190 million.", refs, {}, facts, []
    )
    assert [c.kind for c in citations] == ["xbrl", "xbrl"]
    good, bad = citations
    assert good.valid is True and good.verified is True
    assert good.tag == "Revenues"
    assert good.fiscal_year == 2023
    assert good.accn == fact.accn
    assert good.value == pytest.approx(1.577e9)
    assert "Revenues" in good.snippet and fact.accn in good.snippet
    assert len(good.snippet) <= SNIPPET_MAX_CHARS
    assert bad.valid is False and bad.verified is False
    assert bad.tag == "NetIncomeLoss"
    assert bad.fiscal_year == 2023
    assert bad.accn == other.accn
    assert bad.value is None
    assert bad.snippet == ""
    assert grounded is False  # $190 million only backed by the unreturned row


def test_xbrl_fact_grounds_answer_number(verifier: CitationVerifier) -> None:
    fact = make_fact()
    _, grounded = verifier.verify(
        "Revenue was $1.58 billion.", [CitationRef(ref=fact.ref)], {}, {fact.ref: fact}, []
    )
    assert grounded is True  # 1.577e9 vs 1.58e9 is within the 1% tolerance


def test_xbrl_payload_accepted_as_fact_key(verifier: CitationVerifier) -> None:
    fact = make_fact()
    _, payload = parse_ref(fact.ref)
    citations, _ = verifier.verify("text", [CitationRef(ref=fact.ref)], {}, {payload: fact}, [])
    assert citations[0].valid is True and citations[0].verified is True


def test_mixed_chunk_and_xbrl_citations(
    verifier: CitationVerifier, chunk: Chunk, chunks: dict[str, Chunk]
) -> None:
    fact = make_fact(tag="LongTermDebt", val=1.2e9)
    refs = [
        CitationRef(ref=f"chunk:{chunk.chunk_id}", quote="Total net sales were $1,577 million"),
        CitationRef(ref=fact.ref),
    ]
    citations, grounded = verifier.verify(
        "Net sales were $1,577 million; long-term debt was $1.2 billion.",
        refs,
        chunks,
        {fact.ref: fact},
        [],
    )
    assert [c.kind for c in citations] == ["chunk", "xbrl"]
    assert all(c.verified and c.valid for c in citations)
    assert grounded is True
