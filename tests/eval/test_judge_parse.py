"""Judge parsing on recorded (synthetic) verdict fixtures, the rule judge, prompts and kappa."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from secqa.core.contracts import Answer, Chunk, FBQuestion, Usage
from secqa.core.errors import ConfigError
from secqa.core.ids import sha256_hex
from secqa.eval.judge import (
    JUDGE_PROMPT_NAMES,
    JUDGE_SCHEMA,
    JUDGE_VERSION,
    PASSAGE_MAX_CHARS,
    PROMPTS_DIR,
    JudgeParseError,
    LLMJudge,
    RuleJudge,
    answer_from_record,
    build_correctness_prompt,
    build_faithfulness_prompt,
    cited_passages,
    cohen_kappa,
    human_agreement,
    judge_correctness,
    judge_faithfulness,
    judge_prompt_hashes,
    judge_swap,
    load_judge_prompt,
    make_judge,
    parse_correctness,
    read_human_labels,
)
from secqa.grounding.verifier import SNIPPET_MAX_CHARS
from secqa.rag.prompts import prompt_hashes
from secqa.store import DuckDBStore
from tests.eval.conftest import (
    DIM,
    FIXTURES,
    TOP_DOC,
    VERDICTS_DIR,
    FixedProvider,
    make_record,
    verified_citation,
    write_predictions,
)

SNAPSHOT = FIXTURES / "rag_prompt_hashes.json"
"""The one prompt registry (CONTRACTS rule 11): judge prompts are pinned next to the rag ones."""


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((VERDICTS_DIR / name).read_text(encoding="utf-8"))


def _question() -> FBQuestion:
    return FBQuestion(
        id="q1",
        company="Fixture Corp",
        doc_name="FIXTURE_2023_10K",
        question_type="metrics-generated",
        question="What were total net sales in fiscal 2023?",
        answer="$1,577 million",
        justification="Stated on page one.",
        evidence=[],
    )


def _answer(text: str = "Total net sales were $1,577 million.", **kw: Any) -> Answer:
    fields: dict[str, Any] = {
        "request_id": "r1",
        "question": "What were total net sales in fiscal 2023?",
        "text": text,
        "value": 1_577_000_000.0,
        "unit": "USD",
        "abstained": False,
        "citations": [verified_citation()],
        "grounded": True,
        "retrieved": [],
        "trace": [],
        "usage": Usage(),
        "cost_usd": 0.0,
        "latency_ms": 1.0,
        "provider": "openai",
        "model": "gpt-test",
        "mode": "rag",
        "terminated_by": "single_shot",
    }
    fields.update(kw)
    return Answer(**fields)


# ---- prompts ------------------------------------------------------------------------------


def test_judge_prompts_exist_and_are_pinned() -> None:
    for name in JUDGE_PROMPT_NAMES:
        text = load_judge_prompt(name)
        assert text.strip() and "JSON" in text
    hashes = judge_prompt_hashes()
    assert set(hashes) == set(JUDGE_PROMPT_NAMES)
    for name, digest in hashes.items():
        assert digest == sha256_hex((PROMPTS_DIR / name).read_bytes())
    assert hashes == {name: prompt_hashes()[name] for name in JUDGE_PROMPT_NAMES}
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert hashes == {name: expected[name] for name in JUDGE_PROMPT_NAMES}, (
        "judge prompts changed; if intended, bump JUDGE_VERSION and regenerate "
        "tests/fixtures/rag_prompt_hashes.json"
    )
    with pytest.raises(ConfigError, match="not found"):
        load_judge_prompt("nope.md")


def test_prompt_builders_hide_gold_from_faithfulness() -> None:
    q, a = _question(), _answer()
    correctness = build_correctness_prompt(q, a)
    assert q.answer in correctness and q.justification in correctness
    assert "Prediction value: 1577000000.0" in correctness
    faith = build_faithfulness_prompt(a)
    assert "Reference answer" not in faith and q.justification not in faith  # gold hidden
    assert "[1] (chunk:" in faith and "p.1" in faith
    assert cited_passages([verified_citation(False)])  # snippet still shown (store text)
    invalid = verified_citation().model_copy(update={"valid": False})
    assert cited_passages([invalid]) == []
    assert "(no valid cited passages)" in build_faithfulness_prompt(_answer(citations=[]))


def _store_with_chunk(text: str) -> DuckDBStore:
    """An in-memory index holding one chunk with the citation id ``verified_citation`` uses."""
    db = DuckDBStore(":memory:", embed_dim=DIM)
    db.init_schema("hashing", DIM)
    chunk = Chunk(
        chunk_id="a" * 40,
        doc_name=TOP_DOC,
        page_num=1,
        chunk_idx=0,
        section=None,
        text=text,
        n_tokens=len(text.split()),
    )
    db.add_chunks([chunk], np.zeros((1, DIM), dtype=np.float32))
    return db


def test_faithfulness_judge_sees_the_whole_cited_chunk_not_the_display_snippet() -> None:
    """The verifier's ``snippet`` is a <=300-char display prefix. The faithfulness judge must see
    the whole cited chunk from the index, or a claim supported by the tail of the chunk (and not
    quoted verbatim) is scored unsupported and the metric degrades to 'quoted verbatim'."""
    head = "Total net sales were $1,577 million in fiscal 2023, an increase of 12% over 2022. "
    tail = "Operating income was $245 million, down from $260 million in the prior year."
    filler = "Segment results are discussed below.\n" * 12  # pushes the tail past 300 chars
    chunk_text = head + filler + tail
    citation = verified_citation()
    assert " ".join(chunk_text.split()).startswith(citation.snippet)  # snippet = chunk prefix
    assert tail not in citation.snippet and len(citation.snippet) <= SNIPPET_MAX_CHARS
    answer = _answer(text="Operating income was $245 million.", citations=[citation])

    with _store_with_chunk(chunk_text) as store:
        (passage,) = cited_passages(answer.citations, store)
        assert passage.startswith(f"({citation.ref} {TOP_DOC} p.1) {citation.quote} ... ")
        assert tail in passage and "\n" not in passage  # whole chunk, whitespace collapsed
        provider = FixedProvider(text='{"claims": [{"claim": "x", "supported": true}]}')
        judge = LLMJudge(provider)
        assert judge.faithfulness(answer, store=store) is not None
        assert tail in provider.calls[0]["messages"][0].content
        assert build_faithfulness_prompt(answer, store) == provider.calls[0]["messages"][0].content

    # Without a store (or when the chunk left the index) only the display snippet is available.
    (fallback,) = cited_passages(answer.citations)
    assert citation.snippet in fallback and tail not in fallback
    with _store_with_chunk("unrelated chunk text") as other:
        missing = verified_citation().model_copy(update={"chunk_id": "b" * 40})
        (degraded,) = cited_passages([missing], other)
        assert missing.snippet in degraded and "unrelated" not in degraded


def test_cited_passages_cap_and_xbrl_rendering() -> None:
    oversized = "word " * (PASSAGE_MAX_CHARS // 2)
    with _store_with_chunk(oversized) as store:
        (passage,) = cited_passages([verified_citation(False)], store)
        body = passage.split(") ", 1)[1]
        assert body == " ".join(oversized.split())[:PASSAGE_MAX_CHARS]
    fact = verified_citation().model_copy(
        update={
            "ref": "xbrl:Revenues|FY2023|0000000000-23-000001",
            "kind": "xbrl",
            "chunk_id": None,
            "doc_name": None,
            "page_num": None,
            "quote": "",
            "snippet": "us-gaap:Revenues FY2023 = 1,577,000,000.0 USD (FIX 10-K accn ...)",
        }
    )
    with _store_with_chunk("irrelevant") as store:
        assert cited_passages([fact], store) == [f"({fact.ref}) {fact.snippet}"]


# ---- recorded verdicts --------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["correct_parsed.json", "incorrect_fenced.json", "abstain_text.json"]
)
def test_correctness_parses_recorded_replies(name: str) -> None:
    fixture = _fixture(name)
    provider = FixedProvider(text=fixture["text"], parsed=fixture["parsed"])
    verdict = judge_correctness(_question(), _answer(), provider)
    assert verdict.label == fixture["expected_label"]
    assert verdict.judge_model == "anthropic:claude-test"
    assert verdict.judge_version == JUDGE_VERSION
    assert verdict.usage.input_tokens == 500
    call = provider.calls[0]
    assert call["json_schema"] == JUDGE_SCHEMA and call["effort"] == "low"
    assert call["system"] == load_judge_prompt("judge_correctness.md")


def test_malformed_reply_and_refusal_raise() -> None:
    fixture = _fixture("malformed.json")
    with pytest.raises(JudgeParseError, match="not a JSON object"):
        judge_correctness(_question(), _answer(), FixedProvider(text=fixture["text"]))
    with pytest.raises(JudgeParseError, match="refused"):
        judge_correctness(_question(), _answer(), FixedProvider(stop_reason="refusal"))
    with pytest.raises(JudgeParseError, match="label"):
        parse_correctness({"label": "maybe", "rationale": ""}, "")


def test_faithfulness_parses_recorded_reply() -> None:
    fixture = _fixture("faith_two_claims.json")
    provider = FixedProvider(text=fixture["text"])
    verdict = judge_faithfulness(_answer(), provider)
    assert verdict.claims == fixture["expected_claims"]
    assert verdict.supported == fixture["expected_supported"]
    assert verdict.score == pytest.approx(0.5)
    empty = judge_faithfulness(_answer(), FixedProvider(text='{"claims": []}'))
    assert empty.claims == 0 and empty.score is None
    with pytest.raises(JudgeParseError, match="claims"):
        judge_faithfulness(_answer(), FixedProvider(text='{"label": "correct"}'))
    with pytest.raises(JudgeParseError, match="supported"):
        judge_faithfulness(_answer(), FixedProvider(text='{"claims": [{"claim": "x"}]}'))


# ---- rule judge and factory ---------------------------------------------------------------


def test_rule_judge_decides_only_by_rule() -> None:
    judge = RuleJudge()
    q = _question()
    assert judge.correctness(q, _answer()).label == "correct"  # type: ignore[union-attr]
    wrong = judge.correctness(q, _answer(value=42.0))
    assert wrong is not None and wrong.label == "incorrect" and wrong.judge_model == "rule"
    abstain = judge.correctness(q, _answer(text="INSUFFICIENT EVIDENCE", abstained=True))
    assert abstain is not None and abstain.label == "abstain"
    assert judge.correctness(q, _answer(value=None)) is None  # free text: unscored
    assert judge.faithfulness(_answer()) is None


def test_rule_judge_rationale_names_a_scaled_match() -> None:
    """A gold table figure quoted without its 'in millions' header matches the base-unit value
    one way only, and the rationale says so; the reverse direction is scored incorrect."""
    judge = RuleJudge()
    q = _question().model_copy(update={"answer": "$1577.00"})
    scaled = judge.correctness(q, _answer(value=1_577_000_000.0))
    assert scaled is not None and scaled.label == "correct"
    assert "x1,000,000" in scaled.rationale
    plain = judge.correctness(_question(), _answer(value=1_577_000_000.0))
    assert plain is not None and plain.label == "correct" and "understated" not in plain.rationale
    reverse = judge.correctness(q, _answer(value=1.577))
    assert reverse is not None and reverse.label == "incorrect"


def test_make_judge_and_llm_judge_skips_uncitable_answers() -> None:
    assert isinstance(make_judge("rule"), RuleJudge)
    with pytest.raises(ConfigError, match="fabricate"):
        make_judge("mock")
    with pytest.raises(ConfigError, match="provider"):
        make_judge("anthropic:claude-test")
    provider = FixedProvider(text='{"claims": []}')
    llm = make_judge("anthropic:claude-test", provider)
    assert isinstance(llm, LLMJudge) and llm.name == "anthropic:claude-test"
    assert llm.faithfulness(_answer(abstained=True, text="INSUFFICIENT EVIDENCE")) is None
    assert llm.faithfulness(_answer(citations=[])) is None
    assert provider.calls == []
    assert llm.faithfulness(_answer()) is not None
    assert len(provider.calls) == 1


# ---- agreement ----------------------------------------------------------------------------


def test_cohen_kappa_values() -> None:
    assert cohen_kappa([], []) is None
    assert cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == pytest.approx(1.0)
    assert cohen_kappa(["a", "a"], ["a", "a"]) == pytest.approx(1.0)  # no variation, all agree
    assert cohen_kappa(["a", "b"], ["b", "a"]) == pytest.approx(-1.0)
    mixed = cohen_kappa(["a", "a", "b", "b"], ["a", "b", "b", "b"])
    assert mixed is not None and 0.0 < mixed < 1.0
    with pytest.raises(ValueError):
        cohen_kappa(["a"], [])


def test_judge_swap_and_human_agreement(tmp_path: Path) -> None:
    q1 = _question()
    q2 = q1.model_copy(update={"id": "q2", "answer": "$245 million"})
    records = [
        make_record("q1", numeric=True, judge_label="correct"),
        make_record("q2", numeric=None, judge_label="incorrect"),
        make_record("q3", numeric=True, judge_label="correct", error="provider: down"),
    ]
    pred = write_predictions(tmp_path / "run", records)
    swap = FixedProvider(text='{"label": "correct", "rationale": "same"}')
    with pytest.raises(ConfigError, match="questions"):
        judge_swap(pred, swap)
    report = judge_swap(pred, swap, [q1, q2])
    assert report.n == 2 and report.disagreements == ["q2"]
    assert report.agreement == pytest.approx(0.5)
    assert report.name_b == "anthropic:claude-test" and report.run_id == records[0].run_id
    assert (tmp_path / "run" / "judge_swap_anthropic_claude-test.json").is_file()
    assert report.provisional  # kappa undefined/low on two records

    labels = tmp_path / "labels.csv"
    labels.write_text(
        "# protocol comment\n"
        "run_id,financebench_id,label,annotator,labelled_at,notes\n"
        f"{records[0].run_id},q1,correct,wc,2026-09-11,fine\n"
        f"{records[0].run_id},q2,incorrect,wc,2026-09-11,fine\n"
        "other_run,q1,incorrect,wc,2026-09-11,ignored\n"
        ",q3,correct,wc,2026-09-11,errored record is skipped\n",
        encoding="utf-8",
    )
    human = human_agreement(pred, labels)
    assert human.n == 2 and human.kappa == pytest.approx(1.0) and not human.provisional
    assert human.confusion == {"correct": {"correct": 1}, "incorrect": {"incorrect": 1}}
    assert (tmp_path / "run" / "human_agreement.json").is_file()
    labels.write_text("financebench_id,label\nq1,maybe\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="label"):
        read_human_labels(labels)
    labels.write_text("financebench_id,label\nzz,correct\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="no human label"):
        human_agreement(pred, labels)


def test_shipped_human_labels_file_is_valid_and_empty() -> None:
    shipped = Path(__file__).resolve().parents[2] / "src" / "secqa" / "eval" / "human_labels.csv"
    assert read_human_labels(shipped) == []


def test_answer_from_record_round_trip() -> None:
    rec = make_record("q9", abstained=True)
    answer = answer_from_record(rec, "question?")
    assert answer.abstained and answer.text == "INSUFFICIENT EVIDENCE"
    assert answer.mode == "rag" and answer.question == "question?"
