"""Unit tests for the API request models (validation that does not need a running app)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from secqa.api.schemas import AskRequest, AskResponse, SearchRequest, SqlRequest
from secqa.core.contracts import Answer, RetrievalFilters, Usage


def test_ask_request_defaults_and_normalisation() -> None:
    request = AskRequest(question="  What was revenue?  ", ticker="nvda", doc_names=[" A ", "A"])
    assert request.question == "What was revenue?"
    assert request.ticker == "NVDA"
    assert request.doc_names == ["A"]
    assert request.mode == "rag" and request.k == 8 and request.include_trace is True
    assert request.provider is None and request.max_cost_usd is None
    assert request.filters() == RetrievalFilters(ticker="NVDA", doc_names=["A"])


def test_ask_request_without_filters_gives_none() -> None:
    assert AskRequest(question="q").filters() is None


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "q", "k": 0},
        {"question": "q", "k": 21},
        {"question": "q", "mode": "oracle"},
        {"question": "q", "provider": "   "},
        {"question": "q", "doc_names": [""]},
        {"question": "q", "doc_names": ["x" * 201]},
        {"question": "q", "ticker": "1BAD"},
        {"question": "q", "extra": True},
    ],
)
def test_ask_request_rejects(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        AskRequest(**payload)  # type: ignore[arg-type]


def test_search_request_form_filter() -> None:
    request = SearchRequest(query="net sales", form="10-K", strategy="bm25")
    assert request.retrieval_filters() == RetrievalFilters(form="10-K")
    assert SearchRequest(query="net sales").retrieval_filters() is None
    with pytest.raises(ValidationError):
        SearchRequest(query="x", strategy="faiss")  # type: ignore[arg-type]


def test_sql_request_limits() -> None:
    assert SqlRequest(sql=" SELECT 1 ").sql == "SELECT 1"
    with pytest.raises(ValidationError):
        SqlRequest(sql="x" * 5001)


def test_ask_response_is_an_answer() -> None:
    answer = Answer(
        request_id="r",
        question="q",
        text="INSUFFICIENT EVIDENCE",
        abstained=True,
        citations=[],
        grounded=True,
        retrieved=[],
        trace=[],
        usage=Usage(),
        cost_usd=0.0,
        latency_ms=1.0,
        provider="mock",
        model="mock-extractive",
        mode="rag",
        terminated_by="empty_retrieval",
    )
    response = AskResponse.model_validate(answer.model_dump())
    assert isinstance(response, Answer)
    assert response.model_dump() == answer.model_dump()
