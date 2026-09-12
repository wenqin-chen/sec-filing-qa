"""ToolRuntime: every tool's happy path and error path, the ledger, result bounding, and TOOLS."""

from __future__ import annotations

import json
from typing import Any

import pytest

from secqa.agent import TOOLS, ToolRuntime
from secqa.agent.runtime import page_to_chunk
from secqa.agent.tools import (
    FINAL_ANSWER_SCHEMA,
    PAGE_MAX_CHARS,
    TOOL_NAMES,
    FinalAnswer,
    decode_final_answer,
    parse_final_answer_text,
)
from secqa.core.contracts import CitationRef, RetrievalFilters, ToolCall, ToolResult
from secqa.core.ids import chunk_id
from secqa.retrieval import Retriever
from secqa.store import DuckDBStore
from tests.agent.conftest import (
    ACCN_FY2023,
    INJECTION_TEXT,
    LONG_PAGE_TEXT,
    NET_SALES_SENTENCE,
    OTHER_DOC,
    PAGE_TEXTS,
    REVENUE_REF,
    REVENUE_USD,
    TICKER,
    TOP_DOC,
    net_sales_chunk_id,
)


def call(name: str, **arguments: Any) -> ToolCall:
    return ToolCall(id=f"call-{name}", name=name, arguments=arguments)


def payload(result: ToolResult) -> dict[str, Any]:
    data = json.loads(result.content)
    assert isinstance(data, dict)
    return data


# ---- tool specs -----------------------------------------------------------------------------


def test_tool_specs_are_strict_and_complete() -> None:
    assert TOOL_NAMES == (
        "search_filings",
        "get_pages",
        "lookup_company",
        "lookup_fact",
        "query_xbrl",
        "calculate",
        "final_answer",
    )
    for tool in TOOLS:
        schema = tool.input_schema
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"]), tool.name
        assert tool.description.strip()
        json.dumps(schema)  # serialisable as sent to vendors
    final = next(t for t in TOOLS if t.name == "final_answer")
    assert set(final.input_schema["properties"]) == set(FINAL_ANSWER_SCHEMA["properties"])
    assert "title" not in final.input_schema


def test_query_xbrl_description_warns_that_financials_rows_are_not_citable() -> None:
    tool = next(t for t in TOOLS if t.name == "query_xbrl")
    assert "NOT citable" in tool.description
    assert "lookup_fact" in tool.description


# ---- construction and reset -----------------------------------------------------------------


def test_constructor_validates_limits(store: DuckDBStore, retriever: Retriever) -> None:
    with pytest.raises(ValueError, match="max_result_chars"):
        ToolRuntime(store, retriever, max_result_chars=10)
    with pytest.raises(ValueError, match="sql_timeout_s"):
        ToolRuntime(store, retriever, sql_timeout_s=0)


def test_reset_clears_the_ledger(runtime: ToolRuntime) -> None:
    runtime.dispatch(call("search_filings", query="net sales"))
    runtime.dispatch(call("calculate", expression="1 + 1"))
    assert runtime.seen_chunks and runtime.calc_results and runtime.tool_log
    runtime.reset(RetrievalFilters(ticker="OTHR"))
    assert not runtime.seen_chunks and not runtime.seen_facts
    assert not runtime.calc_results and not runtime.calc_log and not runtime.tool_log
    assert not runtime.retrieved and runtime.retrieval_ms == 0.0
    assert runtime.base_filters == RetrievalFilters(ticker="OTHR")


# ---- search_filings -------------------------------------------------------------------------


def test_search_returns_citable_hits_and_records_them(runtime: ToolRuntime) -> None:
    result = runtime.dispatch(call("search_filings", query="total net sales fiscal 2023"))
    assert not result.is_error and result.name == "search_filings"
    data = payload(result)
    assert data["n"] == len(data["hits"]) > 0
    top = data["hits"][0]
    assert top["ref"] == f"chunk:{top['chunk_id']}"
    assert top["doc_name"] == TOP_DOC and top["page_num"] == 1
    assert top["snippet"] == NET_SALES_SENTENCE + " Growth was broad-based."
    assert set(runtime.seen_chunks) == {hit["chunk_id"] for hit in data["hits"]}
    assert [view.chunk_id for view in runtime.retrieved] == [h["chunk_id"] for h in data["hits"]]
    assert runtime.retrieval_ms > 0.0
    assert runtime.tool_log[-1].kind == "tool" and runtime.tool_log[-1].error is None


def test_search_honours_ticker_filter(runtime: ToolRuntime) -> None:
    result = runtime.dispatch(call("search_filings", query="revenue", ticker="OTHR"))
    data = payload(result)
    assert data["hits"], "the OTHER document mentions revenue"
    assert {hit["doc_name"] for hit in data["hits"]} == {OTHER_DOC}
    assert data["filters"] == {"ticker": "OTHR"}
    assert all(chunk.doc_name == OTHER_DOC for chunk in runtime.seen_chunks.values())


def test_search_merges_request_level_filters(runtime: ToolRuntime) -> None:
    runtime.reset(RetrievalFilters(doc_names=[OTHER_DOC], ticker=TICKER))
    # The model may override ticker; doc_names from the request always applies.
    data = payload(runtime.dispatch(call("search_filings", query="revenue", ticker="OTHR")))
    assert data["filters"] == {"ticker": "OTHR", "doc_names": [OTHER_DOC]}
    assert {hit["doc_name"] for hit in data["hits"]} == {OTHER_DOC}
    data = payload(runtime.dispatch(call("search_filings", query="net sales")))
    assert data["filters"]["ticker"] == TICKER and data["filters"]["doc_names"] == [OTHER_DOC]
    assert data["hits"] == [] and "note" in data


def test_search_blank_query_and_unknown_ticker_are_empty_not_errors(
    runtime: ToolRuntime,
) -> None:
    for arguments in ({"query": "   "}, {"query": "net sales", "ticker": "NOPE"}):
        result = runtime.dispatch(call("search_filings", **arguments))
        assert not result.is_error
        assert payload(result)["hits"] == []
    assert runtime.seen_chunks == {}


def test_search_rejects_bad_k_and_unknown_arguments(runtime: ToolRuntime) -> None:
    result = runtime.dispatch(call("search_filings", query="net sales", k=11))
    assert result.is_error and "between 1 and 10" in payload(result)["error"]
    result = runtime.dispatch(call("search_filings", query="net sales", k=0))
    assert result.is_error
    result = runtime.dispatch(call("search_filings", query="net sales", limit=3))
    assert result.is_error and "invalid arguments" in payload(result)["error"]
    assert runtime.tool_log[-1].error is not None


def test_search_k_bounds_the_hit_count(runtime: ToolRuntime) -> None:
    data = payload(runtime.dispatch(call("search_filings", query="revenue income cash", k=2)))
    assert len(data["hits"]) == 2


# ---- get_pages ------------------------------------------------------------------------------


def test_get_pages_returns_exact_text_with_citable_refs(runtime: ToolRuntime) -> None:
    result = runtime.dispatch(call("get_pages", doc_name=TOP_DOC, pages=[2, 1]))
    assert not result.is_error
    data = payload(result)
    assert data["doc_name"] == TOP_DOC
    assert [page["page_num"] for page in data["pages"]] == [1, 2]
    assert data["pages"][0]["text"] == PAGE_TEXTS[TOP_DOC][0][1]
    assert data["pages"][1]["text"] == PAGE_TEXTS[TOP_DOC][1][1]
    assert data["pages"][0]["truncated"] is False
    assert "missing_pages" not in data
    for page in data["pages"]:
        expected_id = chunk_id(TOP_DOC, page["page_num"], 0, page["text"])
        assert page["ref"] == f"chunk:{expected_id}"
        assert runtime.seen_chunks[expected_id].text == page["text"]
    # The page-1 pseudo-chunk shares its id with the page-1 store chunk (same text).
    assert net_sales_chunk_id() in runtime.seen_chunks


def test_get_pages_truncates_long_pages_but_keeps_full_text_in_ledger(
    runtime: ToolRuntime,
) -> None:
    data = payload(runtime.dispatch(call("get_pages", doc_name=TOP_DOC, pages=[4])))
    (page,) = data["pages"]
    assert page["truncated"] is True and page["chars"] == len(LONG_PAGE_TEXT)
    assert page["text"].endswith("[truncated]")
    assert len(page["text"]) <= PAGE_MAX_CHARS + len(" [truncated]")
    assert page["text"][: PAGE_MAX_CHARS - 20] == LONG_PAGE_TEXT[: PAGE_MAX_CHARS - 20]
    chunk = runtime.seen_chunks[page["ref"].removeprefix("chunk:")]
    assert chunk.text == LONG_PAGE_TEXT  # quotes from the visible part verify against this


def test_get_pages_reports_missing_pages(runtime: ToolRuntime) -> None:
    data = payload(runtime.dispatch(call("get_pages", doc_name=OTHER_DOC, pages=[3, 9])))
    assert [page["page_num"] for page in data["pages"]] == [3]
    assert data["missing_pages"] == [9]


@pytest.mark.parametrize(
    ("arguments", "reason"),
    [
        ({"doc_name": TOP_DOC, "pages": [1, 2, 3, 4]}, "at most 3 pages"),
        ({"doc_name": TOP_DOC, "pages": []}, "at least one"),
        ({"doc_name": TOP_DOC, "pages": [0]}, "1-based"),
        ({"doc_name": "NOPE_2020_10K", "pages": [1]}, "unknown document"),
        ({"doc_name": TOP_DOC}, "invalid arguments"),
    ],
)
def test_get_pages_errors(runtime: ToolRuntime, arguments: dict[str, Any], reason: str) -> None:
    result = runtime.dispatch(call("get_pages", **arguments))
    assert result.is_error
    assert reason in payload(result)["error"]
    assert runtime.seen_chunks == {}


def test_page_to_chunk_matches_oracle_ids() -> None:
    from secqa.core.contracts import Page

    page = Page(doc_name=TOP_DOC, page_num=3, text="Some page text of a filing.")
    chunk = page_to_chunk(page)
    assert chunk.chunk_id == chunk_id(TOP_DOC, 3, 0, page.text)
    assert chunk.chunk_idx == 0 and chunk.section is None and chunk.text == page.text


# ---- lookup_company -------------------------------------------------------------------------


def test_lookup_company_by_ticker_name_and_prefix(runtime: ToolRuntime) -> None:
    data = payload(runtime.dispatch(call("lookup_company", name_or_ticker="fixt")))
    (match,) = data["matches"]
    assert match["ticker"] == TICKER and match["cik"] == "0001234567"
    assert match["fiscal_years"] == [2023]
    assert [d["doc_name"] for d in match["documents"]] == [TOP_DOC]
    assert match["documents"][0]["form"] == "10-K" and match["documents"][0]["n_pages"] == 4

    data = payload(runtime.dispatch(call("lookup_company", name_or_ticker="corp")))
    assert {m["ticker"] for m in data["matches"]} == {TICKER, "OTHR"}

    data = payload(runtime.dispatch(call("lookup_company", name_or_ticker="other_2022")))
    assert [m["ticker"] for m in data["matches"]] == ["OTHR"]


def test_lookup_company_miss_and_blank(runtime: ToolRuntime) -> None:
    result = runtime.dispatch(call("lookup_company", name_or_ticker="zzz"))
    assert not result.is_error
    data = payload(result)
    assert data["matches"] == [] and "no indexed filings" in data["note"]
    result = runtime.dispatch(call("lookup_company", name_or_ticker="  "))
    assert result.is_error and "blank" in payload(result)["error"]


# ---- lookup_fact ----------------------------------------------------------------------------


def test_lookup_fact_returns_accession_and_registers_ref(runtime: ToolRuntime) -> None:
    result = runtime.dispatch(
        call("lookup_fact", ticker="fixt", metric="revenue", fiscal_year=2023)
    )
    assert not result.is_error
    data = payload(result)
    assert data["ticker"] == TICKER and data["n"] == 1
    (fact,) = data["facts"]
    assert fact["accn"] == ACCN_FY2023 and fact["filed"] == "2024-02-15"
    assert fact["ref"] == REVENUE_REF
    assert fact["value"] == REVENUE_USD and fact["unit"] == "USD"
    assert fact["fiscal_year"] == 2023 and fact["fp"] == "FY" and fact["form"] == "10-K"
    assert fact["concept"] == "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
    assert fact["start_date"] == "2023-01-01" and fact["end_date"] == "2023-12-31"
    assert runtime.seen_facts[REVENUE_REF].accn == ACCN_FY2023
    assert runtime.invalid_refs([CitationRef(ref=REVENUE_REF)]) == []


def test_lookup_fact_miss_and_errors(runtime: ToolRuntime) -> None:
    result = runtime.dispatch(
        call("lookup_fact", ticker=TICKER, metric="gross_profit", fiscal_year=2023)
    )
    assert not result.is_error
    data = payload(result)
    assert data["facts"] == [] and "no annual 10-K value" in data["note"]

    result = runtime.dispatch(call("lookup_fact", ticker="!!", metric="revenue", fiscal_year=2023))
    assert result.is_error and "invalid ticker" in payload(result)["error"]
    result = runtime.dispatch(
        call("lookup_fact", ticker=TICKER, metric="not a metric", fiscal_year=2023)
    )
    assert result.is_error and "unknown metric" in payload(result)["error"]
    result = runtime.dispatch(call("lookup_fact", ticker=TICKER, metric="revenue"))
    assert result.is_error and "invalid arguments" in payload(result)["error"]
    assert runtime.seen_facts == {}


# ---- query_xbrl -----------------------------------------------------------------------------


def test_query_xbrl_rejects_writes_with_an_error_string(
    runtime: ToolRuntime, store: DuckDBStore
) -> None:
    before = store.counts()["facts"]
    for sql in (
        "DROP TABLE xbrl_facts",
        "DELETE FROM xbrl_facts",
        "SELECT * FROM chunks",
        "SELECT * FROM read_csv('/etc/passwd')",
        "SELECT val FROM xbrl_facts; SELECT 1",
        "SELECT nonexistent_column FROM xbrl_facts",
        "",
    ):
        result = runtime.dispatch(call("query_xbrl", sql=sql))
        assert result.is_error, sql
        error = payload(result)["error"]
        assert isinstance(error, str) and error
    assert store.counts()["facts"] == before
    assert runtime.seen_facts == {}


def test_query_xbrl_rows_with_tag_fy_accn_val_get_citable_refs(runtime: ToolRuntime) -> None:
    result = runtime.dispatch(
        call(
            "query_xbrl",
            sql=(
                "SELECT tag, fy, accn, val, unit, end_date FROM xbrl_facts WHERE tag = 'Revenues' "
                f"AND accn = '{ACCN_FY2023}' AND end_date = '2022-12-31'"
            ),
        )
    )
    assert not result.is_error
    data = payload(result)
    assert data["columns"] == ["tag", "fy", "accn", "val", "unit", "end_date", "ref"]
    assert data["row_count"] == 1 and data["truncated"] is False
    assert data["sql"].endswith("LIMIT 200")
    (row,) = data["rows"]
    tag, fy, accn, val, _unit, _end, ref = row
    assert ref == f"xbrl:{tag}|FY{fy}|{accn}" == f"xbrl:Revenues|FY2023|{ACCN_FY2023}"
    assert runtime.seen_facts[ref].val == val == 1_400_000_000.0
    assert runtime.seen_facts[ref].end_date is not None
    assert runtime.invalid_refs([CitationRef(ref=ref)]) == []
    assert "note" not in data


def test_query_xbrl_ambiguous_refs_are_not_citable(runtime: ToolRuntime) -> None:
    """Comparative periods in one 10-K share tag, fy (filing focus) and accn: two different
    values would collide on one ref, so neither gets one and the ledger stays clean."""
    data = payload(
        runtime.dispatch(
            call(
                "query_xbrl",
                sql=(
                    "SELECT tag, fy, accn, val FROM xbrl_facts WHERE tag = 'Revenues' "
                    f"AND accn = '{ACCN_FY2023}' ORDER BY end_date"
                ),
            )
        )
    )
    assert data["row_count"] == 2
    assert [row[-1] for row in data["rows"]] == [None, None]
    assert {row[3] for row in data["rows"]} == {1_210_000_000.0, 1_400_000_000.0}
    assert "2 row(s) have ref null" in data["note"]
    assert runtime.seen_facts == {}
    # A ref already in the ledger with a different value is not overwritten either.
    runtime.dispatch(
        call(
            "query_xbrl",
            sql=(
                "SELECT tag, fy, accn, val FROM xbrl_facts WHERE tag = 'Revenues' "
                f"AND accn = '{ACCN_FY2023}' AND end_date = '2022-12-31'"
            ),
        )
    )
    assert runtime.seen_facts[f"xbrl:Revenues|FY2023|{ACCN_FY2023}"].val == 1_400_000_000.0
    data = payload(
        runtime.dispatch(
            call(
                "query_xbrl",
                sql=(
                    "SELECT tag, fy, accn, val FROM xbrl_facts WHERE tag = 'Revenues' "
                    f"AND accn = '{ACCN_FY2023}' AND end_date = '2021-12-31'"
                ),
            )
        )
    )
    assert data["rows"][0][-1] is None and "note" in data
    assert runtime.seen_facts[f"xbrl:Revenues|FY2023|{ACCN_FY2023}"].val == 1_400_000_000.0


def test_query_xbrl_financials_rows_are_not_citable(runtime: ToolRuntime) -> None:
    data = payload(
        runtime.dispatch(
            call(
                "query_xbrl",
                sql="SELECT fiscal_year, revenue FROM financials WHERE ticker = 'FIXT'",
            )
        )
    )
    assert data["columns"] == ["fiscal_year", "revenue"]
    assert "not citable" in data["note"]
    assert runtime.seen_facts == {}


# ---- calculate ------------------------------------------------------------------------------


def test_calculate_records_results_and_errors(runtime: ToolRuntime) -> None:
    result = runtime.dispatch(call("calculate", expression="(1577 - 1408) / 1408 * 100"))
    assert not result.is_error
    data = payload(result)
    assert data["expression"] == "(1577 - 1408) / 1408 * 100"
    assert abs(data["result"] - 12.0028409) < 1e-6
    assert runtime.calc_results == [data["result"]]
    assert runtime.rendered_calculation() == "(1577 - 1408) / 1408 * 100 = 12.0028"

    result = runtime.dispatch(call("calculate", expression="__import__('os')"))
    assert result.is_error and "__import__" in payload(result)["error"]
    assert len(runtime.calc_results) == 1


def test_rendered_calculation_is_none_without_calls(runtime: ToolRuntime) -> None:
    assert runtime.rendered_calculation() is None


# ---- final_answer ---------------------------------------------------------------------------


def test_final_answer_rejects_unknown_refs_then_accepts_known_ones(runtime: ToolRuntime) -> None:
    result = runtime.dispatch(
        call(
            "final_answer",
            answer="x",
            value=None,
            unit=None,
            citations=[{"ref": "chunk:" + "0" * 40, "quote": "nothing"}, {"ref": "bogus"}],
            calculation=None,
            abstain=False,
        )
    )
    assert result.is_error
    error = payload(result)["error"]
    assert "chunk:" + "0" * 40 in error and "bogus" in error and "unknown citation" in error

    runtime.dispatch(call("search_filings", query="net sales"))
    ref = f"chunk:{net_sales_chunk_id()}"
    result = runtime.dispatch(
        call("final_answer", answer="x", citations=[{"ref": ref, "quote": "Total net sales"}])
    )
    assert not result.is_error and payload(result) == {"status": "accepted", "abstain": False}
    final = runtime.final_answer_from(
        {"answer": "x", "citations": [{"ref": ref, "quote": "Total net sales"}]}
    )
    assert isinstance(final, FinalAnswer) and final.citations[0].ref == ref


def test_final_answer_rejects_unknown_fields(runtime: ToolRuntime) -> None:
    result = runtime.dispatch(call("final_answer", answer="x", confidence=0.9))
    assert result.is_error and "confidence" in payload(result)["error"]


def test_decode_and_parse_final_answer() -> None:
    final = decode_final_answer({"answer": "INSUFFICIENT EVIDENCE", "citations": None})
    assert final.abstained and final.text == "INSUFFICIENT EVIDENCE"
    final = decode_final_answer({"answer": "", "abstain": True, "unit": "  "})
    assert final.text == "INSUFFICIENT EVIDENCE" and final.unit is None
    with pytest.raises(ValueError, match="JSON object"):
        decode_final_answer([1, 2])
    with pytest.raises(ValueError, match="value"):
        decode_final_answer({"answer": "x", "value": "many"})

    parsed, why = parse_final_answer_text('```json\n{"answer": "42 widgets", "value": 42}\n```')
    assert parsed is not None and parsed.value == 42.0 and why is None
    assert parse_final_answer_text("just prose")[0] is None
    assert parse_final_answer_text("")[0] is None
    assert parse_final_answer_text("[1]")[0] is None


# ---- dispatch invariants --------------------------------------------------------------------


def test_unknown_tool_is_an_error_result(runtime: ToolRuntime) -> None:
    result = runtime.dispatch(call("shell", command="rm -rf /"))
    assert result.is_error and result.tool_call_id == "call-shell"
    assert "unknown tool 'shell'" in payload(result)["error"]
    assert runtime.tool_log[-1].name == "shell" and runtime.tool_log[-1].error


def test_dispatch_never_raises(runtime: ToolRuntime, monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("database on fire")

    monkeypatch.setattr(runtime.retriever, "retrieve", explode)
    result = runtime.dispatch(call("search_filings", query="net sales"))
    assert result.is_error
    assert "RuntimeError: database on fire" in payload(result)["error"]


def test_results_are_bounded_and_stay_valid_json(store: DuckDBStore, retriever: Retriever) -> None:
    small = ToolRuntime(store, retriever, max_result_chars=1000)
    result = small.dispatch(call("get_pages", doc_name=TOP_DOC, pages=[4]))
    assert len(result.content) <= 1000
    data = payload(result)  # still a JSON object
    assert not result.is_error and data["truncated"] is True
    assert data["pages"][0]["truncated"] is True
    result = small.dispatch(call("search_filings", query="revenue income cash sales", k=6))
    assert len(result.content) <= 1000 and payload(result)["truncated"] is True
    tiny = ToolRuntime(store, retriever, max_result_chars=200)
    result = tiny.dispatch(call("get_pages", doc_name=TOP_DOC, pages=[4]))
    assert len(result.content) <= 200 and result.is_error
    assert "narrow the request" in payload(result)["error"]


def test_tool_log_records_every_dispatch_in_order(runtime: ToolRuntime) -> None:
    runtime.dispatch(call("calculate", expression="1 + 1"))
    runtime.dispatch(call("lookup_company", name_or_ticker="fixt"))
    runtime.dispatch(call("nope"))
    assert [step.name for step in runtime.tool_log] == ["calculate", "lookup_company", "nope"]
    assert [step.step for step in runtime.tool_log] == [1, 2, 3]
    assert all(step.kind == "tool" for step in runtime.tool_log)
    assert runtime.tool_log[0].arguments == {"expression": "1 + 1"}
    assert runtime.tool_log[0].result_preview.startswith('{"expression"')


def test_injected_text_is_returned_as_data(runtime: ToolRuntime) -> None:
    data = payload(runtime.dispatch(call("get_pages", doc_name=OTHER_DOC, pages=[2])))
    assert data["pages"][0]["text"] == INJECTION_TEXT
