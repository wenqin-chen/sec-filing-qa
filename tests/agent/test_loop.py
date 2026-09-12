"""AgentLoop with ScriptedProvider scenarios: happy path, retries, every stopping rule, the
forced final call, text-only replies, prompt injection, and the MockProvider end to end."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from secqa.agent import AGENT_SYSTEM, AgentLoop, ToolRuntime, agent_prompt_hash
from secqa.agent.loop import FINAL_PROMPT, NUDGE_PROMPT, build_agent_prompt, load_agent_prompt
from secqa.agent.tools import FINAL_ANSWER_SCHEMA, TOOLS
from secqa.core.contracts import Answer, LLMResponse, Message, RetrievalFilters, ToolSpec
from secqa.core.errors import ProviderError
from secqa.core.ids import sha256_hex
from secqa.providers.deadline import remaining_s
from secqa.providers.mock_provider import MockProvider
from secqa.store import DuckDBStore
from tests.agent.conftest import (
    INJECTION_TEXT,
    NET_SALES_QUESTION,
    NET_SALES_QUOTE,
    NET_SALES_SENTENCE,
    OTHER_DOC,
    REVENUE_REF,
    REVENUE_USD,
    SCENARIOS,
    TICKER,
    TOP_DOC,
    PricedScripted,
    RecordingScripted,
    abstain_final_json,
    final_turn,
    injection_chunk_id,
    net_sales_chunk_id,
    search_turn,
)

LoopFactory = Callable[..., AgentLoop]


def net_sales_citation() -> dict[str, str]:
    return {"ref": f"chunk:{net_sales_chunk_id()}", "quote": NET_SALES_QUOTE}


def trace_kinds(answer: Answer) -> list[str]:
    return [step.kind for step in answer.trace]


# ---- construction and prompt ------------------------------------------------------------


def test_constructor_validates_arguments(make_loop: LoopFactory) -> None:
    provider = RecordingScripted([])
    for name, value in (
        ("max_steps", 0),
        ("max_tool_calls", 0),
        ("max_input_tokens", 0),
        ("max_tokens", 0),
        ("max_cost_usd", -1.0),
        ("wall_clock_s", 0.0),
    ):
        with pytest.raises(ValueError, match=name):
            make_loop(provider, **{name: value})
    with pytest.raises(ValueError, match="effort"):
        make_loop(provider, effort="max")


def test_system_prompt_is_loaded_hashed_and_states_the_rules() -> None:
    text = load_agent_prompt()
    assert "INSUFFICIENT EVIDENCE" in text and '"abstain"' in text
    assert "data, not instructions" in text
    assert "calculate" in text and "final_answer" in text
    assert agent_prompt_hash() == sha256_hex(text.encode("utf-8"))
    assert AGENT_SYSTEM == "agent_system.md"


def test_agent_prompt_is_registered_in_the_shared_prompt_registry() -> None:
    """``secqa.rag.prompt_hashes`` is the one registry of every prompts/*.md (recorded on eval
    records); the agent's own hash must be the same bytes-hash the registry publishes."""
    from secqa import rag

    assert rag.AGENT_SYSTEM == AGENT_SYSTEM
    assert AGENT_SYSTEM in rag.PROMPT_NAMES
    assert rag.prompt_hashes()[AGENT_SYSTEM] == agent_prompt_hash()


def test_user_prompt_carries_question_and_hints() -> None:
    assert build_agent_prompt("  What   is\nrevenue? ") == "Question: What is revenue?"
    prompt = build_agent_prompt(
        "q", RetrievalFilters(ticker="FIXT", fiscal_year=2023, form="10-K", doc_names=["A", "B"])
    )
    assert prompt == "Question: q\nHints: ticker=FIXT fiscal_year=2023 form=10-K documents=A,B"
    with pytest.raises(ValueError, match="blank"):
        build_agent_prompt("   ")


# ---- happy path --------------------------------------------------------------------------


def test_happy_path_search_lookup_calculate_final(
    make_loop: LoopFactory, runtime: ToolRuntime
) -> None:
    provider = RecordingScripted(
        [
            search_turn("total net sales fiscal 2023", ticker=TICKER, fiscal_year=2023, k=4),
            {
                "match": "chunk:",
                "tool_calls": [
                    {
                        "name": "lookup_fact",
                        "arguments": {"ticker": TICKER, "metric": "revenue", "fiscal_year": 2023},
                    }
                ],
            },
            {
                "match": REVENUE_REF.replace("|", r"\|"),
                "tool_calls": [
                    {"name": "calculate", "arguments": {"expression": "1577000000 / 1000000"}}
                ],
            },
            final_turn(
                NET_SALES_SENTENCE,
                citations=[net_sales_citation(), {"ref": REVENUE_REF, "quote": ""}],
                calculation="1577000000 / 1000000",
                match='"result": 1577',
            ),
        ]
    )
    answer = make_loop(provider).run(NET_SALES_QUESTION, request_id="req-1")

    assert answer.mode == "agent" and answer.terminated_by == "final_answer"
    assert answer.request_id == "req-1" and answer.question == NET_SALES_QUESTION
    assert answer.text == NET_SALES_SENTENCE and not answer.abstained
    assert answer.value == REVENUE_USD and answer.unit == "USD"
    assert [c.ref for c in answer.citations] == [f"chunk:{net_sales_chunk_id()}", REVENUE_REF]
    assert all(c.valid and c.verified for c in answer.citations)
    assert answer.citations[0].doc_name == TOP_DOC and answer.citations[0].page_num == 1
    assert answer.citations[0].snippet.startswith("Total net sales")
    assert answer.citations[1].kind == "xbrl" and answer.citations[1].value == REVENUE_USD
    assert answer.grounded is True
    assert answer.calculation == "1577000000 / 1000000 = 1577"
    assert answer.steps == 4 and answer.tool_calls == 4
    assert trace_kinds(answer) == ["llm", "tool"] * 4 + ["verify"]
    assert [step.step for step in answer.trace] == list(range(1, 10))
    assert [s.name for s in answer.trace if s.kind == "tool"] == [
        "search_filings",
        "lookup_fact",
        "calculate",
        "final_answer",
    ]
    assert all(s.usage is not None for s in answer.trace if s.kind == "llm")
    assert answer.usage.input_tokens > 0 and answer.usage.output_tokens > 0
    assert answer.cost_usd == 0.0  # scripted providers are free
    assert answer.retrieved and answer.retrieved[0].doc_name == TOP_DOC
    assert answer.retrieval_ms > 0 and answer.latency_ms > 0
    assert answer.prompt_hashes == {AGENT_SYSTEM: agent_prompt_hash()}
    assert answer.provider == "scripted"
    assert provider.turns_consumed == 4
    assert runtime.seen_facts[REVENUE_REF].accn.endswith("-24-000010")


class CachedScripted(RecordingScripted):
    """Every reply looks like a cassette hit: ``cached=True`` with a recorded latency."""

    RECORDED_MS = 400.0

    def complete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        tools: list[ToolSpec] | None = None,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 2048,
        effort: Any = None,
    ) -> LLMResponse:
        response = super().complete(
            messages,
            system=system,
            tools=tools,
            json_schema=json_schema,
            max_tokens=max_tokens,
            effort=effort,
        )
        return response.model_copy(update={"cached": True, "latency_ms": self.RECORDED_MS})


def test_replayed_calls_report_the_recorded_llm_latency(make_loop: LoopFactory) -> None:
    """Cassette hits answer in microseconds; ``llm_ms`` sums the recorded per-call latency."""
    provider = CachedScripted(
        [
            search_turn("total net sales fiscal 2023"),
            final_turn(NET_SALES_SENTENCE, citations=[net_sales_citation()]),
        ]
    )
    answer = make_loop(provider).run(NET_SALES_QUESTION)

    assert answer.steps == 2
    assert answer.llm_ms == 2 * CachedScripted.RECORDED_MS
    assert answer.latency_ms >= answer.llm_ms
    assert [s.latency_ms for s in answer.trace if s.kind == "llm"] == [400.0, 400.0]


def test_message_protocol_one_tool_message_per_step(make_loop: LoopFactory) -> None:
    provider = RecordingScripted(
        [
            {
                "tool_calls": [
                    {"name": "calculate", "arguments": {"expression": "1 + 1"}},
                    {"name": "calculate", "arguments": {"expression": "2 + 2"}},
                    {"name": "lookup_company", "arguments": {"name_or_ticker": "fixt"}},
                ]
            },
            final_turn("INSUFFICIENT EVIDENCE", value=None, unit=None, abstain=True),
        ]
    )
    loop = make_loop(provider, effort="low", max_tokens=512)
    answer = loop.run("q", filters=RetrievalFilters(ticker=TICKER))

    first, second = provider.calls
    assert first["system"] == load_agent_prompt()
    assert first["tools"] == TOOLS and first["json_schema"] is None
    assert first["effort"] == "low" and first["max_tokens"] == 512
    assert [m.role for m in first["messages"]] == ["user"]
    assert first["messages"][0].content == f"Question: q\nHints: ticker={TICKER}"

    roles = [m.role for m in second["messages"]]
    assert roles == ["user", "assistant", "tool"]
    assistant, tool = second["messages"][1], second["messages"][2]
    assert [c.name for c in assistant.tool_calls] == ["calculate", "calculate", "lookup_company"]
    assert [r.tool_call_id for r in tool.tool_results] == [c.id for c in assistant.tool_calls]
    assert [r.name for r in tool.tool_results] == ["calculate", "calculate", "lookup_company"]
    assert not any(r.is_error for r in tool.tool_results)
    assert answer.tool_calls == 4 and answer.steps == 2
    assert answer.calculation == "1 + 1 = 2; 2 + 2 = 4"
    assert answer.abstained and answer.terminated_by == "final_answer"


# ---- final_answer validation -------------------------------------------------------------


def test_invalid_ref_gets_error_result_then_retry_succeeds(make_loop: LoopFactory) -> None:
    provider = RecordingScripted(
        [
            search_turn("net sales"),
            final_turn(citations=[{"ref": "chunk:" + "f" * 40, "quote": NET_SALES_QUOTE}]),
            final_turn(citations=[net_sales_citation()], match="unknown citation ref"),
        ]
    )
    answer = make_loop(provider).run(NET_SALES_QUESTION)
    assert answer.terminated_by == "final_answer" and not answer.abstained
    assert answer.steps == 3 and answer.tool_calls == 3
    tool_steps = [s for s in answer.trace if s.kind == "tool"]
    assert [s.error is not None for s in tool_steps] == [False, True, False]
    assert "unknown citation ref" in (tool_steps[1].error or "")
    assert [c.verified for c in answer.citations] == [True]
    error_result = provider.calls[2]["messages"][-1].tool_results[0]
    assert error_result.is_error and error_result.name == "final_answer"


def test_final_answer_rejected_twice_abstains_without_final_call(
    make_loop: LoopFactory,
) -> None:
    bad = {"ref": "chunk:" + "f" * 40, "quote": NET_SALES_QUOTE}
    provider = RecordingScripted(
        [search_turn("net sales"), final_turn(citations=[bad]), final_turn(citations=[bad])]
    )
    answer = make_loop(provider).run(NET_SALES_QUESTION)
    assert answer.abstained and answer.text == "INSUFFICIENT EVIDENCE"
    assert answer.terminated_by == "error"
    assert answer.steps == 3 and provider.turns_consumed == 3  # no tools-off final call
    assert answer.citations == []
    assert "rejected 2 times" in (answer.trace[-1].error or "")


def test_unknown_refs_in_forced_final_call_are_kept_invalid(make_loop: LoopFactory) -> None:
    provider = RecordingScripted(
        [
            search_turn("net sales"),
            {
                "match": "Stop using tools",
                "parsed": {
                    "answer": "Net sales were $1,577 million.",
                    "value": REVENUE_USD,
                    "unit": "USD",
                    "citations": [{"ref": "chunk:" + "a" * 40, "quote": NET_SALES_QUOTE}],
                    "calculation": None,
                    "abstain": False,
                },
            },
        ]
    )
    answer = make_loop(provider, max_steps=1).run(NET_SALES_QUESTION)
    assert answer.terminated_by == "max_steps" and not answer.abstained
    (citation,) = answer.citations
    assert citation.valid is False and citation.verified is False and citation.snippet == ""
    assert answer.grounded is False


# ---- stopping rules ----------------------------------------------------------------------


def test_max_steps_forces_final_call_from_yaml_scenario(make_loop: LoopFactory) -> None:
    provider = RecordingScripted(SCENARIOS / "max_steps.yaml")
    answer = make_loop(provider, max_steps=2).run(NET_SALES_QUESTION)
    assert answer.terminated_by == "max_steps"
    assert answer.abstained and answer.text == "INSUFFICIENT EVIDENCE"
    assert answer.steps == 3 and answer.tool_calls == 2
    final_call = provider.calls[-1]
    assert final_call["tools"] is None and final_call["json_schema"] == FINAL_ANSWER_SCHEMA
    assert final_call["messages"][-1].role == "user"
    assert final_call["messages"][-1].content == FINAL_PROMPT.format(reason="max_steps")
    llm_steps = [s for s in answer.trace if s.kind == "llm"]
    assert [s.arguments["tools"] for s in llm_steps] == [True, True, False]
    assert trace_kinds(answer) == ["llm", "tool", "llm", "tool", "llm", "verify"]
    assert answer.retrieved  # evidence gathered before the abort is still reported


def test_three_identical_calls_abort(make_loop: LoopFactory, runtime: ToolRuntime) -> None:
    same = search_turn("net sales", k=3)
    provider = RecordingScripted(
        [same, same, same, {"match": "Stop using tools", "text": abstain_final_json()}]
    )
    answer = make_loop(provider).run(NET_SALES_QUESTION)
    assert answer.terminated_by == "error"
    assert answer.steps == 4 and answer.tool_calls == 3
    assert len(runtime.tool_log) == 2  # the third identical call was not executed
    third = [s for s in answer.trace if s.kind == "tool"][2]
    assert third.error and "identical arguments" in third.error
    assert "identical arguments" in provider.calls[-1]["messages"][-1].content
    assert answer.abstained


def test_same_tool_failing_twice_in_a_row_aborts(
    make_loop: LoopFactory, store: DuckDBStore
) -> None:
    before = store.counts()["facts"]
    provider = RecordingScripted(
        [
            {"tool_calls": [{"name": "query_xbrl", "arguments": {"sql": "DROP TABLE xbrl_facts"}}]},
            {
                "match": "allowed",
                "tool_calls": [
                    {"name": "query_xbrl", "arguments": {"sql": "DELETE FROM xbrl_facts"}}
                ],
            },
            {"match": "Stop using tools", "text": abstain_final_json()},
        ]
    )
    answer = make_loop(provider).run("How many facts are there?")
    assert answer.terminated_by == "error"
    assert "query_xbrl failed twice" in (answer.trace[-1].error or "")
    assert answer.steps == 3 and answer.tool_calls == 2
    assert store.counts()["facts"] == before


def test_failures_of_different_tools_do_not_abort(make_loop: LoopFactory) -> None:
    provider = RecordingScripted(
        [
            {"tool_calls": [{"name": "query_xbrl", "arguments": {"sql": "DROP TABLE xbrl_facts"}}]},
            {"tool_calls": [{"name": "calculate", "arguments": {"expression": "x"}}]},
            {"tool_calls": [{"name": "query_xbrl", "arguments": {"sql": "SELECT 1 AS one"}}]},
            final_turn("INSUFFICIENT EVIDENCE", value=None, unit=None, abstain=True),
        ]
    )
    answer = make_loop(provider).run("q")
    assert answer.terminated_by == "final_answer" and answer.steps == 4


def test_tool_call_cap_exempts_final_answer(make_loop: LoopFactory) -> None:
    provider = RecordingScripted(
        [
            search_turn("net sales"),
            final_turn(citations=[net_sales_citation()]),
        ]
    )
    answer = make_loop(provider, max_tool_calls=1).run(NET_SALES_QUESTION)
    assert answer.terminated_by == "final_answer" and answer.tool_calls == 2


def test_tool_call_cap_aborts_with_budget(make_loop: LoopFactory) -> None:
    provider = RecordingScripted(
        [
            search_turn("net sales"),
            search_turn("operating income"),
            {"match": "Stop using tools", "text": abstain_final_json()},
        ]
    )
    answer = make_loop(provider, max_tool_calls=1).run(NET_SALES_QUESTION)
    assert answer.terminated_by == "budget"
    second_tool = [s for s in answer.trace if s.kind == "tool"][1]
    assert second_tool.error and "budget of 1 exhausted" in second_tool.error
    assert answer.steps == 3 and answer.tool_calls == 2


def test_cost_cap_aborts_before_the_next_call(make_loop: LoopFactory) -> None:
    usage = {"input_tokens": 100_000, "output_tokens": 1_000}  # $0.416 per call at gpt-test
    provider = PricedScripted(
        [
            {**search_turn("net sales"), "usage": usage},
            {"match": "Stop using tools", "text": abstain_final_json(), "usage": usage},
            {**search_turn("never reached"), "usage": usage},
        ]
    )
    answer = make_loop(provider, max_cost_usd=0.5).run(NET_SALES_QUESTION)
    assert answer.terminated_by == "budget"
    assert "projected cost" in (answer.trace[-1].error or "")
    assert answer.steps == 2 and provider.turns_consumed == 2
    assert answer.cost_usd == pytest.approx(0.832)
    assert answer.provider == "openai" and answer.model == "gpt-test"


class SnapshotEchoScripted(PricedScripted):
    """``openai:gpt-test`` whose responses echo a dated snapshot id, as real vendors do."""

    def complete(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        response = super().complete(messages, **kwargs)
        return response.model_copy(update={"model": f"{self.model}-2026-06-01"})


def test_cost_is_priced_on_the_configured_id_not_the_vendor_echo(make_loop: LoopFactory) -> None:
    """Regression: pricing keyed on ``response.model`` raised ConfigError for the dated
    snapshot id OpenAI echoes (``gpt-5.5-2026-06-01``) after the call had been paid for."""
    usage = {"input_tokens": 1_000, "output_tokens": 100}  # $0.0056 per call at gpt-test
    provider = SnapshotEchoScripted(
        [
            {**search_turn("net sales"), "usage": usage},
            {**final_turn(citations=[net_sales_citation()]), "usage": usage},
        ]
    )
    answer = make_loop(provider, max_cost_usd=0.5).run(NET_SALES_QUESTION)
    assert answer.terminated_by == "final_answer"
    assert answer.cost_usd == pytest.approx(0.0112)
    assert answer.provider == "openai" and answer.model == "gpt-test"
    llm_steps = [step for step in answer.trace if step.kind == "llm"]
    assert llm_steps and all(step.name == "openai:gpt-test-2026-06-01" for step in llm_steps)


def test_cost_cap_lets_a_cheap_run_finish(make_loop: LoopFactory) -> None:
    usage = {"input_tokens": 1_000, "output_tokens": 100}  # $0.0056 per call
    provider = PricedScripted(
        [
            {**search_turn("net sales"), "usage": usage},
            {**final_turn(citations=[net_sales_citation()]), "usage": usage},
        ]
    )
    answer = make_loop(provider, max_cost_usd=0.5).run(NET_SALES_QUESTION)
    assert answer.terminated_by == "final_answer"
    assert answer.cost_usd == pytest.approx(0.0112)


def test_input_token_cap_counts_cached_tokens(make_loop: LoopFactory) -> None:
    provider = RecordingScripted(
        [
            {
                **search_turn("net sales"),
                "usage": {"input_tokens": 100, "cache_read_tokens": 950, "output_tokens": 10},
            },
            {"match": "Stop using tools", "text": abstain_final_json()},
        ]
    )
    answer = make_loop(provider, max_input_tokens=1000).run(NET_SALES_QUESTION)
    assert answer.terminated_by == "budget"
    assert "prompt tokens 1050 exceeded 1000" in (answer.trace[-1].error or "")
    assert answer.usage.cache_read_tokens == 950


def test_wall_clock_abort_before_any_tool_call(make_loop: LoopFactory) -> None:
    provider = RecordingScripted([{"match": "Stop using tools", "text": abstain_final_json()}])
    answer = make_loop(provider, wall_clock_s=1e-9).run(NET_SALES_QUESTION)
    assert answer.terminated_by == "budget" and answer.abstained
    assert answer.steps == 1 and answer.tool_calls == 0 and answer.retrieved == []
    assert "wall clock" in (answer.trace[-1].error or "")


def test_forced_final_call_that_does_not_parse_abstains(make_loop: LoopFactory) -> None:
    provider = RecordingScripted(
        [search_turn("net sales"), {"match": "Stop using tools", "text": "I cannot say."}]
    )
    answer = make_loop(provider, max_steps=1).run(NET_SALES_QUESTION)
    assert answer.terminated_by == "max_steps" and answer.abstained
    assert answer.text == "INSUFFICIENT EVIDENCE" and answer.citations == []


# ---- text-only replies -------------------------------------------------------------------


def test_text_only_reply_accepted_when_it_parses_with_known_refs(
    make_loop: LoopFactory,
) -> None:
    final_json = json.dumps(
        {
            "answer": NET_SALES_SENTENCE,
            "value": REVENUE_USD,
            "unit": "USD",
            "citations": [net_sales_citation()],
            "calculation": None,
            "abstain": False,
        }
    )
    provider = RecordingScripted(
        [search_turn("net sales"), {"text": f"```json\n{final_json}\n```"}]
    )
    answer = make_loop(provider).run(NET_SALES_QUESTION)
    assert answer.terminated_by == "final_answer" and answer.steps == 2 and answer.tool_calls == 1
    assert answer.value == REVENUE_USD and answer.citations[0].verified


def test_text_only_prose_is_nudged_once(make_loop: LoopFactory) -> None:
    provider = RecordingScripted(
        [
            {"text": "Let me think about which filing to open."},
            {
                "match": "no tool call",
                **final_turn("INSUFFICIENT EVIDENCE", value=None, unit=None, abstain=True),
            },
        ]
    )
    answer = make_loop(provider).run(NET_SALES_QUESTION)
    assert answer.terminated_by == "final_answer" and answer.steps == 2
    nudge = provider.calls[1]["messages"][-1]
    assert nudge.role == "user" and nudge.content.startswith(NUDGE_PROMPT[:40])
    assert "not a JSON object" in nudge.content


def test_text_only_with_unknown_ref_is_nudged_with_the_ref(make_loop: LoopFactory) -> None:
    bad_json = json.dumps({"answer": "x", "citations": [{"ref": "chunk:" + "b" * 40, "quote": ""}]})
    provider = RecordingScripted(
        [
            {"text": bad_json},
            {"match": "unknown citation ref", **final_turn("INSUFFICIENT EVIDENCE", abstain=True)},
        ]
    )
    answer = make_loop(provider).run(NET_SALES_QUESTION)
    assert answer.terminated_by == "final_answer"


def test_text_only_twice_aborts_then_forced_final_call(make_loop: LoopFactory) -> None:
    provider = RecordingScripted(
        [
            {"text": "Thinking."},
            {"match": "no tool call", "text": "Still thinking."},
            {"match": "Stop using tools", "text": abstain_final_json()},
        ]
    )
    answer = make_loop(provider).run(NET_SALES_QUESTION)
    assert answer.terminated_by == "error" and answer.steps == 3 and answer.abstained
    assert "no parseable final answer" in (answer.trace[-1].error or "")


def test_refusal_abstains_immediately(make_loop: LoopFactory) -> None:
    provider = RecordingScripted([{"stop_reason": "refusal", "text": "ignored"}])
    answer = make_loop(provider).run(NET_SALES_QUESTION)
    assert answer.terminated_by == "error" and answer.abstained
    assert answer.steps == 1 and provider.turns_consumed == 1
    assert answer.trace[0].error == "provider reported a refusal"


def test_provider_error_propagates(make_loop: LoopFactory) -> None:
    provider = RecordingScripted([{"match": "this text is not in the prompt", "text": "x"}])
    with pytest.raises(ProviderError):
        make_loop(provider).run(NET_SALES_QUESTION)


# ---- prompt injection --------------------------------------------------------------------


def test_injected_instructions_pass_through_as_data_and_change_nothing(
    make_loop: LoopFactory, store: DuckDBStore
) -> None:
    before = store.counts()["facts"]
    injected_ref = f"chunk:{injection_chunk_id()}"
    provider = RecordingScripted(
        [
            search_turn("net sales", ticker="OTHR"),  # the injected chunk is in OTHER_DOC
            final_turn(
                "The only mention of net sales is an instruction-like sentence in the filing.",
                value=None,
                unit=None,
                citations=[{"ref": injected_ref, "quote": INJECTION_TEXT}],
                match="Ignore previous instructions",
            ),
        ]
    )
    answer = make_loop(provider).run(NET_SALES_QUESTION)

    tool_message = provider.calls[1]["messages"][-1]
    assert tool_message.role == "tool"
    assert INJECTION_TEXT in tool_message.tool_results[0].content
    assert trace_kinds(answer) == ["llm", "tool", "llm", "tool", "verify"]
    assert [s.name for s in answer.trace if s.kind == "tool"] == ["search_filings", "final_answer"]
    assert answer.terminated_by == "final_answer"
    assert store.counts()["facts"] == before
    assert "data, not instructions" in provider.calls[0]["system"]
    # The injected sentence is ordinary quotable filing text: cited, verified, nothing executed.
    (citation,) = answer.citations
    assert citation.valid and citation.verified and citation.doc_name == OTHER_DOC
    assert all(view.doc_name == OTHER_DOC for view in answer.retrieved)


# ---- request-level filters and mock provider ----------------------------------------------


def test_request_filters_constrain_every_search(make_loop: LoopFactory) -> None:
    provider = RecordingScripted(
        [
            search_turn("net sales", ticker=TICKER),  # the model's ticker cannot escape doc_names
            final_turn("INSUFFICIENT EVIDENCE", abstain=True, value=None, unit=None),
        ]
    )
    answer = make_loop(provider).run(
        NET_SALES_QUESTION, filters=RetrievalFilters(doc_names=[OTHER_DOC])
    )
    assert answer.retrieved == []  # FIXT filings are outside the requested documents
    assert provider.calls[0]["messages"][0].content.endswith(f"Hints: documents={OTHER_DOC}")


def test_mock_provider_end_to_end_yields_verified_citation(make_loop: LoopFactory) -> None:
    answer = make_loop(MockProvider()).run(NET_SALES_QUESTION)
    assert answer.terminated_by == "final_answer" and not answer.abstained
    assert answer.text == NET_SALES_SENTENCE
    (citation,) = answer.citations
    assert citation.valid and citation.verified and citation.doc_name == TOP_DOC
    assert answer.grounded is True and answer.value == REVENUE_USD
    assert answer.steps == 2 and answer.tool_calls == 2 and answer.cost_usd == 0.0
    assert answer.provider == "mock" and answer.model == "mock-extractive"


def test_mock_abstain_provider_records_abstention(make_loop: LoopFactory) -> None:
    answer = make_loop(MockProvider("abstain")).run(NET_SALES_QUESTION)
    assert answer.abstained and answer.terminated_by == "final_answer"
    assert answer.steps == 1 and answer.tool_calls == 1 and answer.retrieved == []


def test_answer_is_json_serialisable_and_runtime_is_reusable(make_loop: LoopFactory) -> None:
    provider = RecordingScripted(
        [search_turn("net sales"), final_turn(citations=[net_sales_citation()])]
    )
    loop = make_loop(provider)
    first = loop.run(NET_SALES_QUESTION)
    data = json.loads(first.model_dump_json())
    assert data["mode"] == "agent" and data["citations"][0]["verified"] is True
    assert len(first.request_id) == 32

    provider.reset()
    provider.calls.clear()
    second = loop.run(NET_SALES_QUESTION)
    assert second.request_id != first.request_id
    assert second.tool_calls == first.tool_calls == 2
    assert [s.step for s in second.trace] == [1, 2, 3, 4, 5]


# ---- the wall clock is enforced in flight ---------------------------------------------------


class DeadlineRecordingScripted(RecordingScripted):
    """Records the request deadline the vendor adapters would see at each call."""

    def __init__(self, scenario: list[dict[str, Any]]) -> None:
        super().__init__(scenario)
        self.remaining: list[float | None] = []

    def complete(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.remaining.append(remaining_s())
        return super().complete(messages, **kwargs)


def test_tool_enabled_calls_run_under_the_wall_clock_deadline(make_loop: LoopFactory) -> None:
    """Regression: the wall clock was only checked between steps, so an in-flight vendor call
    could block for its full SDK timeout (times retries) after the budget was gone. Every
    tool-enabled call must now see a deadline no larger than ``wall_clock_s``."""
    provider = DeadlineRecordingScripted(
        [search_turn("net sales"), final_turn(citations=[net_sales_citation()])]
    )
    answer = make_loop(provider, wall_clock_s=30.0).run(NET_SALES_QUESTION)
    assert answer.terminated_by == "final_answer"
    assert len(provider.remaining) == 2
    assert all(left is not None and 0.0 < left <= 30.0 for left in provider.remaining)
    assert remaining_s() is None, "the deadline does not leak out of run()"


def test_final_call_after_an_abort_runs_outside_the_deadline(make_loop: LoopFactory) -> None:
    """The tools-off final call is what turns an abort into an answer; it gets the provider's own
    timeout window rather than the (already spent) wall clock."""
    provider = DeadlineRecordingScripted(
        [search_turn("net sales"), {"match": "Stop using tools", "text": abstain_final_json()}]
    )
    answer = make_loop(provider, max_steps=1, wall_clock_s=30.0).run(NET_SALES_QUESTION)
    assert answer.terminated_by == "max_steps"
    first, final = provider.remaining
    assert first is not None and 0.0 < first <= 30.0
    assert final is None


def test_wall_clock_abort_final_call_has_no_deadline(make_loop: LoopFactory) -> None:
    provider = DeadlineRecordingScripted(
        [{"match": "Stop using tools", "text": abstain_final_json()}]
    )
    answer = make_loop(provider, wall_clock_s=1e-9).run(NET_SALES_QUESTION)
    assert answer.terminated_by == "budget"
    assert provider.remaining == [None]
