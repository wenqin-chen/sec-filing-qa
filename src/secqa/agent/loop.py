"""``AgentLoop``: the manual, provider-neutral tool loop with hard stopping rules (SPEC 5).

One run is a sequence of steps. Each step is one ``provider.complete(messages, system=...,
tools=TOOLS)`` call; every tool call the model made in that step is executed through
:class:`~secqa.agent.runtime.ToolRuntime` and all results go back in ONE ``Message(role='tool')``
(CONTRACTS rule 1). The assistant message is appended exactly as the provider returned it so
the Anthropic adapter can replay its own signed turns. The loop ends when a ``final_answer``
call is accepted, or when a stopping rule fires:

===========================================  =================  ==================================
Rule                                         ``terminated_by``  What happens
===========================================  =================  ==================================
``final_answer`` accepted                    ``final_answer``   answer verified and returned
``max_steps`` LLM calls made                 ``max_steps``      one final call with ``tools=None``
``max_tool_calls`` evidence calls made       ``budget``         one final call with ``tools=None``
prompt tokens > ``max_input_tokens``         ``budget``         one final call with ``tools=None``
projected cost > ``max_cost_usd``            ``budget``         one final call with ``tools=None``
wall clock > ``wall_clock_s``                ``budget``         one final call with ``tools=None``
identical ``(tool, arguments)`` three times  ``error``          one final call with ``tools=None``
same tool fails twice in a row               ``error``          one final call with ``tools=None``
no tool call and no parseable answer, twice  ``error``          one final call with ``tools=None``
``final_answer`` rejected twice (bad refs)   ``error``          abstain, no further call
provider refusal                             ``error``          abstain, no further call
===========================================  =================  ==================================

The final call after an abort asks the model to answer from the evidence already gathered (or
abstain) as a JSON object matching :data:`~secqa.agent.tools.FINAL_ANSWER_SCHEMA`; its citations
are verified like any other, so an invented ref shows up as ``valid=False`` rather than being
dropped. A text-only reply during the loop is accepted as final only if it parses into that same
schema and its refs are known; otherwise the model is nudged once.

The wall clock is also enforced *in flight*: tool-enabled calls run inside
``secqa.providers.deadline.deadline(wall_clock_s)``, so the vendor adapters shrink each HTTP
attempt (and drop retries) to the time left rather than blocking for their full timeout after
the budget is gone. The tools-off final call runs outside that deadline, with the provider's own
timeout, so an abort still produces an answer.

Provider failures (:class:`~secqa.core.errors.ProviderError`) propagate, exactly as in the rag
pipeline: the API maps them to 502 and the harness records them per question.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Literal

from secqa.agent.runtime import ToolRuntime
from secqa.agent.tools import (
    FINAL_ANSWER,
    FINAL_ANSWER_SCHEMA,
    TOOLS,
    FinalAnswer,
    abstain_answer,
    decode_final_answer,
    parse_final_answer_text,
)
from secqa.core.contracts import (
    Answer,
    LLMProvider,
    LLMResponse,
    Message,
    RetrievalFilters,
    Terminated,
    ToolCall,
    ToolResult,
    TraceStep,
    Usage,
)
from secqa.core.errors import ConfigError
from secqa.core.ids import request_id as make_request_id
from secqa.core.ids import sha256_hex
from secqa.core.logging import get_logger
from secqa.grounding import CitationVerifier
from secqa.providers.deadline import deadline
from secqa.providers.pricing import PriceTable

log = get_logger(__name__)

Effort = Literal["low", "medium", "high"]
EFFORTS: tuple[str, ...] = ("low", "medium", "high")

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
AGENT_SYSTEM = "agent_system.md"

DEFAULT_MAX_STEPS = 8
DEFAULT_MAX_TOOL_CALLS = 12
DEFAULT_MAX_COST_USD = 0.5
DEFAULT_MAX_INPUT_TOKENS = 200_000
DEFAULT_WALL_CLOCK_S = 90.0
DEFAULT_EFFORT: Effort = "medium"
DEFAULT_MAX_TOKENS = 2048
IDENTICAL_CALL_LIMIT = 3
FINAL_ANSWER_ATTEMPTS = 2

WALL_CLOCK_ABORT_PREFIX = "wall clock"
COST_ABORT_PREFIX = "projected cost"
"""The two ``budget`` abort reasons callers tell apart (the API maps them to 504 / 402); each
is the leading text of the final ``verify`` :class:`TraceStep`'s ``error`` for that abort."""

NUDGE_PROMPT = (
    "Your last message contained no tool call and was not a valid final answer ({reason}). "
    "Either call a tool to gather evidence or call final_answer with your answer."
)
FINAL_PROMPT = (
    "Stop using tools: {reason}. Using ONLY the evidence already gathered above, answer the "
    "question now as a single JSON object with the fields answer, value, unit, citations, "
    "calculation and abstain. Cite only refs that tools returned in this conversation. If the "
    "evidence is insufficient, set abstain to true and answer exactly INSUFFICIENT EVIDENCE."
)
_PREVIEW_CHARS = 200


@cache
def load_agent_prompt() -> str:
    """Return the agent system prompt (``prompts/agent_system.md``; cached per process).

    Raises:
        ConfigError: if the file is missing or empty.
    """
    path = PROMPTS_DIR / AGENT_SYSTEM
    if not path.is_file():
        raise ConfigError(f"prompt file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ConfigError(f"prompt file is empty: {path}")
    return text


def agent_prompt_hash() -> str:
    """SHA-256 of the agent system prompt bytes (recorded in ``Answer.prompt_hashes``)."""
    return sha256_hex(load_agent_prompt().encode("utf-8"))


def build_agent_prompt(question: str, filters: RetrievalFilters | None = None) -> str:
    """The first user message: the question plus any request-level hints (ticker, year ...)."""
    if not question.strip():
        raise ValueError("question must not be blank")
    lines = [f"Question: {' '.join(question.split())}"]
    hints = _format_hints(filters)
    if hints:
        lines.append(f"Hints: {hints}")
    return "\n".join(lines)


@dataclass
class _RunState:
    """Mutable bookkeeping for one :meth:`AgentLoop.run` call."""

    started: float
    messages: list[Message]
    trace: list[TraceStep] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    last_call_cost: float = 0.0
    prompt_tokens: int = 0
    llm_ms: float = 0.0  # reported LLM time: recorded latency for cassette hits, else measured
    llm_wall_ms: float = 0.0  # measured LLM time (what the wall clock above actually contains)
    llm_calls: int = 0
    tool_calls: int = 0
    evidence_calls: int = 0
    call_counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    last_failed_tool: str | None = None
    nudged: bool = False
    final_rejections: int = 0
    final: FinalAnswer | None = None
    terminated_by: Terminated | None = None
    abort_reason: str | None = None
    needs_final_call: bool = False

    def add_trace(self, step: TraceStep) -> None:
        self.trace.append(step.model_copy(update={"step": len(self.trace) + 1}))


class AgentLoop:
    """Answer questions with tools under a fixed budget; see the module docstring for the rules.

    Args:
        provider: Any :class:`~secqa.core.contracts.LLMProvider`.
        runtime: Tool runtime bound to the store / retriever; reset at the start of every run.
        verifier: Citation verifier shared with the rag pipeline and the API.
        prices: Price table used to turn usage into ``cost_usd`` and to enforce ``max_cost_usd``.
        max_steps: Maximum LLM calls with tools offered.
        max_tool_calls: Maximum evidence-gathering tool calls (``final_answer`` is exempt so a
            model at the cap can still finish).
        max_cost_usd: Abort when the cost so far plus the cost of one more call like the last
            would exceed this.
        max_input_tokens: Abort when cumulative prompt tokens (uncached + cache read + cache
            write) exceed this.
        wall_clock_s: Abort when the run has taken longer than this; also the in-flight
            deadline of every tool-enabled provider call.
        effort: Reasoning effort forwarded to providers that support it.
        max_tokens: Output token cap per LLM call.
    """

    def __init__(
        self,
        provider: LLMProvider,
        runtime: ToolRuntime,
        verifier: CitationVerifier,
        prices: PriceTable,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
        max_cost_usd: float = DEFAULT_MAX_COST_USD,
        max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
        wall_clock_s: float = DEFAULT_WALL_CLOCK_S,
        effort: str = DEFAULT_EFFORT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        _check_positive_int("max_steps", max_steps)
        _check_positive_int("max_tool_calls", max_tool_calls)
        _check_positive_int("max_input_tokens", max_input_tokens)
        _check_positive_int("max_tokens", max_tokens)
        if max_cost_usd < 0:
            raise ValueError(f"max_cost_usd must be >= 0, got {max_cost_usd!r}")
        if wall_clock_s <= 0:
            raise ValueError(f"wall_clock_s must be positive, got {wall_clock_s!r}")
        if effort not in EFFORTS:
            raise ValueError(f"effort must be one of {', '.join(EFFORTS)}; got {effort!r}")
        self.provider = provider
        self.runtime = runtime
        self.verifier = verifier
        self.prices = prices
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls
        self.max_cost_usd = max_cost_usd
        self.max_input_tokens = max_input_tokens
        self.wall_clock_s = wall_clock_s
        self.effort: Effort = effort  # type: ignore[assignment]  # validated against EFFORTS
        self.max_tokens = max_tokens

    def run(
        self,
        question: str,
        *,
        filters: RetrievalFilters | None = None,
        request_id: str | None = None,
    ) -> Answer:
        """Run the loop for one question and return an ``Answer`` with ``mode='agent'``."""
        rid = request_id or make_request_id()
        system = load_agent_prompt()
        self.runtime.reset(filters)
        state = _RunState(
            started=time.perf_counter(),
            messages=[Message(role="user", content=build_agent_prompt(question, filters))],
        )
        log.info(
            "agent_run_started",
            request_id=rid,
            provider=self.provider.provider,
            model=self.provider.model,
            filters=filters.model_dump(exclude_none=True) if filters else {},
        )

        # The wall clock is checked between steps AND enforced in flight: inside this block the
        # vendor adapters cap every HTTP attempt (and drop retries) to the time left, so a
        # tool-enabled call can never outlive ``wall_clock_s``. The tools-off final call below
        # runs outside it, with the provider's own timeout, so an abort still yields an answer.
        with deadline(self.wall_clock_s):
            while state.final is None and state.terminated_by is None:
                reason = self._budget_exceeded(state)
                if reason is not None:
                    self._abort(state, "max_steps" if reason == "max_steps" else "budget", reason)
                    break
                response = self._complete(state, system, tools=True)
                if response.stop_reason == "refusal":
                    state.terminated_by = "error"
                    state.abort_reason = "provider refused the request"
                    state.final = abstain_answer()
                    break
                state.messages.append(
                    Message(role="assistant", content=response.text, tool_calls=response.tool_calls)
                )
                if response.tool_calls:
                    self._execute_step(state, response.tool_calls)
                else:
                    self._handle_text_only(state, response)

        if state.needs_final_call and state.final is None:
            state.final = self._final_call(state, system)

        return self._assemble(rid, question, state)

    # ---- one step ----------------------------------------------------------------------------

    def _complete(self, state: _RunState, system: str, *, tools: bool) -> LLMResponse:
        """One provider call; records usage, cost and a trace step (never swallows errors)."""
        llm_started = time.perf_counter()
        try:
            response = self.provider.complete(
                state.messages,
                system=system,
                tools=TOOLS if tools else None,
                json_schema=None if tools else FINAL_ANSWER_SCHEMA,
                max_tokens=self.max_tokens,
                effort=self.effort,
            )
        except Exception as exc:
            log.error(
                "agent_llm_failed",
                provider=self.provider.provider,
                model=self.provider.model,
                step=state.llm_calls + 1,
                error=str(exc),
            )
            raise
        wall_ms = (time.perf_counter() - llm_started) * 1000.0
        # A cassette hit answers in microseconds; the only real measurement of that call is the
        # one the recording run stored in ``response.latency_ms``, so report that instead.
        elapsed = response.latency_ms if response.cached else wall_ms
        # Price on the configured id (validated by the pre-flight checks), not the vendor echo:
        # OpenAI answers with a dated snapshot id that is not a key in models.yaml.
        cost = self.prices.cost_usd(self.provider.provider, self.provider.model, response.usage)
        state.llm_calls += 1
        state.llm_ms += elapsed
        state.llm_wall_ms += wall_ms
        state.usage = state.usage + response.usage
        state.cost_usd += cost
        state.last_call_cost = cost
        state.prompt_tokens += (
            response.usage.input_tokens
            + response.usage.cache_read_tokens
            + response.usage.cache_write_tokens
        )
        state.add_trace(
            TraceStep(
                step=0,
                kind="llm",
                name=f"{response.provider}:{response.model}",
                arguments={
                    "tools": tools,
                    "effort": self.effort,
                    "max_tokens": self.max_tokens,
                    "n_messages": len(state.messages),
                },
                result_preview=_response_preview(response),
                latency_ms=elapsed,
                usage=response.usage,
                error=_llm_error(response),
            )
        )
        return response

    def _execute_step(self, state: _RunState, calls: list[ToolCall]) -> None:
        """Execute every tool call of one step and append ONE tool message with all results."""
        results: list[ToolResult] = []
        for call in calls:
            state.tool_calls += 1
            result = self._dispatch_guarded(state, call)
            results.append(result)
            if call.name == FINAL_ANSWER:
                self._track_final(state, call, result)
            else:
                self._track_failures(state, call, result)
        state.messages.append(Message(role="tool", tool_results=results))

    def _dispatch_guarded(self, state: _RunState, call: ToolCall) -> ToolResult:
        """Apply the per-call rules (identical calls, tool-call cap) before dispatching."""
        key = (call.name, json.dumps(call.arguments, sort_keys=True, default=str))
        state.call_counts[key] += 1
        if state.call_counts[key] >= IDENTICAL_CALL_LIMIT:
            reason = f"{call.name} called {IDENTICAL_CALL_LIMIT} times with identical arguments"
            self._abort(state, "error", reason)
            return self._error_result(state, call, reason)
        if call.name != FINAL_ANSWER:
            if state.evidence_calls >= self.max_tool_calls:
                reason = f"tool call budget of {self.max_tool_calls} exhausted"
                self._abort(state, "budget", reason)
                return self._error_result(state, call, reason)
            state.evidence_calls += 1
        result = self.runtime.dispatch(call)
        state.add_trace(self.runtime.tool_log[-1])
        return result

    def _error_result(self, state: _RunState, call: ToolCall, reason: str) -> ToolResult:
        """An error result the loop produces itself (the tool is not executed)."""
        content = json.dumps({"error": reason})
        state.add_trace(
            TraceStep(
                step=0,
                kind="tool",
                name=call.name,
                arguments=dict(call.arguments),
                result_preview=content,
                error=reason,
            )
        )
        return ToolResult(tool_call_id=call.id, name=call.name, content=content, is_error=True)

    def _track_final(self, state: _RunState, call: ToolCall, result: ToolResult) -> None:
        if not result.is_error:
            state.final = self.runtime.final_answer_from(call.arguments)
            state.terminated_by = "final_answer"
            return
        state.final_rejections += 1
        if state.final_rejections >= FINAL_ANSWER_ATTEMPTS:
            # SPEC: invalid refs -> error result, one retry, then abstain (no further call).
            state.terminated_by = "error"
            state.abort_reason = f"final_answer rejected {state.final_rejections} times"
            state.final = abstain_answer()
            log.warning("agent_final_answer_rejected", attempts=state.final_rejections)

    def _track_failures(self, state: _RunState, call: ToolCall, result: ToolResult) -> None:
        if not result.is_error:
            state.last_failed_tool = None
            return
        if state.last_failed_tool == call.name and state.terminated_by is None:
            self._abort(state, "error", f"{call.name} failed twice in a row")
        state.last_failed_tool = call.name

    def _handle_text_only(self, state: _RunState, response: LLMResponse) -> None:
        """A reply without tool calls: accept it as final if it parses and cites known refs."""
        final, reason = parse_final_answer_text(response.text)
        if final is not None:
            invalid = self.runtime.invalid_refs(final.citations)
            if invalid:
                reason = f"unknown citation ref(s): {', '.join(invalid)}"
                final = None
        if final is not None:
            state.final = final
            state.terminated_by = "final_answer"
            return
        if not state.nudged:
            state.nudged = True
            state.messages.append(Message(role="user", content=NUDGE_PROMPT.format(reason=reason)))
            log.info("agent_nudged", reason=reason)
            return
        self._abort(state, "error", f"no tool call and no parseable final answer ({reason})")

    # ---- stopping rules ----------------------------------------------------------------------

    def _budget_exceeded(self, state: _RunState) -> str | None:
        """Reason the next tool-enabled LLM call must not happen, or ``None``."""
        if state.llm_calls >= self.max_steps:
            return "max_steps"
        elapsed = time.perf_counter() - state.started
        if elapsed > self.wall_clock_s:
            return f"{WALL_CLOCK_ABORT_PREFIX} {elapsed:.1f}s exceeded {self.wall_clock_s:g}s"
        if state.prompt_tokens > self.max_input_tokens:
            return (
                f"cumulative prompt tokens {state.prompt_tokens} exceeded {self.max_input_tokens}"
            )
        projected = state.cost_usd + state.last_call_cost
        if state.llm_calls > 0 and projected > self.max_cost_usd:
            return f"{COST_ABORT_PREFIX} ${projected:.4f} exceeds cap ${self.max_cost_usd:.2f}"
        return None

    def _abort(self, state: _RunState, terminated_by: Terminated, reason: str) -> None:
        if state.terminated_by is not None:
            return
        state.terminated_by = terminated_by
        state.abort_reason = reason
        state.needs_final_call = True
        log.warning("agent_abort", terminated_by=terminated_by, reason=reason)

    def _final_call(self, state: _RunState, system: str) -> FinalAnswer:
        """The one tools-off call after an abort; falls back to abstention if it cannot parse."""
        reason = state.abort_reason or "budget exhausted"
        state.messages.append(Message(role="user", content=FINAL_PROMPT.format(reason=reason)))
        response = self._complete(state, system, tools=False)
        state.messages.append(Message(role="assistant", content=response.text))
        if response.stop_reason == "refusal":
            return abstain_answer()
        final, why = _decode_parsed(response.parsed)
        if final is None:
            final, why = parse_final_answer_text(response.text)
        if final is None:
            log.warning("agent_final_call_unparsed", reason=why)
            return abstain_answer()
        return final

    # ---- assembly ----------------------------------------------------------------------------

    def _assemble(self, rid: str, question: str, state: _RunState) -> Answer:
        final = state.final or abstain_answer()
        terminated_by: Terminated = state.terminated_by or "error"
        refs = list(final.citations)
        verify_started = time.perf_counter()
        citations, grounded = self.verifier.verify(
            final.text,
            refs,
            self.runtime.seen_chunks,
            self.runtime.seen_facts,
            self.runtime.calc_results,
        )
        state.add_trace(
            TraceStep(
                step=0,
                kind="verify",
                name="citation_verifier",
                arguments={
                    "n_refs": len(refs),
                    "n_seen_chunks": len(self.runtime.seen_chunks),
                    "n_seen_facts": len(self.runtime.seen_facts),
                    "n_calc_results": len(self.runtime.calc_results),
                },
                result_preview=(
                    f"{sum(c.verified for c in citations)}/{len(citations)} verified, "
                    f"grounded={grounded}"
                ),
                latency_ms=(time.perf_counter() - verify_started) * 1000.0,
                error=state.abort_reason,
            )
        )
        answer = Answer(
            request_id=rid,
            question=question,
            text=final.text,
            value=final.value,
            unit=final.unit,
            abstained=final.abstained,
            citations=citations,
            grounded=grounded,
            calculation=self.runtime.rendered_calculation(),
            retrieved=list(self.runtime.retrieved),
            trace=state.trace,
            usage=state.usage,
            cost_usd=round(state.cost_usd, 8),
            # Whole-question wall clock with the measured LLM segments swapped for the reported
            # ones, so a replayed answer still satisfies ``latency_ms >= llm_ms``.
            latency_ms=(time.perf_counter() - state.started) * 1000.0
            - state.llm_wall_ms
            + state.llm_ms,
            retrieval_ms=self.runtime.retrieval_ms,
            llm_ms=state.llm_ms,
            provider=self.provider.provider,
            model=self.provider.model,
            mode="agent",
            steps=state.llm_calls,
            tool_calls=state.tool_calls,
            terminated_by=terminated_by,
            prompt_hashes={AGENT_SYSTEM: agent_prompt_hash()},
        )
        log.info(
            "agent_answer",
            request_id=rid,
            provider=answer.provider,
            model=answer.model,
            terminated_by=answer.terminated_by,
            abort_reason=state.abort_reason,
            abstained=answer.abstained,
            grounded=answer.grounded,
            steps=answer.steps,
            tool_calls=answer.tool_calls,
            n_retrieved=len(answer.retrieved),
            n_citations=len(answer.citations),
            n_verified=sum(c.verified for c in answer.citations),
            input_tokens=answer.usage.input_tokens,
            output_tokens=answer.usage.output_tokens,
            cost_usd=answer.cost_usd,
            latency_ms=round(answer.latency_ms, 1),
            retrieval_ms=round(answer.retrieval_ms, 1),
            llm_ms=round(answer.llm_ms, 1),
        )
        return answer


# ---- helpers ---------------------------------------------------------------------------------


def _decode_parsed(parsed: dict[str, object] | None) -> tuple[FinalAnswer | None, str | None]:
    """Decode a provider-parsed JSON object; ``(None, reason)`` when absent or invalid."""
    if parsed is None:
        return None, "provider returned no parsed JSON"
    try:
        return decode_final_answer(parsed), None
    except ValueError as exc:
        return None, str(exc)


def _format_hints(filters: RetrievalFilters | None) -> str:
    if filters is None:
        return ""
    fields: list[str] = []
    if filters.ticker:
        fields.append(f"ticker={filters.ticker}")
    if filters.fiscal_year is not None:
        fields.append(f"fiscal_year={filters.fiscal_year}")
    if filters.form:
        fields.append(f"form={filters.form}")
    if filters.doc_names:
        fields.append(f"documents={','.join(filters.doc_names)}")
    return " ".join(fields)


def _response_preview(response: LLMResponse) -> str:
    if response.tool_calls:
        calls = ", ".join(
            f"{call.name}({json.dumps(call.arguments, sort_keys=True, default=str)})"
            for call in response.tool_calls
        )
        return _preview(calls)
    return _preview(response.text)


def _llm_error(response: LLMResponse) -> str | None:
    if response.stop_reason == "max_tokens":
        return "output truncated at max_tokens"
    if response.stop_reason == "refusal":
        return "provider reported a refusal"
    return None


def _preview(text: str, limit: int = _PREVIEW_CHARS) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "..."


def _check_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an int >= 1, got {value!r}")


__all__ = [
    "AGENT_SYSTEM",
    "COST_ABORT_PREFIX",
    "DEFAULT_EFFORT",
    "DEFAULT_MAX_COST_USD",
    "DEFAULT_MAX_INPUT_TOKENS",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MAX_TOOL_CALLS",
    "DEFAULT_WALL_CLOCK_S",
    "FINAL_PROMPT",
    "NUDGE_PROMPT",
    "PROMPTS_DIR",
    "WALL_CLOCK_ABORT_PREFIX",
    "AgentLoop",
    "agent_prompt_hash",
    "build_agent_prompt",
    "load_agent_prompt",
]
