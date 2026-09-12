"""secqa.agent: provider-neutral tool loop over the local index with hard stopping rules.

Public surface: :data:`TOOLS` (the seven tool specs), :class:`ToolRuntime` (dispatches tool
calls, never raises, keeps the ledger of chunks / facts / calculations this run actually saw),
:func:`safe_calculate` (AST-whitelisted calculator) and :class:`AgentLoop` (the manual loop
that turns a question into an :class:`~secqa.core.contracts.Answer` with ``mode='agent'``).
"""

from secqa.agent.calc import safe_calculate
from secqa.agent.loop import AGENT_SYSTEM, AgentLoop, agent_prompt_hash, load_agent_prompt
from secqa.agent.runtime import ToolRuntime
from secqa.agent.tools import (
    FINAL_ANSWER,
    FINAL_ANSWER_SCHEMA,
    TOOL_NAMES,
    TOOLS,
    FinalAnswer,
    decode_final_answer,
    parse_final_answer_text,
)

__all__ = [
    "AGENT_SYSTEM",
    "FINAL_ANSWER",
    "FINAL_ANSWER_SCHEMA",
    "TOOLS",
    "TOOL_NAMES",
    "AgentLoop",
    "FinalAnswer",
    "ToolRuntime",
    "agent_prompt_hash",
    "decode_final_answer",
    "load_agent_prompt",
    "parse_final_answer_text",
    "safe_calculate",
]
