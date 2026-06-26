"""Pydantic models for the agent's reasoning trace and result.

``conclusion`` reuses :class:`app.schemas.UnderstandingResult` so the existing
UI rendering works unchanged — the agent is a drop-in upgrade for the
understanding layer, just with a richer (traced) provenance.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..schemas import UnderstandingResult


class ToolCall(BaseModel):
    """One tool invocation the brain requested this round."""

    id: str
    name: str
    arguments: dict


class AgentStep(BaseModel):
    """One round of the ReAct loop: a thought, the tool calls it made, and the
    observations each tool returned."""

    round: int
    thought: str | None = Field(
        None, description="The brain's reasoning text this round (may be empty "
                          "when the model emits only tool calls)."
    )
    tool_calls: list[ToolCall] = Field(default_factory=list)
    observations: list[dict] = Field(
        default_factory=list,
        description="Per-tool-call results, aligned with tool_calls by index.",
    )


class AgentResult(BaseModel):
    """Final output of an agent run: a structured conclusion + full trace."""

    conclusion: UnderstandingResult
    trace: list[AgentStep] = Field(default_factory=list)
    rounds: int = 0
    model: str = ""
    fallback: bool = Field(
        False,
        description="True when the agent loop failed and we degraded to a "
                    "single-shot VLM call (trace will be empty).",
    )
    error: str | None = Field(
        None, description="Populated on fallback / failure, for diagnostics."
    )
