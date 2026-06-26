"""Reasoning brain: qwen3-max with function calling (OpenAI-compatible).

The brain is text-only — it never sees pixels. It reasons over the conversation
history (which contains tool outputs as text) and decides which tool to call
next, or that it's done. We use non-streaming tool-calling, the documented,
well-trodden path on DashScope's compatible endpoint.
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from openai import OpenAI

from .schemas import ToolCall

if TYPE_CHECKING:
    from ..config import Settings

logger = logging.getLogger(__name__)


class ReasoningLLM:
    """Thin wrapper over qwen3-max for the agent's function-calling step."""

    def __init__(self, settings: "Settings") -> None:
        if not settings.vlm_api_key:
            raise RuntimeError("OCR_VLM_API_KEY is not set; the agent brain needs it.")
        self._client = OpenAI(
            api_key=settings.vlm_api_key,
            base_url=settings.vlm_base_url,
        )
        self._model = settings.agent_llm_model

    def step(
        self,
        messages: list[dict],
        tools_schema: list[dict],
    ) -> tuple[list[ToolCall] | None, str | None]:
        """One reasoning step. Returns ``(tool_calls, text)``.

        - ``tool_calls`` is non-None (possibly empty) when the model requested
          tool execution; ``text`` is its reasoning narration.
        - When ``tool_calls`` is None, the model produced only text — that's the
          signal it's done (the loop checks ``not tool_calls`` to terminate).
        """
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools_schema,
            tool_choice="auto",
            temperature=0.2,  # a touch of sampling helps the brain explain itself
            max_tokens=2048,
        )
        msg = resp.choices[0].message
        text = (getattr(msg, "content", None) or "").strip() or None
        raw_calls = getattr(msg, "tool_calls", None) or []
        if not raw_calls:
            return None, text

        calls: list[ToolCall] = []
        for tc in raw_calls:
            try:
                name = tc.function.name
                args = _parse_arguments(tc.function.arguments)
                calls.append(ToolCall(id=tc.id, name=name, arguments=args))
            except Exception as exc:  # noqa: BLE001 — keep the loop alive
                logger.warning("llm.step: bad tool_call skipped: %s", exc)
        return calls, text


def _parse_arguments(raw: object) -> dict:
    """Parse a tool_call's ``arguments`` into a dict, tolerating DashScope's
    double-JSON-encoding quirk (a JSON string whose value is itself a JSON
    string). Returns ``{}`` when there's nothing to parse."""
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        value: object = raw
        # Unwrap nested JSON strings up to a couple of levels.
        for _ in range(3):
            if isinstance(value, dict):
                return value  # type: ignore[return-value]
            if not isinstance(value, str):
                break
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                break
        if isinstance(value, dict):
            return value  # type: ignore[return-value]
    return {}
