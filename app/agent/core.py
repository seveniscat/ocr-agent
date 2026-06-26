"""The agent loop — ReAct (Reason + Act) over function-calling tools.

Each round: the brain reasons and either requests tools or declares it's done.
On tool requests we dispatch, append results to history, and loop. On a plain
text answer we treat it as the final conclusion and parse it into a structured
result. The whole loop is wrapped so any failure degrades to a single-shot VLM
call — the endpoint always returns something.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..schemas import UnderstandingResult
from ..tiling import load_image
from ..understanding import _safe_parse  # reuse the tolerant JSON parser
from .llm import ReasoningLLM
from .schemas import AgentResult, AgentStep
from .tools import ToolRegistry

if TYPE_CHECKING:
    from ..config import Settings
    from ..pipeline import Pipeline

logger = logging.getLogger(__name__)


_AGENT_SYSTEM_PROMPT = """你是一位资深的包装图像分析专家。你正在分析一张包装模切图/展开图/标签/实物包装图。

重要:你无法直接看到图片。你只能通过调用工具(look / ocr_text / describe)去观察它,工具会返回文字描述。请像真正的专家一样逐步分析:

1. 先用 look() 看整张图(region 留空),判断这是什么品类、有几个面板、整体布局和主色调。
2. 针对你判断的关键区域(如 logo、品名、文字密集处、条码/二维码区)用 describe() 或 ocr_text() 深入了解。区域坐标用归一化 [x,y,w,h]。
3. 每一步调用工具前,先说明你的推理:为什么看这里、你预期会看到什么。
4. 当你确信已充分理解图片,停止调用工具,直接输出最终结论。

最终结论必须是 JSON 对象,字段如下(只输出 JSON,不要其它内容):
{
  "category": "包装品类,如 食品-饮料 / 日化 / 药品 / 其他",
  "category_confidence": 0.0到1.0,
  "panel_count_estimate": 整数,你估计的面板/画面数量,
  "style_keywords": ["风格关键词,最多5个"],
  "dominant_colors": ["主色十六进制,最多4个"],
  "key_elements": [{"kind": "logo|product_image|text_block|barcode|qr|nutrition_table|color_block|other", "description": "描述", "location": [x,y,w,h]}],
  "summary": "一句话概述这张图是什么"
}

location 用归一化坐标 [x,y,w,h](0–1)。关键元素应基于你实际观察到的内容,不要编造。

记住:宁可多调几次工具看清楚,也不要凭空猜测。你的结论必须基于工具的观察结果。"""


def run_agent(
    image_data: bytes, settings: "Settings", pipeline: "Pipeline"
) -> AgentResult:
    """Run the ReAct agent loop. Always returns an AgentResult.

    On any unrecoverable failure (brain can't start, repeated errors), degrades
    to a single-shot VLM call so the endpoint never 500s. ``fallback=True`` and
    an empty trace mark the degraded path.
    """
    try:
        img = load_image(image_data)
    except Exception as exc:  # noqa: BLE001
        return _fallback(settings, pipeline, f"图像解码失败: {exc}")

    h, w = img.shape[:2]
    try:
        registry = ToolRegistry(img, w, h, pipeline, settings)
        llm = ReasoningLLM(settings)
    except Exception as exc:  # noqa: BLE001
        return _fallback(settings, pipeline, f"Agent 初始化失败: {exc}")

    history: list[dict] = [
        {"role": "system", "content": _AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": "请分析这张包装图片。"},
    ]
    trace: list[AgentStep] = []
    final_text: str | None = None

    for round_idx in range(settings.agent_max_rounds):
        try:
            tool_calls, text = llm.step(history, registry.schemas)
        except Exception as exc:  # noqa: BLE001
            logger.warning("agent: brain step failed (round %d): %s", round_idx, exc)
            return _fallback(settings, pipeline, f"推理模型调用失败: {exc}")

        step = AgentStep(round=round_idx + 1, thought=text, tool_calls=tool_calls or [])
        trace.append(step)

        # No tool calls → the brain is done; its text is the conclusion.
        if not tool_calls:
            final_text = text
            break

        # Append the assistant's tool-request message to history (OpenAI shape).
        history.append(_assistant_tool_message(tool_calls, text))

        # Dispatch each tool call and feed results back.
        for call in tool_calls:
            result = registry.dispatch(call.name, call.arguments)
            step.observations.append(result)
            history.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": _stringify(result),
            })
    else:
        # Loop exhausted without the brain declaring done — force a conclusion.
        logger.warning("agent: reached max_rounds=%d, forcing conclusion", settings.agent_max_rounds)
        try:
            tool_calls, final_text = llm.step(
                history + [{"role": "user", "content":
                    "已达到分析轮数上限。请基于你已有的观察,直接输出最终 JSON 结论,不要再调用工具。"}],
                [],  # no tools → forces a text answer
            )
        except Exception as exc:  # noqa: BLE001
            return _fallback(settings, pipeline, f"收尾推理失败: {exc}")

    # Parse the brain's final text into a structured conclusion.
    model_tag = f"{settings.agent_llm_model} + {settings.agent_vlm_model}"
    conclusion = _safe_parse(final_text or "", model=model_tag)
    return AgentResult(
        conclusion=conclusion, trace=trace, rounds=len(trace), model=model_tag,
    )


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _assistant_tool_message(tool_calls, text: str | None) -> dict:
    """Rebuild the assistant message carrying tool_calls in OpenAI shape, so the
    next request's history matches what the API expects."""
    return {
        "role": "assistant",
        "content": text,
        "tool_calls": [
            {
                "id": c.id,
                "type": "function",
                "function": {
                    "name": c.name,
                    # arguments must be a JSON string on the wire
                    "arguments": _dumps(c.arguments),
                },
            }
            for c in tool_calls
        ],
    }


def _stringify(result: dict) -> str:
    """Tool results go back as the tool message content (a string)."""
    import json

    try:
        return json.dumps(result, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(result)


def _dumps(args: dict) -> str:
    import json

    try:
        return json.dumps(args, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"


def _fallback(settings: "Settings", pipeline: "Pipeline", message: str) -> AgentResult:
    """Degrade gracefully when the agent loop can't run at all.

    We can't fall back to a single-shot VLM here because callers reach this
    path only after decode/init/brain failures (where we don't have a usable
    image or a working brain). Instead we return a structured error so the UI
    can show what went wrong; the trace is empty and fallback=True.
    """
    logger.warning("agent: degrading (%s)", message)
    err_conclusion = UnderstandingResult(
        category="未知",
        category_confidence=0.0,
        panel_count_estimate=1,
        summary=f"Agent 失败,已降级:{message}",
        raw_note=message,
        model=f"{settings.agent_llm_model} (fallback)",
    )
    return AgentResult(
        conclusion=err_conclusion, trace=[], rounds=0,
        model=f"{settings.agent_llm_model} + {settings.agent_vlm_model}",
        fallback=True, error=message,
    )
