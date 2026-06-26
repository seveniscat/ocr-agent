"""Agent tools — the agent's eyes and hands.

The brain (qwen3-max) is text-only; it learns about the image by calling these
tools. Each tool takes normalized ``[x, y, w, h]`` region coords (0–1 fractions)
so the brain doesn't need to reason in pixels, and returns results in
original-image pixel coords so positions are meaningful across tools.

Tools never raise — on error they return ``{"error": "..."}`` so the brain can
adapt (e.g. retry a region with describe after ocr_text failed).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

from ..tiling import TileSpec, crop_tile, offset_polygon

if TYPE_CHECKING:
    import numpy as np
    from ..config import Settings
    from ..pipeline import Pipeline

logger = logging.getLogger(__name__)

# Region coords are JSON arrays of 4 floats; reused across tool schemas.
_REGION_SCHEMA = {
    "type": "array",
    "items": {"type": "number"},
    "description": "归一化坐标 [x, y, w, h],取值 0–1,表示区域左上角和宽高占整图的比例。",
}


class ToolRegistry:
    """Holds the decoded image + engines and exposes the agent's tools.

    The image is decoded once (in the agent loop) and shared by all tools so we
    don't re-decode per call. Engines come from the pipeline's lazy loaders.
    """

    def __init__(
        self,
        image: "np.ndarray",
        width: int,
        height: int,
        pipeline: "Pipeline",
        settings: "Settings",
    ) -> None:
        self.img = image
        self.W, self.H = width, height
        self.pipeline = pipeline
        self.settings = settings

    # ------------------------------------------------------------------
    # Tool schemas (OpenAI function format, sent to qwen3-max as `tools=`)
    # ------------------------------------------------------------------
    @property
    def schemas(self) -> list[dict]:
        return [_LOOK_SCHEMA, _OCR_SCHEMA, _DESCRIBE_SCHEMA]

    # ------------------------------------------------------------------
    # Dispatch — the agent loop calls this with a tool name + parsed args
    # ------------------------------------------------------------------
    def dispatch(self, name: str, args: dict) -> dict:
        fn = self._handlers().get(name)
        if fn is None:
            return {"error": f"未知工具: {name}"}
        try:
            return fn(**args)
        except TypeError as exc:
            return {"error": f"参数错误 ({name}): {exc}"}
        except Exception as exc:  # noqa: BLE001 — tools must not kill the loop
            logger.warning("tool %s failed: %s", name, exc)
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _handlers(self) -> dict[str, Callable]:
        return {"look": self.look, "ocr_text": self.ocr_text, "describe": self.describe}

    # ------------------------------------------------------------------
    # Geometry helper — normalized region → cropped array + pixel box
    # ------------------------------------------------------------------
    def _crop_norm(self, region: list | None) -> tuple["np.ndarray", list[int]]:
        """Crop a normalized [x,y,w,h] region. Returns (crop_array, [x0,y0,x1,y1])
        in original-image pixel coords (the offset to map local→global)."""
        import numpy as np  # local; avoid hard dep at import time

        if region is None:
            # Whole image: crop is a no-op copy (so callers can mutate safely).
            return self.img.copy(), [0, 0, self.W, self.H]

        if not (isinstance(region, list) and len(region) == 4):
            raise ValueError("region 必须是 [x,y,w,h] 形式")
        nx, ny, nw, nh = (float(v) for v in region)
        x0 = max(0, int(round(nx * self.W)))
        y0 = max(0, int(round(ny * self.H)))
        x1 = min(self.W, int(round((nx + nw) * self.W)))
        y1 = min(self.H, int(round((ny + nh) * self.H)))
        if x1 <= x0 or y1 <= y0:
            raise ValueError("region 区域无效或为空")
        crop = crop_tile(self.img, TileSpec(0, x0, y0, x1, y1))
        return np.ascontiguousarray(crop), [x0, y0, x1, y1]

    # ------------------------------------------------------------------
    # Tool: look — whole-image / region overview via VLM
    # ------------------------------------------------------------------
    def look(self, region: list | None = None, focus: str = "整体") -> dict:
        """看整张图或某个区域的概览(VLM)。适合先了解'这是什么'。"""
        crop, box = self._crop_norm(region)
        data_url = self._encode(crop)
        prompt = f"请客观描述这张图片。重点:{focus}。" if focus else "请客观描述这张图片。"
        text, _conf = self._vlm().ask_image(
            data_url, prompt, model_override=self.settings.agent_vlm_model,
        )
        return {
            "description": text or "(无描述)",
            "region_pixel": box,
            "region_norm": region,
        }

    # ------------------------------------------------------------------
    # Tool: ocr_text — precise text read via PaddleOCR
    # ------------------------------------------------------------------
    def ocr_text(self, region: list) -> dict:
        """识别指定区域内所有文字(PaddleOCR)。返回文字+原图坐标。"""
        crop, (x0, y0, _x1, _y1) = self._crop_norm(region)
        dets = self._ocr().detect_and_recognize(crop, granularity="line")
        texts = []
        for d in dets:
            poly = offset_polygon(d.polygon, x0, y0)
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            texts.append({
                "text": d.text,
                "bbox": [min(xs), min(ys), max(xs), max(ys)],
                "confidence": round(float(d.confidence), 3),
            })
        return {"texts": texts, "count": len(texts)}

    # ------------------------------------------------------------------
    # Tool: describe — VLM free-form Q&A on a region
    # ------------------------------------------------------------------
    def describe(self, region: list, question: str) -> dict:
        """用 VLM 回答关于指定区域的问题(颜色/物体/布局/异常)。"""
        crop, box = self._crop_norm(region)
        data_url = self._encode(crop)
        text, _conf = self._vlm().ask_image(
            data_url, question or "描述这个区域的内容",
            model_override=self.settings.agent_vlm_model,
        )
        return {"answer": text or "(无回答)", "region_pixel": box}

    # ------------------------------------------------------------------
    # Engine + encoding accessors
    # ------------------------------------------------------------------
    def _vlm(self):
        return self.pipeline._get_vlm()  # noqa: SLF001 — reuse lazy loader

    def _ocr(self):
        return self.pipeline._get_ocr()  # noqa: SLF001 — reuse lazy loader

    def _encode(self, arr: "np.ndarray") -> str:
        """Downscale to the agent's look max side and JPEG-encode to a data URL.
        Reuses the understanding layer's encoder so encoding is consistent."""
        from ..understanding import _encode_for_vlm

        return _encode_for_vlm(arr, max_side=self.settings.agent_look_max_side)


# ----------------------------------------------------------------------
# Tool schemas (module-level constants — they don't depend on instance state)
# ----------------------------------------------------------------------
_LOOK_SCHEMA = {
    "type": "function",
    "function": {
        "name": "look",
        "description": "看整张图或某个区域的概览(通过视觉模型)。适合先了解'这是什么品类/几个面板/整体布局'。返回对该区域的整体描述。",
        "parameters": {
            "type": "object",
            "properties": {
                "region": _REGION_SCHEMA,
                "focus": {
                    "type": "string",
                    "description": "想重点了解什么,如'这是什么品类的包装'、'有几个面板'。不填=整体概述。",
                },
            },
            "required": [],
        },
    },
}

_OCR_SCHEMA = {
    "type": "function",
    "function": {
        "name": "ocr_text",
        "description": "精确识别指定区域内所有文字(PaddleOCR)。比'看'更准,适合读配料表/净含量/品名等密集文字。返回文字内容及在原图中的坐标。",
        "parameters": {
            "type": "object",
            "properties": {"region": _REGION_SCHEMA},
            "required": ["region"],
        },
    },
}

_DESCRIBE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "describe",
        "description": "用视觉模型回答关于指定区域的任意问题(颜色/物体/布局/是否有 logo 等)。比 look 更聚焦,必须指定区域和具体问题。",
        "parameters": {
            "type": "object",
            "properties": {
                "region": _REGION_SCHEMA,
                "question": {
                    "type": "string",
                    "description": "要问的问题,如'这里有没有品牌 logo?是什么?'、'这块区域的背景色是什么?'",
                },
            },
            "required": ["region", "question"],
        },
    },
}
