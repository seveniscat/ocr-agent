"""Application configuration loaded from environment / .env.

All tuning knobs live here so pipeline modules stay declarative.
Prefix is ``OCR_`` to avoid clashing with system env vars.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Granularity = Literal["word", "line", "paragraph"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="OCR_", extra="ignore"
    )

    # ---- Tiling ----
    tile_target_size: int = Field(
        1600, description="Target long-edge px of each tile"
    )
    tile_overlap: float = Field(
        0.15, ge=0.0, lt=0.5, description="Overlap ratio between adjacent tiles"
    )
    small_image_threshold: int = Field(
        2000, description="Images with both dims below this skip tiling"
    )

    # ---- OCR detection tuning (DB++ text detector) ----
    ocr_threshold: float = Field(
        0.3, ge=0.0, le=1.0,
        description="Pixel prob threshold for the DB shrink map. "
                    "Lower → more boxes (recall up) but more noise.",
    )
    ocr_box_thresh: float = Field(
        0.6, ge=0.0, le=1.0,
        description="Min average score inside a candidate box for it to be kept. "
                    "Lower → keeps faint art text.",
    )
    ocr_unclip_ratio: float = Field(
        1.8, ge=0.5, le=5.0,
        description="Expand each detected box by this ratio. Larger → looser "
                    "boxes (helps catch art text whose glyphs bleed past the shrink map).",
    )
    ocr_det_limit_side_len: int = Field(
        960, ge=320, le=4096,
        description="DB detector resizes the long edge to this before inference. "
                    "Larger → sharper small text but more memory.",
    )
    ocr_det_limit_type: Literal["max", "min"] = Field(
        "max",
        description="max=scale down only (keep detail on small images); "
                    "min=scale up only (speed up large images).",
    )
    rec_confidence_fallback: float = Field(
        0.6, ge=0.0, le=1.0,
        description="Recognition confidence below which we route the crop to the VLM",
    )

    # ---- Output granularity (box level) ----
    ocr_granularity: Granularity = Field(
        "line",
        description="word = per-token boxes (PaddleOCR return_word_box); "
                    "line = default text-line boxes (most common); "
                    "paragraph = line boxes merged into paragraph blocks by "
                    "geometric proximity (good for packaging copy blocks).",
    )
    ocr_paragraph_gap_ratio: float = Field(
        0.6, ge=0.0, le=3.0,
        description="[paragraph mode] Two lines merge into one block when the "
                    "vertical gap between them is <= gap_ratio * line_height, "
                    "AND their x-ranges overlap. Larger → more aggressive merging.",
    )
    ocr_paragraph_x_overlap: float = Field(
        0.3, ge=0.0, le=1.0,
        description="[paragraph mode] Min horizontal overlap ratio (IoU of "
                    "x-ranges) for two vertically-adjacent lines to merge.",
    )

    # ---- VLM fallback ----
    vlm_enabled: bool = True
    vlm_provider: str = "qwen"
    vlm_api_key: str = ""
    vlm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    vlm_model: str = "qwen-vl-max"

    # ---- Preprocessing (die-line auto-crop) ----
    preprocess_autocrop: bool = Field(
        True,
        description="Crop the blank margins surrounding the die-line artwork "
                    "before tiling/OCR. Disable for non-die-line images or when "
                    "the original frame must be preserved.",
    )
    preprocess_autocrop_threshold: int = Field(
        240, ge=0, le=255,
        description="Grayscale cutoff for 'ink' pixels (background is near-white). "
                    "Pixels strictly below this are treated as content. Lower → "
                    "only darker ink counts (ignores faint scan noise); higher → "
                    "more aggressive (catches light grey guides but risks eating "
                    "near-white art).",
    )
    preprocess_autocrop_padding: int = Field(
        0, ge=0, le=500,
        description="Extra margin (px) kept on every side after cropping, so the "
                    "artwork doesn't sit flush against the tile edge.",
    )

    # ---- Async ----
    large_image_threshold: int = Field(
        4000, description="Long edge above this => async processing"
    )

    # ---- Annotator ----
    annotator_line_width: int = 3


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
