"""Application configuration loaded from environment / .env.

All tuning knobs live here so pipeline modules stay declarative.
Prefix is ``OCR_`` to avoid clashing with system env vars.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # ---- OCR ----
    ocr_threshold: float = Field(0.3, ge=0.0, le=1.0)
    rec_confidence_fallback: float = Field(
        0.6, ge=0.0, le=1.0,
        description="Recognition confidence below which we route to VLM",
    )

    # ---- VLM fallback ----
    vlm_enabled: bool = True
    vlm_provider: str = "qwen"
    vlm_api_key: str = ""
    vlm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    vlm_model: str = "qwen-vl-max"

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
