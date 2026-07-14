"""Application configuration loaded from environment / .env.

All tuning knobs live here so pipeline modules stay declarative.
Prefix is ``OCR_`` to avoid clashing with system env vars.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Granularity = Literal["word", "line", "paragraph"]
OcrVersion = Literal["PP-OCRv6", "PP-OCRv5"]
OcrEngine = Literal["paddleocr", "vlm"]

# Canonical project .env location (project root = parent of app/). config reads
# from here and envstore writes to here, so edits made via the UI /config/vlm
# endpoint are guaranteed to be picked up on the next settings read, regardless
# of the process's CWD.
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH), env_prefix="OCR_", extra="ignore"
    )

    # ---- Tiling (v1 scope: long edge ≤ 4000px → single PaddleOCR predict) ----
    tile_target_size: int = Field(
        4000,
        description="Target long-edge px of each tile when tiling is required "
                    "(long edge > small_image_threshold).",
    )
    tile_overlap: float = Field(
        0.15, ge=0.0, lt=0.5, description="Overlap ratio between adjacent tiles"
    )
    tile_merge_x_thres: int = Field(
        50, ge=1, le=500,
        description="[dedupe] PaddleOCR slice-style horizontal merge threshold (px). "
                    "Adjacent seam boxes within this distance merge.",
    )
    tile_merge_y_thres: int = Field(
        35, ge=1, le=500,
        description="[dedupe] PaddleOCR slice-style vertical merge threshold (px).",
    )
    small_image_threshold: int = Field(
        4000,
        description="Long edge ≤ this → one tile, direct PaddleOCR.predict() on "
                    "the full image (current product scope).",
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
        1216, ge=320, le=4096,
        description="DB detector resizes the long edge to this before inference. "
                    "1216 is the PaddleOCR 3.x recommendation for higher-res inputs "
                    "(≤4000px long edge, single-tile path).",
    )
    ocr_det_limit_type: Literal["max", "min"] = Field(
        "max",
        description="max=scale down only (keep detail on small images); "
                    "min=scale up only (speed up large images).",
    )
    # Keep detector boxes the recognizer dropped. The PP-OCRv6 rec model's dict
    # has no Hangul (and other non-{ch/en/japan/latin} scripts), so Korean text
    # gets a near-zero rec score and is filtered out by the pipeline — its box
    # vanishes too, since we only read `rec_polys`. With this on, we read the
    # detector's full `dt_polys` set as well and emit the unmatched boxes as
    # `recognized=False` items (with an optional base64 crop) so a downstream
    # script-aware model can re-read them. This is how Korean regions stop being
    # silently lost.
    ocr_emit_crops: bool = Field(
        True,
        description="For boxes the recognizer dropped (low confidence — mainly "
                    "scripts the rec model can't read, e.g. Korean under PP-OCRv6), "
                    "keep the detector polygon AND emit a base64 PNG crop. Off → "
                    "the boxes are still kept (recognized=False) but with no crop_b64.",
    )
    rec_confidence_fallback: float = Field(
        0.94, ge=0.0, le=1.0,
        description="[vlm_ocr_fallback] Recognition confidence below which a crop "
                    "is re-read by the VLM. Default 0.94 — any text box PaddleOCR "
                    "isn't 94% sure of gets cropped and sent to the VLM for a second "
                    "read (only when vlm_ocr_fallback_enabled + vlm_enabled are on). "
                    "Pairs with rec_confidence_drop: boxes in [drop, fallback) are "
                    "VLM re-read, boxes below drop are discarded (POST /analyze only).",
    )
    rec_confidence_drop: float = Field(
        0.60, ge=0.0, le=1.0,
        description="[/analyze] Text boxes whose FINAL confidence is below this are "
                    "discarded from /analyze results (after the VLM fallback pass). "
                    "Default 0.60 — a box PaddleOCR scored <60% that the VLM couldn't "
                    "lift above 60% is dropped. Only text items are dropped; qr/barcode "
                    "are kept. Applied on POST /analyze only (/verify keeps all boxes). "
                    "Must be ≤ rec_confidence_fallback.",
    )

    # ---- Circular / ring-shaped text detection (hard region for line OCR) ----
    # Characters arranged on an arc (around logos, seals, badges, caps) break the
    # recognizer's left-to-right reading assumption. detect_circular_regions
    # finds such rings with pure geometry (HoughCircles + angle-spread check);
    # the pipeline then crops each ring's bounding box and sends it to the VLM
    # with a circular-aware prompt. Gain-only: if no circles are found (the
    # common case), this stage is a no-op and the main OCR path is unchanged.
    circular_detect_enabled: bool = Field(
        False,
        description="Detect circular/ring-shaped text regions and re-read them "
                    "via the VLM. OFF by default — the HoughCircles-based detector "
                    "currently over-reports on real packaging (false rings from "
                    "scattered horizontal text). Enable only for testing until the "
                    "detection criteria are tightened.",
    )
    circular_min_members: int = Field(
        4, ge=2, le=30,
        description="Minimum text boxes whose centers must sit on the ring's "
                    "annulus for it to count as a circular region. Lower → more "
                    "sensitive (more false rings); higher → only dense rings found.",
    )
    circular_band_ratio: float = Field(
        0.25, ge=0.05, le=0.8,
        description="Half-width of the ring's annulus as a fraction of the "
                    "radius. A box center is 'on the ring' when |dist - r| ≤ "
                    "band_ratio*r. Larger → more tolerant of imprecise circle "
                    "fits; smaller → tighter to the exact circle.",
    )
    ocr_version: OcrVersion = Field(
        "PP-OCRv6",
        description="PaddleOCR pipeline version. Default PP-OCRv6 loads "
                    "PP-OCRv6_medium_det + PP-OCRv6_medium_rec (≈50-language rec).",
    )
    ocr_lang: str = Field(
        "ch",
        description="PaddleOCR lang tag. Default ch with PP-OCRv6 uses the "
                    "multilingual medium rec pack (简中/繁中/英/日 + Latin).",
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
    # Same-line overlap merge: when a single text line mixes scripts the detector
    # often splits it into two boxes (e.g. a Latin half + a Hangul half) whose
    # unclipped edges then overlap. The cross-tile dedupe won't merge them
    # (different text → low similarity), so the two boxes both survive and
    # overlap. This stage merges any two boxes on the same line whose x-ranges
    # overlap by ≥ this ratio into one box (text concatenated left-to-right).
    same_line_merge_x_overlap: float = Field(
        0.3, ge=0.0, le=1.0,
        description="Two boxes on the same line whose x-ranges overlap by ≥ this "
                    "ratio are merged into one (eliminates detector-split overlaps "
                    "on mixed-script lines like English+Korean). 0 = always merge "
                    "same-line neighbors; >0.5 effectively disables.",
    )

    # ---- VLM (opt-in cloud vision; OCR path is PaddleOCR-only by default) ----
    # NOTE: model defaults to qwen3.7-plus — qwen-vl-max / qwen-vl-plus are
    # scheduled for deprecation on 2026-07-13 (DashScope notice 118178). The
    # OpenAI-compatible endpoint and our code are unchanged; only the model
    # name moved. Qwen3.x supports an optional "thinking" mode (deep reasoning
    # before answering) — useful for future QA reasoning, but disabled by
    # default because it can't combine with response_format=json_object.
    vlm_enabled: bool = Field(
        False,
        description="Master switch for cloud VLM endpoints (/understand, /agent, "
                    "/panels/vlm). OCR analyze does not require this.",
    )
    vlm_ocr_fallback_enabled: bool = Field(
        True,
        description="Re-read low-confidence PaddleOCR crops (< rec_confidence_fallback, "
                    "default 0.95) via VLM during POST /analyze. On by default so the "
                    "second-pass VLM recognition kicks in the moment the VLM is enabled "
                    "(vlm_enabled + API key). The pipeline guards on BOTH switches, so "
                    "with the default vlm_enabled=false this stays dormant — no key, no "
                    "calls, no behavior change.",
    )
    vlm_provider: str = "qwen"
    vlm_api_key: str = ""
    vlm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    vlm_model: str = "qwen3.7-plus"
    vlm_enable_thinking: bool = Field(
        False,
        description="Enable Qwen3.x thinking mode (deep reasoning before the "
                    "answer). Off by default: thinking mode is incompatible with "
                    "response_format=json_object, so JSON output relies on our "
                    "tolerant parser instead. Turn on for QA-reasoning tasks.",
    )

    # ---- VLM OCR engine (Qwen-VL grounding OCR; alternative to PaddleOCR) ----
    # When selected (per-request ``engine=vlm`` or ocr_engine_default="vlm"),
    # the pipeline runs Qwen-VL on each tile instead of PaddleOCR. The VLM
    # returns normalized [x1,y1,x2,y2] boxes for every text/art_text/qr/barcode
    # it sees, mapped back to pixel coords. Codes are also run through pyzbar
    # (reliable decoding) and merged by the standard dedupe stage.
    # Requires vlm_enabled + API key. PaddleOCR stays the default.
    vlm_ocr_enabled: bool = Field(
        False,
        description="Master switch for the VLM OCR engine (POST /analyze with "
                    "engine=vlm). Requires vlm_enabled + API key.",
    )
    vlm_ocr_model: str = Field(
        "",
        description="Vision model for VLM OCR grounding. Empty = reuse "
                    "vlm_model. Pin a grounding-strong model (e.g. "
                    "qwen2.5-vl-72b-instruct) independent of the art-text "
                    "fallback / understanding model.",
    )
    vlm_ocr_max_side: int = Field(
        2048, ge=512, le=4096,
        description="Long-edge px each tile is downscaled to before the VLM "
                    "grounds boxes. The pipeline tiles the image so each tile "
                    "is already ≤ this; downscaling only further trims tokens. "
                    "Normalized coords are scaled by the TILE size (pre-downscale) "
                    "so precision is bounded by tile_target_size, not this.",
    )
    vlm_ocr_confidence: float = Field(
        0.8, ge=0.0, le=1.0,
        description="Flat confidence assigned to VLM-grounded items (Qwen-VL "
                    "gives no native score). 0.8 mirrors the art-text fallback "
                    "convention.",
    )

    # ---- OCR engine default ----
    # Which engine /analyze runs when the request omits ``engine``. The UI can
    # still override per-request. PaddleOCR (local, deterministic) is the safe
    # default; set to "vlm" to default to the cloud vision model.
    ocr_engine_default: OcrEngine = Field(
        "paddleocr",
        description="Default OCR engine when the request doesn't specify one: "
                    "paddleocr (local DB++ det+rec) or vlm (Qwen-VL grounding).",
    )

    # ---- AI Native understanding layer (``POST /understand``) ----
    # Reuses OCR_VLM_* for the provider/key/url/model — no separate VLM creds.
    # The understanding layer is the first AI-Native capability: the VLM looks
    # at the whole image and answers "what is this" instead of just recognizing
    # art text. Level-1 = one whole-image call; per-panel calls are a future
    # extension (level 2).
    understand_enabled: bool = Field(
        False,
        description="Toggle the AI understanding layer (POST /understand). "
                    "Requires vlm_enabled + API key.",
    )
    understand_max_side: int = Field(
        1080, ge=256, le=4096,
        description="Long-edge px the image is downscaled to before asking the "
                    "VLM. Dielines run 1000-10000px but VLMs effectively resolve "
                    "~2Kpx; for whole-image 'what is this' understanding a single "
                    "downscale is enough. Larger → sharper but more tokens.",
    )

    # ---- AI Native agent layer (``POST /agent/understand``) ----
    # The agent is qwen3-max (reasoning brain, text-only, function calling) +
    # VLM/OCR as tools it calls. The brain never sees pixels directly — it
    # learns about the image through tool outputs (text descriptions). This
    # multi-round, targeted inspection beats a single-shot VLM call for complex
    # packaging images (the cause of the earlier "荒诞不准确" results).
    agent_llm_model: str = Field(
        "qwen3-max",
        description="Reasoning brain model (text-only, must support function "
                    "calling on the OpenAI-compatible DashScope endpoint).",
    )
    agent_vlm_model: str = Field(
        "qwen3.7-plus",
        description="Vision model used by the look/describe tools (the agent's "
                    "'eyes'). Separate from vlm_model so the agent can pin a "
                    "known-good vision model independently of the art-text "
                    "fallback path.",
    )
    agent_max_rounds: int = Field(
        8, ge=1, le=30,
        description="Max ReAct loop iterations. Prevents runaway loops; on "
                    "exhaustion the model is forced to conclude with what it has.",
    )
    agent_look_max_side: int = Field(
        1080, ge=256, le=4096,
        description="Long-edge px the look/describe tools downscale crops to "
                    "before asking the VLM.",
    )

    # ---- Panel splitting (VLM cut-line detection) ----
    panels_vlm_max_side: int = Field(
        512, ge=128, le=2048,
        description="Long-edge px the image is downscaled to before the VLM "
                    "looks for cut lines. Cut lines are a low-detail task (just "
                    "outlines), so 512 is enough and saves tokens vs. the "
                    "understanding layer's 1080.",
    )

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

    # ---- URL input (``url`` form field on every image endpoint) ----
    # Endpoints accept either a multipart ``file`` or a ``url`` form field.
    # These knobs bound the URL fetch so a slow/huge URL can't hang or OOM the
    # service. No SSRF guard — callers are trusted internal systems; add an
    # allow-host check in app/fetch.py if this service is ever exposed publicly.
    url_fetch_timeout: float = Field(
        30.0, gt=0,
        description="Connect/read timeout (seconds) for downloading a URL image.",
    )
    url_fetch_max_bytes: int = Field(
        104_857_600, gt=0,
        description="Abort the URL download once the body exceeds this size "
                    "(100MB default — die-line images can be large).",
    )

    # ---- Copy verification (POST /verify; deterministic rules, no cloud model) ----
    # Verification compares OCR'd text against a caller-supplied standard-copy
    # list. The metric is recall (fraction of a standard entry's characters
    # found, in order, in the OCR text). These two thresholds carve it into
    # matched / partial / missing. No `verify_enabled` switch: the path is fully
    # local (no API key, no cost), so it's always on like /analyze.
    verify_match_threshold: float = Field(
        0.85, ge=0.0, le=1.0,
        description="Recall ≥ this → a standard entry is 'matched' (present). "
                    "Lower → more lenient (tolerates OCR noise / minor rewording).",
    )
    verify_partial_threshold: float = Field(
        0.60, ge=0.0, le=1.0,
        description="Recall ≥ this (but < match) → 'partial' (something close is "
                    "there; flag for review). Below this → 'missing'.",
    )

    # ---- HTTP / CORS ----
    # Comma-separated list of origins allowed by the browser for cross-origin
    # XHR/fetch calls (e.g. an internal web UI on a different host/port). "*"
    # allows any origin — fine for a trusted intranet; for production set an
    # explicit allow-list, e.g. "http://10.1.93.196:3000,http://foo.corp".
    # Empty string disables CORS entirely (no middleware attached).
    cors_origins: str = Field(
        "*",
        description="Comma-separated allowed CORS origins, or '*' for any. "
                    "Empty to disable CORS.",
    )

    # ---- Annotator ----
    annotator_line_width: int = 3


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
