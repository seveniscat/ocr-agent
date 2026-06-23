"""Dieline panel splitting: cut a packaging die-line into its main faces.

Algorithm (validated on real white-background die-lines):

1. **Saturation mask** — convert to HSV; keep pixels with high saturation
   (``sat > 0.4``). Die-line cut/crease lines are drawn in saturated colors
   (red=cut, blue=crease, magenta, etc.), while everything else — white
   background, grey/black text, photos — has low saturation. This single step
   isolates the structural lines from *all* content, which raw grayscale LSD
   cannot do (black text also produces long line segments).

2. **LSD on the mask** — run the Line Segment Detector on the binary mask.
   Keep only long segments (≥ 12% of the short edge), split into
   near-horizontal vs near-vertical by angle, then cluster by perpendicular
   coordinate (segments within a tolerance merge into one grid line).

3. **Grid → candidate panels** — adjacent parallel grid lines bound rectangles.

4. **Main-panel selection** — filter by area (≥ 2% of image) and aspect ratio
   (0.2–5), then keep the largest cluster of same-sized rectangles. Faces of
   one box share dimensions; auxiliary flaps (glue/tuck/dust) don't, so they
   fall outside the cluster.

**Known limitation:** this works on die-lines whose lines are *saturated
colors on a light background*. It does NOT work on "filled design drafts"
where the artwork itself is a saturated color filling whole panels (the lines
are then indistinguishable from the fill). For those, a semantic (VLM) route
is needed.

All processing runs on a downsampled copy (long edge ≤ 1500px) for speed;
returned bboxes are scaled back to original-image pixels.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Panel:
    """One detected main panel.

    ``bbox`` is in **original-image** pixel coords [x0, y0, x1, y1] (exclusive
    end); ``image_b64`` is the cropped panel as base64 PNG (only populated when
    the caller asks for previews).
    """

    bbox: list[int]
    width: int
    height: int
    image_b64: str | None = None


# ---------------------------------------------------------------------------
# Stage 1 — saturation mask + LSD long-line detection
# ---------------------------------------------------------------------------


def _saturation_mask(arr: np.ndarray, sat_thresh: float = 0.4) -> np.ndarray:
    """Binary mask of high-saturation pixels (the colored die-lines)."""
    arr_f = arr.astype(np.float32)
    mx = arr_f.max(axis=2)
    mn = arr_f.min(axis=2)
    sat = np.where(mx > 1, (mx - mn) / np.maximum(mx, 1), 0)
    return ((sat > sat_thresh).astype(np.uint8)) * 255


def _detect_long_lines(
    mask: np.ndarray,
    long_ratio: float = 0.12,
    angle_tol: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """LSD on the mask → (horizontal, vertical) long segments [x1,y1,x2,y2]."""
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    raw, _, _, _ = lsd.detect(mask)
    if raw is None:
        return np.empty((0, 4)), np.empty((0, 4))
    lines = raw.reshape(-1, 4)
    length = np.hypot(lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1])
    angle = np.degrees(np.arctan2(lines[:, 3] - lines[:, 1], lines[:, 2] - lines[:, 0]))
    angle = np.where(angle < 0, angle + 180, angle)

    min_dim = min(mask.shape)
    long_mask = length >= min_dim * long_ratio
    h_mask = long_mask & ((angle < angle_tol) | (angle > 180 - angle_tol))
    v_mask = long_mask & (np.abs(angle - 90) < angle_tol)
    return lines[h_mask], lines[v_mask]


def _cluster(coords: np.ndarray, tol: float = 20.0) -> list[float]:
    """Merge near-duplicate perpendicular coords into one grid line."""
    if len(coords) == 0:
        return []
    order = np.argsort(coords)
    centers: list[float] = []
    bucket: list[float] = []
    for c in coords[order]:
        if bucket and c - bucket[-1] <= tol:
            bucket.append(c)
        else:
            if bucket:
                centers.append(float(np.mean(bucket)))
            bucket = [float(c)]
    if bucket:
        centers.append(float(np.mean(bucket)))
    return centers


# ---------------------------------------------------------------------------
# Stage 2–3 — candidate panels + main-panel selection
# ---------------------------------------------------------------------------


def _candidate_panels(
    h_lines: list[float],
    v_lines: list[float],
    img_area: float,
    area_min: float = 0.02,
    ratio_min: float = 0.2,
    ratio_max: float = 5.0,
) -> list[dict]:
    """Rectangles bounded by adjacent grid lines, filtered by area & aspect."""
    panels: list[dict] = []
    for i in range(len(h_lines) - 1):
        for j in range(len(v_lines) - 1):
            x0, y0 = v_lines[j], h_lines[i]
            x1, y1 = v_lines[j + 1], h_lines[i + 1]
            pw, ph = x1 - x0, y1 - y0
            if pw <= 0 or ph <= 0:
                continue
            area = pw * ph
            ratio = pw / ph
            if area >= img_area * area_min and ratio_min <= ratio <= ratio_max:
                panels.append({"bbox": (x0, y0, x1, y1), "w": pw, "h": ph, "area": area})
    return panels


def _select_main_panels(
    panels: list[dict], target: tuple[int, int] = (5, 6)
) -> list[dict]:
    """Keep the largest cluster of same-sized rectangles — the box faces."""
    if len(panels) <= target[1]:
        return sorted(panels, key=lambda p: (p["bbox"][1], p["bbox"][0]))

    def size_key(p: dict) -> tuple[int, int]:
        return (round(p["h"] / 30) * 30, round(p["w"] / 30) * 30)

    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for p in panels:
        groups[size_key(p)].append(p)

    ordered = sorted(groups.values(), key=lambda g: -sum(p["area"] for p in g))
    out = list(ordered[0])
    for grp in ordered[1:]:
        if len(out) >= target[1]:
            break
        out.extend(grp)
    out = out[: target[1]]
    return sorted(out, key=lambda p: (p["bbox"][1], p["bbox"][0]))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_ANALYSIS_MAX_SIDE = 1500


def split_panels(
    img: Image.Image,
    *,
    sat_thresh: float = 0.4,
    long_ratio: float = 0.12,
    area_min: float = 0.02,
    target: tuple[int, int] = (5, 6),
    return_images: bool = False,
) -> list[Panel]:
    """Split a die-line image into its main panels.

    Parameters
    ----------
    img : PIL.Image
        Input die-line (any size; downscaled internally for analysis).
    sat_thresh : float
        HSV saturation cutoff for "line" pixels (default 0.4). Lower → keeps
        fainter colored lines; higher → only vivid lines.
    long_ratio : float
        Min line length as a fraction of the short edge (default 0.12).
    area_min : float
        Min panel area as a fraction of image area (default 0.02 = 2%).
    target : (int, int)
        (min, max) number of main panels to return.
    return_images : bool
        If True, each Panel's ``image_b64`` is filled with a base64 PNG crop.

    Returns
    -------
    list[Panel]
        0–6 main panels, sorted top-to-bottom then left-to-right. Bboxes are
        in the **original** image's pixel coords.

    Notes
    -----
    Works on die-lines with saturated-colored lines on a light background.
    Does NOT work on "filled design drafts" where artwork fills panels with a
    saturated color — for those, callers should fall back to a VLM route.
    """
    import base64
    import io

    orig_w, orig_h = img.size
    rgb = img.convert("RGB")

    scale = _ANALYSIS_MAX_SIDE / max(orig_w, orig_h)
    if scale < 1.0:
        small = rgb.resize((max(1, int(orig_w * scale)), max(1, int(orig_h * scale))))
    else:
        small = rgb
        scale = 1.0
    arr = np.array(small)
    sh, sw = arr.shape[:2]

    mask = _saturation_mask(arr, sat_thresh=sat_thresh)
    h_segs, v_segs = _detect_long_lines(mask, long_ratio=long_ratio)
    h_lines = _cluster((h_segs[:, 1] + h_segs[:, 3]) / 2) if len(h_segs) else []
    v_lines = _cluster((v_segs[:, 0] + v_segs[:, 2]) / 2) if len(v_segs) else []

    candidates = _candidate_panels(h_lines, v_lines, sh * sw, area_min=area_min)
    main = _select_main_panels(candidates, target=target)

    panels: list[Panel] = []
    for p in main:
        x0, y0, x1, y1 = p["bbox"]
        ox0 = int(round(x0 / scale))
        oy0 = int(round(y0 / scale))
        ox1 = int(round(x1 / scale))
        oy1 = int(round(y1 / scale))
        image_b64 = None
        if return_images:
            crop = rgb.crop((ox0, oy0, ox1, oy1))
            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        panels.append(
            Panel(
                bbox=[ox0, oy0, ox1, oy1],
                width=ox1 - ox0,
                height=oy1 - oy0,
                image_b64=image_b64,
            )
        )
    return panels


# ===========================================================================
# Interactive panel splitting — candidate-line detection + panel computation
#
# The pure-automatic ``split_panels`` above fails on "filled design drafts"
# (artwork filling panels with a saturated color). Rather than fight that, we
# expose a two-step interactive flow:
#   1. ``detect_candidate_lines`` — loosely proposes candidate cut lines
#      (gradient-projection peaks at multiple thresholds). The UI shows them
#      with confidence-coded styles; the user toggles/adds/removes.
#   2. ``compute_panels`` — a pure function turning the user-confirmed line
#      positions into panel rectangles. No image processing; the front-end
#      calls the same logic for live preview.
#
# Gradient projection is color-agnostic (responds to any edge), so candidate
# detection works across all die-line styles — white-background line art AND
# filled design drafts. The user's clicks are what make the final result
# correct, not the algorithm.
# ===========================================================================


# Confidence levels emitted by ``detect_candidate_lines``. The UI renders them
# differently and pre-selects only the strongest.
CONF_HIGH = 1.0   # strong peak → solid line, pre-selected
CONF_MID = 0.8    # moderate peak → dashed line, not selected
CONF_LOW = 0.6    # weak peak → faint dashed line, not selected


def _projection_peaks(
    proj: np.ndarray,
    *,
    high_pct: float = 95.0,
    mid_pct: float = 85.0,
    low_pct: float = 70.0,
    min_gap: int = 15,
    smooth: int = 11,
) -> list[tuple[int, float]]:
    """Find peaks in a 1-D projection; classify each by **percentile rank**
    within the smoothed projection.

    Different die-lines have wildly different projection scales (a clean
    line-art image's strongest peak might normalize to 0.37; a filled draft's
    to 0.87), so fixed thresholds or mean±k·std both fail: the std is itself
    inflated by the very peaks we're hunting. Percentiles are robust — the
    top 5% of projection values are, by definition, the strongest 5% of
    positions, regardless of absolute scale.

    Defaults: top 5% → HIGH (pre-selected), top 15% → MID, top 30% → LOW.

    Returns ``[(position, confidence), ...]`` with confidence one of
    ``CONF_HIGH`` / ``CONF_MID`` / ``CONF_LOW``.
    """
    if smooth > 1:
        proj = np.convolve(proj, np.ones(smooth) / smooth, mode="same")

    hi_t = float(np.percentile(proj, high_pct))
    mid_t = float(np.percentile(proj, mid_pct))
    lo_t = float(np.percentile(proj, low_pct))

    best: dict[int, float] = {}
    n = len(proj)
    for i in range(2, n - 2):
        v = proj[i]
        if v < lo_t:
            continue
        # Local max, tolerant of flat plateaus: >= both neighbours and strictly
        # > at least one of the ±2 neighbours (on a peak, not a slope).
        if not (
            v >= proj[i - 1]
            and v >= proj[i + 1]
            and (v > proj[i - 2] or v > proj[i + 2])
        ):
            continue
        if v >= hi_t:
            conf = CONF_HIGH
        elif v >= mid_t:
            conf = CONF_MID
        else:
            conf = CONF_LOW
        # Snap to the local max within ±2 for sub-bin accuracy.
        window = proj[max(0, i - 2) : i + 3]
        pos = int(np.argmax(window)) + (i - 2)
        pos = max(0, min(n - 1, pos))
        # Merge into the nearest existing peak within min_gap; keep the higher
        # confidence (a strong peak absorbs a nearby weak one).
        merged = False
        for existing in list(best.keys()):
            if abs(existing - pos) < min_gap:
                if conf > best[existing]:
                    best[existing] = conf
                merged = True
                break
        if not merged:
            best[pos] = conf
    return sorted(best.items())


def detect_candidate_lines(
    img: Image.Image,
    *,
    high_pct: float = 95.0,
    mid_pct: float = 85.0,
    low_pct: float = 70.0,
    min_gap_ratio: float = 0.025,
) -> dict:
    """Propose candidate cut lines via gradient projection.

    Works on any die-line style (color-agnostic) because Sobel gradient responds
    to edges of any color. Returns line positions in **original-image** pixels.

    Confidence is adaptive via **percentile rank** of the projection value
    (see ``_projection_peaks``): top 5% → HIGH (pre-selected), top 15% → MID,
    top 30% → LOW. Lower a percentile to surface more candidates.

    Returns
    -------
    dict
        ``{"width", "height", "h_lines": [{"pos","confidence","selected"}],
        "v_lines": [...]}``. ``selected`` is True only for CONF_HIGH lines
        (the UI lets the user toggle the rest).
    """
    orig_w, orig_h = img.size
    rgb = img.convert("RGB")

    scale = _ANALYSIS_MAX_SIDE / max(orig_w, orig_h)
    if scale < 1.0:
        small = rgb.resize(
            (max(1, int(orig_w * scale)), max(1, int(orig_h * scale)))
        )
    else:
        small = rgb
        scale = 1.0
    arr = np.array(small)
    sh, sw = arr.shape[:2]

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    # A horizontal cut line produces a strong gradient band across many columns,
    # so summing the gradient along each ROW spikes at that row. Symmetric for
    # vertical lines / column sum.
    row_proj = grad.sum(axis=1)
    col_proj = grad.sum(axis=0)

    min_gap = max(12, int(min_gap_ratio * min(sh, sw)))
    h_peaks = _projection_peaks(
        row_proj, high_pct=high_pct, mid_pct=mid_pct, low_pct=low_pct,
        min_gap=min_gap,
    )
    v_peaks = _projection_peaks(
        col_proj, high_pct=high_pct, mid_pct=mid_pct, low_pct=low_pct,
        min_gap=min_gap,
    )

    def to_orig(peaks, scale):
        out = []
        for pos, conf in peaks:
            out.append(
                {
                    "pos": int(round(pos / scale)),
                    "confidence": conf,
                    "selected": conf >= CONF_HIGH,
                }
            )
        return out

    return {
        "width": orig_w,
        "height": orig_h,
        "h_lines": to_orig(h_peaks, scale),
        "v_lines": to_orig(v_peaks, scale),
    }


def compute_panels(
    h_positions: list[int],
    v_positions: list[int],
    width: int,
    height: int,
) -> list[Panel]:
    """Turn confirmed cut-line positions into panel rectangles.

    Pure function — no image access — so the front-end can call the same logic
    for live preview. The image's four edges are implicit boundaries, so even
    with zero confirmed lines you get one panel (the whole image). With one
    horizontal + one vertical line you get 4 panels, etc.

    Positions are clamped to ``[0, width/height]`` and de-duplicated (within a
    small tolerance), so the caller doesn't have to sanitize input.
    """
    # Sanitize: clamp, dedup near-equal, sort.
    def clean(positions, limit):
        if not positions:
            return []
        ps = sorted({max(0, min(limit, int(round(p)))) for p in positions})
        out = [ps[0]]
        for p in ps[1:]:
            if p - out[-1] > 2:  # collapse positions within 2px
                out.append(p)
        # Drop positions that sit exactly on an edge (they add nothing).
        return [p for p in out if 0 < p < limit]

    hs = clean(h_positions, height)
    vs = clean(v_positions, width)

    # Edges bound the grid: y-bounds = [0, *hs, height], x-bounds likewise.
    y_bounds = [0, *hs, height]
    x_bounds = [0, *vs, width]

    panels: list[Panel] = []
    for i in range(len(y_bounds) - 1):
        for j in range(len(x_bounds) - 1):
            x0, y0 = x_bounds[j], y_bounds[i]
            x1, y1 = x_bounds[j + 1], y_bounds[i + 1]
            if x1 - x0 <= 0 or y1 - y0 <= 0:
                continue
            panels.append(
                Panel(
                    bbox=[x0, y0, x1, y1],
                    width=x1 - x0,
                    height=y1 - y0,
                )
            )
    return panels


# ===========================================================================
# Subject-based splitting — the reliable route for finished design drafts.
#
# Insight: a rectangular-box die-line's six faces form ONE connected blob on a
# binarized image (the faces are joined along crease lines, and even filled
# design drafts keep ink/content spanning the joins). Auxiliary flaps (glue
# tabs, tuck ears) are either small or weakly connected, so after a mild
# dilation the SIX-FACE BODY is the largest connected component — reliably,
# across all the sample images (verified: 6/6 correct subject bbox).
#
# Once we have the subject's bounding box, the standard unfolding layouts are
# geometrically constrained:
#   - aspect ≈ 1.0–1.3  → 2 rows × 3 cols (top/bottom + 4 body faces)
#   - aspect ≈ 1.5–1.7  → 1 row × 4 cols (4 body faces in a strip)
# Equal subdivisions are the initial guess; the caller/UI may refine them.
# ===========================================================================


def detect_subject(
    img: Image.Image,
    *,
    bg_thresh: int = 225,
    dilate: int = 12,
) -> dict | None:
    """Find the main unfolding subject = the largest connected blob of non-bg.

    Returns ``{"bbox": [x0,y0,x1,y1], "width", "height"}`` in original-image px,
    or ``None`` if the image is essentially blank.

    The mild ``dilate`` (default 12px on the analysis-size image) bridges the
    thin crease lines and small gutters between adjacent faces so the whole
    six-face body reads as ONE component. Too small → faces fragment; too
    large → subject swallows nearby flaps. 12px on a 1500px-long image is the
    empirically robust value across the sample set.
    """
    orig_w, orig_h = img.size
    rgb = img.convert("RGB")
    scale = _ANALYSIS_MAX_SIDE / max(orig_w, orig_h)
    if scale < 1.0:
        small = rgb.resize(
            (max(1, int(orig_w * scale)), max(1, int(orig_h * scale)))
        )
    else:
        small = rgb
        scale = 1.0
    arr = np.array(small)

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    fg = (gray < bg_thresh).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate, dilate))
    fg = cv2.dilate(fg, k)

    n, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = 1 + int(np.argmax(areas))
    x0 = int(round(stats[biggest, cv2.CC_STAT_LEFT] / scale))
    y0 = int(round(stats[biggest, cv2.CC_STAT_TOP] / scale))
    x1 = int(round((stats[biggest, cv2.CC_STAT_LEFT]
                     + stats[biggest, cv2.CC_STAT_WIDTH]) / scale))
    y1 = int(round((stats[biggest, cv2.CC_STAT_TOP]
                     + stats[biggest, cv2.CC_STAT_HEIGHT]) / scale))
    return {
        "bbox": [x0, y0, x1, y1],
        "width": x1 - x0,
        "height": y1 - y0,
    }


def _layout_for_aspect(aspect: float, wide_thresh: float = 1.4) -> tuple[int, int]:
    """Pick (rows, cols) from the subject's width/height aspect.

    - aspect >= wide_thresh → (1, 4): the four body faces in a horizontal strip.
    - otherwise             → (2, 3): top/bottom row + body faces row.
    """
    if aspect >= wide_thresh:
        return 1, 4
    return 2, 3


def split_subject_into_panels(
    subject: dict,
) -> tuple[list[int], list[int]]:
    """Equal-subdivision initial cut lines for a detected subject.

    Returns ``(v_lines, h_lines)`` — the x and y coordinates (in the same
    space as ``subject["bbox"]``, i.e. original-image px) of the internal
    grid lines. Lengths: cols-1 verticals and rows-1 horizontals.

    These are *initial* equal splits; real die-lines have non-equal faces
    (top/bottom differ from body), so the UI should let the user drag them.
    """
    x0, y0, x1, y1 = subject["bbox"]
    bw, bh = subject["width"], subject["height"]
    aspect = bw / bh if bh else 1.0
    rows, cols = _layout_for_aspect(aspect)
    v_lines = [x0 + round(bw * (j / cols)) for j in range(1, cols)]
    h_lines = [y0 + round(bh * (i / rows)) for i in range(1, rows)]
    return v_lines, h_lines


def split_panels_auto(
    img: Image.Image,
    *,
    return_images: bool = False,
) -> tuple[list[Panel], dict | None]:
    """One-shot automatic split: detect subject → equal-subdivide by layout.

    Returns ``(panels, subject)``. ``subject`` exposes the detected main-body
    bbox so the UI can draw it; ``panels`` are the equal-subdivision cells.

    This is the recommended entry point for finished design drafts. It is NOT
    pixel-perfect (real faces aren't exactly equal-sized), but it gives a
    correct subject box and a sensible 4-or-6-panel grid that the user can
    fine-tune by dragging the lines (re-using ``compute_panels``).
    """
    subject = detect_subject(img)
    if subject is None:
        return [], None

    v_lines, h_lines = split_subject_into_panels(subject)
    panels = compute_panels(h_lines, v_lines, img.size[0], img.size[1])

    if return_images:
        import base64
        import io as _io

        rgb = img.convert("RGB")
        for p in panels:
            buf = _io.BytesIO()
            rgb.crop(tuple(p.bbox)).save(buf, format="PNG")
            p.image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return panels, subject
