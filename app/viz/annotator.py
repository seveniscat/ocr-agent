"""Draw detected polygons + their text/content onto the image.

Color-coded by type so you can eyeball bbox quality at a glance — this is the
single most useful debugging tool for the pipeline, so it ships in v1.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image, ImageDraw, ImageFont

from ..config import Settings
from ..schemas import Item

if TYPE_CHECKING:
    import numpy as np


_COLORS = {
    "text": (50, 200, 50),       # green
    "art_text": (255, 165, 0),   # orange — VLM fallback
    "qr": (50, 120, 255),        # blue
    "barcode": (180, 80, 255),   # purple
}


def annotate_image(
    image: "np.ndarray", items: list[Item], settings: Settings
) -> Image.Image:
    """Return a PIL image with polygons + labels overlaid."""
    pil = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(pil, "RGBA")

    width, height = pil.size
    # font size scales with image so labels stay readable on 10000px images.
    font_size = max(14, int(min(width, height) * 0.012))
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial.ttf", font_size
        )
    except OSError:  # non-macOS / no font
        font = ImageFont.load_default()

    line_w = max(1, settings.annotator_line_width)
    for item in items:
        color = _COLORS.get(item.type, (255, 255, 255))
        # quad polygon
        poly = [(float(p[0]), float(p[1])) for p in item.polygon]
        draw.line(poly + [poly[0]], fill=color, width=line_w)

        # label background + text
        label = item.text or item.content or ""
        if len(label) > 40:
            label = label[:37] + "…"
        if label:
            x0 = min(p[0] for p in poly)
            y0 = min(p[1] for p in poly) - font_size - 2
            tb = draw.textbbox((x0, max(0, y0)), label, font=font)
            draw.rectangle(
                [tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2],
                fill=(0, 0, 0, 180),
            )
            draw.text((x0, max(0, y0)), label, fill=color, font=font)
    return pil
