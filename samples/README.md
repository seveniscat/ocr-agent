# Samples

Drop test packaging-dieline images here (any resolution, 100–10000px).
Used for manual / integration testing of the pipeline.

Suggested set:
- `small.png` — < 2000px, exercises the no-tile fast path.
- `medium.jpg` — ~4000px, exercises tiling + overlap dedup.
- `huge.png` — ~10000px, exercises the async (202) path and VRAM ceiling.
- `art_text.jpg` — contains curved / stylized text, exercises VLM fallback.
- `qr.png` — contains a QR code, exercises the pyzbar channel.
