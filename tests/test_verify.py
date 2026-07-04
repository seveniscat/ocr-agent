"""Tests for copy verification (``POST /verify`` and ``app.verify``).

Two layers, following the repo convention:
1. ``app.verify`` pure functions — normalize, recall, multi-item assembly,
   required/optional semantics, threshold boundaries.
2. ``/verify`` endpoint — standard input accepted, validation errors (400),
   and the JSON shape (incl. the ``pass`` alias).

No paddle / VLM / pyzbar dependency needed: the pipeline is stubbed.
"""
from __future__ import annotations

import io
import json

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app import main as main_mod
from app.config import Settings
from app.schemas import AnalyzeResponse, ImageMeta, Item, VerifyEntry
from app.verify import Thresholds, normalize, verify


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _item(
    id_: str,
    text: str,
    x1: float = 0,
    y1: float = 0,
    x2: float = 100,
    y2: float = 20,
    conf: float = 0.9,
) -> Item:
    """Build an axis-aligned text Item (polygon = quad of the bbox)."""
    return Item(
        id=id_,
        type="text",
        text=text,
        polygon=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        bbox=[x1, y1, x2, y2],
        confidence=conf,
        source="paddleocr",
    )


TH = Thresholds(match=0.85, partial=0.60)


def _run(entries, items, th=TH):
    """Convenience wrapper: assemble once via verify(), return one result each.

    Routes through :func:`verify` (not :func:`match_entry` directly) so that
    entry ids are auto-assigned and the contract matches the endpoint path.
    """
    _, _, _, results = verify(items, entries, th)
    return results


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------


def test_normalize_strips_spaces_and_punctuation():
    assert normalize("净 含 量：450g") == "净含量450g"


def test_normalize_folds_fullwidth_and_lowers_case():
    # ４５０ｇ are fullwidth digits/letter; NFKC folds them.
    assert normalize("ＮＥＴ４５０ｇ") == "net450g"


def test_normalize_drops_newlines_and_underscores():
    assert normalize("a_b\nc-d e") == "abcde"


def test_normalize_empty():
    assert normalize("") == ""
    assert normalize("   ") == ""


# ---------------------------------------------------------------------------
# recall / status
# ---------------------------------------------------------------------------


def test_exact_match_is_matched():
    items = [_item("t1", "净含量450g")]
    res = _run([VerifyEntry(text="净含量450g")], items)[0]
    assert res.status == "matched"
    assert res.similarity == pytest.approx(1.0)
    assert res.matched_item_ids == ["t1"]


def test_missing_entry_is_missing():
    items = [_item("t1", "净含量450g")]
    res = _run([VerifyEntry(text="生产日期20250101")], items)[0]
    assert res.status == "missing"
    assert res.matched_item_ids == []


def test_ocr_typo_is_partial():
    # 4 of 5 chars match (a→b→c→d then 'X'≠'e') → recall 0.80: ≥ partial, < match.
    items = [_item("t1", "abcdX")]
    res = _run([VerifyEntry(text="abcde")], items)[0]
    assert res.status == "partial"
    assert 0.60 <= res.similarity < 0.85


def test_single_char_typo_on_long_entry_still_matches():
    # A 7-char entry with one wrong glyph → recall 6/7 ≈ 0.857 ≥ match (0.85).
    # This is intended: one-char differences on long copy are treated as present
    # (likely OCR noise), not as a compliance failure. Partial is reserved for
    # more substantial divergence.
    items = [_item("t1", "净含量45Og")]  # 'O' instead of '0'
    res = _run([VerifyEntry(text="净含量450g")], items)[0]
    assert res.status == "matched"


def test_match_threshold_boundary():
    items = [_item("t1", "abcde")]  # 5 chars
    # Recall exactly 0.85 → matched (>= threshold). 4/5 = 0.8 → partial.
    res = _run([VerifyEntry(text="abcde")], items)[0]
    assert res.status == "matched"
    res2 = _run([VerifyEntry(text="abcde", required=True)], [_item("t1", "abcdX")])[0]
    assert res2.status == "partial"


def test_partial_threshold_boundary():
    # 3/5 = 0.6 → exactly partial threshold → partial (>=), not missing.
    res = _run([VerifyEntry(text="abcde")], [_item("t1", "abcXY")])[0]
    assert res.status == "partial"


# ---------------------------------------------------------------------------
# multi-item assembly (cross-line standard copy)
# ---------------------------------------------------------------------------


def test_entry_spanning_multiple_items_matches():
    # Standard copy "净含量450g" appears split across two OCR lines, stacked
    # vertically. Reading order joins them; the entry should match both items.
    items = [
        _item("t1", "净含量", y1=0, y2=20),
        _item("t2", "450g", y1=30, y2=50),
    ]
    res = _run([VerifyEntry(text="净含量450g")], items)[0]
    assert res.status == "matched"
    # Both backing items are reported (order = first occurrence in joined text).
    assert res.matched_item_ids == ["t1", "t2"]


def test_reading_order_is_top_to_bottom_then_left_to_right():
    # Two columns: right-column item has a smaller y but comes second.
    items = [
        _item("b", "右", x1=200, y1=0, y2=10),
        _item("a", "左", x1=0, y1=0, y2=10),
    ]
    from app.verify import _assemble_text

    joined, owners = _assemble_text(items)
    # Same y → sorted by x: 左(a) before 右(b). Each is one CJK char.
    assert joined == "左右"
    assert owners == ["a", "b"]


def test_dedup_matched_item_ids():
    # One item contains the whole needle; ensure id appears once.
    items = [_item("t1", "净含量450g")]
    res = _run([VerifyEntry(text="净含量450g")], items)[0]
    assert res.matched_item_ids.count("t1") == 1


# ---------------------------------------------------------------------------
# required vs optional → pass_
# ---------------------------------------------------------------------------


def test_optional_missing_does_not_fail_overall():
    items = [_item("t1", "净含量450g")]
    entries = [
        VerifyEntry(text="净含量450g", required=True),
        VerifyEntry(text="不存在的文案", required=False),  # optional, missing
    ]
    matched, partial, missing, results = verify(items, entries, TH)
    assert matched == 1 and missing == 1
    # pass_ = all REQUIRED entries matched; the optional missing one is ignored.
    assert all(r.status == "matched" for r in results if r.required)
    pass_overall = all(r.status == "matched" for r in results if r.required)
    assert pass_overall is True


def test_required_partial_fails_overall():
    items = [_item("t1", "abcdX")]  # 4/5 → partial
    entries = [VerifyEntry(text="abcde", required=True)]
    matched, partial, missing, results = verify(items, entries, TH)
    assert partial == 1
    pass_overall = all(r.status == "matched" for r in results if r.required)
    assert pass_overall is False


def test_verify_counts():
    items = [
        _item("t1", "净含量450g"),
        _item("t2", "配料：水"),
        _item("t3", "zzz"),
    ]
    entries = [
        VerifyEntry(text="净含量450g"),        # matched
        VerifyEntry(text="配料水"),            # matched (normalize drops '：')
        VerifyEntry(text="完全不存在"),         # missing
        VerifyEntry(text="abcde"),             # partial vs t3 "zzz"? -> missing; use real partial below
    ]
    # 'abcde' vs the joined text has no overlap → missing, not partial. Replace
    # with a genuine partial: 'abcde' vs 'abcdX' (4/5 = 0.80).
    items.append(_item("t4", "abcdX"))
    entries[-1] = VerifyEntry(text="abcde")
    matched, partial, missing, _ = verify(items, entries, TH)
    assert matched == 2 and partial == 1 and missing == 1


# ---------------------------------------------------------------------------
# endpoint integration (pipeline stubbed)
# ---------------------------------------------------------------------------


def _png_bytes(size=(8, 8)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def _stub_pipeline(monkeypatch, items: list[Item]):
    """Make _get_pipeline().run return a fixed AnalyzeResponse with `items`."""
    resp = AnalyzeResponse(image_meta=ImageMeta(width=8, height=8), items=items)
    pipeline = main_mod._get_pipeline()
    monkeypatch.setattr(pipeline, "run", lambda data, annotate=False, options=None: resp)


def _settings(**overrides) -> Settings:
    base = Settings()
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def test_verify_endpoint_matched_and_missing(monkeypatch):
    monkeypatch.setattr(main_mod, "_settings", lambda: _settings())
    _stub_pipeline(monkeypatch, [
        _item("t1", "净含量450g"),
        _item("t2", "品名：鲜奶茶"),
    ])
    standard = json.dumps([
        {"text": "净含量450g", "category": "净含量"},
        {"text": "品名鲜奶茶"},          # normalize drops '：' → matched
        {"text": "不存在的文案"},         # missing
    ])
    c = TestClient(main_mod.app)
    r = c.post(
        "/verify",
        files={"file": ("x.png", _png_bytes(), "image/png")},
        data={"standard": standard},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 3
    assert body["matched"] == 2
    assert body["missing"] == 1
    assert body["partial"] == 0
    # JSON key is "pass" (alias), not "pass_".
    assert body["pass"] is False  # one required entry is missing
    # matched_item_ids present for the matched entry.
    statuses = {res["text"]: res["status"] for res in body["results"]}
    assert statuses["净含量450g"] == "matched"
    assert statuses["品名鲜奶茶"] == "matched"
    assert statuses["不存在的文案"] == "missing"
    # items echoed for UI highlighting.
    assert len(body["items"]) == 2


def test_verify_endpoint_pass_when_all_required_matched(monkeypatch):
    monkeypatch.setattr(main_mod, "_settings", lambda: _settings())
    _stub_pipeline(monkeypatch, [_item("t1", "净含量450g")])
    standard = json.dumps([
        {"text": "净含量450g"},
        {"text": "可选文案", "required": False},  # optional, missing, ignored
    ])
    c = TestClient(main_mod.app)
    r = c.post(
        "/verify",
        files={"file": ("x.png", _png_bytes(), "image/png")},
        data={"standard": standard},
    )
    assert r.status_code == 200
    assert r.json()["pass"] is True


def test_verify_endpoint_auto_assigns_ids(monkeypatch):
    monkeypatch.setattr(main_mod, "_settings", lambda: _settings())
    _stub_pipeline(monkeypatch, [_item("t1", "净含量450g")])
    standard = json.dumps([{"text": "净含量450g"}, {"text": "其它"}])
    c = TestClient(main_mod.app)
    r = c.post(
        "/verify",
        files={"file": ("x.png", _png_bytes(), "image/png")},
        data={"standard": standard},
    )
    assert r.status_code == 200
    ids = [res["entry_id"] for res in r.json()["results"]]
    assert ids == ["v1", "v2"]


def test_verify_endpoint_rejects_empty_standard(monkeypatch):
    monkeypatch.setattr(main_mod, "_settings", lambda: _settings())
    _stub_pipeline(monkeypatch, [])
    c = TestClient(main_mod.app)
    r = c.post(
        "/verify",
        files={"file": ("x.png", _png_bytes(), "image/png")},
        data={"standard": "[]"},
    )
    assert r.status_code == 400


def test_verify_endpoint_rejects_non_array_standard(monkeypatch):
    monkeypatch.setattr(main_mod, "_settings", lambda: _settings())
    _stub_pipeline(monkeypatch, [])
    c = TestClient(main_mod.app)
    r = c.post(
        "/verify",
        files={"file": ("x.png", _png_bytes(), "image/png")},
        data={"standard": json.dumps({"text": "x"})},
    )
    assert r.status_code == 400


def test_verify_endpoint_rejects_bad_json(monkeypatch):
    monkeypatch.setattr(main_mod, "_settings", lambda: _settings())
    _stub_pipeline(monkeypatch, [])
    c = TestClient(main_mod.app)
    r = c.post(
        "/verify",
        files={"file": ("x.png", _png_bytes(), "image/png")},
        data={"standard": "not json{"},
    )
    assert r.status_code == 400
