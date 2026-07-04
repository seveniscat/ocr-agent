"""Copy verification — compare OCR text against a maintained standard.

The verification path is **deterministic** (no cloud model): given the list of
recognized :class:`Item` objects from the OCR pipeline and a list of standard
copy entries, decide for each entry whether it is present on the image.

Why deterministic: packaging copy verification needs results that are cheap,
offline, and reproducible. A cloud LLM can be added later for the ambiguous
"partial" band, but the headline answer (matched / missing) must be stable.

The two ideas here:

1. **Normalize** both sides so layout/OCR noise can't mask a true match:
   NFKC fold (全角→半角) → lowercase → strip everything that isn't a letter or
   digit (spaces, punctuation, newlines). What remains is a bare character
   stream; "净 含 量：450g" and "净含量450g" both become "净含量450g".

2. **Recall via SequenceMatcher**: the standard entry is the *needle* and the
   joined OCR text is the *haystack*. We measure what fraction of the needle's
   characters appear, in order, somewhere in the haystack — i.e. recall. This
   tolerates the needle spanning several OCR items (multi-line copy) and the
   haystack containing lots of unrelated text. (Precision is irrelevant here:
   extra text on the package is not the standard's problem.)

A character→item map is built while joining, so each matching block can be
traced back to the concrete OCR item(s) it came from — that drives the UI
highlighting of the region that satisfies a standard entry.
"""
from __future__ import annotations

import difflib
import logging
import re
import unicodedata
from dataclasses import dataclass

from .schemas import Item, VerifyEntry, VerifyEntryResult, VerifyEntryStatus

logger = logging.getLogger(__name__)

# A run of characters to KEEP after normalization: Unicode word characters
# (CJK ideographs, letters, digits) but NOT the underscore. ``[^\W_]`` reads as
# "a word char that is not '_'" — \w already includes underscore, so we carve it
# back out. Everything else (spaces, punctuation, symbols) is dropped by taking
# only these runs and joining them.
_KEEP = re.compile(r"[^\W_]+", flags=re.UNICODE)


def normalize(s: str) -> str:
    """Fold a string to a bare character stream for comparison.

    NFKC first (全角→半角, compatibility decompositions), then lowercase, then
    keep only runs of word characters joined together. Spaces/punctuation are
    discarded, so "品名: 鲜奶茶" and "品名鲜奶茶" normalize identically.
    """
    if not s:
        return ""
    folded = unicodedata.normalize("NFKC", s).lower()
    return "".join(_KEEP.findall(folded))


@dataclass(frozen=True)
class Thresholds:
    """Status cutoffs for recall (fraction of the needle matched).

    recall ≥ match           → matched   (the entry is on the image)
    partial ≤ recall < match → partial   (something close is there, but not a
                                          clean hit — OCR error / paraphrase /
                                          real difference; flag for review)
    recall < partial         → missing   (the entry is not on the image)
    """

    match: float
    partial: float


# ---------------------------------------------------------------------------
# Reading-order assembly + char→item map
# ---------------------------------------------------------------------------


def _reading_order(items: list[Item]) -> list[Item]:
    """Sort OCR items into approximate reading order (top-to-bottom, L-to-R).

    Packaging copy is mostly axis-aligned, so sorting by top-y then by x is a
    good enough reading order for joining text across items. We key on the
    *top* of each box (bbox[1]) so vertically stacked lines keep their order.
    """
    return sorted(items, key=lambda it: (round(it.bbox[1]), it.bbox[0]))


def _assemble_text(items: list[Item]) -> tuple[str, list[str]]:
    """Join item texts in reading order; return (joined, char_to_item_id).

    ``char_to_item_id`` is parallel to the joined string: index ``i`` holds the
    id of the item that contributed character ``i``. Items with no text are
    skipped (their characters would never match anything).
    """
    chars: list[str] = []
    owners: list[str] = []
    for it in _reading_order(items):
        norm = normalize(it.text or "")
        if not norm:
            continue
        chars.append(norm)
        owners.extend([it.id] * len(norm))
    return "".join(chars), owners


def _owners_in_range(owners: list[str], start: int, end: int) -> list[str]:
    """Distinct item ids owning chars in ``[start, end)`` of the joined text.

    De-duplicated but order-preserving (first occurrence wins), so the UI gets a
    stable, minimal set of boxes to highlight for one matching block.
    """
    seen: set[str] = set()
    out: list[str] = []
    for i in range(start, min(end, len(owners))):
        oid = owners[i]
        if oid not in seen:
            seen.add(oid)
            out.append(oid)
    return out


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def _recall(needle: str, haystack: str) -> tuple[float, list[tuple[int, int, int]]]:
    """Return (recall, matching_blocks) for needle ⊆ haystack.

    ``matching_blocks`` are difflib blocks ``(haystack_start, needle_start,
    size)`` restricted to blocks of size > 0. Recall = matched needle chars /
    len(needle). When the needle is empty recall is 0 (nothing to match).
    """
    if not needle:
        return 0.0, []
    sm = difflib.SequenceMatcher(a=haystack, b=needle, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
    matched = sum(b.size for b in blocks)
    return matched / len(needle), blocks


def match_entry(
    entry: VerifyEntry,
    items: list[Item],
    joined: str,
    owners: list[str],
    th: Thresholds,
) -> VerifyEntryResult:
    """Compute the verification result for one standard entry.

    ``joined``/``owners`` are precomputed by :func:`_assemble_text` so a batch of
    entries shares one assembly pass (the expensive part).
    """
    needle = normalize(entry.text)
    recall, blocks = _recall(needle, joined)

    if recall >= th.match:
        status: VerifyEntryStatus = "matched"
    elif recall >= th.partial:
        status = "partial"
    else:
        status = "missing"

    # Collect the OCR items that back the matched characters (for highlighting).
    # For a *missing* entry any coincidental overlap (e.g. "50" appearing in
    # both "450g" and "20250101") is meaningless, so we only report matched ids
    # for entries the user should actually act on (matched / partial).
    matched_ids: list[str] = []
    matched_chars: list[str] = []
    if status != "missing":
        for b in blocks:
            # difflib block (a=haystack, b=needle): a[i:i+size] == b[j:j+size].
            hs_start = b.a
            matched_ids.extend(_owners_in_range(owners, hs_start, hs_start + b.size))
            matched_chars.append(joined[hs_start : hs_start + b.size])

    # De-dup matched_ids keeping order (a block may span the same item twice).
    seen: set[str] = set()
    dedup_ids: list[str] = []
    for mid in matched_ids:
        if mid not in seen:
            seen.add(mid)
            dedup_ids.append(mid)

    return VerifyEntryResult(
        entry_id=entry.id,
        text=entry.text,
        required=entry.required,
        category=entry.category,
        status=status,
        similarity=round(recall, 4),
        matched_item_ids=dedup_ids,
        matched_text="".join(matched_chars) if matched_chars else None,
    )


def verify(
    items: list[Item],
    entries: list[VerifyEntry],
    thresholds: Thresholds,
) -> tuple[int, int, int, list[VerifyEntryResult]]:
    """Verify all entries against the OCR items.

    Returns ``(matched_count, partial_count, missing_count, results)``. Assembly
    of the joined OCR text happens once here; per-entry work is then O(needle).

    Entry ids are auto-assigned (``v1``, ``v2``, …) when the caller didn't
    supply one, so the results are always correlatable — done here (not at the
    HTTP layer) so direct callers of :func:`verify` get the same guarantee.
    """
    joined, owners = _assemble_text(items)
    results: list[VerifyEntryResult] = []
    m = p = g = 0
    for i, e in enumerate(entries, start=1):
        if not e.id:
            e.id = f"v{i}"
        r = match_entry(e, items, joined, owners, thresholds)
        results.append(r)
        if r.status == "matched":
            m += 1
        elif r.status == "partial":
            p += 1
        else:
            g += 1
    return m, p, g, results
