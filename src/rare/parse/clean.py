"""Normalise line-wrap artefacts inside a single region's extracted text.

pdfplumber positions glyphs, then flattens the layout into a string, inserting a
newline at every visual line break. Within one paragraph region that produces
two artefacts that are not "real" text, only a side effect of digitising a
printed column:

* soft hyphens (U+00AD) — discretionary break hints the typesetter embeds and
  pdfplumber extracts literally; never real text, so always removed
* hyphenated word-splits at line ends — ``"informa-\ntion"`` → ``"information"``
* mid-paragraph newlines — a wrapped line that should just be a space

This module removes both. It is the per-region counterpart to
``merge.merge_flowing_paragraphs`` (which joins *across* regions): run this
first so the merge sees clean single-line text.

Structured regions (tables, lists) are intentionally left untouched — their
newlines carry structure, so callers should skip those categories.
"""

from __future__ import annotations

import re

# Zero-width / formatting characters that carry no text: zero-width space,
# ZWNJ, ZWJ, word joiner, and BOM / zero-width no-break space. Always stripped.
_ZERO_WIDTH = re.compile(r"[​‌‍⁠﻿]")
# Unicode spaces (non-breaking, en/em quad, thin, narrow nbsp, ideographic …)
# folded to a plain space so the whitespace-collapse below catches them.
_UNICODE_SPACES = re.compile(r"[  -   　]")

# Soft hyphen (U+00AD)
_SOFT_HYPHEN_WRAP = re.compile(r"­[ \t]*\r?\n[ \t]*")
# Any remaining soft hyphen (mid-line, no break) is simply stripped.
_SOFT_HYPHEN = re.compile(r"­")

# A word character, then a hyphen/dash at a line break, then the continuation.
# Captures the chars either side (and the dash) so the join logic can inspect
# both the dash type and the continuation's case. Tolerates stray spaces/tabs
# around the newline (pdfplumber often emits a trailing space at line ends) so
# the word still closes up with no gap. Dash class: ASCII hyphen, Unicode
# hyphen / non-breaking hyphen, figure/en/em dash, horizontal bar.
_HYPHEN_WRAP = re.compile(
    r"(\w)([-‐‑‒–—―])[ \t]*\r?\n[ \t]*(\w)"
)
# En/em dashes and bars are real marks (compound or punctuation): keep them.
_KEEP_DASHES = frozenset("‒–—―")
_NEWLINES = re.compile(r"\s*\r?\n\s*")
_SPACES = re.compile(r"[ \t]+")

# Categories whose internal newlines are meaningful — do not normalise these.
STRUCTURED_LABELS: frozenset[str] = frozenset(
    {"Table", "OrderedList", "UnorderedList"}
)


def _dehyphenate(match: re.Match) -> str:
    before, dash, after = match.group(1), match.group(2), match.group(3)
    # En/em dashes & bars are real marks (e.g. the compound "vzgojno—varstvenih"
    # broken at the dash): keep the dash, just close up the wrap so no space
    # sneaks in from the collapsed newline.
    if dash in _KEEP_DASHES:
        return before + dash + after
    # Hyphen class. Lowercase continuation -> a wrapped single word: drop the
    # hyphen. Uppercase/digit continuation is likely a real compound broken at
    # its hyphen ("ZDA-" / "well-Known"): keep the hyphen, just close the line.
    if after.islower():
        return before + after
    return before + "-" + after


def normalize_text(text: str) -> str:
    """Collapse line-wrap newlines and de-hyphenate wrapped words in prose."""
    if not text:
        return text
    # Drop zero-width/formatting chars; fold Unicode spaces to plain spaces.
    text = _ZERO_WIDTH.sub("", text)
    text = _UNICODE_SPACES.sub(" ", text)
    # Soft hyphens first: drop the break ones (join the word), then any leftover.
    text = _SOFT_HYPHEN_WRAP.sub("", text)
    text = _SOFT_HYPHEN.sub("", text)
    text = _HYPHEN_WRAP.sub(_dehyphenate, text)
    text = _NEWLINES.sub(" ", text)
    text = _SPACES.sub(" ", text)
    return text.strip()