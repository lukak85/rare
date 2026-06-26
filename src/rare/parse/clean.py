"""Normalise line-wrap artefacts inside a single region's extracted text.

pdfplumber positions glyphs, then flattens the layout into a string, inserting a
newline at every visual line break. Within one paragraph region that produces
two artefacts that are not "real" text, only a side effect of digitising a
printed column:

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

# A word character, then a hyphen at a line break, then the continuation.
# Captures the chars either side so we can inspect the continuation's case.
_HYPHEN_WRAP = re.compile(r"(\w)-\r?\n(\w)")
_NEWLINES = re.compile(r"\s*\r?\n\s*")
_SPACES = re.compile(r"[ \t]+")

# Categories whose internal newlines are meaningful — do not normalise these.
STRUCTURED_LABELS: frozenset[str] = frozenset(
    {"Table", "OrderedList", "UnorderedList"}
)


def _dehyphenate(match: re.Match) -> str:
    before, after = match.group(1), match.group(2)
    # Lowercase continuation -> a wrapped single word: drop the hyphen.
    # Uppercase/digit continuation is likely a real compound broken at its
    # hyphen ("ZDA-" / "well-Known"): keep the hyphen, just close the line.
    if after.islower():
        return before + after
    return before + "-" + after


def normalize_text(text: str) -> str:
    """Collapse line-wrap newlines and de-hyphenate wrapped words in prose."""
    if not text:
        return text
    text = _HYPHEN_WRAP.sub(_dehyphenate, text)
    text = _NEWLINES.sub(" ", text)
    text = _SPACES.sub(" ", text)
    return text.strip()