"""Normalise entity surface forms into comparable matching keys.

Slovenian is heavily inflected, so the same person appears as "Haller",
"Hallerja", "Hallerjem", "Hallerjeva" across one article. Entity overlap only
means something once those collapse to a single key.

This is a deliberately small suffix-stripping heuristic, not a lemmatiser: it
only has to make two mentions of the *same* name collide more often than it
makes two *different* names collide. Over-stripping is the dangerous direction,
so the suffix list is short and only applied to sufficiently long tokens.
"""

from __future__ import annotations

import re
import unicodedata

# Longest first — "jema" must be tried before "ja" and "a".
_SUFFIXES: tuple[str, ...] = (
    "jema", "jevi", "jeva", "jevo", "jem", "jev", "jih", "jem",
    "oma", "ama", "ega", "emu", "ovi", "ova", "ovo", "ov", "om", "em",
    "ju", "ja", "je", "mi", "ih", "ah", "am", "u", "a", "e", "i", "o",
)
# Below this length, stripping a suffix destroys the stem more often than it
# helps ("Ana" -> "An").
_MIN_STEM = 5

_NON_WORD = re.compile(r"[^\w\s]+", re.UNICODE)
_WS = re.compile(r"\s+")


def _fold(text: str) -> str:
    """Casefold and strip diacritics, so 'Šivic' and 'Sivic' agree.

    OCR of 1980s print regularly loses carons, so matching must survive it.
    """
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _stem(token: str) -> str:
    for suffix in _SUFFIXES:
        if len(token) - len(suffix) >= _MIN_STEM and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def entity_key(text: str, label: str = "") -> str:
    """Return the canonical matching key for an entity surface form.

    Empty when the surface form carries nothing worth matching on.
    """
    folded = _fold(text)
    folded = _NON_WORD.sub(" ", folded)
    tokens = [t for t in _WS.split(folded) if t]
    if not tokens:
        return ""
    return " ".join(_stem(t) for t in tokens)


def surname_key(text: str, label: str = "") -> str:
    """Secondary key for people: the last token alone.

    Bylines print "MATEJA HALLER" while the body says "Haller"; keying on the
    surname lets those two mentions meet. Only meaningful for PER, and only
    when there is more than one token.
    """
    if label and label.upper() != "PER":
        return ""
    key = entity_key(text, label)
    tokens = key.split()
    if len(tokens) < 2:
        return ""
    return tokens[-1]


def keys_for(text: str, label: str = "") -> list[str]:
    """Every key an entity should be indexed under, primary first."""
    keys = []
    primary = entity_key(text, label)
    if primary:
        keys.append(primary)
    surname = surname_key(text, label)
    if surname and surname not in keys:
        keys.append(surname)
    return keys
