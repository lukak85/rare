from __future__ import annotations

import re

from rare.link.classify import SECTION_TO_GENRE, genre_for_section
from rare.models.registry import register

_SECTION_LINE = re.compile(r"^Rubrika:(.*)$", re.MULTILINE)


@register("classification", "heuristic")
class HeuristicClassification:

    def __init__(self, config: dict | None = None):
        cfg = dict(config or {})
        self.classes = list(cfg.get("classes") or sorted(set(SECTION_TO_GENRE.values())))

    def classify(self, text: str) -> str:
        match = _SECTION_LINE.search(text or "")
        if not match:
            return ""
        return genre_for_section(match.group(1).strip(), self.classes) or ""
