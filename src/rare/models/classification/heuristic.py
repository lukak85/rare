from __future__ import annotations

from rare.link.classify import SECTION_TO_GENRE, genre_for_section
from rare.models.registry import register


@register("classification", "heuristic")
class HeuristicClassification:
    reads_headers = True

    def __init__(self, config: dict | None = None):
        cfg = dict(config or {})
        self.classes = list(cfg.get("classes") or sorted(set(SECTION_TO_GENRE.values())))

    def classify(self, text: str, section: str | None = None, title: str | None = None) -> str:
        """The genre the section names, else the one the title names, else ""."""
        return (
            genre_for_section(section, self.classes)
            or genre_for_section(title, self.classes)
            or ""
        )
