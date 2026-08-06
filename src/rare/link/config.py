"""Tunable thresholds for the linking passes.

Every number the linker decides on lives here rather than inline, so the
behaviour can be inspected and overridden from a JSON config without reading
the algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class LinkConfig:
    # --- caption -> figure (geometry) ---------------------------------------
    # A caption must share this fraction of its width with the figure.
    caption_min_h_overlap: float = 0.25
    # …and sit within this fraction of the page height of it.
    caption_max_gap_frac: float = 0.08
    # Captions print below their figure far more often than above.
    caption_below_bonus: float = 0.15

    # --- orphan caption -> article -------------------------------------------
    # How far a caption may be from an article's nearest item on the page.
    orphan_max_distance_frac: float = 0.5

    # --- entity rarity --------------------------------------------------------
    # A key occurring in more than this share of articles carries no signal —
    # "Ljubljana" in a Slovenian music magazine links nothing.
    max_doc_frequency: float = 0.25
    # Very short keys collide by accident.
    min_key_length: int = 4
    # Fan-out cap for entity-overlap links, per item.
    max_entity_links_per_item: int = 8

    # --- cross-page continuation ----------------------------------------------
    continuation_min_score: float = 0.55
    # A near-identical title means a jump headline or a duplicate detection.
    title_similarity_threshold: float = 0.85
    title_weight: float = 0.60
    untitled_weight: float = 0.25          # continuation carries no headline
    header_weight: float = 0.25            # shared running header
    entity_weight: float = 0.40
    seam_weight: float = 0.35              # sentence runs across the boundary
    # Only consider articles this many pages apart.
    max_page_gap: int = 1

    @classmethod
    def from_dict(cls, config: dict | None) -> "LinkConfig":
        """Build from a JSON config, ignoring keys that aren't ours."""
        if not config:
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in config.items() if k in known})
