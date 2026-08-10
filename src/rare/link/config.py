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

    # --- column splitting -------------------------------------------------------
    # A column (records, letters, news briefs) carries one Headline for the
    # whole page and sets each individual piece under a Subhead; a feature uses
    # Subheads for internal breaks instead. Density tells them apart — a column
    # opens a piece every few items, a feature every twenty or so.
    split_columns: bool = True
    split_min_headings: int = 3
    split_max_items_per_heading: float = 10.0
    # Not every Subhead titles a piece. A quiz sets its quotations under
    # lead-ins ("Leta 1808:", "Iz pogovora z L. Schlosserjem:") and a crossword
    # numbers its clues ("2. Rebusa"); both are internal to one piece.
    split_skip_colon_headings: bool = True
    split_skip_enumerated_headings: bool = True
    # Sections known to run as columns relax both thresholds, so a short column
    # under a familiar running header still splits.
    column_section_min_headings: int = 2
    column_section_max_items_per_heading: float = 12.0
    # Matched against Article.section folded and as a substring, because OCR
    # reads the running header back with mirrored bleed ("IAHHCIO ODMEVI").
    column_sections: tuple[str, ...] = (
        "telegrami",
        "odmevi",
        "novice",
        "od vsepovsod",
        "pisma",
        "plosce",
        "izdaje",
        "mine iz tujih",
        "iz vsebine",
    )

    # --- section changes ---------------------------------------------------------
    # An article cannot run across a change of running header: the section
    # changed, so the piece did too.
    split_section_changes: bool = True
    # Below this token overlap two running headers name different sections.
    # The gap is wide in practice: OCR noise on one printing of a header still
    # overlaps the next ("(PRED)USM ERJENE STRANI" against the misread
    # "CPRED)USMEHJENE STRANI" scores 0.33), while a genuine change of section
    # shares no content word at all and scores 0.0.
    section_change_max_similarity: float = 0.25
    # …and the same test on characters, for a header whose words themselves
    # came back misread. Same-section pairs score 0.63 and up in this corpus,
    # genuine changes 0.47 and below.
    section_change_min_char_similarity: float = 0.55
    # Captions and standfirsts get labelled Header; a long one standing in for
    # a section name would invent a change on every page it appears.
    section_header_max_words: int = 8
    # Cutting one or two stray items off the end of an article buys nothing.
    section_change_min_piece_items: int = 3
    # A change of section also forbids merging across it.
    veto_merge_on_section_change: bool = True

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

    # --- article classification -----------------------------------------------
    # How much of an article's text the classifier sees. Generative backends
    # are charged per token and the genre is evident from the opening.
    classify_max_chars: int = 4000
    # The running header names the section a piece ran in, which is most of the
    # way to its genre — "Telegrami" is news, "Plošče" reviews. Give it to the
    # classifier as context…
    classify_include_section: bool = True
    # …and fall back on it when the classifier says nothing usable, or when a
    # piece is too short to classify at all.
    classify_section_fallback: bool = True

    @classmethod
    def from_dict(cls, config: dict | None) -> "LinkConfig":
        """Build from a JSON config, ignoring keys that aren't ours."""
        if not config:
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in config.items() if k in known})
