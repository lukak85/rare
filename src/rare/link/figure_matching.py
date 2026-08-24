"""
Caption <-> figure pairing for document layout analysis.

Coordinate convention: (x0, y0, x1, y1), top-left origin, y grows DOWNWARD.
"Below" therefore means larger y. If you use native PDF points (y up),
either flip your y values on input or change the `below` line in `pair_cost`.
"""

from dataclasses import dataclass

from collections import defaultdict

from rare.doc.schema import CaptionItem, FigBylineItem, FigureItem, GlasanaDocument, Link
from rare.link._geometry import box_of, page_size
from rare.link.config import LinkConfig


@dataclass(frozen=True)
class Box:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return 0.5 * (self.x0 + self.x1)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y0 + self.y1)


def _axis_gap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Signed 1-D gap between intervals [a0,a1] and [b0,b1].

    > 0  : clear separation (distance between nearest edges)
    <= 0 : the intervals overlap (magnitude = overlap length)
    """
    if a1 <= b0:
        return b0 - a1
    if b1 <= a0:
        return a0 - b1
    return -(min(a1, b1) - max(a0, b0))  # overlapping -> negative


def pair_cost(
    fig: Box,
    cap: Box,
    page_w: float,
    page_h: float,
    max_gap_frac: float = 0.20,
    above_penalty: float = 2.0,
    side_penalty: float = 3.0,
    align_weight: float = 0.5,
):
    """Cost of assigning caption `cap` to figure `fig`.

    Returns a float cost (lower = better) or None if the pair violates the
    distance bound and should never be matched.

    Tunables
    --------
    max_gap_frac : reject if edge separation exceeds this fraction of the page
                   dimension (0.20 -> your 20% rule), checked per axis.
    above_penalty: multiplier when the caption sits above the figure.
    side_penalty : multiplier when the caption sits beside the figure.
    align_weight : weight of cross-axis center misalignment relative to the gap.
    """
    v_gap = _axis_gap(fig.y0, fig.y1, cap.y0, cap.y1)  # vertical separation
    h_gap = _axis_gap(fig.x0, fig.x1, cap.x0, cap.x1)  # horizontal separation

    # --- hard distance bound (per axis) ---
    if v_gap > max_gap_frac * page_h:
        return None
    if h_gap > max_gap_frac * page_w:
        return None

    # normalized edge separation (only positive separation counts)
    gap = max(v_gap, 0.0) / page_h + max(h_gap, 0.0) / page_w

    below = cap.cy >= fig.cy  # <-- flip this comparison for PDF (y-up) coords

    if v_gap >= h_gap:
        # predominantly a vertical (stacked) relationship
        misalign = abs(fig.cx - cap.cx) / max(fig.width, 1e-6)
        directional = 1.0 if below else above_penalty
    else:
        # predominantly a side-by-side relationship
        misalign = abs(fig.cy - cap.cy) / max(fig.height, 1e-6)
        directional = side_penalty

    return (gap + align_weight * misalign) * directional


def match_captions(figures, captions, page_w, page_h, **cost_kwargs):
    """Greedy 1:1 assignment of captions to figures.

    Parameters
    ----------
    figures, captions : sequences of Box
    page_w, page_h    : page dimensions (same units as the boxes)
    cost_kwargs       : forwarded to `pair_cost` (max_gap_frac, penalties, ...)

    Returns
    -------
    pairs           : list of (fig_idx, cap_idx, cost), best first
    unmatched_figs  : figure indices with no valid caption
    unmatched_caps  : caption indices with no valid figure
    """
    cost_kwargs.setdefault("max_gap_frac", 0.2)
    candidates = []
    for fi, fig in enumerate(figures):
        for ci, cap in enumerate(captions):
            c = pair_cost(fig, cap, page_w, page_h, **cost_kwargs)
            if c is not None:
                candidates.append((c, fi, ci))

    candidates.sort(key=lambda t: t[0])  # cheapest first

    used_fig, used_cap, pairs = set(), set(), []
    for c, fi, ci in candidates:
        if fi in used_fig or ci in used_cap:
            continue
        pairs.append((fi, ci, c))
        used_fig.add(fi)
        used_cap.add(ci)

    unmatched_figs = [i for i in range(len(figures)) if i not in used_fig]
    unmatched_caps = [i for i in range(len(captions)) if i not in used_cap]
    return pairs, unmatched_figs, unmatched_caps

def link_captions(doc: GlasanaDocument, config: LinkConfig) -> int:
    """Set `figure_id` on every caption we can place. Returns how many."""
    figures_by_page = defaultdict(list)
    captions_by_page = defaultdict(list)
    bylines_by_page = defaultdict(list)

    for item in doc.items.values():
        page_no = item.provenance.page_no
        if isinstance(item, FigureItem):
            figures_by_page[page_no].append(item)
        elif isinstance(item, CaptionItem):
            captions_by_page[page_no].append(item)
        elif isinstance(item, FigBylineItem):
            bylines_by_page[page_no].append(item)

    linked = 0
    for page_no, figures in figures_by_page.items():
        page_w, page_h = page_size(doc, page_no)
        figure_boxes = [Box(*box_of(figure)) for figure in figures]

        # Captions and photo credits are matched in separate 1:1 rounds. One
        # round would make them compete: `match_captions` gives a figure a
        # single partner, and a figure that carries both a caption and a credit
        # would lose whichever scored worse.
        for texts in (captions_by_page.get(page_no), bylines_by_page.get(page_no)):
            if not texts:
                continue
            pairs, _, _ = match_captions(
                figure_boxes,
                [Box(*box_of(text)) for text in texts],
                page_w,
                page_h,
            )
            for figure_index, text_index, _cost in pairs:
                caption, figure = texts[text_index], figures[figure_index]
                caption.figure_id = figure.item_id
                doc.add_link(
                    Link(
                        kind="caption-of",
                        from_id=caption.item_id,
                        to_id=figure.item_id,
                        method="geometry",
                    )
                )
                linked += 1

    return linked
