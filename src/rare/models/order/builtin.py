"""Reading-order backends.

- top-bottom: sort by region centroid (y, x) — simple, robust baseline.
- left-right: sort by region left edge, reading near-aligned blocks top-down.
"""

from __future__ import annotations

from rare.models.registry import register


@register("order", "top-bottom")
class TopBottomBackend:
    """Sort regions by centroid y (then x)."""

    def order(
        self, layout, *, image=None, page_no=None, pdf_stem=None, ocr_lines=None, img_path=None, pdf_root=None
    ) -> list[int]:
        def centroid(block):
            x1, y1, x2, y2 = block.coordinates
            return (y1 + y2) / 2.0, (x1 + x2) / 2.0

        return [
            i for i, _ in sorted(enumerate(layout), key=lambda iv: centroid(iv[1]))
        ]


@register("order", "left-right")
class LeftRightBackend:
    """Sort regions left-to-right, breaking near-horizontal ties top-down.

    A plain sort on x alone splits a column whenever two blocks in it start a
    pixel or two apart. Instead, blocks are first grouped into vertical bands:
    walking left to right, a block joins the current band while its left edge is
    within `x_tol` pixels of the band's leftmost block, and opens a new band
    otherwise. Anchoring on the band's leftmost member (rather than the previous
    block) keeps a band at most `x_tol` wide, so a slow rightward drift across
    the page cannot chain everything into one band. Each band is then read
    top-down, i.e. of two blocks only a few pixels apart horizontally the upper
    one comes first.

    Config keys:
      x_tol — band width in pixels (default 10).
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self.x_tol = float(cfg.get("x_tol", 10.0))

    def order(
        self, layout, *, image=None, page_no=None, pdf_stem=None, ocr_lines=None, img_path=None, pdf_root=None
    ) -> list[int]:
        n = len(layout)
        if n < 2:
            return list(range(n))

        def left(i):
            return layout[i].coordinates[0]

        def top(i):
            return layout[i].coordinates[1]

        bands: list[tuple[float, list[int]]] = []
        for i in sorted(range(n), key=lambda i: (left(i), top(i))):
            if bands and left(i) - bands[-1][0] <= self.x_tol:
                bands[-1][1].append(i)
            else:
                bands.append((left(i), [i]))

        return [
            i
            for _, band in bands
            for i in sorted(band, key=lambda i: (top(i), left(i)))
        ]