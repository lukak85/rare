"""Reading-order backends.

- top-bottom: sort by region centroid (y, x) — simple, robust baseline.
"""

from __future__ import annotations

from rare.models.registry import register


@register("order", "top-bottom")
class TopBottomBackend:
    """Sort regions by centroid y (then x)."""

    def order(self, layout, *, image=None, page_no=None, pdf_stem=None) -> list[int]:
        def centroid(block):
            x1, y1, x2, y2 = block.coordinates
            return (y1 + y2) / 2.0, (x1 + x2) / 2.0

        return [
            i for i, _ in sorted(enumerate(layout), key=lambda iv: centroid(iv[1]))
        ]

