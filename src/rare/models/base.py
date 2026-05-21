"""Backend protocols shared by the pipeline and VLM tracks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    import layoutparser as lp
    from PIL.Image import Image
    from rare.doc.schema import GlasanaDocument


@runtime_checkable
class LayoutBackend(Protocol):
    """Pipeline DLA backend — returns layout regions for one page image."""

    name: str

    def detect(self, image) -> "lp.Layout": ...


@runtime_checkable
class ReadingOrderBackend(Protocol):
    """Pipeline reading-order backend — returns a permutation of region indices.

    Args:
        layout: the lp.Layout produced by a LayoutBackend.
        image:  optional page image (PIL or ndarray) for vision-based orderers.
        page_no, pdf_stem: optional context for orderers that key off filenames
                           (e.g. a precomputed connections.json).
    Returns:
        A list of indices such that layout[indices[k]] is the k-th region in
        reading order.
    """

    name: str

    def order(
        self,
        layout: "lp.Layout",
        *,
        image: Optional["Image"] = None,
        page_no: Optional[int] = None,
        pdf_stem: Optional[str] = None,
    ) -> list[int]: ...


@runtime_checkable
class VLMBackend(Protocol):
    """VLM backend — takes a PDF path and returns a GlasanaDocument."""

    name: str

    def parse_pdf(self, pdf_path: str) -> "GlasanaDocument": ...
