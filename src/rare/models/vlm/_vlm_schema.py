"""Pydantic schema for the per-region VLM output format.

All VLM backends (chat-based and specialized) emit a `VLMDocument` and then
the assembler converts it into a full `GlasanaDocument`. This keeps the
schema VLMs are asked to follow simpler than the full GlasanaDocument
(no discriminated unions, no Provenance, no Article grouping).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class VLMRegion(BaseModel):
    label: str  # one of RegionCategory values; see rare.models.vlm.prompts.GLASANA_LABELS
    text: str = ""
    bbox_norm_1000: Optional[list[float]] = None  # [x0, y0, x1, y1], top-left origin
    detection_score: Optional[float] = None
    image_path: Optional[str] = None  # for FigureItem when the VLM/parser cropped one


class VLMPage(BaseModel):
    page_no: int
    width: int = 0      # rendered page dimensions in pixels, 0 if unknown
    height: int = 0
    regions: list[VLMRegion] = Field(default_factory=list)


class VLMDocument(BaseModel):
    pages: list[VLMPage] = Field(default_factory=list)
