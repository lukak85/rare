"""Shared helpers for writing parse outputs (HTML / MD / JSON / figures/)."""

from __future__ import annotations

from pathlib import Path

from rare.doc.renderers import to_html, to_markdown, to_markdown_pages
from rare.doc.schema import GlasanaDocument


def write_outputs(
    doc: GlasanaDocument,
    output_root: str | Path,
    per_page: bool = False,
) -> Path:
    """Write `{stem}.{html,md}` and `{stem}_doc.json` to `<output_root>/<stem>/`.

    When `per_page` is True, also write one Markdown file per page into a
    `pages/` subdirectory as `{stem}_{page_no}.md` (the `<stem>_<page>` naming
    the OmniDocBench VLM track expects). Pages with no body content still get an
    (empty) file so the prediction page set stays aligned with the ground truth.

    Returns the per-document output directory.
    """
    stem = doc.source_pdf or "document"
    out_dir = Path(output_root) / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}_doc.json").write_text(doc.model_dump_json(indent=2))
    (out_dir / f"{stem}.md").write_text(to_markdown(doc))
    (out_dir / f"{stem}.html").write_text(to_html(doc))

    if per_page:
        pages_dir = out_dir / "pages"
        pages_dir.mkdir(exist_ok=True)
        for page_no, page_md in to_markdown_pages(doc).items():
            (pages_dir / f"{stem}_{page_no}.md").write_text(page_md)

    return out_dir
