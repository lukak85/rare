"""Shared helpers for writing parse outputs (HTML / MD / JSON / figures/)."""

from __future__ import annotations

from pathlib import Path

from rare.doc.renderers import to_html, to_markdown
from rare.doc.schema import GlasanaDocument


def write_outputs(doc: GlasanaDocument, output_root: str | Path) -> Path:
    """Write `{stem}.{html,md}` and `{stem}_doc.json` to `<output_root>/<stem>/`.

    Returns the per-document output directory.
    """
    stem = doc.source_pdf or "document"
    out_dir = Path(output_root) / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}_doc.json").write_text(doc.model_dump_json(indent=2))
    (out_dir / f"{stem}.md").write_text(to_markdown(doc))
    (out_dir / f"{stem}.html").write_text(to_html(doc))
    return out_dir
