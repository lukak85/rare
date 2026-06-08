#!/usr/bin/env python3
"""
Normalize a model's prediction output into the *flat* markdown layout that
OmniDocBench's end2end loader expects, namely one file per page named
`<image_stem>.md` directly inside the output directory.

OmniDocBench's `End2EndDataset._resolve_prediction_path` only ever looks for
`<pred_folder>/<image_stem>.md` (plus `.mmd` / `<image_name>.md` fallbacks).
Anything else is silently treated as an empty page and scores the maximum
normalized edit distance of 1.0. Different tools emit different layouts, so we
normalize them all here before evaluation.

Recognized inputs (auto-detected, or force with --layout):

  1. odb.json  — a single JSON file in OmniDocBench's prediction-JSON shape:
       {"<image>.jpg": [{"content": "...", ...}, ...], ...}
     Each page's `content` fields are concatenated (in array order) into one
     markdown file. Used by e.g. our MinerU export.

  2. paddlex   — a directory of per-page bundle subfolders, as produced by
     PaddleX's `PPStructureV3.save_to_markdown` (OmniDocBench's
     tools/model_infer/PaddleOCR_img2md.py):
       <dir>/<stem>/<stem>.md   (+ <stem>_res.json, imgs/)
     The inner `.md` is lifted out to `<out>/<stem>.md`.

  3. flat      — a directory that already contains `<stem>.md` / `<stem>.mmd`
     files. Copied through unchanged (normalized to `.md`).

Single whole-document markdown (e.g. Docling run on a multi-page PDF) is NOT a
recognized layout: it has no per-page boundaries to split on. Re-export it from
the tool with a page-break marker, e.g. Docling's

    doc.export_to_markdown(page_break_placeholder="<!-- PAGE -->")

then split it here with `--split-on`. Each chunk becomes `<input_stem>_<i>.md`
(0-indexed), matching OmniDocBench's `<docid>_<page>.jpg` page naming. Empty
chunks are kept so page indices stay aligned with the GT.

Usage:
  scripts/omnidocbench/normalize_pred.py <input> <out_dir> [--layout auto|odb-json|paddlex|flat] [--dry-run]
  scripts/omnidocbench/normalize_pred.py <doc.md> <out_dir> --split-on "<!-- PAGE -->" [--dry-run]

Then point the evaluator at <out_dir>:
  scripts/omnidocbench/run.sh <gt.json> <out_dir> <result_dir>
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PARAGRAPH_SEP = "\n\n"
MD_SUFFIXES = (".md", ".mmd")


def _stem(name: str) -> str:
    """Image filename -> stem, matching OmniDocBench's `img_name[:-4]` logic
    for the common 4-char extensions (.jpg/.png/...) while staying correct for
    anything `Path.stem` handles."""
    return Path(name).stem


def detect_layout(input_path: Path) -> str:
    """Best-effort classification of the input into one of the recognized
    layouts. Raises if nothing matches."""
    if input_path.is_file():
        if input_path.suffix.lower() == ".json":
            return "odb-json"
        if input_path.suffix.lower() in MD_SUFFIXES:
            return "flat"  # a lone markdown file; handled by the flat path
        raise SystemExit(f"Unrecognized input file type: {input_path}")

    if not input_path.is_dir():
        raise SystemExit(f"Input does not exist: {input_path}")

    # Flat if there are markdown files directly inside.
    if any(p.suffix.lower() in MD_SUFFIXES for p in input_path.iterdir() if p.is_file()):
        return "flat"

    # PaddleX-style if subdirectories contain markdown one level down.
    for sub in input_path.iterdir():
        if sub.is_dir() and any(
            p.suffix.lower() in MD_SUFFIXES for p in sub.iterdir() if p.is_file()
        ):
            return "paddlex"

    raise SystemExit(
        f"Could not detect layout for {input_path}. "
        f"Pass --layout explicitly (odb-json|paddlex|flat)."
    )


def from_odb_json(input_path: Path) -> dict[str, str]:
    """odb.json -> {stem: markdown}. Concatenates each page's `content` fields
    (skipping null/empty, e.g. image-only blocks) in array order."""
    data = json.loads(input_path.read_text())
    if not isinstance(data, dict):
        raise SystemExit(
            f"{input_path} is not OmniDocBench prediction-JSON (expected a "
            f"dict keyed by image name, got {type(data).__name__})."
        )
    pages: dict[str, str] = {}
    for image_name, blocks in data.items():
        parts = [b["content"] for b in blocks if isinstance(b, dict) and b.get("content")]
        pages[_stem(image_name)] = PARAGRAPH_SEP.join(parts)
    return pages


def _pick_markdown(files: list[Path], prefer_stem: str | None) -> Path | None:
    """Choose the markdown file from `files`. Prefer `<prefer_stem>.md`, then a
    sole `.md`, then a sole `.mmd`; return None if ambiguous or none."""
    md = [p for p in files if p.suffix.lower() == ".md"]
    mmd = [p for p in files if p.suffix.lower() == ".mmd"]
    if prefer_stem is not None:
        for p in md + mmd:
            if p.stem == prefer_stem:
                return p
    if len(md) == 1:
        return md[0]
    if not md and len(mmd) == 1:
        return mmd[0]
    return None


def from_paddlex(input_path: Path) -> dict[str, Path]:
    """PaddleX bundle dir -> {stem: md_path}. Each immediate subdirectory is one
    page; its inner markdown becomes `<subdir_name>.md`."""
    pages: dict[str, Path] = {}
    for sub in sorted(p for p in input_path.iterdir() if p.is_dir()):
        files = [p for p in sub.iterdir() if p.is_file()]
        chosen = _pick_markdown(files, prefer_stem=sub.name)
        if chosen is None:
            print(f"  WARNING: no unambiguous markdown in {sub}, skipping", file=sys.stderr)
            continue
        pages[sub.name] = chosen
    return pages


def from_flat(input_path: Path) -> dict[str, Path]:
    """Flat dir (or a single .md file) -> {stem: md_path}."""
    if input_path.is_file():
        return {input_path.stem: input_path}
    pages: dict[str, Path] = {}
    for p in sorted(input_path.iterdir()):
        if p.is_file() and p.suffix.lower() in MD_SUFFIXES:
            pages[p.stem] = p
    return pages


def split_single_markdown(input_path: Path, marker: str) -> dict[str, str]:
    """Single whole-document markdown -> {f'{stem}_{i}': chunk}. Splits on
    `marker` (the page-break placeholder re-exported from the tool). Empty
    chunks are kept so page indices stay aligned with the GT's `_<page>`
    numbering."""
    if not input_path.is_file() or input_path.suffix.lower() not in MD_SUFFIXES:
        raise SystemExit(f"--split-on requires a single .md/.mmd file, got {input_path}")
    if not marker:
        raise SystemExit("--split-on marker must be a non-empty string")
    text = input_path.read_text()
    if marker not in text:
        raise SystemExit(
            f"Marker {marker!r} not found in {input_path}. Re-export the document "
            f"with a page-break placeholder (e.g. Docling's "
            f"export_to_markdown(page_break_placeholder=...))."
        )
    chunks = text.split(marker)
    stem = input_path.stem
    return {f"{stem}_{i}": chunk.strip("\n") for i, chunk in enumerate(chunks)}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Normalize model predictions into OmniDocBench's flat "
        "<stem>.md layout.",
    )
    ap.add_argument("input", type=Path, help="odb.json file, or a prediction directory")
    ap.add_argument("out_dir", type=Path, help="destination directory for flat <stem>.md files")
    ap.add_argument(
        "--layout",
        choices=["auto", "odb-json", "paddlex", "flat"],
        default="auto",
        help="input layout (default: auto-detect)",
    )
    ap.add_argument(
        "--split-on",
        metavar="MARKER",
        help="split a single whole-document markdown on this page-break marker "
        "into <stem>_<i>.md files (bypasses --layout)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be written without touching the filesystem",
    )
    args = ap.parse_args()

    # Build a unified {stem: content-or-source-path} plan.
    text_pages: dict[str, str] = {}
    file_pages: dict[str, Path] = {}
    if args.split_on is not None:
        print(f"Splitting {args.input} on marker {args.split_on!r}")
        text_pages = split_single_markdown(args.input, args.split_on)
    else:
        layout = detect_layout(args.input) if args.layout == "auto" else args.layout
        print(f"Input layout: {layout}  ({args.input})")
        if layout == "odb-json":
            text_pages = from_odb_json(args.input)
        elif layout == "paddlex":
            file_pages = from_paddlex(args.input)
        elif layout == "flat":
            file_pages = from_flat(args.input)
        else:  # pragma: no cover - argparse restricts choices
            raise SystemExit(f"Unknown layout: {layout}")

    total = len(text_pages) + len(file_pages)
    if total == 0:
        raise SystemExit("No prediction pages found; nothing to write.")

    if args.dry_run:
        for stem in sorted([*text_pages, *file_pages]):
            src = "(from content)" if stem in text_pages else f"<- {file_pages[stem]}"
            print(f"  would write {args.out_dir / (stem + '.md')}  {src}")
        print(f"Dry run: {total} page(s) would be written to {args.out_dir}")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for stem, content in sorted(text_pages.items()):
        (args.out_dir / f"{stem}.md").write_text(content)
        written += 1
    for stem, src in sorted(file_pages.items()):
        shutil.copyfile(src, args.out_dir / f"{stem}.md")
        written += 1

    print(f"Wrote {written} markdown file(s) to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())