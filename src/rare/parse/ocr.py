"""OCR fallback for regions the PDF text layer left empty.

The corpus PDFs are scans: one full-page image per page with an invisible OCR
text layer on top. Where that upstream OCR gave up, `rare.parse.text` extracts
nothing — not because the extraction is wrong, but because there are no glyphs
under the box to extract. Running headers are a frequent casualty (11% of them
come back empty), and unlike a missing paragraph a missing header is invisible
in the rendered output: the page simply loses its section marker.

This module re-reads those regions from the pixels. It runs *after*
`extract_text_for_page` and only on regions that came back empty, so a region
whose text the PDF already carries is never second-guessed — the text layer,
where it exists, is better than anything re-OCR'ing a 1971 halftone will give.

Tesseract is driven through its CLI rather than pytesseract: TSV output already
carries per-word confidences, and the binary is the only dependency. Note that
Slovenian is a separate package from Tesseract itself:

    sudo apt install tesseract-ocr tesseract-ocr-slv

Construction fails loudly when the requested language is missing, rather than
falling back to English and quietly filling the document with text whose
diacritics are wrong ("kvartet" read as "vartel").
"""

from __future__ import annotations

import csv
import io
import logging
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional, Sequence

from PIL import Image, ImageOps

from rare.parse.clean import STRUCTURED_LABELS, normalize_text
from rare.parse.figures import crop_region

logger = logging.getLogger(__name__)

# Labels re-read when their text comes back empty. Deliberately narrow: this is
# the set the OCR fallback has been measured on. Widening it is a config
# change, not a code change — see `--ocr-labels`.
DEFAULT_OCR_LABELS: frozenset[str] = frozenset({"Header"})

# Regions that are a single line of type by construction. Page segmentation
# mode is not a detail here: a one-line header read as a uniform block (psm 6)
# comes back split or empty, and a multi-line region read as a single line
# (psm 7) comes back as one run-on line.
SINGLE_LINE_LABELS: frozenset[str] = frozenset({
    "Author", "Byline", "Dateline", "FigByline", "Footer", "Header",
    "Kicker", "PageNum", "Section", "Subhead", "Subsubhead", "Translator",
})
PSM_SINGLE_LINE = 7
PSM_BLOCK = 6

# Regions that hold no text to find. Never OCR'd, whatever the label set says:
# a halftone read as a block of type comes back as pages of plausible-looking
# noise at low confidence.
NON_TEXT_LABELS: frozenset[str] = frozenset({"Figure", "Form", "Abandon"})


def available_languages() -> set[str]:
    """Languages the installed Tesseract can use. Empty if it isn't installed."""
    try:
        out = subprocess.run(
            ["tesseract", "--list-langs"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return set()
    # First line is a header ("List of available languages ...").
    return {line.strip() for line in out.splitlines()[1:] if line.strip()}


class TesseractOCR:
    """Reads one cropped region at a time.

    `dpi` is the resolution the *crop* is taken at, not the parse DPI: 200 (the
    parse default) leaves body-size type in these scans marginal for Tesseract,
    and upscaling a 200-DPI crop is not the same as rendering at 400.
    """

    name = "tesseract"

    def __init__(
        self,
        lang: str = "slv",
        dpi: int = 400,
        padding: int = 10,
        min_height: int = 90,
        min_confidence: float = 40.0,
    ):
        languages = available_languages()
        if not languages:
            raise RuntimeError(
                "tesseract is not installed or not on PATH "
                "(sudo apt install tesseract-ocr)."
            )
        if lang not in languages:
            raise RuntimeError(
                f"tesseract has no '{lang}' language data "
                f"(installed: {', '.join(sorted(languages))}). "
                f"Install it with `sudo apt install tesseract-ocr-{lang}`, or "
                f"pass a different language explicitly."
            )
        self.lang = lang
        self.dpi = dpi
        self.padding = padding
        self.min_height = min_height
        self.min_confidence = min_confidence

    # --- image preparation -------------------------------------------------

    def _preprocess(self, crop: Image.Image) -> Image.Image:
        """Grayscale, stretch contrast, upscale small crops.

        Deliberately stops short of binarising: these are halftone scans, and a
        global threshold eats thin strokes and light display faces — the very
        type the text layer already failed on. Tesseract's own Otsu pass does
        better on contrast-stretched grayscale than on a pre-binarised image.
        """
        gray = ImageOps.autocontrast(crop.convert("L"))
        if 0 < gray.height < self.min_height:
            factor = self.min_height / gray.height
            gray = gray.resize(
                (max(1, int(gray.width * factor)), self.min_height),
                Image.LANCZOS,
            )
        return gray

    # --- recognition -------------------------------------------------------

    def recognize(self, crop: Image.Image, *, single_line: bool = False) -> tuple[str, float]:
        """Read one crop. Returns (text, mean word confidence in 0-100).

        Crops twice as tall as they are wide are retried rotated: captions and
        credits set up the side of a photograph are common here, and Tesseract
        reads them one character per line because it never considers that the
        page might be turned. Which way it turns cannot be known from the
        geometry (both directions occur), so both are tried and the more
        confident reading wins.
        """
        if crop.width < 2 or crop.height < 2:
            return "", 0.0

        prepared = self._preprocess(crop)
        psm = PSM_SINGLE_LINE if single_line else PSM_BLOCK
        text, confidence = self._run(prepared, psm)

        if prepared.height >= 2 * prepared.width:
            for angle in (90, 270):
                # Rotated type is a single line, whatever the label implies.
                rotated, rotated_confidence = self._run(
                    prepared.rotate(angle, expand=True), PSM_SINGLE_LINE
                )
                if rotated_confidence > confidence:
                    text, confidence = rotated, rotated_confidence

        return text, confidence

    def _run(self, image: Image.Image, psm: int) -> tuple[str, float]:
        """One Tesseract invocation over a prepared image.

        TSV output is used instead of plain text because it carries per-word
        confidences; lines are rebuilt from the block/paragraph/line columns so
        the result keeps the region's line structure the way `words_to_text`
        does for the pdfplumber path.
        """
        buf = io.BytesIO()
        image.save(buf, format="PNG")

        try:
            proc = subprocess.run(
                ["tesseract", "stdin", "stdout", "-l", self.lang, "--psm", str(psm), "tsv"],
                input=buf.getvalue(), capture_output=True, check=True,
            )
        except (OSError, subprocess.CalledProcessError):
            logger.exception("tesseract failed on a %sx%s crop", image.width, image.height)
            return "", 0.0

        rows = csv.DictReader(
            io.StringIO(proc.stdout.decode("utf-8", "replace")),
            delimiter="\t", quoting=csv.QUOTE_NONE,
        )

        lines: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        confidences: list[float] = []
        for row in rows:
            word = (row.get("text") or "").strip()
            if not word:
                continue
            try:
                confidence = float(row["conf"])
            except (KeyError, TypeError, ValueError):
                continue
            if confidence < 0:               # -1 marks the non-word rows
                continue
            lines[(row["block_num"], row["par_num"], row["line_num"])].append(word)
            confidences.append(confidence)

        text = "\n".join(" ".join(words) for words in lines.values())
        mean = sum(confidences) / len(confidences) if confidences else 0.0
        return text, mean


# --- the pipeline hook ------------------------------------------------------

def fill_empty_regions(
    regions: Sequence[dict],
    texts: dict[str, str],
    *,
    recognizer: TesseractOCR,
    pdf_path: str | Path,
    page_no: int,
    labels: Iterable[str] = DEFAULT_OCR_LABELS,
    page_image: Optional[Image.Image] = None,
    page_image_dpi: Optional[int] = None,
) -> int:
    """Fill in `texts` for regions of `labels` that came back empty. Mutates
    `texts`; returns how many regions were filled.

    Regions that already have text are left alone, as are labels outside
    `labels` and anything in NON_TEXT_LABELS. A reading below the recognizer's
    `min_confidence` is discarded — an empty header is a smaller problem than a
    header filled with noise.

    Filled regions are marked with `text_source` / `ocr_confidence`, which
    `assemble_page` carries onto the item's provenance so OCR'd text stays
    distinguishable from text the PDF actually carried.

    The page is rendered lazily and only once, so pages with no gap in them
    cost nothing. A `page_image` the caller already has is reused when it was
    rendered at least as finely as the recognizer asks for.
    """
    wanted = frozenset(labels) - NON_TEXT_LABELS
    gaps = [
        region for region in regions
        if region.get("label") in wanted
        and not texts.get(region["region_id"], "").strip()
    ]
    if not gaps:
        return 0

    if page_image is not None and (page_image_dpi or 0) >= recognizer.dpi:
        image = page_image
    else:
        from rare.parse.pdf import render_page
        image = render_page(pdf_path, page_no, dpi=recognizer.dpi)

    filled = 0
    for region in gaps:
        label = region["label"]
        crop = crop_region(image, region["bbox_norm_1000"], recognizer.padding)
        text, confidence = recognizer.recognize(
            crop, single_line=label in SINGLE_LINE_LABELS
        )
        text = text.strip()

        if not text:
            continue
        if confidence < recognizer.min_confidence:
            logger.debug(
                "page %d: dropped %s OCR at confidence %.1f (%r)",
                page_no, label, confidence, text[:60],
            )
            continue

        # Same treatment the pdfplumber path gives: structured regions keep
        # their newlines, prose gets unwrapped and de-hyphenated.
        if label not in STRUCTURED_LABELS:
            text = normalize_text(text)

        texts[region["region_id"]] = text
        region["text_source"] = f"ocr:{recognizer.name}"
        region["ocr_confidence"] = confidence
        filled += 1

    if filled:
        logger.info("page %d: filled %d empty region(s) by OCR", page_no, filled)
    return filled