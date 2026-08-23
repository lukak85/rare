"""OCR fallback for regions the PDF text layer left empty.

The corpus PDFs are scans: one full-page image per page with an invisible OCR
text layer on top. Where that upstream OCR gave up, `rare.parse.text` extracts
nothing — not because the extraction is wrong, but because there are no glyphs
under the box to extract. Running headers are a frequent casualty (11% of them
come back empty), and unlike a missing paragraph a missing header is invisible
in the rendered output: the page simply loses its section marker.

This module re-reads those regions from the pixels. It runs *after*
`extract_text_for_page`, and by default only on regions that came back empty,
so a region whose text the PDF already carries is never second-guessed — the
text layer, where it exists, is better than anything re-OCR'ing a 1971 halftone
will give.

That default misses the failures that hurt most. A region the upstream OCR got
*wrong* rather than missed is not empty — the headline JIŘÍ KYLIÁN comes
through as "W Z7" — so no emptiness test will ever reach it. Pass `retry` a set
of `rare.parse.quality` reasons (`junk`, `sparse`, `alien`) and those regions
are re-read too. Replacing existing text is held to a higher standard than
filling a hole: see `fill_failed_regions`.

Two backends are available. Tesseract is the default and the one measured
here; `rare.parse.ocr_ppocr` adds PP-OCRv5, which fails in the opposite
direction on letterspaced display type and can be run alongside Tesseract
through `BestOfOCR`.

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

from rare.parse import quality
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
        min_replace_confidence: float = 60.0,
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
        # Filling an empty region needs `min_confidence`; overwriting text the
        # PDF already carried needs this, deliberately higher.
        self.min_replace_confidence = min_replace_confidence

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


# --- running more than one backend ------------------------------------------

class BestOfOCR:
    """Runs several recognizers on the same crop and keeps the better reading.

    Tesseract and PP-OCR fail in opposite directions on this corpus — see
    `rare.parse.ocr_ppocr` — so on the regions that reach OCR at all, which are
    by definition the ones something already failed on, neither is reliably the
    one to ask. Reading with both costs twice the time per region and nothing
    per page, since the render is shared.

    The choice is made on the text, not on the confidence alone: a reading that
    scores as junk loses to one that does not, however sure the model was.
    Confidence only breaks the tie. The label is not available here, so this
    test is the coarse version of `_is_noise`; the authoritative one runs in
    `fill_failed_regions`, which knows the label and has the final say over
    whether the winner is used at all.
    """

    def __init__(self, recognizers: Sequence):
        if not recognizers:
            raise ValueError("BestOfOCR needs at least one recognizer.")
        self.recognizers = list(recognizers)
        self.name = "+".join(r.name for r in self.recognizers)
        # The page is rendered once, so it has to satisfy the hungriest
        # backend; the gates take the strictest setting asked for.
        self.dpi = max(r.dpi for r in self.recognizers)
        self.padding = max(r.padding for r in self.recognizers)
        self.min_confidence = min(r.min_confidence for r in self.recognizers)
        self.min_replace_confidence = max(
            getattr(r, "min_replace_confidence", REPLACE_MIN_CONFIDENCE)
            for r in self.recognizers
        )

    def recognize(self, crop: Image.Image, *, single_line: bool = False) -> tuple[str, float]:
        best_text, best_confidence, best_rank = "", 0.0, -1.0
        for recognizer in self.recognizers:
            text, confidence = recognizer.recognize(crop, single_line=single_line)
            text = text.strip()
            if not text:
                continue
            clean = (
                quality.junk_ratio(text) < quality.JUNK_RATIO
                and quality.alien_ratio(text) < quality.ALIEN_RATIO
            )
            rank = (1.0 if clean else 0.0) * 1000.0 + confidence
            if rank > best_rank:
                best_text, best_confidence, best_rank = text, confidence, rank
            logger.debug(
                "%s read %r at %.1f%s",
                recognizer.name, text[:60], confidence, "" if clean else " (junk)",
            )
        return best_text, best_confidence


# --- the pipeline hook ------------------------------------------------------

# A reading that *replaces* existing text is held to a higher standard than one
# that fills a hole: the text it overwrites came from the publisher's own OCR
# pass and, wrong as it may be, was produced from the original film rather than
# from a re-render of a re-scan.
REPLACE_MIN_CONFIDENCE = 60.0

# A replacement may not throw away most of the region's content. PP-OCR in
# particular sometimes answers a broken region with a short, clean-looking,
# high-confidence string that has silently dropped half of it — a Subhead
# reading "b e s e d a u r e d n i š t v a" came back as "enista" at 67. Every
# test above this one passes that: it is not junk, and its confidence is fine.
#
# The floor is counted in alphanumeric characters, so the spaces letterspacing
# adds do not inflate the original. 0.4 is where it sits because at 0.5 it
# starts refusing good repairs — "(PRED)USMERJENE STRANI" recovered from
# "ustva.is ajstaraansnCaand) (P R E D )U S M E R" loses 45% of the characters
# and is exactly right. Measured over 51 replacements on four documents.
MIN_REPLACEMENT_ALNUM_RATIO = 0.4


def _alnum_count(text: str) -> int:
    return sum(1 for ch in text if ch.isalnum())


def _is_noise(text: str, label: str) -> bool:
    """Is this reading worse than nothing?

    Only the token-level tests are asked, never `sparse`: a short reading of a
    big box is the *good* case here — one word recovered out of a headline the
    text layer lost entirely — while "V 0 S P R poe J U" is Tesseract finding
    letters in a display face it cannot read. Tesseract reports the latter at
    83% confidence, so confidence alone will not catch it.
    """
    if label in quality.NO_JUNK_CHECK_LABELS:
        return False
    return (
        quality.junk_ratio(text, label) >= quality.JUNK_RATIO
        or quality.alien_ratio(text) >= quality.ALIEN_RATIO
    )


def _better_reading(
    new_text: str,
    old_text: str,
    label: str,
    box: quality.Box,
    page_w: float,
    page_h: float,
) -> bool:
    """Is `new_text` an improvement on the text already there?

    Only asked when the existing text was flagged, and only of readings
    `_is_noise` already passed, so the bar is not "is this good" but "is this
    less bad". A reading that throws away most of the region's characters is
    refused first, whatever else it has going for it — see
    `MIN_REPLACEMENT_ALNUM_RATIO`. Otherwise a reading that scores clean is
    taken. When the new one is also
    flagged — `sparse`, in practice, since junk is gone by here — it wins only
    if it is at least twice as long, which is the case where the upstream layer
    caught one word of a headline and Tesseract caught the line. Anything else
    is churn that would destroy the evidence the region still needs a human.
    """
    old_alnum = _alnum_count(old_text)
    if old_alnum and _alnum_count(new_text) < MIN_REPLACEMENT_ALNUM_RATIO * old_alnum:
        return False

    new_flags, _ = quality.assess(new_text, label, box, page_w, page_h)
    if not new_flags:
        return True
    return len(new_text) >= 2 * len(old_text)


def fill_failed_regions(
    regions: Sequence[dict],
    texts: dict[str, str],
    *,
    recognizer: TesseractOCR,
    pdf_path: str | Path,
    page_no: int,
    page_w: float,
    page_h: float,
    labels: Iterable[str] = DEFAULT_OCR_LABELS,
    retry: Iterable[str] = (),
    page_image: Optional[Image.Image] = None,
    page_image_dpi: Optional[int] = None,
    replace_min_confidence: Optional[float] = None,
) -> int:
    """Re-read regions of `labels` whose text is missing or (with `retry`) bad.
    Mutates `texts`; returns how many regions were rewritten.

    `retry` names `rare.parse.quality` reasons — ``junk``, ``sparse``,
    ``alien``. Empty (the default) means only empty regions are filled, which is
    the behaviour this has always had. Labels outside `labels` are left alone,
    as is anything in NON_TEXT_LABELS.

    A reading below the recognizer's `min_confidence` is discarded, as is one
    that scores as junk whatever its confidence — an empty header is a smaller
    problem than a header filled with noise. Overwriting
    text that already exists additionally needs `replace_min_confidence` and has
    to survive `_better_reading`. `replace_min_confidence` overrides the
    recognizer's own.

    Rewritten regions are marked with `text_source` / `ocr_confidence`, which
    `assemble_page` carries onto the item's provenance so OCR'd text stays
    distinguishable from text the PDF actually carried. A region whose text was
    *replaced* also keeps `text_before_ocr` and the `text_flags` that condemned
    it, so the change can be reviewed after the fact rather than taken on trust.

    The page is rendered lazily and only once, so pages with no gap in them
    cost nothing. A `page_image` the caller already has is reused when it was
    rendered at least as finely as the recognizer asks for.
    """
    wanted = frozenset(labels) - NON_TEXT_LABELS
    retry_reasons = frozenset(retry) - {"empty"}

    gaps: list[tuple[dict, str, list[str]]] = []
    for region in regions:
        label = region.get("label")
        if label not in wanted:
            continue
        current = texts.get(region["region_id"], "").strip()
        if not current:
            gaps.append((region, "", ["empty"]))
        elif retry_reasons:
            flags = quality.is_suspect(
                current, label,
                quality.box_from_norm_1000(region["bbox_norm_1000"], page_w, page_h),
                page_w, page_h,
                reasons=retry_reasons,
            )
            if flags:
                gaps.append((region, current, flags))

    if not gaps:
        return 0

    if page_image is not None and (page_image_dpi or 0) >= recognizer.dpi:
        image = page_image
    else:
        from rare.parse.pdf import render_page
        image = render_page(pdf_path, page_no, dpi=recognizer.dpi)

    filled = 0
    for region, previous, flags in gaps:
        label = region["label"]
        crop = crop_region(image, region["bbox_norm_1000"], recognizer.padding)
        text, confidence = recognizer.recognize(
            crop, single_line=label in SINGLE_LINE_LABELS
        )
        text = text.strip()

        if not text:
            continue

        floor = recognizer.min_confidence if not previous else max(
            recognizer.min_confidence,
            replace_min_confidence if replace_min_confidence is not None
            else getattr(recognizer, "min_replace_confidence", REPLACE_MIN_CONFIDENCE),
        )
        if confidence < floor:
            logger.debug(
                "page %d: dropped %s OCR at confidence %.1f (%r)",
                page_no, label, confidence, text[:60],
            )
            continue

        # Same treatment the pdfplumber path gives: structured regions keep
        # their newlines, prose gets unwrapped and de-hyphenated.
        if label not in STRUCTURED_LABELS:
            text = normalize_text(text)

        if _is_noise(text, label):
            logger.debug(
                "page %d: dropped %s OCR as noise at confidence %.1f (%r)",
                page_no, label, confidence, text[:60],
            )
            continue

        if previous:
            box = quality.box_from_norm_1000(region["bbox_norm_1000"], page_w, page_h)
            if not _better_reading(text, previous, label, box, page_w, page_h):
                logger.debug(
                    "page %d: kept %s text %r over OCR %r",
                    page_no, label, previous[:40], text[:40],
                )
                continue
            region["text_before_ocr"] = previous
            region["text_flags"] = "|".join(flags)

        texts[region["region_id"]] = text
        region["text_source"] = f"ocr:{recognizer.name}"
        region["ocr_confidence"] = confidence
        filled += 1

    if filled:
        logger.info("page %d: re-read %d region(s) by OCR", page_no, filled)
    return filled
