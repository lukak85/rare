"""PP-OCRv5 as a second opinion where Tesseract is not enough.

Tesseract and PP-OCR fail on this corpus in *opposite* directions, which is the
whole reason for having both. On letterspaced display type — the commonest
failure left after `--ocr-retry` — Tesseract explodes the word into single
letters ("0 GL A S N A D E S K A") while PP-OCR collapses it into one
("yemavugankah" for a header reading "Tema v ugankah"). On an outline face
Tesseract reads JIŘÍ KYLIÁN as "JA VLA" at confidence 25; PP-OCR returns
"JIRI KYLIÁN" at 0.91. Neither wins everywhere, so `rare.parse.ocr` can run
both and keep the better reading.

Lifted from `examples/manual/svrt/regions.py`, which is where all of this was
measured and which still imports the two image operations from here.

Two things about this backend differ from the Tesseract one:

* **Lines are cut before recognition, not by the model.** PaddleOCR's
  `TextRecognition` is a *line* recogniser: hand it a three-line headline and
  it returns one garbled line. The obvious cutter is PP-OCR's own
  `TextDetection`, which aborts on this machine's paddle build (3.3.1) with
  "Intel oneMKL function load error" and returns nonsense when salvaged. So
  lines come from `split_lines_by_projection` — the classical horizontal ink
  profile, reliable here because a layout region is already one block of one
  column. Nothing in this module loads a detection model.

* **The recognition model decides which alphabet you get back.**
  `PP-OCRv5_server_rec` is the Chinese/English head and has no č, š or ž in its
  character dictionary — it transliterates them away silently.
  `latin_PP-OCRv5_mobile_rec` is the Latin-script multilingual head and is the
  default here for that reason.

Confidence is reported on Tesseract's 0–100 scale rather than PaddleOCR's 0–1,
so that `--ocr-min-confidence` means one thing whichever backend is running.
The two scales are only roughly comparable — a 0.90 from SVTR is not the same
evidence as a 90 from Tesseract — but both behave like "how sure am I", and the
gates that use them are coarse.

Needs PaddleOCR, which the README recommends installing into its own conda
environment (it clashes with several of the other model extras):

    pip install -e ".[pp-doclayoutv3]"      # paddleocr[all]
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Latin-script multilingual head: the one that has the Slovenian diacritics in
# its character dictionary. See the module docstring.
REC_MODEL = "latin_PP-OCRv5_mobile_rec"

# SVTR's recogniser input height. Crops shorter than this would be upscaled by
# the model anyway; doing it here with a good filter beats its own resize.
REC_INPUT_HEIGHT = 48


def _cv2():
    """Import OpenCV lazily, so importing this module costs nothing."""
    import cv2

    return cv2


# --- image operations -------------------------------------------------------

def split_lines_by_projection(
    image: np.ndarray,
    min_ink_ratio: float = 0.01,
    pad: int = 4,
) -> list[np.ndarray]:
    """Cut a region crop into horizontal line strips, top to bottom.

    A model-free stand-in for `TextDetection`. It exists because PP-OCR's
    detector is both unnecessary and unavailable here: unnecessary because a
    layout region is already one block of one column, so its lines are
    separated by full-width horizontal whitespace; unavailable because
    `PP-OCRv5_mobile_det` aborts on this machine's paddle build with "Intel
    oneMKL function load error".

    The method is the classical horizontal projection profile: count inked
    pixels per row, and every run of rows above the ink threshold is a line. A
    row needs more than `min_ink_ratio` of the width inked to count, which is
    what stops a descender or a speck of scan noise from welding two lines
    together.

    Returns the whole image as a single strip when it finds nothing to cut.
    """
    cv2 = _cv2()
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    per_row = (ink > 0).sum(axis=1)
    threshold = max(1.0, min_ink_ratio * ink.shape[1])
    inked = per_row > threshold

    bands: list[tuple[int, int]] = []
    start: int | None = None
    for y, is_inked in enumerate(inked):
        if is_inked and start is None:
            start = y
        elif not is_inked and start is not None:
            bands.append((start, y))
            start = None
    if start is not None:
        bands.append((start, len(inked)))

    if len(bands) < 2:
        return [image]

    # Rejoin a band that is not a line of its own. Diacritics floating clear
    # above the cap line are the case that matters: JIŘÍ KYLIÁN's carons and
    # acutes form a 62px band over a 260px band of letters, and cut apart they
    # are recognised as their own line ("V") while the letters below lose their
    # accents.
    #
    # The tell is the height disparity, not the gap. Two real lines are set in
    # the same type and come out within a few percent of each other (93 vs 94
    # for a two-line headline, 39–49 down a whole paragraph), whereas an accent
    # strip is a fraction of the line it belongs to. The gap only has to be
    # small enough to rule out a genuinely separate short line further down the
    # region.
    merged: list[tuple[int, int]] = [bands[0]]
    for top, bottom in bands[1:]:
        previous_top, previous_bottom = merged[-1]
        height, previous_height = bottom - top, previous_bottom - previous_top
        tallest = max(height, previous_height)
        if (
            min(height, previous_height) < 0.5 * tallest
            and (top - previous_bottom) < 0.5 * tallest
        ):
            merged[-1] = (previous_top, bottom)
        else:
            merged.append((top, bottom))

    bands = merged
    if len(bands) < 2:
        return [image]

    strips = []
    for top, bottom in bands:
        y0 = max(0, top - pad)
        y1 = min(image.shape[0], bottom + pad)
        strip = image[y0:y1]
        if strip.shape[0] < REC_INPUT_HEIGHT:
            f = REC_INPUT_HEIGHT / strip.shape[0]
            strip = cv2.resize(strip, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        strips.append(strip)
    return strips


def fill_outlines(image: np.ndarray) -> np.ndarray:
    """Solidify hollow (outline) letterforms.

    Otsu-binarise, flood-fill the background inwards from a corner, and treat
    whatever the flood never reached as an enclosed counter to be inked. On a
    filled face this is close to a no-op; on an outline face it turns rings
    into letters.

    Worth having because an outline face is exactly what defeats the text
    layer: only the contour is inked, so every recogniser sees rings. On the
    JIŘÍ KYLIÁN headline it moves PP-OCR from "JIRI KYLIÁN" (0.91) to
    "JİŘI KYLIÁN" (0.88) — recovering the caron at the cost of the I. Neither
    is exact; both are legible. Off by default for that reason.
    """
    cv2 = _cv2()
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    flooded = ink.copy()
    mask = np.zeros((ink.shape[0] + 2, ink.shape[1] + 2), np.uint8)
    cv2.floodFill(flooded, mask, (0, 0), 255)
    solid = ink | cv2.bitwise_not(flooded)

    return cv2.cvtColor(cv2.bitwise_not(solid), cv2.COLOR_GRAY2BGR)


# --- the recognizer ---------------------------------------------------------

class PPOCRRecognizer:
    """Reads one cropped region at a time, to the same contract as
    `rare.parse.ocr.TesseractOCR`.

    Construction loads the model, which takes a few seconds and prints a wall
    of PaddleX logging, so it is done once per run and never per page.
    """

    name = "ppocr"

    def __init__(
        self,
        rec_model: str = REC_MODEL,
        dpi: int = 400,
        padding: int = 10,
        min_confidence: float = 40.0,
        min_replace_confidence: float = 60.0,
        *,
        solidify: bool = False,
        enable_mkldnn: bool = False,
        batch_size: int = 8,
    ):
        try:
            from paddleocr import TextRecognition
        except ImportError as exc:                        # pragma: no cover
            raise RuntimeError(
                "paddleocr is not installed. Install it with "
                '`pip install -e ".[pp-doclayoutv3]"`, ideally into its own '
                "conda environment — it clashes with several of the other "
                "model extras."
            ) from exc

        # mkldnn is off by default because it is what makes PP-OCR's *detection*
        # model abort on this paddle build; recognition is unaffected, but the
        # default stays conservative.
        self._rec = TextRecognition(model_name=rec_model, enable_mkldnn=enable_mkldnn)
        self.rec_model = rec_model
        self.dpi = dpi
        self.padding = padding
        self.min_confidence = min_confidence
        self.min_replace_confidence = min_replace_confidence
        self.solidify = solidify
        self.batch_size = batch_size

    def recognize(
        self, crop: Image.Image, *, single_line: bool = False
    ) -> tuple[str, float]:
        """Read one crop. Returns (text, mean line score scaled to 0–100).

        `single_line` skips the line cutter, which is what a Header or a Byline
        wants: a one-line region put through the projection profile can still
        be split by a descender, and re-joining the halves costs accuracy the
        cut never bought.

        Crops twice as tall as they are wide are retried rotated, for the same
        reason the Tesseract path does it: captions and credits set up the side
        of a photograph are common here, both directions occur, and which way
        this one turns cannot be read off the geometry.
        """
        cv2 = _cv2()
        if crop.width < 2 or crop.height < 2:
            return "", 0.0

        image = cv2.cvtColor(np.array(crop.convert("RGB")), cv2.COLOR_RGB2BGR)
        if self.solidify:
            image = fill_outlines(image)

        if image.shape[0] >= 2 * image.shape[1]:
            best = ("", -1.0)
            for angle in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
                # Rotated type is a single line, whatever the label implies.
                candidate = self._recognize([cv2.rotate(image, angle)])
                if candidate[1] > best[1]:
                    best = candidate
            return best if best[1] >= 0 else ("", 0.0)

        lines = [image] if single_line else split_lines_by_projection(image)
        return self._recognize(lines)

    def _recognize(self, lines: list[np.ndarray]) -> tuple[str, float]:
        """Recognise already-cut line strips and join them top to bottom.

        Lines are joined with ``\\n`` rather than spaces, matching the
        pdfplumber and Tesseract paths, so that `normalize_text` can
        de-hyphenate across the line break instead of finding a welded word.
        """
        if not lines:
            return "", 0.0

        try:
            results = list(self._rec.predict(lines, batch_size=self.batch_size))
        except Exception:                                 # noqa: BLE001
            logger.exception("PP-OCR failed on %d line strip(s)", len(lines))
            return "", 0.0

        per_line = [(r["rec_text"], float(r["rec_score"])) for r in results]
        text = "\n".join(t for t, _ in per_line if t.strip())
        mean = sum(s for _, s in per_line) / len(per_line) if per_line else 0.0
        # PaddleOCR scores 0–1; the pipeline's thresholds are Tesseract's 0–100.
        return text, mean * 100.0


def make_recognizer(
    rec_model: str = REC_MODEL,
    dpi: int = 400,
    min_confidence: float = 40.0,
    min_replace_confidence: float = 60.0,
    *,
    solidify: bool = False,
) -> PPOCRRecognizer:
    """Constructor used by the CLI. Kept separate so `rare.parse.ocr` can name
    this backend without importing paddle at module load."""
    return PPOCRRecognizer(
        rec_model=rec_model,
        dpi=dpi,
        min_confidence=min_confidence,
        min_replace_confidence=min_replace_confidence,
        solidify=solidify,
    )
