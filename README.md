# Razčlenjevalnik Revij (RaRe)

RaRe is a parsing toolkit for Slovene magazines (primarily *Glasbena Mladina*), built on top of a fork of [layoutparser](https://github.com/Layout-Parser/layout-parser).
It exposes two tracks for parsing PDFs and comparing models on annotated data:

- **Pipeline track** — DLA model → reading-order → assembled `GlasanaDocument` → HTML / Markdown / JSON.
- **VLM track** — vision-language model (cloud or locally-served) producing HTML / Markdown / JSON directly.

Pipelines track (at the moment) assumes presence of previously OCR-ed PDFs.

## Installation

Before running either of these, check [Additional model-specific setup](#additional-model-specific-setup) below for
additional per-model requirements. Then run (if the code below doesn't include the desired model, no additional
dependencies are needed):

```bash
pip install -e .                   # core package + 'rare' command

# Model specific dependencies
pip install -e ".[detr]" # + DETR dependencies
pip install -e ".[dit]" # + DiT dependencies
pip install -e ".[doclayout-yolo]" # + DocLayout-YOLO dependencies
pip install -e ".[faster-rcnn]" # + Faster R-CNN dependencies
pip install -e ".[layoutlmv3]" # + LayoutLMv3 dependencies 
pip install -e ".[mask-rcnn]" # + Mask R-CNN dependencies
pip install -e ".[pp-doclayoutv3]" --extra-index-url https://download.pytorch.org/whl/cpu # + PP-DocLayoutV3 dependencies
pip install -e ".[rf-detr]" # + RF-DETR dependencies
pip install -e ".[vgt]" # + VGT dependencies

pip install -e ".[marker]" # + Marker dependencies
```

Furthermore install LayoutParser fork:

```bash
pip install -e layout-parser
```

It is recommended to create a separate Conda environment for each of the intended models in order to avoid library and
version clashes between different model's dependencies.

## Usage

The single `rare` command exposes three subcommands.

### `rare parse` — parse a PDF

```bash
# Pipeline track
rare parse <pdf> --layout doclayout-yolo --order top-bottom

# VLM track (mutually exclusive with --layout)
rare parse <pdf> --vlm claude

# Pipeline track, with per-page Markdown for OmniDocBench's end2end evaluator
rare parse <pdf> --layout doclayout-yolo --emit-omnidocbench

# Choose the NER backend used by the linking stage
rare parse <pdf> --layout doclayout-yolo --ner rudar-slv

# Re-read regions the PDF's text layer left empty (running headers, by default)
rare parse <pdf> --layout doclayout-yolo --ocr tesseract

# Discover backends
rare parse --list-models
```

Outputs are stored in `outputs/parsed/<pdf_stem>/{<stem>.html, <stem>.md, <stem>_doc.json, <stem>_articles.json, <stem>_articles.md, figures/}`.

#### Linking

After a document is assembled, a whole-document pass fills in the relationships a single page cannot show: named
entities on every text region, captions bound to their figure (or, when there is no figure, to the closest article),
articles made complete and ordered, and pieces continuing across a page break merged into one article. Every inference
is also recorded in `doc.links` with the method, score and evidence behind it.

`--ner rudar-slv` needs the NER extra (`pip install -e ".[ner]"`).

#### OCR fallback for failed regions (`--ocr`)

The corpus PDFs are scans: one full-page image per page with an invisible OCR text layer over it. Where that upstream
OCR gave up, per-region extraction yields nothing — there are no glyphs under the box to extract. Running headers are
the frequent casualty.

`--ocr tesseract` re-reads those regions from the pixels, cropping them out of the page re-rendered at
`--ocr-dpi` (400 by default — 200 is marginal for Tesseract on these scans). By default it runs **only** on regions
that came back empty, so text the PDF actually carries is never second-guessed, and only on the labels named by
`--ocr-labels` (default: `Header`). Figures are never OCR'd whatever the label set says. Readings below
`--ocr-min-confidence`, or that score as junk at any confidence, are discarded — an empty header is a smaller problem
than a header full of noise.

Filled regions carry `provenance.text_source: "ocr:tesseract"` and `provenance.ocr_confidence` in the output JSON, so
OCR'd text stays distinguishable from text the PDF carried. Everything else keeps `"pdf"`.

##### Text that is present but wrong (`--ocr-retry`)

Emptiness is the easy failure. The expensive one is a region the upstream OCR got *wrong* rather than missed: the
headline JIŘÍ KYLIÁN arrives as `W Z7`, which is not empty and so is never re-read. `rare.parse.quality` scores a
region's text against its box and its label and names what is wrong with it:

| reason   | what it catches                                                                                                                                                                                           |
|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `junk`   | most tokens are not words — no vowel, letters and digits mixed, or a run of bare single letters where letterspaced display type was read glyph by glyph (`0 GL A S N A D E S K A`). `W Z7` scores 2 of 2. |
| `sparse` | far less text than a box that shape holds, measured by aspect ratio for one-line labels so type size drops out                                                                                            |
| `alien`  | too many characters from outside the Slovene alphabet — the `□` class of failure                                                                                                                          |
| `empty`  | empty region                                                                                                                                                                                              |

`--ocr-retry` (bare, or with a subset like `--ocr-retry junk,alien`) sends those regions for a second reading too.
Overwriting text is held to a higher standard than filling a hole, because the text being overwritten came from the
publisher's own pass over the original film:

* the reading needs `--ocr-min-replace-confidence` (60) rather than `--ocr-min-confidence` (40);
* a reading that is itself junk is refused, so `W Z7` is never traded for `VV Z]` — and the region stays visibly broken
  for a human instead of looking repaired;
* a reading that throws away most of the region's letters is refused whatever its confidence — PP-OCR in particular
  answers a broken region with a short, clean, confident string often enough to matter (`b e s e d a u r e d n i š t v a`
  came back as `enista` at 67). The floor counts letters rather than characters, so the spaces letterspacing adds do
  not make a genuine repair look like a loss;
* when both readings score badly the new one wins only if it is at least twice as long, which is the case where the
  text layer caught one word of a headline and Tesseract caught the line.

A region whose text was replaced keeps `provenance.text_before_ocr` and `provenance.text_flags` alongside the usual
markers, so every replacement can be reviewed after the run rather than taken on trust:

```bash
jq -r '.items[].provenance | select(.text_before_ocr) | "\(.text_flags)\t\(.text_before_ocr)"' \
    outputs/parsed/<stem>/<stem>_doc.json
```

`examples/manual/audit/run.py` scores an OmniDocBench export with the same module, listing every failed region as a CSV
with a blank `corrected_text` column — for the ones OCR cannot rescue and a person has to type.

Needs the binary and the Slovenian language data, which is a separate package:

```bash
sudo apt install tesseract-ocr tesseract-ocr-slv
```

Parsing fails immediately if the requested `--ocr-lang` is not installed, rather than falling back to English and
filling the document with plausible-looking text whose diacritics are wrong.

##### A second opinion (`--ocr ppocr`, `--ocr tesseract,ppocr`)

Tesseract is not always enough. The two engines available here fail in *opposite* directions on the failure that is
left once the empty regions are handled — letterspaced display type. Tesseract explodes the word into single letters
(`0 GL A S N A D E S K A`); PP-OCRv5 collapses it into one (`yemavugankah` for a header reading "Tema v ugankah"). On
an outline face Tesseract reads JIŘÍ KYLIÁN as `JA VLA` at confidence 25, while PP-OCR returns `JIRI KYLIÁN` at 0.91.

`--ocr ppocr` swaps the backend; `--ocr tesseract,ppocr` reads every region with both and keeps the better answer — a
reading that scores as junk loses to one that does not, and confidence only breaks the tie. That costs twice the
recognition time per region and nothing per page, since the render is shared. The final say still belongs to the same
gates: a winner that is junk for its label, or under the confidence floor, is discarded like any other reading.

The PP-OCR path comes from `examples/manual/svrt/regions.py`, which stays the place to look at one region at a time and
see why a reading came out the way it did. Two things it taught, both now baked in:

* **Lines are cut before recognition, not by a model.** PaddleOCR's `TextRecognition` is a *line* recogniser — hand it
  a three-line headline and it returns one garbled line. Its own `TextDetection` is the obvious cutter and aborts on
  this paddle build (3.3.1) with `Intel oneMKL function load error`, so lines come from a horizontal ink profile
  instead. That is reliable here because a layout region is already one block of one column. No detection model is
  ever loaded.
* **`--ocr-rec-model` decides which alphabet you get back.** The default `latin_PP-OCRv5_mobile_rec` is the
  Latin-script multilingual head. The Chinese/English heads have no č, š or ž in their character dictionary and
  transliterate them away without saying so.

`--ocr-fill-outlines` solidifies hollow letterforms before recognition — flood-fill the background and ink whatever it
never reached. On the outline headline above it recovers the caron at the cost of the I (`JİŘI KYLIÁN`, 0.88). Off by
default, since it trades one error for another on filled type.

PP-OCR confidences are reported on Tesseract's 0–100 scale so `--ocr-min-confidence` means one thing either way. Needs
PaddleOCR, which is best installed into its own conda environment — it clashes with several of the other model extras:

```bash
pip install -e ".[pp-doclayoutv3]"      # paddleocr[all]
```

##### Measuring what it bought (`rare evaluate --ocr`)

The same flags exist on `rare evaluate --track pipeline`, so a configuration can be evaluated exactly as it is parsed.
There they apply to the **predicted** text only. The ground truth keeps whatever text the corpus carries — one text
source feeding both sides would compare OCR against itself, and `text_block` Edit_dist would barely move. If the ground
truth should carry corrected text too, hand it in with `--omnidocbench-ground`; `examples/manual/audit/run.py --apply`
writes exactly that file.

Run the same models twice, **into two separate run directories**, and compare the reports:

```bash
rare evaluate --track pipeline --dataset glasbena_mladina --layout doclayout-yolo \
    --run-omnidocbench --omnidocbench-eval end2end --run-id base
rare evaluate --track pipeline --dataset glasbena_mladina --layout doclayout-yolo \
    --run-omnidocbench --omnidocbench-eval end2end --run-id base-ocr \
    --ocr tesseract,ppocr --ocr-retry
```

Reusing one `--run-id` accumulates *different* models into one report, and these two runs are the same model — same
`--layout`, same `--order`, hence the same `<layout>__<order>` name. The second would overwrite the first's
`per_model/<model>.json` and its `markdown_pred_<model>/` pages rather than sit beside them.

The OCR pass feeds the end2end scoring, so it needs `--run-omnidocbench` with `--omnidocbench-eval end2end` (or
`both`) — the detection pass scores boxes and is unaffected by text. Each run prints how many predicted regions were
rewritten and records it as `ocr_regions_filled` in the per-model results. The per-page prediction markdown under
`<run>/omnidocbench/markdown_pred_<model>/` is written before the container starts, so the two runs can be diffed
directly even without Docker.

**Prefer `--ocr-retry junk,alien` when the boxes come from a detector.** `sparse` asks whether a box holds less text
than a box that shape usually does, and its per-label medians are measured on the hand-drawn ground truth, whose boxes
are tight. A detector's box is not, so a short but perfectly correct line inside a generous box reads as sparse: on one
28-page document it re-read `Ime in priimek reševalca` and replaced it with `resšea.l.ca..........`. Dropping `sparse`
removed that and cost nothing else. It stays worth having on `--coco` runs, where the boxes are the annotated ones.

On the pipeline track, `--emit-omnidocbench` additionally writes one Markdown file per page to `outputs/parsed/<pdf_stem>/omnidocbench/<stem>_<page>.md` — the flat `<image_stem>.md` layout OmniDocBench's end-to-end evaluator mounts at `data_md/predictions`. These pages are rendered from the regions **as the DLA model detected them**, before the heuristic pass that re-joins paragraphs split across columns or pages, so the score reflects the model's own segmentation. The regular `<stem>.md` (and `--per-page` output under `pages/`) stay merged.

### `rare evaluate` — score one model against a dataset

```bash
# Pipeline track — layout mAP + reading-order Kendall tau
rare evaluate --track pipeline --dataset glasbena_mladina \
    --layout doclayout-yolo --order top-bottom \
    [--run-id myrun-2026-05] [--limit 5]

# VLM track — F1 + edit-distance ratio against gold markdown
rare evaluate --track vlm --dataset glasbena_mladina \
    --vlm claude \
    [--pdfs-dir dataset/pdfs] [--run-id myrun-2026-05]
```

Each invocation runs **one model**. Re-invoke with the same `--run-id` to accumulate models; `report.md` regenerates from every per-model JSON in the run directory.

Outputs are stored in `outputs/evaluations/<run_id>/{report.md, scores.csv, per_model/}`.

#### Figure/caption → article attachment (`--track figure-link`)

There's two concerns with images in reading order:
1. The models tested other than LayotutReader perform worse than heuristic on our dataset, and LayoutReader works with
text lines, so providing it with figures would break the result.
2. Figures can mostly be viewed in any order (at the start). Only their affiliation to the article is important. 

Linking is done using 3 heuristics:
1. Figure + caption's proximity to the article.
2. NER similarity between caption and article.
3. Position of the figure relative to the article and its header (their preferred position is to the upper-left direction).

Evaluation of the resulting links:
```bash
# Linking alone — documents are rebuilt from the ground-truth boxes and order,
# so detection and reading order are perfect and only rare.link is measured.
rare evaluate --track figure-link --dataset glasbena_mladina \
    --pdfs-dir datasets/glasbena_mladina/pdfs/eval
```

#### Page type vs article genre (`--track page-genre`)

Compares annotation of `page_type` to the `genre` determined by the classifier. Currently supported types (in alignment
with selected page types):

| Page type          | Expected genre              |
|--------------------|-----------------------------|
| `ArticlePage`      | `članek`                    |
| `NewsPage`         | `novice`                    |
| `InterviewPage`    | `intervju`                  |
| `RecordsPage`      | `recenzija`                 |
| `LettersPage`      | `pisma`                     |
| `QuizPage`         | `kviz`                      |
| `EventsPage`       | `dogodki`                   |
| `ImagesPage`       | `slike`                     |
| `TOCPage`          | `kazalo`                    |
| `AdvertPage`       | `reklama`                   |
| `CoverPage`        | `naslovnica`                |
| `SpecialPage`      | *not scored* (manual check) |
| `FrontPage`        | *not scored*                |
| `BackPage`         | *not scored*                |

```bash
# Score a real parse; genres come from whatever --classification produced.
rare evaluate --track page-genre --classification gams \
    --dataset glasbena_mladina \
    --pdfs-dir datasets/glasbena_mladina/pdfs/eval
```

Because a page can holds pieces of several genres, result is reported from three angles:
1. `accuracy_dominant` (the  article holding most of the page has the expected genre, the way the pages were annotated
in the first place), `accuracy_any` (some article on the page does) and
2. `article_accuracy` over every (page, article) pair — the truth for a mixed page is between the first two.
3. `genre_coverage` is the share of scored pages carrying any genre at all; a low headline accuracy usually means
articles went unclassified, not that they were misclassified.

`page_genre_summary.json` holds the **confusion matrix** of page type against the predicted genres.

#### OmniDocBench Layout detection metrics (`--run-omnidocbench`)

The pipeline track can run [OmniDocBench](https://github.com/opendatalab/OmniDocBench)'s layout evaluator, including **mAP**. Pass `--run-omnidocbench`; this runs the pinned OmniDocBench Docker image against the artifacts emitted under `outputs/evaluations/<run_id>/omnidocbench/` (so **Docker must be installed**). Use `--omnidocbench-image` to override the image.

Before running, clone the [OmniDocBench](https://github.com/opendatalab/OmniDocBench) repository:
```bash
git clone https://github.com/opendatalab/OmniDocBench.git
```
Switch to `v1_5` branch:
```bash
git switch v1_5
```
Copy the Dockerfile from [OmnoDocBench-Dockerfile](./OmniDocBench-Dockerfile) to the root of the cloned repository and build the Docker image:
```bash
docker build -t omnidocbench-v15 .
```

Then run:
```bash
# Pipeline track — implies --emit-omnidocbench. With --pdfs-dir, ground-truth
# region text is filled from the PDF (real text Edit distance); without it,
# stub tokens are used and only reading-order box placement is measured.
rare evaluate --track pipeline --dataset glasbena_mladina \
    --layout doclayout-yolo --order top-bottom \
    --run-omnidocbench --pdfs-dir datasets/glasbena_mladina/pdfs
```

#### OmniDocBench Edit distance (`--run-omnidocbench`)

Both tracks can run [OmniDocBench](https://github.com/opendatalab/OmniDocBench)'s end-to-end evaluator and fold the `text_block` and `reading_order` **Edit distance** into `report.md`. Pass `--run-omnidocbench`; this runs the pinned OmniDocBench Docker image against the artifacts emitted under `outputs/evaluations/<run_id>/omnidocbench/` (so **Docker must be installed**). Use `--omnidocbench-image` to override the image.

On the pipeline track this runs *two* container passes, selectable with `--omnidocbench-eval`:

| Pass | Metrics | Prefix in `report.md` |
| --- | --- | --- |
| `detection` | COCODet mAP/AP over the predicted boxes | `bbox_` |
| `end2end` | `text_block` / `reading_order` Edit distance over per-page Markdown | `odb_` |
| `both` (default) | both of the above | — |

The two passes ship in different images, so they are configured separately: `--omnidocbench-image` for end2end, `--omnidocbench-layout-image` for detection.

The `end2end` pass renders each page's detected regions through the *same* `to_markdown` renderer the VLM track is scored with.

Before running, pull the following image:
```bash
docker pull ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204
```

Then run:
```bash
# Pipeline track — implies --emit-omnidocbench. With --pdfs-dir, ground-truth
# region text is filled from the PDF (real text Edit distance); without it,
# stub tokens are used and only reading-order box placement is measured.
rare evaluate --track pipeline --dataset glasbena_mladina \
    --layout doclayout-yolo --order top-bottom \
    --run-omnidocbench --pdfs-dir datasets/glasbena_mladina/pdfs

# VLM track — REQUIRES --pdfs-dir. The VLM emits real OCR text, so the ground
# truth must also carry real text (extracted from the PDF); without a resolvable
# PDF directory the container step is skipped with a warning.
rare evaluate --track vlm --dataset glasbena_mladina \
    --vlm dots-ocr \
    --run-omnidocbench --pdfs-dir datasets/glasbena_mladina/pdfs
```

Results land in `omnidocbench/results_<model>/` and surface as `odb_text_block_edit` / `odb_reading_order_edit` columns in `report.md`. The container scores `text_block` and `reading_order` only; the formula CDM metric is intentionally omitted (irrelevant for formula-free magazines and it needs the heavy in-container LaTeX stack). Lower Edit distance is better.

> **Note on coverage:** ground truth covers the whole dataset while predictions cover only the samples you ran, so combining `--run-omnidocbench` with `--limit` leaves unmatched GT pages that score the maximum Edit distance of 1.0. Run the full set for headline numbers.

### `rare tools` — annotation utilities

```bash
rare tools -m count-annotations -a dataset/annotations.json
rare tools -m join-annotations -p results/<doc_hash>/ -o merged.json
rare tools -m prepare-annotations -a merged.json -o cleaned.json
rare tools -m review-annotations -a cleaned.json -s reviewed/
```

`rare tools -h` prints the full flag list (same as the old `helper.py`).

## Supported Models

The supported models (and therefore given Python version recommendations) were tested using:
- Ubuntu 24.04
- CUDA 12.8

### Pipeline track — layout backends

| Model                                                                                                       | CLI name         | Type                | Recommended Python version |
|-------------------------------------------------------------------------------------------------------------|------------------|---------------------|----------------------------|
| **[DETR](https://huggingface.co/cmarkea/detr-layout-detection)**                                            | `detr`           | Vision transformers | 3.14.3                     |
| **[DiT](https://github.com/microsoft/unilm/tree/master/dit)**                                               | `dit`            | Vision transformers | 3.8                        |
| **[DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)**                                         | `doclayout-yolo` | Object detection    | 3.10                       |
| **Faster R-CNN***                                                                                           | `faster-rcnn`    | CNN-based           | 3.12                       |
| **[LayoutLMv3](https://github.com/microsoft/unilm/tree/master/layoutlmv3)**                                 | `layoutlmv3`     | Multimodal          | 3.7                        |
| **Mask R-CNN***                                                                                             | `mask-rcnn`      | CNN-based           | 3.12                       |
| **[Nemotron-Page-Elements-v3](https://huggingface.co/nvidia/nemotron-page-elements-v3)**                    | `nemotron`       | Object detection    | 3.14                       |
| **[PP-DocLayoutV3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3)**                                    | `pp-doclayoutv3` | Vision transformers | 3.12                       |
| **[RF-DETR](https://huggingface.co/neka-nat/rfdetr-doclayout)**                                             | `rf-detr`        | Vision transformers | 3.14                       |
| **[SwinDocSegmenter](https://github.com/ayanban011/SwinDocSegmenter)**                                      | `swindocseg`     | Vision transformers | 3.8                        |
| **[VGT](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/DocumentUnderstanding/VGT)** | `vgt`            | Multimodal          | 3.8                        |

\* Included in LayoutParser with detectron2

### Pipeline track — reading-order backends

| Model                                                                    | CLI name         | Type                | Recommended Python version |
|--------------------------------------------------------------------------|------------------|---------------------|----------------------------|
| Top-bottom                                                               | _Default_        | Rule based          | Any                        |
| Left-right                                                               | `left-right`     | Rule based          | Any                        |
| **[PaddleX's Improved XY-Cut](https://github.com/PaddlePaddle/PaddleX)** | `paddlex-xy-cut` | Rule based          | Any                        |
| **[LayoutReader](https://github.com/FreeOCR-AI/layoutreader)**           | `layoutreader`   | Vision transformers | Any                        |

### VLM track

| Model                                                                               | CLI name      | Type               | Recommended Python version |
|-------------------------------------------------------------------------------------|---------------|--------------------|----------------------------|
| **[DeepSeek-OCR-2](https://github.com/deepseek-ai/DeepSeek-OCR-2)**                 | `deepseekocr` | Specialized VLMs   | 3.12.9                     |
| **[Docling](https://github.com/docling-project/docling)**                           | `docling`     | Specialized VLMs   | 3.14                       |
| **[Dolphin](https://github.com/bytedance/Dolphin)**                                 | `dolphin`     | Specialized VLMs   | 3.13                       |
| **[dots.ocr](https://github.com/rednote-hilab/dots.ocr)**                           | `dots-ocr`    | Specialized VLMs   | 3.12                       |
| **[GLM-OCR](https://github.com/zai-org/GLM-OCR)**                                   | `glm-ocr`     | Specialized VLMs   | 3.13                       |
| **[Marker](https://github.com/datalab-to/marker)**                                  | `marker`      | Specialized VLMs   | 3.10                       |
| **[MinerU](https://github.com/opendatalab/mineru)**                                 | `mineru`      | Specialized VLMs   | 3.13                       |
| **[Nemotron-Parse-v1.2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.2)** | `nemotron`    | Specialized VLMs   | 3.13                       |
| **[PaddleOCR](https://github.com/PADDLEPADDLE/PADDLEOCR)**                          | `paddleocr`   | Specialized VLMs   | 3.12                       |
| **[Qwen3-VL](https://huggingface.co/collections/Qwen/qwen3-vl)**                    | `qwen`        | Local general VLMs | 3.12                       |
| **[Youtu-Parsing](https://github.com/PADDLEPADDLE/PADDLEOCR)**                      | `youtu`       | Specialized VLMs   | 3.10                       |

## Outputs

`outputs/parsed/<pdf_stem>/<stem>.json` is a `GlasanaDocument`:

```json
{
  "source_pdf": "ac30fbcf...",
  "pages":     {"0": {"page_no": 0, "width": ..., "height": ...}, ...},
  "items":     {"<uuid>": {"category": "Headline", "text": "...", "provenance": {...},
                           "entities": [{"text": "Mateja Haller", "label": "PER", "key": "matej haller"}]}, ...},
  "body_order": ["<uuid>", ...],
  "articles":  {"<uuid>": {"title": "...", "item_ids": [...], "page_nos": [3, 4],
                           "section": "ODMEVI", "entity_keys": [...], "continued": true}},
  "links":     [{"kind": "caption-of", "from_id": "<uuid>", "to_id": "<uuid>",
                 "method": "geometry", "score": 0.94, "evidence": []}]
}
```

`outputs/parsed/<pdf_stem>/<stem>_articles.json` is the denormalised counterpart — one entry per article with its items inlined in reading order, ready to render without joining `items` against `body_order`. `<stem>_articles.md` is the same grouping as Markdown.

`outputs/evaluations/<run_id>/report.md` is a Markdown table — one row per model, one column per metric:

```
| Model                      | map    | map_50 | kendall_tau |
|---|---|---|---|
| doclayout-yolo__top-bottom | 0.6231 | 0.8104 | 0.7402      |
| rf-detr__top-bottom        | 0.5984 | 0.7891 | 0.6951      |
```

## Project Structure

```
rare/                         # installable package — entry point: rare = "rare.cli:main"
├── cli.py                    # rare parse | evaluate | tools
├── doc/{schema,renderers}.py # GlasanaDocument + 43 region classes + HTML/MD renderers
├── models/
│   ├── base.py               # LayoutBackend / ReadingOrderBackend / VLMBackend protocols
│   ├── registry.py           # lazy registry; sets LAYOUTPARSER_BACKEND env var
│   ├── layout/               # layout detection model/method classes
│   ├── order/                # order detection model/method classes
│   └── vlm/                  # visual language model document parsing classes
├── parse/                    # PDF → pages → layout → order → text → GlasanaDocument
├── evaluate/                 # dataset loaders + pipeline/VLM/figure-link metrics + runner + report
├── tools/_helper.py          # annotation utilities
└── utils/                    # eval / display / file / conversion / character helpers
configs/                      # JSON configs per model
data/                         # default path for model weights and model files
datasets/                     # default path for datasets
layout-parser/                # git submodule (layoutparser fork)
outputs/                      # outputs/parsed/* + outputs/evaluations/*
```

## Ground markdown

_TODO (VLM track)_

## Additional model-specific setup

<details>
<summary><b>Additional model-specific setup</b></summary>

### DiT

The installation roughly follows that of [DiT install notes](https://github.com/microsoft/unilm/tree/master/dit#setup).
Install Pytorch via:
```bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

Due to Detectron2 backbone, install it via:
```bash
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
```

### LayoutLMv3

The installation rougly follows that of [LayoutLMv3 install notes](https://github.com/microsoft/unilm/tree/master/layoutlmv3#installation).
Install Pytorch via:
```bash
pip install torch==1.10.0+cu111 torchvision==0.11.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

Due to Detectron2 backbone, install it via:
```bash
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
```

Check Pytorch version. If not 1.10.0+cu111, run the Pytorch installation command again.

Then clone the [unilm repository](https://github.com/microsoft/unilm/tree/master):
```bash
git clone https://github.com/microsoft/unilm.git
```

And inside [/unilm/layoutlmv3](/unilm/layoutlmv3) run:
```bash
pip install -e .
```

Inside [configs/layoutlmv3/yaml](./configs/layoutlmv3/yaml) place [cascade_layoutlmv3.yaml](https://github.com/microsoft/unilm/blob/c45389eda88e14c57de2c07472e3f49383a6dab0/layoutlmv3/examples/object_detection/cascade_layoutlmv3.yaml),
and change WEIGHTS path to the path with weights on your system.

### PP-DocLayoutV3

Based on your CUDA version, use the fitting command from [PaddlePaddle install page](https://www.paddlepaddle.org.cn/en/install)
to install `paddlepaddle`. For example, given CUDA version 12.8:

```bash
python -m pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

If NVCC is not is not available, it must be installed. For example, given CUDA version 12.8:

```bash
conda install nvidia::cuda-nvcc==12.8.93
```

### RF-DETR

Given CUDA version 12.8, install torch via:

```bash
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

### VGT

The installation instructions largely follow [VGT install notes](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/DocumentUnderstanding/VGT#install-requirements).

After installing RaRe and VGT dependencies, install Pytorch:
````bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
````

Also install `detectron2`:
```bash
python -m pip install detectron2==0.6 -f  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
```

This method requires `.pkl` grid file for each input image. Therefore before running, generate `pkl` grid information by
running `create_grid_input.py` from [VGT's Generating grid information](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/DocumentUnderstanding/VGT#generating-grid-information)
section. First, an installation of transformers is needed:
```bash
pip install transformers
```

And then run (for each PDF):
```bash
python create_grid_input.py \
--pdf 'path-to-pdf-file' \
--output 'path-to-output-folder' \
--tokenizer 'google-bert/bert-base-uncased' \
--model 'doclaynet'
```
Then point `rare parse` or `rare evaluate` at it via `--config {"grid_root": "<path>"}`.

---

### DeepSeek-OCR-2

Follow the installation instructions on [DeepSeek-OCR-2 GitHub repository](https://github.com/deepseek-ai/DeepSeek-OCR-2).

### Docling

Follow the [installation instructions](https://www.docling.ai/). If you have a NVIDIA GPU with CUDA version 12.8, run:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### dots.ocr

Install the appropriate Pytorch version according to your CUDA version. E.g., for CUDA 12.8:
```bash
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

For faster inference, also install `flash-attn` and its required packages:
```bash
pip install psutil
pip install flash-attn --no-build-isolation
```

Then clone the [dots.ocr](https://github.com/rednote-hilab/dots.ocr) repository and insall:
````bash
pip install -e .
````

Then install vLLM using `uv`:
```bash
pip install uv # If uv is not previously installed
uv pip install vllm --torch-backend=cu128
```

If using GPU, vLLM also requires `nvcc`. If not available, install via:
````bash
conda install nvidia::cuda-nvcc==12.8.93
````

And run the vLLM server:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve rednote-hilab/dots.mocr --tensor-parallel-size 1 --gpu-memory-utilization 0.9 --chat-template-content-format string --served-model-name model --trust-remote-code
```

<details>
<summary><b>Additional troubleshooting</b></summary>
If your Pytorch and driver CUDA version are mismatched, try installing Pytorch using the commands above again.

If needed, use a vLLM version below `0.20`:
```bash
pip install "vllm<0.20"
```
</details>


### GLM-OCR

Follow the [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR#vllm) installation instructions. 

Additionally, install `zai-sdk` for evaluation using OmniDocBench:
```bash
pip install zai-sdk
```

If using GPU, vLLM also requires `nvcc`. If not available, install via:
````bash
conda install nvidia::cuda-nvcc==12.8.93
````

Then run a vLLM server:
```bash
vllm serve zai-org/GLM-OCR --allowed-local-media-path / --port 8080
```

<details>
<summary><b>Additional troubleshooting</b></summary>
Install an appropriate Pytorch version, if there is a mismatch, e.g. for CUDA 12.8:

```bash
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

And reinstall vllm:
```bash
pip uninstall vllm
pip install "vllm<0.20"
```
</details>


### Marker

To enable GPU inference, use torch built with CUDA. Given GPU with CUDA 12.8:
```bash
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```


### MinerU

As per MinerU [installation instructions](https://github.com/opendatalab/mineru#install-mineru), run the following commands:
```bash
pip install --upgrade pip
pip install uv
uv pip install -U "mineru[all]"
```

Install CUDA driver compatible Pytorch version. If you have a NVIDIA GPU with CUDA version 12.8, run:
```bash
pip install torch==2.9.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Install `vllm` module for inference acceleration:
```bash
uv pip install "mineru[core,vllm]"
```

<details>
<summary><b>Additional troubleshooting</b></summary>
In our tests, we encountered the following errors, and fixed them the following ways:

- `RuntimeError: flashinfer-cubin version (0.6.8.post1) does not match flashinfer version (0.5.3). Please install the same version of both packages. Set FLASHINFER_DISABLE_VERSION_CHECK=1 to bypass this check.`
    ```bash
    pip install -U "flashinfer-python==0.6.8" "flashinfer-cubin==0.6.8"
    ```

- `Permission denied: 'nvcc'`
    ```bash
    conda install nvidia::cuda-nvcc==12.8.93
    ```
</details>

### Nemotron-Parse

Following the [NVIDIA-Nemotron-Parse-v1.2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.2) quick start, in
addition to `transformers`, install:
```bash
pip install albumentations timm open_clip_torch
```

After running the following download script:
```bash
hf download nvidia/NVIDIA-Nemotron-Parse-v1.2 chat_template.jinja --local-dir . 
```

Run via vLLM:
```bash
vllm serve nvidia/NVIDIA-Nemotron-Parse-v1.2 \
    --dtype bfloat16 \
    --max-num-seqs 8 \
    --limit-mm-per-prompt '{"image": 1}' \
    --trust-remote-code \
    --port 8000 \
    --chat-template chat_template.jinja
```


### PaddleOCR

As per instructions, given CUDA 12.8, install:
```bash
 python -m pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

And then:
```bash
python -m pip install "paddleocr[all]"
```

## Youtu

Follow the instructions on [Youtu's GitHub repository](https://github.com/TencentCloudADP/youtu-parsing).

---

### Qwen3-VL

Follow the instructions of model of choice on [Qwen3-VL's Hugging Face repository](https://huggingface.co/collections/Qwen/qwen3-vl).

Our tests were done with the following model and command:
```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --gpu-memory-utilization 0.95 \
  --max-model-len 12288 \ 
  --max-num-seqs 2 \
  --limit-mm-per-prompt '{"image": 1}' \
  --mm-processor-kwargs '{"max_pixels": 1003520}'
```

</details>

Note: for Claude, ChatGPT and Gemini, the user must have an account and API key, which is pasted into the config file,
which must then be passed as a parameter. The examples for each of them are present in ther respective
[configs](configs) directories.

## Evaluation

Two approaches to evaluation are present:
- manual (hand written functions for computation of mAP, normalized edit distance within the project)
- using [OmniDocBench](https://github.com/opendatalab/OmniDocBench) — run automatically as part of `rare evaluate` via `--run-omnidocbench` (see [the usage section](#omnidocbench-edit-distance---run-omnidocbench))

Current results are temporary and subject to change with further testing.

# OmniDocBench Evaluation Results

The following results were obtained by evaluating detections made by the following models on ground truths of
manually annotated Glasbena Mladina magazines. 

## Layout Analysis

| Model (_detection backbone_) | Pretrained (or model size) / fine-tuned on | Score threshold | mAP / mAP50 / mAP75 / mAP-s / mAP-m / mAP-l (%)                             | Title / text / figure / figure caption AP (%)       |
|------------------------------|--------------------------------------------|-----------------|-----------------------------------------------------------------------------|-----------------------------------------------------|
| DETR                         | DocLayNet                                  | 0.4             | 41.58 / 60.31 / 43.46 / 4.94 / 24.10 / 51.10                                | 28.59 / 54.56 / 68.43 / 14.73                       |
| DiT (_Cascade R-CNN_)        | Large / PubLayNet                          | 0.5             | 33.05 / 42.76 / 35.55 / 3.46 / 22.47 / 38.33                                | 16.34 / 59.34 / 23.46 / -                           |
| DocLayout-YOLO (_YOLOv10_)   | DocSynth300k / D4LA                        | 0               | 53.44 / 67.81 / 56.52 / 11.64 / 33.86 / <ins>67.49</ins>                    | 36.40 / 68.21 / 77.13 / **32.02**                   |
| DocLayout-YOLO (_YOLOv10_)   | DocSynth300k / DocLayNet                   | 0               | 48.22 / 65.59 / 50.06 / 8.28 / 31.92 / 61.82                                | 35.99 / 65.42 / 69.82 / 21.64                       |
| DocLayout-YOLO (_YOLOv10_)   | DocSynth300k / DocStructBench              | 0               | <ins>55.06</ins> / 65.98 / <ins>58.61</ins> / **18.02** / **38.38** / 64.11 | **48.43** / **71.72** / 69.91 / 30.17               |
| LayoutLMv3 (_Cascade R-CNN_) | Base / PubLayNet                           | 0.1             | 40.88 / 54.08 / 44.12 / 8.35 / 27.80 / 45.13                                | 26.12 / 64.03 / 32.50 / -                           |
| RF-DETR (_RF-DETR_)          | DocLayNet                                  | 0               | 31.37 / 44.34 / 31.96 / 4.98 / 15.59 / 45.60                                | 24.73 / 38.10 / 52.38 / 10.26                       |
| PP-DocLayoutV3               | _In-house_                                 | 0               | **64.24** / **73.04** / **67.75** / <ins>16.24</ins> / 33.98 / **76.98**    | <ins>42.46</ins> / <ins>71.09</ins> / **79.15** / - |
| SwinDocSegmenter             | DocLayNet                                  | 0.2             | 21.42 / 27.72 / 22.69 / 6.23 / 14.43 / 26.27                                | 21.90 / 34.30 / 23.52 / 5.98                        |
| VGT (_Cascade R-CNN_)        | DocLayNet                                  | 0.1             | 50.56 / <ins>70.60</ins> / 50.48 / 9.28 / <ins>34.10</ins> / 64.11          | 36.11 / 65.96 / <ins>78.62</ins> / 21.54            |
| VGT (_Cascade R-CNN_)        | D4LA                                       | 0.1             | 50.15 / 69.24 / 52.66 / 10.70 / 32.37 / 65.42                               | 32.81 / 66.53 / 70.37 / <ins>30.88</ins>            |

<details>
<summary><b>Manual evaluation</b></summary>

Results were obtained using this repo's own implementations for evaluations (used mainly for more controlled, manual
control and checking of calculations):

| Model          | Backbone | Dataset   | mAP / mAP50 / mAP70      | Class agnostic mAP / mAP50 / mAP70 |
|----------------|----------|-----------|--------------------------|------------------------------------|
| RF-DETR        | -        | DocLayNet | 0.3422 / 0.4064 / 0.3678 | 0.4910 / 0.6244 / 0.53606          |
</details>

## Reading Order

| Model                     | Normalized edit distance | BLEU              |
|---------------------------|--------------------------|-------------------|
| Top to bottom             | 0.6556                   | 0.1007            |
| Left to right             | <ins>0.2222</ins>        | 0.6322            |
| PaddleX's Improved XY-Cut | 0.2411                   | <ins>0.6349</ins> |
| LayoutReader              | **0.1696**               | **0.7143**        |


## VLM

### Specialized VLMs:

| Model               | Type                    | Text block NED   | Reading order NED |
|---------------------|-------------------------|------------------|-------------------|
| DeepSeekOCR-2       | -                       | 0.188            | 0.115             |
| Docling             | Default                 | 0.0664           | 0.164             |
| dots.ocr            | dots.mocr               | <ins>0.348</ins> | **0.0765**        |
| Dolphin             | Dolphinv2               | 0.0542           | 0.0896            |
| GLM-OCR             | GLM-4V                  | 0.1379*          | 0.1941*           |
| Marker              | Default                 | 0.0416           | 0.1033            |
| MinerU              | MinerU2.5-Pro-2604-1.2B | 0.181            | 0.137             |
| Nemotron-Parse-v1.2 | -                       | 0.0686           | 0.0914            |
| PaddleOCR           | PaddleOCR-VL-1.6        | 0.115            | 0.170             |
| Youtu-Parsing       | Youtu-LLM-2B-Base       | **0.0306**       | <ins>0.0874</ins> |

\* Only results successfully parsed were scored against ground truth.

### General VLMs

| Model    | Type                 | Text block NED*    | Reading order NED* | Mean cost per page** and token usage                                                                            |
|----------|----------------------|--------------------|--------------------|-----------------------------------------------------------------------------------------------------------------|
| ChatGPT  | GPT 5.5              | <ins>0.05688</ins> | <ins>0.0706</ins>  | \$0.1444 / 0,13€<br/>-\$0.0203 for 4050 tokens at \$5/MTok IN<br/>-\$0.1247 for 4157 at \$30.00/MTok OUT        |
| Claude   | Opus 4.8             | **0.0504**         | 0.0800             | \$0.0935 / 0,082€<br/>-\$0.0279 for 5584 tokens at \$5/MTok IN<br/>-\$0.0656 for 2624 tokens at \$25/MTok OUT   |
| Gemini   | Gemini 3.1 Pro       | 0.0718             | **0.0664**         | \$0.0988 / 0,086€<br/>-\$0.0025 for 1277 tokens at ~\$2/MTok IN<br/>-\$0.0961 for 8007 tokens at ~\$12/MTok OUT |

\* Currently only evaluated on a single PDF.

\** As of 15.7.2026, unoptimized (no use of cache), using similar resolution as seen on OmniDocBench dataset images.

**Note**: NED - Normalized edit distance

#### Page wise breakdown of NED scores for best VLMs

Comparison of the best performing VLMs compared to our implementation, given with normalized edit distance by page type.

| Page type          | Youtu-Parsing: Text | Youtu-Parsing: Order | Marker: Text | Marker: Order | dots.ocr: Text | dots.ocr: Order |
|--------------------|---------------------|----------------------|--------------|---------------|----------------|-----------------|
| Advert             | 0.048               | 0.104                | 0.113        | 0.174         | 0.038          | 0.059           |
| Article            | 0.023               | 0.069                | 0.024        | 0.090         | 0.025          | 0.061           |
| Cover              | 0.009               | 0.125                | 0.052        | 0.125         | 0.026          | 0.000           |
| Events             | 0.055               | 0.248                | 0.136        | 0.266         | 0.037          | 0.229           |
| Images             | 0.172               | 0.092                | 0.401        | 0.192         | 0.185          | 0.083           |
| Interview          | 0.022               | 0.078                | 0.032        | 0.101         | 0.023          | 0.074           |
| Letters            | 0.020               | 0.050                | 0.015        | 0.075         | 0.034          | 0.065           |
| News               | 0.020               | 0.093                | 0.021        | 0.103         | 0.022          | 0.064           |
| Quiz               | 0.058               | 0.068                | 0.080        | 0.102         | 0.089          | 0.070           |
| Records            | 0.017               | 0.110                | 0.015        | 0.084         | 0.032          | 0.071           |
| Special            | 0.175               | 0.186                | 0.222        | 0.297         | 0.145          | 0.201           |
| TOC                | 0.058               | 0.145                | 0.061        | 0.151         | 0.049          | 0.146           |
| **All (page avg)** | **0.031**           | **0.084**            | **0.042**    | **0.103**     | **0.035**      | **0.072**       |
| All (whole)        | 0.025               | 0.081                | 0.028        | 0.099         | 0.029          | 0.068           |
| All (sample avg)   | 0.039               | 0.084                | 0.055        | 0.103         | 0.043          | 0.072           |

\* Due to ground truth being obtained using bounding boxes but same OCR-ed letters, the text block NED is not directly
comparable to the other VLMs, but is included for reference.

### Figure linking results

Percentage of figures, captions and figure bylines correctly linked to their respsective articles using proximity,
NER and additional heuristics.


### Classification results

Number of correctly classified articles.

| Model             | Matching |
|-------------------|----------|
| Gams-12B-Instruct | 0.5968   |


# Demo

_TODO_

# TODO

Top priority:
- [x] Add mappings from other datasets (PubLayNet, DocBank, DocLayNet) to OmniDocBench schema for evaluation
- [x] Add OmniDocBench evaluation support for pipeline track
  - [x] Fix classes and other issues in OmniDocBench layout evaluation
- [X] Add specialized VLM support:
  - [x] Marker
- [x] Add general VLM support, among others:
  - [x] Qwen3-VL
  - [x] GPT 5.5
  - [x] Gemini Pro 3.1
  - [x] Anthropic Claude Opus 4.8
- [x] Evaluate all models: 
  - [x] Pipeline
  - [x] Specialized VLM
  - [x] General VLM

# Limitations and Further Work

Pipeline based track:
- Built layout detection and reading order detection tasks are evaluated separately (reading order is evaluated using
ground bounding boxes).
- Currently RaRe only supports inference; possible extension includes training of the available models.
- Adding support for Paragraph2Graph, M2Doc
- VLM track currently only supports output in the formats given by each of the model itself. Further improvement could
see its integration into rare and outputting in an arbitrary format (such as JSON, HTML etc.)

# Acknowledgements

Thanks for the work of the authors of these projects:
- [PaddleX](https://github.com/PaddlePaddle/PaddleX) — the improved XY-Cut reading-order backend is vendored from PaddleX (Apache-2.0); see `NOTICE` and `licenses/LICENSE-PADDLEX`.
- [OmniDocBench](https://github.com/opendatalab/OmniDocBench) — the end-to-end Edit-distance evaluator (run via `--run-omnidocbench`) and the specialized VLM `img2md` parsing backends are adapted from OmniDocBench (Apache-2.0); see `NOTICE` and `licenses/LICENSE-OMNIDOCBENCH`.
- [layoutreader](https://github.com/FreeOCR-AI/layoutreader) — the `layoutreader` reading-order backend uses the LayoutLMv3 inference helpers and the `hantian/layoutreader` checkpoint from Hantian Pang's faster LayoutReader (**CC BY-NC-SA 4.0**); see `NOTICE` and `licenses/LICENSE-LAYOUTREADER`, and the licensing note below.
- [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)

<details>
<summary><b>LayoutReader specifics</b></summary>

RaRe uses Apache License 2.0 (see `LICENSE`), **with one
exception**: `src/rare/models/order/layoutreader_helpers/helpers.py` is
vendored verbatim from [FreeOCR-AI/layoutreader](https://github.com/FreeOCR-AI/layoutreader)
and is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/),
as is the `hantian/layoutreader` checkpoint that the backend downloads at
runtime. Consequently the `layoutreader` reading-order backend may be used for
**non-commercial purposes only**, and adaptations of that file must be shared
under the same license. Every other backend, including the `xy-cut` reading-order
backends, is unaffected and remains Apache-2.0. See `NOTICE` for the full
per-component breakdown.

</details>

# Citation

```BibTeX
TODO
```

<details>
<summary><b>LayoutReader specifics</b></summary>

If you use the `layoutreader` reading-order backend, please cite the upstream
implementation and the original LayoutReader paper:

```BibTeX
@software{Pang_Faster_LayoutReader_based_2024,
  author  = {Pang, Hantian},
  month   = feb,
  title   = {{Faster LayoutReader based on LayoutLMv3}},
  url     = {https://github.com/ppaanngggg/layoutreader},
  version = {1.0.0},
  year    = {2024}
}

@inproceedings{wang-etal-2021-layoutreader,
  title     = {{L}ayout{R}eader: Pre-training of Text and Layout for Reading Order Detection},
  author    = {Wang, Zilong and Xu, Yiheng and Cui, Lei and Shang, Jingbo and Wei, Furu},
  booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  month     = nov,
  year      = {2021},
  address   = {Online and Punta Cana, Dominican Republic},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2021.emnlp-main.389/},
  doi       = {10.18653/v1/2021.emnlp-main.389},
  pages     = {4735--4744}
}
```

</details>
