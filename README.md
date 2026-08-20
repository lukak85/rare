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

#### OCR fallback for empty regions (`--ocr`)

The corpus PDFs are scans: one full-page image per page with an invisible OCR text layer over it. Where that upstream
OCR gave up, per-region extraction yields nothing — there are no glyphs under the box to extract. Running headers are
the frequent casualty.

`--ocr tesseract` re-reads those regions from the pixels, cropping them out of the page re-rendered at
`--ocr-dpi` (400 by default — 200 is marginal for Tesseract on these scans). It runs **only** on regions that came back
empty, so text the PDF actually carries is never second-guessed, and only on the labels named by `--ocr-labels`
(default: `Header`). Figures are never OCR'd whatever the label set says. Readings below `--ocr-min-confidence` are
discarded — an empty header is a smaller problem than a header full of noise.

Filled regions carry `provenance.text_source: "ocr:tesseract"` and `provenance.ocr_confidence` in the output JSON, so
OCR'd text stays distinguishable from text the PDF carried. Everything else keeps `"pdf"`.

Needs the binary and the Slovenian language data, which is a separate package:

```bash
sudo apt install tesseract-ocr tesseract-ocr-slv
```

Parsing fails immediately if the requested `--ocr-lang` is not installed, rather than falling back to English and
filling the document with plausible-looking text whose diacritics are wrong.

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

The annotations carry no article grouping, but their reading order places every Figure, Caption and FigByline
immediately after the piece it illustrates. That gives a ground truth for the linking stage without annotating
articles: **a visual belongs to the same article as the last body block before it in reading order** (its *anchor*,
found by skipping the other visuals in its own run).

```bash
# Linking alone — documents are rebuilt from the ground-truth boxes and order,
# so detection and reading order are perfect and only rare.link is measured.
rare evaluate --track figure-link --dataset glasbena_mladina \
    --pdfs-dir datasets/glasbena_mladina/pdfs/eval

# End to end — score the *_doc.json of a real parse; items are matched back to
# the annotations by IoU, so misdetections and reading-order errors count too.
rare evaluate --track figure-link --dataset glasbena_mladina \
    --docs-dir outputs/parsed
```

| Metric | Meaning |
|---|---|
| `attachment_accuracy` | visual and its anchor in the same predicted article, over the visuals where both were found |
| `attachment_recall`   | same, over every visual the annotation places — an undetected figure counts against it |
| `separation`          | visual/block pairs on opposite sides of a Headline that ended up in **different** articles |
| `attachment_score`    | harmonic mean of accuracy and separation |
| `caption_figure_accuracy` | a Caption/FigByline that follows a Figure carries that figure's `figure_id` |

Accuracy and separation are reported together because each is trivial to win alone: one article for the whole
magazine scores 1.0 on accuracy and 0.0 on separation, one article per block does the opposite.

Results land in `outputs/evaluations/<run_id>/`: `attachment_summary.json` (overall plus per label, per page type
and per document), `attachment_cases.jsonl` (one row per annotated visual — status, anchor, predicted article title)
and the usual `report.md` / `scores.csv`.

#### Page type vs article genre (`--track page-genre`)

A first, deliberately blunt check on the classification pass: does the annotated `page_type` of a page agree with the
`genre` of the articles predicted on it? The two vocabularies describe different things, so the comparison runs
through one editable table, `PAGE_TYPE_TO_GENRE` in `rare/evaluate/page_genre.py`:

| Page type | Expected genre | | Page type | Expected genre |
|---|---|---|---|---|
| `ArticlePage`   | `članek`    | | `ImagesPage` | `slike`      |
| `NewsPage`      | `novice`    | | `TOCPage`    | `kazalo`     |
| `InterviewPage` | `intervju`  | | `AdvertPage` | `reklama`    |
| `RecordsPage`   | `recenzija` | | `CoverPage`  | `naslovnica` |
| `LettersPage`   | `pisma`     | | `SpecialPage`| *not scored* |
| `QuizPage`      | `kviz`      | | `BackPage`   | *not scored* |
| `EventsPage`    | `dogodki`   | | *(no `page_type`)* | *not scored* |

`SpecialPage` and `BackPage` say how a page is laid out and where it sits in the issue, not what the piece on it is,
so they are mapped to `null` and left out of the totals rather than counted as failures. Change any of that from a
JSON file of `{page_type: genre | [genres] | null}` — a list accepts several genres, `null` retires a page type:

```bash
# Score a real parse; genres come from whatever --classification produced.
rare evaluate --track page-genre --dataset glasbena_mladina \
    --docs-dir outputs/parsed [--page-type-map my_map.json]

# Ground-truth layout instead. With no --classification, genres still come from
# the running-header fallback in rare.link.classify — the cheap way to try a
# change to the table.
rare evaluate --track page-genre --dataset glasbena_mladina \
    --pdfs-dir datasets/glasbena_mladina/pdfs/eval
```

A page can hold pieces of several genres, so the result is reported from three angles: `accuracy_dominant` (the
article holding most of the page has the expected genre), `accuracy_any` (some article on the page does) and
`article_accuracy` over every (page, article) pair — the truth for a mixed page is between the first two.
`genre_coverage` is the share of scored pages carrying any genre at all; a low headline accuracy usually means
articles went unclassified, not that they were misclassified.

`page_genre_summary.json` also holds the **confusion matrix** of page type against the genre actually predicted,
which is what to read when deciding how the table should change; `page_genre_pages.jsonl` has one row per scored page.

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

_TODO - introduce Edit distance metric for pipeline track, as it currently only works for VLM track._

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

| Model                     | Normalized edit distance | Kendall Tau       | BLEU              |
|---------------------------|--------------------------|-------------------|-------------------|
| Top to bottom             | 0.6556                   | 0.3763            | 0.1007            |
| Left to right             | <ins>0.2222</ins>        | 0.5857            | 0.6322            |
| PaddleX's Improved XY-Cut | 0.2411                   | <ins>0.8107</ins> | <ins>0.6349</ins> |
| LayoutReader              | **0.1696**               | **0.8599**        | **0.7143**        |


## VLM

### Specialized VLMs:

| Model               | Type                    | Text block NED    | Reading order NED |
|---------------------|-------------------------|-------------------|-------------------|
| DeepSeekOCR-2       | -                       | 0.188             | 0.115             |
| Docling             | Default                 | 0.0664            | 0.164             |
| dots.ocr            | dots.mocr               | <ins>0.0420</ins> | **0.0765**        |
| Dolphin             | Dolphinv2               | 0.0542            | 0.0896            |
| GLM-OCR             | GLM-4V                  | 0.1379*           | 0.1941*           |
| Marker              | Default                 | 0.0461            | 0.1033            |
| MinerU              | MinerU2.5-Pro-2604-1.2B | 0.181             | 0.137             |
| Nemotron-Parse-v1.2 | -                       | 0.0686            | 0.0914            |
| PaddleOCR           | PaddleOCR-VL-1.6        | 0.115             | 0.170             |
| Youtu-Parsing       | Youtu-LLM-2B-Base       | **0.0383**        | <ins>0.0874</ins> |

\* Only results successfully parsed were scored against ground truth.

### General VLMs

| Model    | Type                 | Text block NED*    | Reading order NED* | Mean cost per page** and token usage                                                                            |
|----------|----------------------|--------------------|--------------------|-----------------------------------------------------------------------------------------------------------------|
| ChatGPT  | GPT 5.5              | <ins>0.05688</ins> | <ins>0.0706</ins>  | \$0.1444 / 0,13€<br/>-\$0.0203 for 4050 tokens at \$5/MTok IN<br/>-\$0.1247 for 4157 at \$30.00/MTok OUT        |
| Claude   | Opus 4.8             | **0.0504**         | 0.0800             | \$0.0935 / 0,082€<br/>-\$0.0279 for 5584 tokens at \$5/MTok IN<br/>-\$0.0656 for 2624 tokens at \$25/MTok OUT   |
| Gemini   | Gemini 3.1 Pro       | 0.0718             | **0.0664**         | \$0.0988 / 0,086€<br/>-\$0.0025 for 1277 tokens at ~\$2/MTok IN<br/>-\$0.0961 for 8007 tokens at ~\$12/MTok OUT |
| Qwen3-VL | Qwen3-VL-8B-Instruct | TODO               | TODO               | -                                                                                                               |

\* Currently only evaluated on a single PDF.

\** As of 15.7.2026, unoptimised (no use of cache), using similar resolution as seen on OmniDocBench dataset images.

**Note**: NED - Normalized edit distance

#### Page wise breakdown of NED scores for best VLMs

Comparisson of the best performing VLMs compared to our implementation

| Model         | Type                          | Text block NED    | Reading order NED |
|---------------|-------------------------------|-------------------|-------------------|
| dots.ocr      | dots.mocr                     | <ins>0.0420</ins> | **0.0765**        |
| Marker        | Default                       | 0.0461            | 0.1033            |
| Youtu-Parsing | Youtu-LLM-2B-Base             | **0.0383**        | <ins>0.0874</ins> |
| Ours*         | DocLayout-YOLO + LayoutReader | 0.0694            | 0.1899            |

\* Due to ground truth being obtained using bounding boxes but same OCR-ed letters, the text block NED is not directly
comparable to the other VLMs, but is included for reference.

<ins>**Page wise breakdown**</ins>:

TODO

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
    - [ ] Qwen3-VL

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
