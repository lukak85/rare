# Razčlenjevalnik Revij (RaRe)

RaRe is a parsing toolkit for Slovene magazines (primarily *Glasbena Mladina*), built on top of a fork of [layoutparser](https://github.com/Layout-Parser/layout-parser).
It exposes two tracks for parsing PDFs and comparing models on annotated data:

- **Pipeline track** — DLA model → reading-order → assembled `GlasanaDocument` → HTML / Markdown / JSON.
- **VLM track** — vision-language model (cloud or locally-served) producing HTML / Markdown / JSON directly.

Pipelines track (at the moment) assumes presence of previously OCR-ed PDFs.

## Installation

Before running either of these, check [Additional model-specific setup](#additional-model-specific-setup) below for
additional per-model requirements. Then run:

```bash
pip install -e .                   # core package + 'rare' command

# Model specific dependencies
pip install -e ".[doclayout-yolo]" # + DocLayout-YOLO dependencies
pip install -e ".[pp-doclayoutv3]" --extra-index-url https://download.pytorch.org/whl/cpu # + PP-DocLayoutV3 dependencies 
```

For the LayoutParser fork itself:

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

# Discover backends
rare parse --list-models
```

Outputs are stored in `outputs/parsed/<pdf_stem>/{<stem>.html, <stem>.md, <stem>_doc.json, figures/}`.

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

### `rare tools` — annotation utilities

```bash
rare tools -m count-annotations -a dataset/annotations.json
rare tools -m join-annotations -p results/<doc_hash>/ -o merged.json
rare tools -m prepare-annotations -a merged.json -o cleaned.json
rare tools -m review-annotations -a cleaned.json -s reviewed/
```

`rare tools -h` prints the full flag list (same as the old `helper.py`).

## Supported Models

### Pipeline track — layout backends

| Model                                                                                                       | CLI name         | Type                        | Recommended Python version               |
|-------------------------------------------------------------------------------------------------------------|------------------|-----------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| **[DiT](https://github.com/microsoft/unilm/tree/master/dit)**                                               | `dit`            | Document Image Transformers | _TODO_                                                                                                                              |
| **[DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)**                                         | `doclayout-yolo` | Object detection            | 3.10                                                                                                                              |
| **Faster R-CNN***                                                                                           | `faster-rcnn`    | CNN-Based                   | _TODO_                       |
| **[LayoutLMv3](https://github.com/microsoft/unilm/tree/master/layoutlmv3)**                                 | `layoutlmv3`     | Multimodal                  | _TODO_                       |
| **Mask R-CNN***                                                                                             | `mask-rcnn`      | CNN-Based                   | _TODO_                       |
| **[PP-DocLayoutV3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3)**                                    | `pp-doclayoutv3` | Transformer based           | 3.12                       |
| **[RF-DETR](https://huggingface.co/neka-nat/rfdetr-doclayout)**                                             | `rf-detr`        | Transformer based           | TODO                       |
| **[VGT](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/DocumentUnderstanding/VGT)** | `vgt`            | Multimodal                  | _TODO_                       |
\* Included in LayoutParser with detectron2

### Pipeline track — reading-order backends

| Model      | CLI name                 | Type       | Recommended Python version |
|------------|--------------------------|------------|----------------------------|
| Top-bottom | `top-bottom` *(default)* | Rule based | _TODO_                       |

### VLM track

TODO

## Outputs

`outputs/parsed/<pdf_stem>/<stem>.json` is a `GlasanaDocument`:

```json
{
  "source_pdf": "ac30fbcf...",
  "pages":     {"0": {"page_no": 0, "width": ..., "height": ...}, ...},
  "items":     {"<uuid>": {"category": "Headline", "text": "...", "provenance": {...}}, ...},
  "body_order": ["<uuid>", ...],
  "articles":  {"<uuid>": {"title": "...", "item_ids": [...]}}
}
```

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
│   ├── layout/               # 15 layout adapters (one file per LP model family)
│   ├── order/builtin.py      # top-bottom
│   └── vlm/                  # !!!WIP!!!
├── parse/                    # PDF → pages → layout → order → text → GlasanaDocument
├── evaluate/                 # dataset loaders + pipeline/VLM metrics + runner + report
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

### LayoutLMv3 / DiT

Detectron2 backbones. See the following links:
- [LayoutLMv3 install notes](https://github.com/microsoft/unilm/tree/master/layoutlmv3#installation)
- [DiT install notes](https://github.com/microsoft/unilm/tree/master/dit#setup)

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

### VGT

See the [VGT install notes](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/DocumentUnderstanding/VGT#install-requirements).
This method requires `.pkl` grid file for each input image. Follow the instructions
[VGT - Generating grid information](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/DocumentUnderstanding/VGT#generating-grid-information)
to generate them; point `rare parse` at it via `--config {"grid_root": "<path>"}`.

## Evaluation

Two approaches to evaluation are present:
- manual (hand written functions for computation of mAP, normalized edit distance within the project)
- using [OmniDocBench](/) - _TODO - add_.

# Results

_TODO_

# Demo

_TODO_

# Limitations and Further Work

Pipeline based track:
- Built layout detection and reading order detection tasks are evaluated separately (reading order is evaluated using
ground bounding boxes).
- Currently RaRe only supports inference; possible extension includes training of the available models. 

# Citation

```BibTeX
TODO
```
