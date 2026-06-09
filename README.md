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
pip install -e ".[doclayout-yolo]" # + DocLayout-YOLO dependencies
pip install -e ".[pp-doclayoutv3]" --extra-index-url https://download.pytorch.org/whl/cpu # + PP-DocLayoutV3 dependencies 

pip install -e ".[docling]" # + Docling dependencies
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

#### OmniDocBench Edit distance (`--run-omnidocbench`)

Both tracks can additionally run [OmniDocBench](https://github.com/opendatalab/OmniDocBench)'s end-to-end evaluator and fold the `text_block` and `reading_order` **Edit distance** into `report.md`. Pass `--run-omnidocbench`; this runs the pinned OmniDocBench Docker image against the artifacts emitted under `outputs/evaluations/<run_id>/omnidocbench/` (so **Docker must be installed**). Use `--omnidocbench-image` to override the image.

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

The supported models (and therefore given Python version recommendations) were testeed using:
- Ubuntu 24.04
- CUDA 12.8

### Pipeline track — layout backends

| Model                                                                                                       | CLI name         | Type                | Recommended Python version |
|-------------------------------------------------------------------------------------------------------------|------------------|---------------------|----------------------------|
| **[DiT](https://github.com/microsoft/unilm/tree/master/dit)**                                               | `dit`            | Vision transformers | _TODO_                     |
| **[DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)**                                         | `doclayout-yolo` | Object detection    | 3.10                       |
| **Faster R-CNN***                                                                                           | `faster-rcnn`    | CNN-based           | _TODO_                     |
| **[LayoutLMv3](https://github.com/microsoft/unilm/tree/master/layoutlmv3)**                                 | `layoutlmv3`     | Multimodal          | _TODO_                     |
| **Mask R-CNN***                                                                                             | `mask-rcnn`      | CNN-based           | _TODO_                     |
| **[PP-DocLayoutV3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3)**                                    | `pp-doclayoutv3` | Vision transformers | 3.12                       |
| **[RF-DETR](https://huggingface.co/neka-nat/rfdetr-doclayout)**                                             | `rf-detr`        | Vision transformers | _TODO_                     |
| **[VGT](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/DocumentUnderstanding/VGT)** | `vgt`            | Multimodal          | _TODO_                     |

\* Included in LayoutParser with detectron2

### Pipeline track — reading-order backends

| Model                                                                    | CLI name         | Type       | Recommended Python version |
|--------------------------------------------------------------------------|------------------|------------|----------------------------|
| Top-bottom                                                               | _Default_        | Rule based | Any                        |
| **[PaddleX's Improved XY-Cut](https://github.com/PaddlePaddle/PaddleX)** | `paddlex-xy-cut` | Rule based | Any                        |

### VLM track

| Model                                                      | CLI name    | Type             | Recommended Python version |
|------------------------------------------------------------|-------------|------------------|----------------------------|
| **[Docling](https://github.com/docling-project/docling)**  | `docling`   | Specialized VLMs | 3.14                       |
| **[dots.ocr](https://github.com/rednote-hilab/dots.ocr)**  | `dots-ocr`  | Specialized VLMs | 3.12                       |
| **[GLM-OCR](https://github.com/zai-org/GLM-OCR)**          | `glm-ocr`   | Specialized VLMs | 3.13                       |
| **[MinerU](https://github.com/opendatalab/mineru)**        | `mineru`    | Specialized VLMs | 3.13                       |
| **[PaddleOCR](https://github.com/PADDLEPADDLE/PADDLEOCR)** | `paddleocr` | Specialized VLMs | 3.12                       |

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

---

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
vllm serve zai-org/GLM-OCR --allowed-local-media-path --port 8080
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


### PaddleOCR

As per instructions, given CUDA 12.8, install:
```bash
 python -m pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

And then:
```bash
python -m pip install "paddleocr[all]"
```


## Evaluation

Two approaches to evaluation are present:
- manual (hand written functions for computation of mAP, normalized edit distance within the project)
- using [OmniDocBench](https://github.com/opendatalab/OmniDocBench) — run automatically as part of `rare evaluate` via `--run-omnidocbench` (see [the usage section](#omnidocbench-edit-distance---run-omnidocbench))

Current results are temporary and subject to change with further testing.

# OmniDocBench Evaluation Results

The following results were obtained by evaluating detections made by the following models on ground truths of
manually annotated Glasbena Mladina magazines. 

## Layout Analysis

| Model          | Backbone | mAP/mAP50/mAP70 |
|----------------|----------|-----------------|
| DiT            | TODO     | TODO            |
| DocLayout-YOLO | TODO     | TODO            |
| LayoutLMv3     | TODO     | TODO            |
| DocLayout-YOLO | TODO     | TODO            |

## Reading Order

| Model                     | Backbone | mAP/mAP50/mAP70 |
|---------------------------|----------|-----------------|
| PaddleX's Improved XY-Cut | TODO     | TODO            |


## VLM

| Model     | Type                    | Normalized edit distance |
|-----------|-------------------------|--------------------------|
| Docling   | TODO                    | TODO                     |
| dots.ocr  | TODO                    | TODO                     |
| GLM-OCR   | TODO                    | TODO                     |
| MinerU    | MinerU2.5-Pro-2604-1.2B | 0.268333333              |
| PaddleOCR | TODO                    | TODO                     |


# Demo

_TODO_

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
- [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)

# Citation

```BibTeX
TODO
```
