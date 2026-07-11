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
| **[DiT](https://github.com/microsoft/unilm/tree/master/dit)**                                               | `dit`            | Vision transformers | 3.8                        |
| **[DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)**                                         | `doclayout-yolo` | Object detection    | 3.10                       |
| **Faster R-CNN***                                                                                           | `faster-rcnn`    | CNN-based           | 3.12                       |
| **[LayoutLMv3](https://github.com/microsoft/unilm/tree/master/layoutlmv3)**                                 | `layoutlmv3`     | Multimodal          | 3.7                        |
| **Mask R-CNN***                                                                                             | `mask-rcnn`      | CNN-based           | 3.12                       |
| **[PP-DocLayoutV3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3)**                                    | `pp-doclayoutv3` | Vision transformers | 3.12                       |
| **[RF-DETR](https://huggingface.co/neka-nat/rfdetr-doclayout)**                                             | `rf-detr`        | Vision transformers | 3.14                       |
| **[VGT](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/DocumentUnderstanding/VGT)** | `vgt`            | Multimodal          | 3.8                        |

\* Included in LayoutParser with detectron2

### Pipeline track — reading-order backends

| Model                                                                    | CLI name         | Type       | Recommended Python version |
|--------------------------------------------------------------------------|------------------|------------|----------------------------|
| Top-bottom                                                               | _Default_        | Rule based | Any                        |
| **[PaddleX's Improved XY-Cut](https://github.com/PaddlePaddle/PaddleX)** | `paddlex-xy-cut` | Rule based | Any                        |

### VLM track

| Model                                                                               | CLI name      | Type               | Recommended Python version |
|-------------------------------------------------------------------------------------|---------------|--------------------|----------------------------|
| **[DeepSeek-OCR-2](https://github.com/deepseek-ai/DeepSeek-OCR-2)**                 | `deepseekocr` | Specialized VLMs   | 3.12.9                     |
| **[Docling](https://github.com/docling-project/docling)**                           | `docling`     | Specialized VLMs   | 3.14                       |
| **[dots.ocr](https://github.com/rednote-hilab/dots.ocr)**                           | `dots-ocr`    | Specialized VLMs   | 3.12                       |
| **[GLM-OCR](https://github.com/zai-org/GLM-OCR)**                                   | `glm-ocr`     | Specialized VLMs   | 3.13                       |
| **[Marker](https://github.com/datalab-to/marker)**                                  | `marker`      | Specialized VLMs   | 3.10                       |
| **[MinerU](https://github.com/opendatalab/mineru)**                                 | `mineru`      | Specialized VLMs   | 3.13                       |
| **[Nemotron-Parse-v1.2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.2)** | `nemotron`    | Specialized VLMs   | 3.13                       |
| **[PaddleOCR](https://github.com/PADDLEPADDLE/PADDLEOCR)**                          | `paddleocr`   | Specialized VLMs   | 3.12                       |
| **[Qwen3-VL](https://huggingface.co/collections/Qwen/qwen3-vl)**                    | `paddleocr`   | Local general VLMs | 3.12                       |
| **[Youtu-Parsing](https://github.com/PADDLEPADDLE/PADDLEOCR)**                      | `youtu`       | Specialized VLMs   | 3.10                       |

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
│   ├── layout/               # layout detection model/method classes
│   ├── order/builtin.py      # order detection model/method classes
│   └── vlm/                  # visual language model document parsing classes
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

### Claude

TODO

https://platform.claude.com/docs/en/build-with-claude/pdf-support

### DeepSeek

TODO

### Gemini

TODO

https://ai.google.dev/gemini-api/docs/document-processing

### GPT

TODO

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
| DiT (_Cascade R-CNN_)        | Large / PubLayNet                          | 0.5             | 33.05 / 42.76 / 35.55 / 3.46 / 22.47 / 38.33                                | 16.34 / 59.34 / 23.46 / -                           |
| DocLayout-YOLO (_YOLOv10_)   | DocSynth300k / D4LA                        | 0               | 53.44 / 67.81 / 56.52 / 11.64 / 33.86 / <ins>67.49</ins>                    | 36.40 / 68.21 / 77.13 / **32.02**                   |
| DocLayout-YOLO (_YOLOv10_)   | DocSynth300k / DocLayNet                   | 0               | 48.22 / 65.59 / 50.06 / 8.28 / 31.92 / 61.82                                | 35.99 / 65.42 / 69.82 / 21.64                       |
| DocLayout-YOLO (_YOLOv10_)   | DocSynth300k / DocStructBench              | 0               | <ins>55.06</ins> / 65.98 / <ins>58.61</ins> / **18.02** / **38.38** / 64.11 | **48.43** / **71.72** / 69.91 / 30.17               |
| LayoutLMv3 (_Cascade R-CNN_) | Base / PubLayNet                           | 0.1             | 40.88 / 54.08 / 44.12 / 8.35 / 27.80 / 45.13                                | 26.12 / 64.03 / 32.50 / -                           |
| RF-DETR (_RF-DETR_)          | DocLayNet                                  | 0               | 31.37 / 44.34 / 31.96 / 4.98 / 15.59 / 45.60                                | 24.73 / 38.10 / 52.38 / 10.26                       |
| PP-DocLayoutV3               | _In-house_                                 | 0               | **64.24** / **73.04** / **67.75** / <ins>16.24</ins> / 33.98 / **76.98**    | <ins>42.46</ins> / <ins>71.09</ins> / **79.15** / - |
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

| Model                     | Normalized edit distance | Kendall Tau |
|---------------------------|--------------------------|-------------|
| PaddleX's Improved XY-Cut | 0.2411                   | 0.8107      |


## VLM

### Specialized VLMs:

| Model               | Type                    | Text block NED    | Reading order NED |
|---------------------|-------------------------|-------------------|-------------------|
| DeepSeekOCR-2       | -                       | 0.188             | 0.115             |
| Docling             | Default                 | 0.0664            | 0.164             |
| dots.ocr            | dots.mocr               | <ins>0.0420</ins> | **0.0765**        |
| GLM-OCR             | GLM-4V                  | 0.1379*           | 0.1941*           |
| Marker              | Default                 | 0.0461            | 0.1033            |
| MinerU              | MinerU2.5-Pro-2604-1.2B | 0.181             | 0.137             |
| Nemotron-Parse-v1.2 | -                       | 0.0461            | 0.1033            |
| PaddleOCR           | PaddleOCR-VL-1.6        | 0.115             | 0.170             |
| Youtu-Parsing       | Youtu-LLM-2B-Base       | **0.0383**        | <ins>0.0874</ins> |

\* Only results successfully parsed were scored against ground truth.

### General VLMs

| Model    | Type                 | Text block NED | Reading order NED |
|----------|----------------------|----------------|-------------------|
| ChatGPT  | GPT 5.6              | TODO           | TODO              |
| Claude   | Opus 4.8             | TODO           | TODO              |
| Claude   | Fable 5              | TODO           | TODO              |
| Gemini   | Gemini 3 Pro         | TODO           | TODO              |
| Ovis2.6  | Ovis2.6-30B-A3B      | TODO           | TODO              |
| Qwen3-VL | Qwen3-VL-8B-Instruct | TODO           | TODO              |

**Note**: NED - Normalized edit distance

# Demo

_TODO_

# TODO

Top priority:
- [x] Add mappings from other datasets (PubLayNet, DocBank, DocLayNet) to OmniDocBench schema for evaluation
- [x] Add OmniDocBench evaluation support for pipeline track
  - [x] Fix classes and other issues in OmniDocBench layout evaluation
- [X] Add specialized VLM support:
  - [x] Marker
- [ ] Add general VLM support, among others:
  - [x] Qwen3-VL
  - [ ] GPT 5.5
  - [ ] Gemini Pro 3.1
  - [ ] Anthropic Claude Fable 5 / Opus 4.8
  - [ ] DeepSeek V3
- [ ] Evaluate all models: 
  - [x] Pipeline
  - [x] Specialized VLM
  - [ ] General VLM

Lower priority:
- [ ] Add support for Paragraph2Graph, M2Doc

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
