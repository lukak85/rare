# Adapted from OmniDocBench (https://github.com/opendatalab/OmniDocBench),
# tools/model_infer/DotsOCR_img2md.py. Copyright (c) 2024 OpenDataLab and the
# OmniDocBench authors. Licensed under the Apache License, Version 2.0; see
# licenses/LICENSE-OMNIDOCBENCH and the NOTICE file.
"""dots.ocr backend.

dots.ocr (https://github.com/rednote-hilab/dots.ocr) is a layout+OCR VLM served
through vLLM. For each page image it emits layout cells (category + text + bbox),
which `layoutjson2md` turns into the page's Markdown. That formatter output is
the faithful prediction OmniDocBench compares, so — like Docling — we treat it as
a black box: we don't re-derive regions, we wrap each page's `layoutjson2md`
Markdown in a single region. Two entry points share one client:

- `to_markdown(pdf_dir, image_dir, out_md_dir, ...)` — faithful port of
  OmniDocBench's `DotsOCR_img2md.py`: a folder of page images -> one `<stem>.md`
  per image. Used with the standalone `normalize_pred.py` + `run.sh` path.
- `parse_pdf(pdf) -> GlasanaDocument` — the `evaluate/runner.py::run_vlm`
  contract. Renders the PDF to pages, runs dots.ocr per page, and wraps each
  page's `layoutjson2md` Markdown in one region. Because `raw_markdown = True`,
  the runner joins region text verbatim, reproducing dots.ocr's own per-page
  Markdown for OmniDocBench.

Requires a running dots.ocr vLLM server (set ip/port/model_name via config)::

    pip install dots-ocr   # on the inference box, then serve the model with vLLM
"""

from __future__ import annotations

import os
from pathlib import Path

from tqdm import tqdm

from rare.doc.schema import GlasanaDocument
from rare.models.registry import register
from rare.models.vlm._assembler import assemble_document
from rare.models.vlm._vlm_schema import VLMDocument, VLMPage, VLMRegion

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


@register("vlm", "dots-ocr")
class DotsOCRBackend:
    """Runs dots.ocr over page images and exposes its Markdown for evaluation."""

    name = "dots-ocr"
    # parse_pdf wraps dots.ocr's own per-page Markdown (layoutjson2md) in one
    # region per page, so the runner scores it verbatim (raw join).
    raw_markdown = True

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        # vLLM server endpoint + sampling (defaults mirror DotsOCR_img2md.py).
        self.protocol = cfg.get("protocol", "http")
        self.ip = cfg.get("ip", "localhost")
        self.port = int(cfg.get("port", 8000))
        self.model_name = cfg.get("model_name", "dots.ocr/")
        self.temperature = float(cfg.get("temperature", 0.1))
        self.top_p = float(cfg.get("top_p", 1.0))
        self.max_completion_tokens = int(cfg.get("max_completion_tokens", 16384))
        self.prompt_mode = cfg.get("prompt_mode", "prompt_layout_all_en")
        self.dpi = int(cfg.get("dpi", 200))
        self.min_pixels = cfg.get("min_pixels")
        self.max_pixels = cfg.get("max_pixels")
        self.num_thread = int(cfg.get("num_thread", 16))
        # Drop page header/footer from the Markdown (layoutjson2md's no_page_hf).
        self.no_page_hf = bool(cfg.get("no_page_hf", False))

    # --- inference ---------------------------------------------------------

    def _infer_image(self, origin_image) -> str:
        """Run dots.ocr on one PIL page image and return its Markdown.

        Mirrors `DotsOCR_img2md.py`: infer -> `post_process_output` -> cells,
        then `layoutjson2md`. When post-processing can't parse structured cells
        it returns the raw Markdown string directly (the `filtered` path)."""
        from dots_ocr.model.inference import inference_with_vllm
        from dots_ocr.utils.image_utils import fetch_image
        from dots_ocr.utils.layout_utils import post_process_output
        from dots_ocr.utils.format_transformer import layoutjson2md
        from dots_ocr.utils.prompts import dict_promptmode_to_prompt

        image = fetch_image(
            origin_image, min_pixels=self.min_pixels, max_pixels=self.max_pixels
        )
        prompt = dict_promptmode_to_prompt[self.prompt_mode]
        response = inference_with_vllm(
            image,
            prompt,
            model_name=self.model_name,
            protocol=self.protocol,
            ip=self.ip,
            port=self.port,
            temperature=self.temperature,
            top_p=self.top_p,
            max_completion_tokens=self.max_completion_tokens,
        )
        cells, filtered = post_process_output(
            response,
            self.prompt_mode,
            origin_image,
            image,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        if filtered:
            # cells is already a Markdown string when structured parsing failed.
            return cells if isinstance(cells, str) else ""
        return layoutjson2md(
            origin_image, cells, text_key="text", no_page_hf=self.no_page_hf
        )

    def _infer_images(self, images: list) -> list[str]:
        """Markdown for each page image, in order. Threads across the vLLM
        server (it handles concurrent requests) for throughput."""
        n = min(len(images), self.num_thread)
        if n <= 1:
            return [self._infer_image(im) for im in images]
        from multiprocessing.pool import ThreadPool

        with ThreadPool(n) as pool:
            return list(pool.imap(self._infer_image, images))  # imap keeps order

    # --- faithful reference port -------------------------------------------

    def to_markdown(
        self,
        pdf_dir: str | Path,
        image_dir: str | Path,
        out_md_dir: str | Path,
        skip_existing: bool = False,
    ) -> str | Path:
        """Convert every page image under `image_dir` and write `<stem>.md` into
        `out_md_dir` (one per image), matching `DotsOCR_img2md.py`. `pdf_dir` is
        unused here (dots.ocr scores per page image) but kept for signature
        parity with the other backends."""
        from PIL import Image

        os.makedirs(out_md_dir, exist_ok=True)
        paths = sorted(
            p for p in Path(image_dir).iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
        )
        todo = [
            p for p in paths
            if not (skip_existing and (Path(out_md_dir) / f"{p.stem}.md").exists())
        ]
        print(f"found {len(paths)} images; processing {len(todo)}.")
        images = [Image.open(p).convert("RGB") for p in todo]
        for path, md in zip(todo, tqdm(self._infer_images(images), desc="dots-ocr")):
            (Path(out_md_dir) / f"{path.stem}.md").write_text(md, encoding="utf-8")
        return out_md_dir

    # --- evaluate/runner.py::run_vlm contract ------------------------------

    def parse_pdf(self, pdf_path: str | Path) -> GlasanaDocument:
        """Render `pdf_path` to pages, run dots.ocr, and wrap each page's
        Markdown in a single region for the runner's verbatim (raw) scoring."""
        from dots_ocr.utils.doc_utils import load_images_from_pdf

        pdf_path = Path(pdf_path)
        images = load_images_from_pdf(str(pdf_path), dpi=self.dpi)
        mds = self._infer_images(images)
        pages = [
            VLMPage(
                page_no=page_no,
                width=image.width,
                height=image.height,
                regions=[VLMRegion(label="Paragraph", text=md)],
            )
            for page_no, (image, md) in enumerate(zip(images, mds))
        ]
        return assemble_document(VLMDocument(pages=pages), pdf_path.stem)