# Adapted from OmniDocBench (https://github.com/opendatalab/OmniDocBench),
# tools/model_infer/MinerU2.5_img2md.py. Copyright (c) 2024 OpenDataLab and the
# OmniDocBench authors. Licensed under the Apache License, Version 2.0; see
# licenses/LICENSE-OMNIDOCBENCH and the NOTICE file.
"""MinerU2.5 VLM backend.

Runs the MinerU2.5 vision-language model via
`mineru_vl_utils.MinerUClient.batch_two_step_extract`. Two entry points share
one client:

- `images_to_markdown(...)` — faithful port of OmniDocBench's
  `MinerU2.5_img2md.py`: a folder of page images -> one `<stem>.md` per image,
  each block's `content` joined with blank lines. Used with the standalone
  `normalize_pred.py` + `run.sh` evaluation path.
- `parse_pdf(pdf) -> GlasanaDocument` — the `evaluate/runner.py::run_vlm`
  contract: render a PDF to pages, map blocks to labelled regions, and let the
  shared assembler/`to_markdown` produce the per-page Markdown the runner scores.

Install MinerU's VL utilities on the inference box::

    pip install -U "mineru-vl-utils"   # plus a backend (vllm / transformers)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from rare.doc.schema import GlasanaDocument
from rare.models.registry import register
from rare.models.vlm._assembler import assemble_document
from rare.models.vlm._vlm_schema import VLMDocument, VLMPage, VLMRegion
from rare.models.vlm.prompts import MINERU_LABEL_MAP

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
_DEFAULT_MODEL_PATH = "opendatalab/MinerU2.5-Pro-2604-1.2B"


@register("vlm", "mineru")
class MinerUBackend:
    """Runs the MinerU2.5 VLM over page images and emits per-page Markdown."""

    name = "mineru"
    # parse_pdf stores MinerU's own per-block markdown in each region's text, so
    # the runner scores it verbatim (raw join) instead of re-applying markup —
    # giving the same faithful output as `images_to_markdown`.
    raw_markdown = True

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.model_path = cfg.get("model_path", _DEFAULT_MODEL_PATH)
        self.backend = cfg.get("backend", "vllm-engine")
        self.handle_equation_block = bool(cfg.get("handle_equation_block", False))
        self.dpi = int(cfg.get("dpi", 200))  # PDF render DPI for parse_pdf
        self.label_map = {**MINERU_LABEL_MAP, **(cfg.get("label_overrides") or {})}
        self._client = None  # built lazily on first use

    def _get_client(self):
        """Build the MinerUClient on first use (lazy import keeps registration
        cheap on boxes without `mineru_vl_utils` installed)."""
        if self._client is None:
            from mineru_vl_utils import MinerUClient

            self._client = MinerUClient(
                backend=self.backend,
                model_path=self.model_path,
                handle_equation_block=self.handle_equation_block,
            )
        return self._client

    @staticmethod
    def _load_images(image_dir: str | Path) -> tuple[list[str], list[Image.Image]]:
        """Recursively collect supported images under `image_dir`, returning
        parallel lists of absolute paths and RGB PIL images."""
        image_paths: list[str] = []
        for root, _dirs, files in os.walk(image_dir):
            for file in files:
                if os.path.splitext(file.lower())[1] in SUPPORTED_EXTENSIONS:
                    image_paths.append(os.path.abspath(os.path.join(root, file)))
        image_paths.sort()
        print(f"found {len(image_paths)} image files.")

        images: list[Image.Image] = []
        kept_paths: list[str] = []
        for path in image_paths:
            try:
                images.append(Image.open(path).convert("RGB"))
                kept_paths.append(path)
            except Exception as exc:  # unreadable / corrupt image
                print(f"cannot load {path}: {exc}")
        print(f"successfully loaded {len(images)} images")
        return kept_paths, images

    @staticmethod
    def _blocks_to_markdown(blocks: list[dict]) -> str:
        """Join each block's non-empty `content` with blank lines, matching the
        reference script's Markdown assembly."""
        parts = [b["content"] for b in blocks if b.get("content")]
        return "\n\n".join(parts)

    def to_markdown(
        self,
        pdf_dir: str | Path,
        image_dir: str | Path,
        out_md_dir: str | Path,
        out_json: str | Path | None = None,
        skip_existing: bool = False,
    ) -> dict[str, list[dict]]:
        """Extract every image under `image_dir` and write `<stem>.md` files into
        `out_md_dir`. When `out_json` is given, also dump the raw per-image block
        lists (OmniDocBench prediction-JSON shape, keyed by image path).

        Returns the in-memory `{image_path: blocks}` mapping.
        """
        image_paths, images = self._load_images(image_dir)
        extracted = self._get_client().batch_two_step_extract(images)
        result_dict = dict(zip(image_paths, extracted))

        if out_json is not None:
            os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=4)

        os.makedirs(out_md_dir, exist_ok=True)
        for img_path, blocks in tqdm(result_dict.items(), desc="writing markdown"):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(out_md_dir, f"{stem}.md")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(self._blocks_to_markdown(blocks))

        return result_dict

    # --- evaluate/runner.py::run_vlm contract ------------------------------

    def _block_to_region(self, block: dict) -> VLMRegion:
        """Map one MinerU block to a labelled VLMRegion (text-only; the runner's
        `to_markdown` renders heading/list markup from the label)."""
        label = self.label_map.get(block.get("type", "text"), "Paragraph")
        content = block.get("content") or ""
        if isinstance(content, list):
            content = " ".join(str(c) for c in content if c)
        return VLMRegion(label=label, text="" if label == "Figure" else content)

    def parse_pdf(self, pdf_path: str | Path) -> GlasanaDocument:
        """Render `pdf_path` to page images, run MinerU over them, and assemble a
        GlasanaDocument. This is the entry point `run_vlm` calls; it then renders
        the doc with `to_markdown` / `to_markdown_pages` for OmniDocBench."""
        from rare.parse.pdf import render_pages

        pdf_path = Path(pdf_path)
        images = render_pages(pdf_path, dpi=self.dpi)
        # One block-list per page image, in input order.
        results = self._get_client().batch_two_step_extract(images)

        pages = [
            VLMPage(
                page_no=page_no,
                width=image.size[0],
                height=image.size[1],
                regions=[self._block_to_region(b) for b in (blocks or [])],
            )
            for page_no, (image, blocks) in enumerate(zip(images, results))
        ]
        return assemble_document(VLMDocument(pages=pages), pdf_path.stem)