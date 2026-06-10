# Adapted from OmniDocBench (https://github.com/opendatalab/OmniDocBench),
# tools/model_infer/GLMOCR_img2md.py. Copyright (c) 2024 OpenDataLab and the
# OmniDocBench authors. Licensed under the Apache License, Version 2.0; see
# licenses/LICENSE-OMNIDOCBENCH and the NOTICE file.
"""GLM-OCR backend.

GLM-OCR (https://github.com/zai-org/GLM-OCR) is a single-pass OCR VLM served via
vLLM's OpenAI-compatible `/v1/chat/completions` endpoint. Given a page image and
the `"Text Recognition:"` prompt it returns the page's Markdown directly, so we
treat it as a black box: wrap each page's Markdown in a single region. Two entry
points share one client:

- `to_markdown(pdf_dir, image_dir, out_md_dir, ...)` — faithful port of
  OmniDocBench's `GLMOCR_img2md.py`: a folder of page images -> one `<stem>.md`
  per image. Used with the standalone `normalize_pred.py` + `run.sh` path.
- `parse_pdf(pdf) -> GlasanaDocument` — the `evaluate/runner.py::run_vlm`
  contract. Renders the PDF to pages, runs GLM-OCR per page, and wraps each
  page's Markdown in one region. Because `raw_markdown = True`, the runner joins
  region text verbatim, reproducing GLM-OCR's own per-page Markdown.

Requires a running GLM-OCR vLLM server (set base_url/model via config)::

    vllm serve zai-org/GLM-OCR --port 8000   # on the inference box
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

from openai import OpenAI
from PIL import Image
import mimetypes

from openai import OpenAI

from rare.doc.schema import GlasanaDocument
from rare.models.registry import register
from rare.models.vlm._assembler import assemble_document
from rare.models.vlm._vlm_schema import VLMDocument, VLMPage, VLMRegion
from rare.parse.pdf import render_pages

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


@register("vlm", "glm-ocr")
class GLMOCRBackend:
    """Runs GLM-OCR over page images and exposes its Markdown for evaluation."""

    name = "glm-ocr"
    # parse_pdf wraps GLM-OCR's own per-page Markdown in one region per page, so
    # the runner scores it verbatim (raw join).
    raw_markdown = True

    # GLM-OCR's official single-pass prompt (from the vLLM quickstart / recipe).
    OCR_PROMPT = "Text Recognition:"

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        # vLLM OpenAI-compatible endpoint. `base_url` must end in /v1; `model`
        # must match `curl {base_url}/models` (case-sensitive). `api_key` is
        # ignored by local vLLM but must be a non-empty string.
        self.base_url = cfg.get("base_url", "http://localhost:8000/v1")
        self.model = cfg.get("model", "zai-org/GLM-OCR")
        self.api_key = cfg.get("api_key", "EMPTY")
        self.prompt = cfg.get("prompt", self.OCR_PROMPT)
        self.max_tokens = int(cfg.get("max_tokens", 8192))
        self.temperature = float(cfg.get("temperature", 0.0))  # deterministic
        self.dpi = int(cfg.get("dpi", 200))
        self.num_thread = int(cfg.get("num_thread", 16))
        self._client = None  # built lazily on first use

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    # --- inference ---------------------------------------------------------

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        """PIL image -> base64 PNG data URI (avoids needing vLLM's
        --allowed-local-media-path that file:// URLs would require)."""
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def _infer_image(self, image: Image.Image) -> str:
        """Run GLM-OCR on one PIL page image and return its Markdown."""
        response = self._get_client().chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": self._encode_image(image)}},
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""

    def _infer_images(self, images: list[Image.Image]) -> list[str]:
        """Markdown for each page image, in order. Threads across the vLLM
        server (it handles concurrent requests) for throughput."""
        n = min(len(images), self.num_thread)
        if n <= 1:
            return [self._infer_image(im) for im in images]
        from multiprocessing.pool import ThreadPool

        with ThreadPool(n) as pool:
            return list(pool.imap(self._infer_image, images))  # imap keeps order

    def process_images_to_markdown(
            self,
            image_folder_path,
            markdown_folder_path=None,
            base_url="http://localhost:8000/v1",
            model="zai-org/GLM-OCR",
            api_key="EMPTY",
            max_tokens=8192,
    ):
        """
        Convert every image in a folder to Markdown using a LOCALLY deployed
        GLM-OCR served via vLLM's OpenAI-compatible /v1/chat/completions endpoint.

        Args:
            image_folder_path:    folder containing the input images
            markdown_folder_path: output folder (defaults to <image_folder>/markdown_output)
            base_url:             vLLM server base URL, must end in /v1 and match --port
            model:                the model id EXACTLY as it appears in GET {base_url}/models
                                  (run `curl {base_url}/models` to confirm; case-sensitive,
                                  and if you served from a local path it is that full path)
            api_key:              any non-empty string; local vLLM ignores its value
            max_tokens:           cap on generated tokens; raise for dense full pages
        """
        # Point the standard OpenAI client at the local vLLM server.
        client = OpenAI(base_url=base_url, api_key=api_key)

        if markdown_folder_path is None:
            markdown_folder_path = os.path.join(image_folder_path, "markdown_output")
        os.makedirs(markdown_folder_path, exist_ok=True)

        supported_extensions = {
            ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff", ".tif",
        }

        for filename in sorted(os.listdir(image_folder_path)):
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext not in supported_extensions:
                continue

            image_path = os.path.join(image_folder_path, filename)
            md_filename = os.path.splitext(filename)[0] + ".md"
            md_path = os.path.join(markdown_folder_path, md_filename)

            # Resume support: skip images that were already parsed.
            if os.path.exists(md_path):
                print(f"Skipped: {filename} (Markdown already exists)")
                continue

            print(f"Processing: {filename}")
            try:
                # Read the image and encode it as a base64 data URI. This avoids
                # needing vLLM's --allowed-local-media-path flag (which file:// URLs
                # would require).
                with open(image_path, "rb") as image_file:
                    img_base64 = base64.b64encode(image_file.read()).decode("utf-8")

                mime_type, _ = mimetypes.guess_type(image_path)
                if mime_type is None:
                    fallback = {
                        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                        ".png": "image/png", ".gif": "image/gif",
                        ".bmp": "image/bmp", ".webp": "image/webp",
                        ".tiff": "image/tiff", ".tif": "image/tiff",
                    }
                    mime_type = fallback.get(file_ext, "image/jpeg")

                data_uri = f"data:{mime_type};base64,{img_base64}"

                # Replaces the cloud `client.layout_parsing.create(...)` call with the
                # OpenAI-compatible chat endpoint that vLLM actually serves.
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": data_uri}},
                                {"type": "text", "text": self.prompt},
                            ],
                        }
                    ],
                    max_tokens=max_tokens,
                    temperature=0.0,  # deterministic output for benchmarking
                )

                # vLLM returns chat-completion shape, not the cloud's `.md_results`.
                md_content = response.choices[0].message.content or ""
                print(md_content)

                with open(md_path, "w", encoding="utf-8") as md_file:
                    md_file.write(md_content)
                print(f"  Saved: {md_filename}")

                if response.usage is not None:
                    print(f"  Tokens used: {response.usage.total_tokens}")

            except Exception as e:
                print(f"  Failed {filename}: {e}")
                import traceback
                traceback.print_exc()

        print(f"\nDone. Markdown files saved in: {markdown_folder_path}")
        return markdown_folder_path

    # --- faithful reference port -------------------------------------------

    def to_markdown(
        self,
        pdf_dir: str | Path,
        image_dir: str | Path,
        out_md_dir: str | Path,
        skip_existing: bool = False,
    ) -> str | Path:
        """Convert every page image under `image_dir` and write `<stem>.md` into
        `out_md_dir` (one per image), matching `GLMOCR_img2md.py`. `pdf_dir` is
        unused (GLM-OCR scores per page image) but kept for signature parity."""
        """
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
        images = [Image.open(p) for p in todo]
        for path, md in zip(todo, tqdm(self._infer_images(images), desc="glm-ocr")):
            (Path(out_md_dir) / f"{path.stem}.md").write_text(md, encoding="utf-8")
        return out_md_dir
        """

        IMAGE_FOLDER = "/storage/lukakuzman/datasets/glasbena_mladina/images"  # 请替换为你的图片文件夹路径
        MARKDOWN_FOLDER = "outputs/omnidocbench/glmocr"

        output_folder = process_images_to_markdown(
            image_folder_path=image_dir,
            markdown_folder_path=out_md_dir,
            base_url="http://localhost:8080/v1",  # match your `vllm serve --port`
            model="zai-org/GLM-OCR",  # must match `curl .../v1/models`
        )
        return out_md_dir

    # --- evaluate/runner.py::run_vlm contract ------------------------------

    def parse_pdf(self, pdf_path: str | Path) -> GlasanaDocument:
        """Render `pdf_path` to pages, run GLM-OCR, and wrap each page's Markdown
        in a single region for the runner's verbatim (raw) scoring."""
        pdf_path = Path(pdf_path)
        images = render_pages(pdf_path, dpi=self.dpi)
        mds = self._infer_images(images)
        pages = [
            VLMPage(
                page_no=page_no,
                width=image.size[0],
                height=image.size[1],
                regions=[VLMRegion(label="Paragraph", text=md)],
            )
            for page_no, (image, md) in enumerate(zip(images, mds))
        ]
        return assemble_document(VLMDocument(pages=pages), pdf_path.stem)