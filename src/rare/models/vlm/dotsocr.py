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
import json
from pathlib import Path
from multiprocessing.pool import ThreadPool

from tqdm import tqdm

from dots_ocr.model.inference import inference_with_vllm
from dots_ocr.utils.consts import image_extensions, MIN_PIXELS, MAX_PIXELS
from dots_ocr.utils.image_utils import get_image_by_fitz_doc, fetch_image, smart_resize
from dots_ocr.utils.doc_utils import fitz_doc_to_image, load_images_from_pdf
from dots_ocr.utils.prompts import dict_promptmode_to_prompt
from dots_ocr.utils.layout_utils import post_process_output, draw_layout_on_image, pre_process_bboxes
from dots_ocr.utils.format_transformer import layoutjson2md

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

    def get_prompt(self, prompt_mode, bbox=None, origin_image=None, image=None, min_pixels=None, max_pixels=None):
        prompt = dict_promptmode_to_prompt[prompt_mode]
        if prompt_mode == 'prompt_grounding_ocr':
            assert bbox is not None
            bboxes = [bbox]
            bbox = pre_process_bboxes(origin_image, bboxes, input_width=image.width, input_height=image.height,
                                      min_pixels=min_pixels, max_pixels=max_pixels)[0]
            prompt = prompt + str(bbox)
        return prompt

    def _parse_single_image(
            self,
            origin_image,
            prompt_mode,
            save_dir,
            save_name,
            source="image",
            page_idx=0,
            bbox=None,
            fitz_preprocess=False,
    ):
        min_pixels, max_pixels = self.min_pixels, self.max_pixels
        if prompt_mode == "prompt_grounding_ocr":
            min_pixels = min_pixels or MIN_PIXELS
            max_pixels = max_pixels or MAX_PIXELS
        if min_pixels is not None: assert min_pixels >= MIN_PIXELS, f"min_pixels should >= {MIN_PIXELS}"
        if max_pixels is not None: assert max_pixels <= MAX_PIXELS, f"max_pixels should <= {MAX_PIXELS}"

        if source == 'image' and fitz_preprocess:
            image = get_image_by_fitz_doc(origin_image, target_dpi=self.dpi)
            image = fetch_image(image, min_pixels=min_pixels, max_pixels=max_pixels)
        else:
            image = fetch_image(origin_image, min_pixels=min_pixels, max_pixels=max_pixels)
        input_height, input_width = smart_resize(image.height, image.width)
        prompt = self.get_prompt(prompt_mode, bbox, origin_image, image, min_pixels=min_pixels, max_pixels=max_pixels)
        if self.use_hf:
            response = self._inference_with_hf(image, prompt)
        else:
            response = self._inference_with_vllm(image, prompt)
        result = {'page_no': page_idx,
                  "input_height": input_height,
                  "input_width": input_width
                  }
        if source == 'pdf':
            save_name = f"{save_name}_page_{page_idx}"
        if prompt_mode in ['prompt_layout_all_en', 'prompt_layout_only_en', 'prompt_grounding_ocr']:
            cells, filtered = post_process_output(
                response,
                prompt_mode,
                origin_image,
                image,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            if filtered and prompt_mode != 'prompt_layout_only_en':
                json_file_path = os.path.join(save_dir, f"{save_name}.json")
                with open(json_file_path, 'w', encoding="utf-8") as w:
                    json.dump(response, w, ensure_ascii=False)

                image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
                origin_image.save(image_layout_path)
                result.update({
                    'layout_info_path': json_file_path,
                    'layout_image_path': image_layout_path,
                })

                md_file_path = os.path.join(save_dir, f"{save_name}.md")
                with open(md_file_path, "w", encoding="utf-8") as md_file:
                    md_file.write(cells)
                result.update({
                    'md_content_path': md_file_path
                })
                result.update({
                    'filtered': True
                })
            else:
                try:
                    image_with_layout = draw_layout_on_image(origin_image, cells)
                except Exception as e:
                    print(f"Error drawing layout on image: {e}")
                    image_with_layout = origin_image

                json_file_path = os.path.join(save_dir, f"{save_name}.json")
                with open(json_file_path, 'w', encoding="utf-8") as w:
                    json.dump(cells, w, ensure_ascii=False)

                image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
                image_with_layout.save(image_layout_path)
                result.update({
                    'layout_info_path': json_file_path,
                    'layout_image_path': image_layout_path,
                })
                if prompt_mode != "prompt_layout_only_en":
                    md_content = layoutjson2md(origin_image, cells, text_key='text')
                    md_content_no_hf = layoutjson2md(origin_image, cells, text_key='text', no_page_hf=True)
                    md_file_path = os.path.join(save_dir, f"{save_name}.md")
                    with open(md_file_path, "w", encoding="utf-8") as md_file:
                        md_file.write(md_content)
                    md_nohf_file_path = os.path.join(save_dir, f"{save_name}_nohf.md")
                    with open(md_nohf_file_path, "w", encoding="utf-8") as md_file:
                        md_file.write(md_content_no_hf)
                    result.update({
                        'md_content_path': md_file_path,
                        'md_content_nohf_path': md_nohf_file_path,
                    })
        else:
            image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
            origin_image.save(image_layout_path)
            result.update({
                'layout_image_path': image_layout_path,
            })

            md_content = response
            md_file_path = os.path.join(save_dir, f"{save_name}.md")
            with open(md_file_path, "w", encoding="utf-8") as md_file:
                md_file.write(md_content)
            result.update({
                'md_content_path': md_file_path,
            })

        return result

    def parse_image(self, input_path, filename, prompt_mode, save_dir, bbox=None, fitz_preprocess=False):
        origin_image = fetch_image(input_path)
        result = self._parse_single_image(origin_image, prompt_mode, save_dir, filename, source="image", bbox=bbox,
                                          fitz_preprocess=fitz_preprocess)
        result['file_path'] = input_path
        return [result]

    def parse_pdf(self, input_path, filename, prompt_mode, save_dir):
        print(f"loading pdf: {input_path}")
        images_origin = load_images_from_pdf(input_path, dpi=self.dpi)
        total_pages = len(images_origin)
        tasks = [
            {
                "origin_image": image,
                "prompt_mode": prompt_mode,
                "save_dir": save_dir,
                "save_name": filename,
                "source": "pdf",
                "page_idx": i,
            } for i, image in enumerate(images_origin)
        ]

        def _execute_task(task_args):
            return self._parse_single_image(**task_args)

        if self.use_hf:
            num_thread = 1
        else:
            num_thread = min(total_pages, self.num_thread)
        print(f"Parsing PDF with {total_pages} pages using {num_thread} threads...")

        results = []
        with ThreadPool(num_thread) as pool:
            with tqdm(total=total_pages, desc="Processing PDF pages") as pbar:
                for result in pool.imap_unordered(_execute_task, tasks):
                    results.append(result)
                    pbar.update(1)

        results.sort(key=lambda x: x["page_no"])
        for i in range(len(results)):
            results[i]['file_path'] = input_path
        return results

    def parse_file(self,
                   input_path,
                   output_dir="",
                   prompt_mode="prompt_layout_all_en",
                   bbox=None,
                   fitz_preprocess=False
                   ):
        output_dir = output_dir or self.output_dir
        output_dir = os.path.abspath(output_dir)
        filename, file_ext = os.path.splitext(os.path.basename(input_path))
        save_dir = os.path.join(output_dir, filename)
        os.makedirs(save_dir, exist_ok=True)

        if file_ext == '.pdf':
            results = self.parse_pdf(input_path, filename, prompt_mode, save_dir)
        elif file_ext.lower() in image_extensions:
            results = self.parse_image(input_path, filename, prompt_mode, save_dir, bbox=bbox,
                                       fitz_preprocess=fitz_preprocess)
        else:
            raise ValueError(
                f"file extension {file_ext} not supported, supported extensions are {image_extensions} and pdf")

        print(f"Parsing finished, results saving to {save_dir}")
        jsonl_path = os.path.join(output_dir, filename + '.jsonl')
        with open(jsonl_path, 'w', encoding="utf-8") as w:
            for result in results:
                w.write(json.dumps(result, ensure_ascii=False) + '\n')

        return results

    def _collect_files(input_dir, recursive=False):
        """收集目录下所有支持的 PDF/图片文件。"""
        input_path = Path(input_dir)
        if not input_path.is_dir():
            return []
        supported = {".pdf", *{e.lower() for e in image_extensions}}
        if recursive:
            files = [p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in supported]
        else:
            files = [p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in supported]
        return sorted(files)

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
        if image_dir.is_file():
            # 单文件：保持原有逻辑
            result = self.parse_file(
                str(image_dir),
                prompt_mode="prompt_layout_all_en",
                bbox=('x1', 'y1', 'x2', 'y2'),
                fitz_preprocess=True,
            )
        elif image_dir.is_dir():
            files = self._collect_files(image_dir, recursive=True)
            if not files:
                print(f"目录下未找到支持的 PDF/图片: {image_dir}")
                return
            print(f"在 {image_dir} 下找到 {len(files)} 个文件，输出目录: {out_md_dir}")

            to_process = []
            for f in files:
                stem = f.stem
                save_dir = os.path.join(out_md_dir, stem)
                if skip_existing and os.path.isdir(save_dir) and os.path.isfile(
                        os.path.join(out_md_dir, stem + ".jsonl")):
                    continue
                to_process.append(f)

            skipped = len(files) - len(to_process)
            if skipped:
                print(f"跳过 {skipped} 个(输出已存在)，待处理 {len(to_process)} 个")
            if not to_process:
                print("全部已有输出，无需处理")
                return

            for fp in tqdm(to_process, desc="Processing files"):
                try:
                    self.parse_file(
                        str(fp),
                        output_dir=out_md_dir,
                        prompt_mode="prompt_layout_all_en",
                        bbox=('x1', 'y1', 'x2', 'y2'),
                        fitz_preprocess=True,
                    )
                except Exception as e:
                    tqdm.write(f"失败 {fp.name}: {e}")

        from .helpers.normalize_pred import from_folders

        text_pages = from_folders(out_md_dir)

        for stem, content in sorted(text_pages.items()):
            (Path(out_md_dir) / f"{stem}.md").write_text(content)

        # Remove all folders
        for item in Path(out_md_dir).iterdir():
            if item.is_dir():
                for subitem in item.iterdir():
                    if subitem.is_file():
                        subitem.unlink()
                item.rmdir()

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
