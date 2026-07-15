# Adapted from OmniDocBench (https://github.com/opendatalab/OmniDocBench),
# tools/model_infer/PaddleOCR_img2md.py. Copyright (c) 2024 OpenDataLab and the
# OmniDocBench authors. Licensed under the Apache License, Version 2.0; see
# licenses/LICENSE-OMNIDOCBENCH and the NOTICE file.
"""PaddleOCR (PP-StructureV3) backend.

PP-StructureV3 (https://github.com/PaddlePaddle/PaddleOCR) is a local document
pipeline (layout + OCR + table). For each page it produces a result whose
`save_to_markdown` writes the page's Markdown. That formatter output is the
faithful prediction OmniDocBench compares, so — like Docling/dots.ocr — we treat
it as a black box: wrap each page's Markdown in a single region. Two entry points
share one pipeline:

- `to_markdown(pdf_dir, image_dir, out_md_dir, ...)` — faithful port of
  OmniDocBench's `PaddleOCR_img2md.py`: a folder of page images -> one
  `<stem>.md` per image. Used with the standalone `normalize_pred.py` + `run.sh`
  path.
- `parse_pdf(pdf) -> GlasanaDocument` — the `evaluate/runner.py::run_vlm`
  contract. PP-StructureV3 rasterizes the PDF itself (one result per page); we
  wrap each page's Markdown in one region. Because `raw_markdown = True`, the
  runner joins region text verbatim, reproducing PP-StructureV3's own Markdown.

Install on the inference box::

    pip install -U paddleocr paddlepaddle   # (or paddlepaddle-gpu)
"""

from __future__ import annotations

from pathlib import Path

from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from rare.models.registry import register
from rare.models.vlm.utils.dolphin import *

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif", ".webp"}


@register("vlm", "dolphin")
class DolphinBackend:

    name = "dolphin"
    raw_markdown = True

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self._pipeline_kwargs = dict(cfg.get("pipeline_kwargs") or {})
        # Predict-time flags; defaults mirror PaddleOCR_img2md.py.
        self.predict_flags = {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            **(cfg.get("predict_flags") or {}),
        }
        self._pipeline = None  # built lazily on first use

        self._processor = None
        self._model = None

    def _get_processor(self):
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(
                "ByteDance/Dolphin-v2",
                padding_side="left"
            )
        return self._processor

    def _get_model(self):
        if self._model is None:
            import torch
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "ByteDance/Dolphin-v2",
                # torch_dtype=torch.bfloat16,        # or "auto"
                # attn_implementation="sdpa",        # flash_attention_2 if installed
                device_map="cuda:0"
            )
        return self._model

    def chat(self, prompt, image):
        # Check if we're dealing with a batch
        is_batch = isinstance(image, list)

        if not is_batch:
            # Single image, wrap it in a list for consistent processing
            images = [image]
            prompts = [prompt]
        else:
            # Batch of images
            images = image
            prompts = prompt if isinstance(prompt, list) else [prompt] * len(images)

        assert len(images) == len(prompts)

        # preprocess all images
        processed_images = [resize_img(img) for img in images]
        # generate all messages
        all_messages = []
        for img, question in zip(processed_images, prompts):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img,
                        },
                        {"type": "text", "text": question}
                    ],
                }
            ]
            all_messages.append(messages)

        # prepare all texts
        texts = [
            self._get_processor().apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in all_messages
        ]

        # collect all image inputs
        all_image_inputs = []
        all_video_inputs = None
        for msgs in all_messages:
            image_inputs, video_inputs = process_vision_info(msgs)
            all_image_inputs.extend(image_inputs)

        # prepare model inputs
        inputs = self._get_processor()(
            text=texts,
            images=all_image_inputs if all_image_inputs else None,
            videos=all_video_inputs if all_video_inputs else None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._get_model().device)

        # inference
        generated_ids = self._get_model().generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
            temperature=None,
            repetition_penalty=1.08
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        results = self._get_processor().batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        # Return a single result for single image input
        if not is_batch:
            return results[0]
        return results

    def to_markdown(
        self,
        pdf_dir: str | Path,
        image_dir: str | Path,
        out_md_dir: str | Path,
        skip_existing: bool = False,
    ) -> str | Path:
        if image_dir is None and pdf_dir is not None:
            pdf_files = [f for f in os.listdir(image_dir) if f.endswith(".pdf")]
            for pdf_file in pdf_files:
                convert_pdf_to_images(pdf_file)

        image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png", ".jpeg"))]

        existing_files = []
        new_files = []
        if skip_existing:
            for image_file in image_files:
                output_path = os.path.join(out_md_dir, image_file.split(".")[0] + ".md")
                if os.path.exists(output_path):
                    existing_files.append(image_file)
                else:
                    new_files.append(image_file)
        else:
            new_files = image_files

        print(f"Existing files: {existing_files}")

        for image_file in tqdm(new_files):
            image = Image.open(Path(image_dir) / image_file)
            layout_output = self.chat("Parse the reading order of this document.", image)

            recognition_results = process_elements(layout_output, image, self, 4, out_md_dir, image_file.split(".")[0])

            # Create a dummy image path for save_outputs function
            json_path = save_outputs(recognition_results, image, image_file.split(".")[0], out_md_dir, post_process=False)

        return out_md_dir


def process_document(document_path, model, save_dir, max_batch_size=4, post_process=False):
    """Parse documents with two stages - Handles both images and PDFs"""
    file_ext = os.path.splitext(document_path)[1].lower()

    if file_ext == '.pdf':
        # Convert PDF to images
        images = convert_pdf_to_images(document_path)
        if not images:
            raise Exception(f"Failed to convert PDF {document_path} to images")

        all_results = []

        # Process each page
        for page_idx, pil_image in enumerate(images):
            print(f"Processing page {page_idx + 1}/{len(images)}")

            # Generate output name for this page
            base_name = os.path.splitext(os.path.basename(document_path))[0]
            page_name = f"{base_name}_page_{page_idx + 1:03d}"

            # Process this page (don't save individual page results)
            json_path, recognition_results = process_single_image(
                pil_image, model, save_dir, page_name, max_batch_size, save_individual=False, post_process=post_process
            )

            # Add page information to results
            page_results = {
                "page_number": page_idx + 1,
                "elements": recognition_results
            }
            all_results.append(page_results)

        # Save combined results for multi-page PDF
        combined_json_path = save_combined_pdf_results(all_results, document_path, save_dir, post_process=post_process)

        return combined_json_path, all_results

    else:
        # Process regular image file
        pil_image = Image.open(document_path).convert("RGB")
        base_name = os.path.splitext(os.path.basename(document_path))[0]
        return process_single_image(pil_image, model, save_dir, base_name, max_batch_size, post_process=post_process)


def process_single_image(image, model, save_dir, image_name, max_batch_size=None, save_individual=True, post_process=False):
    """Process a single image (either from file or converted from PDF page)

    Args:
        image: PIL Image object
        model: DOLPHIN model instance
        save_dir: Directory to save results
        image_name: Name for the output file
        max_batch_size: Maximum batch size for processing
        save_individual: Whether to save individual results (False for PDF pages)

    Returns:
        Tuple of (json_path, recognition_results)
    """
    # Stage 1: Page-level layout and reading order parsing
    layout_output = model.chat("Parse the reading order of this document.", image)
    # print(layout_output)

    # Stage 2: Element-level content parsing
    recognition_results = process_elements(layout_output, image, model, max_batch_size, save_dir, image_name)

    # Save outputs only if requested (skip for PDF pages)
    json_path = None
    if save_individual:
        # Create a dummy image path for save_outputs function
        json_path = save_outputs(recognition_results, image, image_name, save_dir, post_process=post_process)

    return json_path, recognition_results


def process_elements(layout_results, image, model, max_batch_size, save_dir=None, image_name=None):
    """Parse all document elements with parallel decoding"""
    layout_results_list = parse_layout_string(layout_results)
    if not layout_results_list or not (layout_results.startswith("[") and layout_results.endswith("]")):
        layout_results_list = [([0, 0, *image.size], 'distorted_page', [])]
    # Check for bbox overlap - if too many overlaps, treat as distorted page
    elif len(layout_results_list) > 1 and check_bbox_overlap(layout_results_list, image):
        print("Falling back to distorted_page mode due to high bbox overlap")
        layout_results_list = [([0, 0, *image.size], 'distorted_page', [])]

    tab_elements = []
    equ_elements = []
    code_elements = []
    text_elements = []
    figure_results = []
    reading_order = 0

    # Collect elements and group
    for bbox, label, tags in layout_results_list:
        try:
            if label == "distorted_page":
                x1, y1, x2, y2 = 0, 0, *image.size
                pil_crop = image
            else:
                # get coordinates in the original image
                x1, y1, x2, y2 = process_coordinates(bbox, image)
                # crop the image
                pil_crop = image.crop((x1, y1, x2, y2))

            if pil_crop.size[0] > 3 and pil_crop.size[1] > 3:
                if label == "fig" and False:
                    figure_filename = save_figure_to_local(pil_crop, save_dir, image_name, reading_order)
                    figure_results.append({
                        "label": label,
                        "text": f"![Figure](figures/{figure_filename})",
                        "figure_path": f"figures/{figure_filename}",
                        "bbox": [x1, y1, x2, y2],
                        "reading_order": reading_order,
                        "tags": tags,
                    })
                else:
                    # Prepare element information
                    element_info = {
                        "crop": pil_crop,
                        "label": label,
                        "bbox": [x1, y1, x2, y2],
                        "reading_order": reading_order,
                        "tags": tags,
                    }

                    if label == "tab":
                        tab_elements.append(element_info)
                    elif label == "equ":
                        equ_elements.append(element_info)
                    elif label == "code":
                        code_elements.append(element_info)
                    else:
                        text_elements.append(element_info)

            reading_order += 1

        except Exception as e:
            print(f"Error processing bbox with label {label}: {str(e)}")
            continue

    recognition_results = figure_results.copy()

    if tab_elements:
        results = process_element_batch(tab_elements, model, "Parse the table in the image.", max_batch_size)
        recognition_results.extend(results)

    if equ_elements:
        results = process_element_batch(equ_elements, model, "Read formula in the image.", max_batch_size)
        recognition_results.extend(results)

    if code_elements:
        results = process_element_batch(code_elements, model, "Read code in the image.", max_batch_size)
        recognition_results.extend(results)

    if text_elements:
        results = process_element_batch(text_elements, model, "Read text in the image.", max_batch_size)
        recognition_results.extend(results)

    recognition_results.sort(key=lambda x: x.get("reading_order", 0))

    return recognition_results


def process_element_batch(elements, model, prompt, max_batch_size=None):
    """Process elements of the same type in batches"""
    results = []

    # Determine batch size
    batch_size = len(elements)
    if max_batch_size is not None and max_batch_size > 0:
        batch_size = min(batch_size, max_batch_size)

    # Process in batches
    for i in range(0, len(elements), batch_size):
        batch_elements = elements[i:i+batch_size]
        crops_list = [elem["crop"] for elem in batch_elements]

        # Use the same prompt for all elements in the batch
        prompts_list = [prompt] * len(crops_list)

        # Batch inference
        batch_results = model.chat(prompts_list, crops_list)

        # Add results
        for j, result in enumerate(batch_results):
            elem = batch_elements[j]
            results.append({
                "label": elem["label"],
                "bbox": elem["bbox"],
                "text": result.strip(),
                "reading_order": elem["reading_order"],
                "tags": elem["tags"],
            })

    return results