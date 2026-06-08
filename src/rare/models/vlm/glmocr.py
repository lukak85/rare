# Adapted from OmniDocBench (https://github.com/opendatalab/OmniDocBench),
# tools/model_infer/GLMOCR_img2md.py. Copyright (c) 2024 OpenDataLab and the
# OmniDocBench authors. Licensed under the Apache License, Version 2.0; see
# licenses/LICENSE-OMNIDOCBENCH and the NOTICE file.
import os
import json
from pathlib import Path
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
import argparse

from dots_ocr.model.inference import inference_with_vllm
from dots_ocr.utils.consts import image_extensions, MIN_PIXELS, MAX_PIXELS
from dots_ocr.utils.image_utils import get_image_by_fitz_doc, fetch_image, smart_resize
from dots_ocr.utils.doc_utils import fitz_doc_to_image, load_images_from_pdf
from dots_ocr.utils.prompts import dict_promptmode_to_prompt
from dots_ocr.utils.layout_utils import post_process_output, draw_layout_on_image, pre_process_bboxes
from dots_ocr.utils.format_transformer import layoutjson2md

from rare.models.registry import register


@register("vlm", "glm-ocr")
class GLMOCRBackend:
    """
    parse image or pdf file
    """

    # GLM-OCR's official single-pass prompt (from the vLLM quickstart / recipe).
    # The model emits Markdown-structured output for a full-page document image.
    OCR_PROMPT = "Text Recognition:"

    def process_images_to_markdown(
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
                                {"type": "text", "text": OCR_PROMPT},
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

    if __name__ == "__main__":
        IMAGE_FOLDER = "/storage/lukakuzman/datasets/glasbena_mladina/images"  # 请替换为你的图片文件夹路径
        MARKDOWN_FOLDER = "outputs/omnidocbench/glmocr"

        output_folder = process_images_to_markdown(
            image_folder_path=IMAGE_FOLDER,
            markdown_folder_path=MARKDOWN_FOLDER,
            base_url="http://localhost:8080/v1",  # match your `vllm serve --port`
            model="zai-org/GLM-OCR",  # must match `curl .../v1/models`
        )
        print(f"Markdown output directory: {output_folder}")
