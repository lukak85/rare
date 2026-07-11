from __future__ import annotations

import base64
import os
from pathlib import Path

from openai import OpenAI, APIConnectionError
from tqdm import tqdm

from rare.models.registry import register

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


@register("vlm", "nemotron")
class NemotronParseBackend:
    """Runs Docling over PDFs and exposes its Markdown for evaluation."""

    name = "docling"
    # parse_pdf wraps Docling's own per-page Markdown in one region per page, so
    # the runner scores it verbatim (raw join) rather than re-applying markup.
    raw_markdown = True

    def __init__(self, config: dict | None = None, base_url: str = None, api_key: str = None):
        self._converter = None  # built lazily on first use
        self.base_url = base_url if base_url is not None else "http://localhost:8000/v1"
        self.api_key = api_key if api_key is not None else "EMPTY"
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
        return self._client

    def process_image(self, client, image_file: Path, image_dir: Path, result_dir):
        with open(Path(image_dir) / Path(image_file), "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        prompt_text = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"

        resp = client.chat.completions.create(
            model="nvidia/NVIDIA-Nemotron-Parse-v1.2",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_text,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=7000,
            temperature=0.0,
            extra_body={
                "repetition_penalty": 1.1,
                "top_k": 1,
                "skip_special_tokens": False,
            },
        )

        image_path = Path(os.path.join(image_dir, image_file))

        rel = image_path.relative_to(image_dir)
        output_path = result_dir / rel.with_suffix(".md")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_path.write_text(self.postprocess_text(resp.choices[0].message.content), encoding="utf-8")

    def postprocess_text(self, generated_text: str) -> str:
        from .helpers.postprocess import extract_classes_bboxes, transform_bbox_to_original, postprocess_text

        classes, bboxes, texts = extract_classes_bboxes(generated_text)

        # Specify output formats for postprocessing
        table_format = 'latex' # latex | HTML | markdown | json | csv
        text_format = 'markdown' # markdown | plain
        blank_text_in_figures = False # remove text inside 'Picture' class
        texts = [postprocess_text(text, cls = cls, table_format=table_format, text_format=text_format, blank_text_in_figures=blank_text_in_figures) for text, cls in zip(texts, classes)]

        str_text = self.to_string(texts)

        return str_text

    def to_string(self, texts: list):
        markdown_string = ""

        for text in texts:
            markdown_string += f"{text}\n\n"

        return markdown_string

    def to_markdown(
        self,
        pdf_dir: str | Path,
        image_dir: str | Path,
        out_md_dir: str | Path,
        skip_existing: bool = False,
    ) -> str | Path:
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
            result = self.process_image(self._get_client(), Path(image_file), Path(image_dir), out_md_dir)

        return out_md_dir
