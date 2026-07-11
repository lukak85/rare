from __future__ import annotations

import base64
import os
from pathlib import Path

import transformers.models.auto.processing_auto   # real error surfaces here
from transformers import AutoModelForImageTextToText, AutoProcessor
from tqdm import tqdm

from rare.models.registry import register
from rare.models.vlm.prompts import OMNIDOCBENCH_PROMPT

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


@register("vlm", "gemma")
class GemmaBackend:

    def __init__(self, config: dict | None = None, base_url: str = None, api_key: str = None, cuda_device: int = 3):
        self.model_name = None

        self._processor = None
        self._model = None
        
        self.pretrained_model_name = "google/gemma-4-E2B-it"

        import torch
        self.cuda_device = torch.device(f"cuda:{cuda_device}")

    def _get_model(self):
        if self._model is None:
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.pretrained_model_name,
                dtype="auto",
                attn_implementation="sdpa"
            ).to(self.cuda_device).eval()
        return self._model

    def _get_processor(self):
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(
                "google/gemma-4-E2B-it",
                padding_side="left"
            )
        return self._processor


    @staticmethod
    def _load_image_paths(image_dir: str | Path) -> tuple[list[str]]:
        """Recursively collect supported images under `image_dir`, returning
        parallel lists of absolute paths and RGB PIL images."""
        image_paths: list[str] = []
        for root, _dirs, files in os.walk(image_dir):
            for file in files:
                if os.path.splitext(file.lower())[1] in SUPPORTED_EXTENSIONS:
                    image_paths.append(os.path.abspath(os.path.join(root, file)))
        image_paths.sort()
        print(f"found {len(image_paths)} image files.")
        return image_paths

    def encode_image(self, image_path):
        """
        Encode the image file to base64 string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def process_image(self, image_file, image_dir, result_dir, model_name):
        image_path = Path(os.path.join(image_dir, image_file))
        base64_image = self.encode_image(image_path)
        data_url = f"data:image/jpeg;base64,{base64_image}"

        messages = [{
            'role':'user',
            'content': [
                {
                    'type': 'text',
                    'text': OMNIDOCBENCH_PROMPT,
                },
                {
                    'type': 'image_url',
                    'image_url': {'url': data_url},
                }
            ],
        }]

        inputs = self._get_processor().apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.cuda_device)
        input_len = inputs["input_ids"].shape[-1]

        output = self._get_model().generate(**inputs, max_new_tokens=4096)

        rel = image_path.relative_to(image_dir)
        output_path = result_dir / rel.with_suffix(".md")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_path.write_text(self._get_processor().decode(output[0][input_len:], skip_special_tokens=True), encoding="utf-8")

        return f"[ok] {rel} -> {output_path.name} ({len(output)} chars)"

    def to_markdown(
        self,
        pdf_dir: str | Path,
        image_dir: str | Path,
        out_md_dir: str | Path,
        skip_existing: bool = False,
    ) -> str | Path:
        image_paths = self._load_image_paths(image_dir)

        image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png", ".jpeg"))]

        existing_files = []
        new_files = []
        if skip_existing:
            for image_file in image_files:
                output_path = os.path.join(out_md_dir, image_file + ".md")
                if os.path.exists(output_path):
                    existing_files.append(image_file)
                else:
                    new_files.append(image_file)
        else:
            new_files = image_files

        for image_file in tqdm(new_files):
            result = self.process_image(Path(image_file), Path(image_dir), out_md_dir, self.model_name)

        return out_md_dir
