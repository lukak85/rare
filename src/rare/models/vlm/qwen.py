from __future__ import annotations

import base64
import os
from pathlib import Path

from openai import OpenAI, APIConnectionError
from tqdm import tqdm

from rare.models.registry import register
from rare.models.vlm.prompts import OMNIDOCBENCH_PROMPT

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".webp": "image/webp",
}


@register("vlm", "qwen")
class QwenParseBackend:

    def __init__(self, config: dict | None = None, base_url: str = None, api_key: str = None, model_name: str = None):
        self._client = None  # built lazily on first use
        self.base_url = base_url if base_url is not None else "http://localhost:8000/v1"
        self.api_key = api_key if api_key is not None else "EMPTY"
        self.model_name = model_name or config.get("model_name")
        self.timeout = config.get("timeout", 600)

    def _get_client(self):
        if self._client is None:
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
        return self._client

    def encode_image(self, image_path):
        """
        Encode the image file to base64 string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """Models often wrap their answer in ```markdown ... ```; unwrap it."""
        stripped = text.strip()
        if not stripped.startswith("```"):
            return stripped
        lines = stripped.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
        return stripped


    def process_image(self, client, image_file: Path, image_dir: Path, result_dir, model_name):
        image_path = Path(os.path.join(image_dir, image_file))
        base64_image = self.encode_image(image_path)
        data_url = f"data:image/jpeg;base64,{base64_image}"
        # from urllib.parse import quote
        # encoded = quote(image_file, safe='')
        # data_url = f"https://huggingface.co/datasets/opendatalab/OmniDocBench/resolve/main/images/{encoded}"
        # print(data_url)

        response = client.chat.completions.create(
            model=model_name,
            messages=[{
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
            }],
            stream=True,
            timeout=self.timeout,
            extra_body={
                "repetition_penalty": 1.08,   # 1.05–1.15; logit penalty, works with greedy
            },
        )

        # result = ""

        chunks: list[str] = []
        finish_reason = None
        for chunk in response:
            """
            # print(chunk)
            # if chunk.choices[0].delta.type == "thought":
            #     continue
            if chunk.choices[0].finish_reason is not None:
                break
            result += chunk.choices[0].delta.content
            """
            if not chunk.choices:  # usage-only chunk when stream_options is set
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            # BUGFIX: delta.content is None on the role chunk and on every
            # reasoning chunk of a thinking model -> `result += None` blew up.
            content = getattr(delta, "content", None)
            if content:
                chunks.append(content)
            # BUGFIX: append *before* breaking, so the final chunk isn't dropped.
            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason
                break

        result = self._strip_code_fence("".join(chunks))

        rel = image_path.relative_to(image_dir)
        output_path = result_dir / rel.with_suffix(".md")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_path.write_text(result, encoding="utf-8")

        return f"[ok] {rel} -> {output_path.name} ({len(result)} chars)"

    def to_markdown(
        self,
        pdf_dir: str | Path,
        image_dir: str | Path,
        out_md_dir: str | Path,
        skip_existing: bool = True,
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
            result = self.process_image(self._get_client(), Path(image_file), Path(image_dir), out_md_dir, self.model_name)

        return out_md_dir
