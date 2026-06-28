from __future__ import annotations

import base64
import os
from pathlib import Path

from openai import OpenAI, APIConnectionError

from rare.doc.schema import GlasanaDocument
from rare.models.registry import register
from rare.models.vlm._assembler import assemble_document
from rare.models.vlm._vlm_schema import VLMDocument, VLMPage, VLMRegion
from rare.models.vlm.prompts import OMNIDOCBENCH_PROMPT

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


@register("vlm", "qwen")
class QwenParseBackend:

    def __init__(self, config: dict | None = None, base_url: str = None, api_key: str = None):
        self._client = None  # built lazily on first use
        self.base_url = base_url if base_url is not None else "http://localhost:8000/v1"
        self.api_key = api_key if api_key is not None else "EMPTY"
        self.model_name = None

    def _get_client(self):
        if self._client is None:
            client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
        return client

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

    def process_image(self, client, image_file, image_dir, result_dir, model_name):
        """
        处理单个图片文件
        """
        try:
            # 检查输出文件是否已存在
            output_path = os.path.join(result_dir, image_file.split(".")[0] + ".md")
            if os.path.exists(output_path):
                return f"⏭ 跳过已存在: {image_file}"

            image_path = os.path.join(image_dir, image_file)
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
                timeout=10000,
            )

            result = ""
            for chunk in response:
                # print(chunk)
                # if chunk.choices[0].delta.type == "thought":
                #     continue
                if chunk.choices[0].finish_reason is not None:
                    break
                result += chunk.choices[0].delta.content

            with open(output_path, "w", encoding='utf-8') as f:
                print(result, file=f)

            return f"✓ 成功处理: {image_file}"
        except APIConnectionError as e:
            return f"✗ 连接超时: {image_file}, 错误: {str(e)}"
        except Exception as e:
            # 保存错误信息到文件
            # output_path = os.path.join(result_dir, image_file + ".md")
            # with open(output_path, "w", encoding='utf-8') as f:
            #     print(f"处理错误: {str(e)}", file=f)
            return f"✗ 处理失败: {image_file}, 错误: {str(e)}"

    def to_markdown(
        self,
        pdf_dir: str | Path,
        image_dir: str | Path,
        out_md_dir: str | Path,
        skip_existing: bool = False,
    ) -> str | Path:
        image_paths = self._load_image_paths(image_dir)

        image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png", ".jpeg"))]

        for image_file in image_files:
            result = self.process_image(self._get_client(), image_file, image_dir, out_md_dir, self.model_name)

        return out_md_dir
