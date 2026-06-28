from __future__ import annotations

import os
from pathlib import Path
import shutil

from tqdm import tqdm

from rare.doc.schema import GlasanaDocument
from rare.models.registry import register
from rare.models.vlm.prompts import OMNIDOCBENCH_PROMPT
from rare.models.vlm._assembler import assemble_document
from rare.models.vlm._vlm_schema import VLMDocument, VLMPage, VLMRegion


@register("vlm", "deepseek-ocr")
class DeepSeekOCRBackend:
    """Runs DeepSeek-OCR over PDFs and exposes its Markdown for evaluation."""

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'

        self._converter = None
        self._tokenizer = None

        self.model_name = 'deepseek-ai/DeepSeek-OCR-2'

    def _get_converter(self):
        """Build the DocumentConverter on first use (lazy import keeps
        registration cheap on boxes without `docling` installed)."""
        import torch
        from transformers import AutoModel

        if self._converter is None:
            self._get_tokenizer()
            model = AutoModel.from_pretrained(self.model_name, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True)
            self._converter = model.eval().cuda().to(torch.bfloat16)
        return self._converter

    def _get_tokenizer(self):
        """Build the DocumentConverter on first use (lazy import keeps
        registration cheap on boxes without `docling` installed)."""
        from transformers import AutoTokenizer

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        return self._tokenizer

    def to_markdown(
        self,
        pdf_dir: str | Path,
        image_dir: str | Path,
        out_md_dir: str | Path,
        skip_existing: bool = False,
    ) -> str | Path:
        prompt = "<image>\n<|grounding|>Convert the document to markdown. "
        for img_name in tqdm(os.listdir(image_dir)):
            out_img_dir = Path(out_md_dir) / img_name.split(".")[0]
            os.mkdir(out_img_dir)
            self._get_converter().infer(self._get_tokenizer(), prompt=prompt,
                                              image_file=os.path.join(image_dir, img_name), output_path = out_img_dir,
                                              base_size = 1024, image_size = 768, crop_mode=True, save_results = True)
            # Rename to {image_name.md} and move it to a joint folder
            shutil.move(Path(out_img_dir) / "result.mmd", f"{Path(out_md_dir) / img_name.split('.')[0]}.md")
        return out_md_dir
