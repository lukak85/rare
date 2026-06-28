from __future__ import annotations

import os
from pathlib import Path

from tqdm import tqdm
from youtu_hf_parser import YoutuOCRParserHF

from rare.doc.schema import GlasanaDocument
from rare.models.registry import register
from rare.models.vlm._assembler import assemble_document
from rare.models.vlm._vlm_schema import VLMDocument, VLMPage, VLMRegion

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


@register("vlm", "youtu")
class YoutuBackend:

    def __init__(self, config: dict | None = None, model_path=None, angle_correct_model_path=None):
        self._converter = None  # built lazily on first use
        self.model_path = model_path
        self.angle_correct_model_path = angle_correct_model_path
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # Force everything on the same device

    def _get_converter(self):
        if self._converter is None:
            self._converter = YoutuOCRParserHF(
                model_path=self.model_path,                    # Path to downloaded model weights
                enable_angle_correct=True,                # Set to False to disable angle correction
                angle_correct_model_path=self.angle_correct_model_path  # If None, model will auto-download to default path; if custom path, manually download https://github.com/TencentCloudADP/youtu-parsing/releases/download/v1.0.0/model.pth to specified location
            )
        return self._converter

    @staticmethod
    def _load_image_paths(image_dir: str | Path) -> list[str]:
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

    def to_markdown(
        self,
        pdf_dir: str | Path,
        image_dir: str | Path,
        out_md_dir: str | Path,
        skip_existing: bool = False,
    ) -> str | Path:
        image_paths = self._load_image_paths(image_dir)

        for image_path in tqdm(image_paths):
            self._get_converter().parse_file(
                input_path=image_path,     # Input document path
                output_dir=out_md_dir      # Output directory for results
            )

        return out_md_dir
