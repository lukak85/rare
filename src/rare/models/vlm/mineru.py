# Adapted from OmniDocBench (https://github.com/opendatalab/OmniDocBench),
# tools/model_infer/MinerU2.5_img2md.py. Copyright (c) 2024 OpenDataLab and the
# OmniDocBench authors. Licensed under the Apache License, Version 2.0; see
# licenses/LICENSE-OMNIDOCBENCH and the NOTICE file.
from __future__ import annotations

import os
import json
from PIL import Image
from tqdm import tqdm
from mineru_vl_utils import MinerUClient

from rare.models.registry import register


@register("vlm", "mineru")
class MinerUBackend:
    """Calls the `mineru` CLI and converts its output to a GlasanaDocument."""

    name = "mineru"

    def __init__(self, config: dict | None = None):

        MODEL_PATH = "opendatalab/MinerU2.5-2509-1.2B"
        IMG_DIR = "/storage/lukakuzman/datasets/glasbena_mladina/images"
        OUT_JSON = "/home/lukakuzman/rare/outputs/omnidocbench/json"
        OUT_MD_DIR = "/home/lukakuzman/rare/outputs/omnidocbench/md"
        os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

        self.client = MinerUClient(
            backend="vllm-engine",
            model_path=MODEL_PATH,
            handle_equation_block=False
        )


    def main(self):
        extracted_blocks_list = self.client.batch_two_step_extract(pil_images)
        result_dict = {}
        for img_path, result_json in zip(image_path_list, extracted_blocks_list):
            result_dict[img_path] = result_json
        with open(OUT_JSON, "w") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=4)


    def load_images(self, ):
        self.SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        image_path_list = []
        for root, dirs, files in os.walk(IMG_DIR):
            for file in files:
                if os.path.splitext(file.lower())[1] in SUPPORTED_EXTENSIONS:
                    img_path = os.path.abspath(os.path.join(root, file))
                    image_path_list.append(img_path)
                else:
                    print(f"skip: {file}")
        print(f"found {len(image_path_list)} image files.")

        pil_images = []
        for image_path in image_path_list:
            try:
                image = Image.open(image_path).convert("RGB")
                pil_images.append(image)
                print(f"successfully load: {os.path.basename(image_path)}")
            except Exception as e:
                print(f"cannot load {image_path}: {e}")
                continue
        print(f"successfully load: {len(pil_images)} images")


    def detect(self, pdf_path):
        self.main()
        # json to markdown
        os.makedirs("/home/lukakuzman/rare/outputs/omnidocbench", exist_ok=True)
        ann_dict = json.load(open(OUT_JSON, "r"))
        for img_path, result_list in tqdm(ann_dict.items()):
            # 获取图像相对路径
            filename = os.path.basename(img_path)[:-4] + ".md"
            # 构建输出路径
            out_path = os.path.join(OUT_MD_DIR, filename)
            os.makedirs(OUT_MD_DIR, exist_ok=True)
            content_list = []
            for bbox_info in result_list:
                type = bbox_info["type"]
                content = bbox_info["content"]
                if content:
                    content_list.append(content)
            md_result = "\n\n".join(content_list)
            # 保存结果到对应子目录
            with open(out_path, "w", encoding="utf-8") as f:
                # 直接写入字符串,不使用json.dump
                f.write(md_result)