from __future__ import annotations

import asyncio
import base64
import os
from pathlib import Path

import aiohttp

from rare.models.registry import register
from rare.models.vlm.prompts import omnidocbench_pdf_prompt


@register("vlm", "gemini")
class GeminiBackend:
    mode = "whole-pdf"

    def __init__(self, config: dict | None = None, ):
        api_key = config.get("api_key", None)
        base_url = config.get("base_url", None)
        model_name = config.get("model", "gemini-3.1-pro-preview")

        if api_key is None:
            raise ValueError("API key must be provided for GeminiBackend.")
        if base_url is None:
            raise ValueError("Base URL must be provided for GeminiBackend.")

        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name

        self.max_retries = 0  # set retry times
        self.request_sleep = 5  # set sleep time(seconds)

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def process_image(self, session, image_path, output_dir, retry_count=0):
        try:
            with open(image_path, "rb") as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')

            async with session.post(
                    url=f"{self.base_url}/chat/completions",
                    json={
                        "model": self.model_name,
                        "messages": [
                            {"role": "user", "content": omnidocbench_pdf_prompt()},
                            {"role": "user", "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                            ]}
                        ],
                    },
                    headers=self.headers
            ) as response:
                await asyncio.sleep(self.request_sleep)
                print(response)

                if response.status == 200:
                    result = await response.json()
                    print(result)
                    markdown_content = result['choices'][0]['message']['content']

                    if not markdown_content or markdown_content.strip() == "":
                        if retry_count < self.max_retries:
                            print(f"Received empty response, retrying ({retry_count + 1}/{self.max_retries}): {image_path}")
                            return await self.process_image(session, image_path, output_dir, retry_count + 1)
                        else:
                            print(
                                f"The maximum number of retries has been reached, and the process failed: {image_path}")
                            return False

                    markdown_content = clean_markdown(markdown_content)

                    file_name = os.path.basename(image_path)
                    base_name = ".".join(file_name.split(".")[:-1])

                    os.makedirs(output_dir, exist_ok=True)
                    md_path = os.path.join(output_dir, f"{base_name}.md")

                    with open(md_path, 'w', encoding='utf-8') as md_file:
                        md_file.write(markdown_content)

                    print(f"Successfully processed {image_path} -> {md_path}")
                    return True
                else:
                    # === 新增这两行 ===
                    error_text = await response.text()
                    print(f"Server Error Details: {error_text}")
                    if retry_count < self.max_retries:
                        print(
                            f"Request failed, status code: {response.status}， retrying ({retry_count + 1}/{self.max_retries}): {image_path}")
                        return await self.process_image(session, image_path, output_dir, retry_count + 1)
                    else:
                        print(f"The maximum number of retries has been reached, and the process failed: {image_path}")
                        return False
        except Exception as e:
            await asyncio.sleep(self.request_sleep)
            if retry_count < self.max_retries:
                print(f"Handling exception, retrying ({retry_count + 1}/{self.max_retries}): {image_path}, Error: {e}")
                return await self.process_image(session, image_path, output_dir, retry_count + 1)
            else:
                print(
                    f"The maximum number of retries has been reached, and the process failed: {image_path}, Error: {e}")
                return False

    def to_markdown(
            self,
            pdf_dir: str | Path,
            image_dir: str | Path,
            out_md_dir: str | Path,
            skip_existing: bool = False,
    ) -> str | Path:
        result = asyncio.run(self.process_directory(image_dir, out_md_dir))

    async def process_directory(self, image_dir, output_dir, file_extensions=None):
        if file_extensions is None:
            file_extensions = ['.jpg', '.jpeg', '.png', '.pdf']

        files_to_process = []
        for root, _, files in os.walk(image_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in file_extensions):
                    files_to_process.append(os.path.join(root, file))

        if not files_to_process:
            print(f"Not find img in {image_dir}")
            return

        print(f"Find  {len(files_to_process)} file to process")

        processed_files = []
        if os.path.exists(output_dir):
            existing_md_files = [os.path.splitext(f)[0] for f in os.listdir(output_dir) if f.endswith('.md')]
            files_to_process = [f for f in files_to_process if
                                os.path.splitext(os.path.basename(f))[0] not in existing_md_files]
            processed_files = len(existing_md_files)
            print(f"Skip {processed_files} files")

        if not files_to_process:
            print("All files have been processed.")
            return

        print(f"Start process {len(files_to_process)} files")

        remaining_files = files_to_process.copy()
        failed_files = []

        timeout_settings = aiohttp.ClientTimeout(total=300, connect=60)
        async with aiohttp.ClientSession(trust_env=True, timeout=timeout_settings) as session:
            while remaining_files:
                current_file = remaining_files.pop(0)
                print(
                    f"Processing ({len(files_to_process) - len(remaining_files)}/{len(files_to_process)}): {current_file}")

                result = await self.process_image(session, current_file, output_dir)
                if not result:
                    failed_files.append(current_file)

        while failed_files:
            print(f"\nThere are  {len(failed_files)} more files that failed to process，retrying...")

            remaining_files = failed_files.copy()
            failed_files = []

            timeout_settings = aiohttp.ClientTimeout(total=300, connect=60)

            async with aiohttp.ClientSession(trust_env=True, timeout=timeout_settings) as session:
                while remaining_files:
                    current_file = remaining_files.pop(0)
                    print(
                        f"Reprocess ({len(files_to_process) - len(remaining_files) - len(failed_files)}/{len(files_to_process)}): {current_file}")

                    result = await self.process_image(session, current_file, output_dir)
                    if not result:
                        failed_files.append(current_file)

        print(f"\n{len(files_to_process)} files has been converted to markdown")

def clean_markdown(markdown_text):
    if markdown_text.strip().startswith("```markdown"):
        markdown_text = markdown_text.strip()[len("```markdown"):].strip()
    if markdown_text.strip().endswith("```"):
        markdown_text = markdown_text.strip()[:-len("```")].strip()
    return markdown_text
