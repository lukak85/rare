"""File I/O utilities for configuration and COCO annotation files."""

import json
from pathlib import Path


def read_json(path):
    """Load a JSON file.

    Args:
        path: Path to the JSON file, or None.

    Returns:
        Parsed JSON, or None if path is None.
    """
    if path is None:
        return None
    with open(path, "r") as f:
        return json.load(f)


def read_config(config_path):
    """Load a JSON configuration file.

    Args:
        config_path: Path to the JSON file, or None.

    Returns:
        Parsed config dict, or None if config_path is None.
    """
    return read_json(config_path)


def save_coco_to_json(coco_data, output_path):
    """Save COCO-format annotation data to a JSON file.

    Creates parent directories if they don't exist.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco_data, f, indent=4)


def split_stem_page(file_name: str) -> tuple[str, int]:
    """Split a "<stem>_<page>.jpg" page-file name into (pdf_stem, page_no).

    Falls back to (full stem, 0) when the trailing token is not an int. This is
    the one naming convention every track shares — COCO `file_name` fields,
    per-page Markdown exports, and the JSON a VLM parser writes per page image
    — so the split lives here rather than in each of them.
    """
    name = Path(file_name).name
    parts = name.rsplit("_", 1)
    if len(parts) == 2:
        try:
            return parts[0], int(parts[1].rsplit(".", 1)[0])
        except ValueError:
            pass
    return Path(name).stem, 0
