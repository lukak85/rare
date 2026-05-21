from pathlib import Path

from rare.utils.fileutils import read_json

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
config = read_json(_PROJECT_ROOT / "configs" / "rare.json")
COCO_ANNO_PATH = str(_PROJECT_ROOT / config["coco_annotations_path"].lstrip("./"))
WEIGHTS_PATH = str(_PROJECT_ROOT / config["weights_path"].lstrip("./"))
