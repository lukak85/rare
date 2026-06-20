"""LayoutLMv3 backend."""

from __future__ import annotations

from rare.config.paths import WEIGHTS_PATH
from rare.models.registry import register


PRED_CATEGORY_MAPS: dict[str, dict[str, str]] = {
    "PubLayNet": {
        "text":     "text_block",
        "title":    "title",
        "list":     "text_block",
        "table":    "table",
        "figure":   "figure",
    }
}

@register("layout", "layoutlmv3")
class LayoutLMv3Backend:
    def __init__(self, config: dict | None = None):
        import layoutparser as lp
        if config is not None:
            self._model = lp.LayoutLMv3LayoutModel(**config)
            label_map = config.get("label_map")
        else:
            self._model = lp.LayoutLMv3LayoutModel(
                WEIGHTS_PATH + "/layoutlmv3/model_final.pth",
                yaml_path="configs/layoutlmv3/yaml/cascade_layoutlmv3.yaml",
            )
            label_map = "Glasana"
        # When predictions already use the Glasbena source vocabulary (the
        # "Glasana" label_map), they share the ground-truth taxonomy, so the
        # runner reuses the GT category map and this stays None.
        self.pred_category_map = PRED_CATEGORY_MAPS.get(label_map)
        self.label_map = self._model.label_map

    def detect(self, image):
        return self._model.detect(image)
