"""LayoutLMv3 backend."""

from __future__ import annotations

from rare.config.paths import WEIGHTS_PATH
from rare.models.registry import register


@register("layout", "layoutlmv3")
class LayoutLMv3Backend:
    def __init__(self, config: dict | None = None):
        import layoutparser as lp
        if config is not None:
            self._model = lp.LayoutLMv3LayoutModel(**config)
        else:
            self._model = lp.LayoutLMv3LayoutModel(
                WEIGHTS_PATH + "/layoutlmv3/model_final.pth",
                yaml_path="configs/layoutlmv3/yaml/cascade_layoutlmv3.yaml",
            )

    def detect(self, image):
        return self._model.detect(image)
