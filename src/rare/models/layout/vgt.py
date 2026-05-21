"""VGT (Vision Grid Transformer) backend."""

from __future__ import annotations

from rare.config.paths import WEIGHTS_PATH
from rare.models.registry import register


@register("layout", "vgt")
class VGTBackend:
    def __init__(self, config: dict | None = None):
        import layoutparser as lp
        if config is not None:
            self._model = lp.VGTLayoutModel(**config)
        else:
            self._model = lp.VGTLayoutModel(
                WEIGHTS_PATH + "/vgt/D4LA_VGT_model.pth",
                grid_root="./data/vgt/grid/",
                yaml_path="configs/vgt/yaml/D4LA_VGT_cascade_PTM.yaml",
            )

    def detect(self, image):
        return self._model.detect(image)
