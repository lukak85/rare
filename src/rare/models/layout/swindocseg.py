"""SwinDocSeg backend."""

from __future__ import annotations

from rare.config.paths import WEIGHTS_PATH
from rare.models.registry import register


@register("layout", "swindocseg")
class SwinDocSegBackend:
    def __init__(self, config: dict | None = None):
        import layoutparser as lp
        if config is not None:
            self._model = lp.SwinDocSegLayoutModel(**config)
        else:
            self._model = lp.SwinDocSegLayoutModel(
                WEIGHTS_PATH + "/swindocseg/model_final_doclay_swindocseg.pth",
                yaml_path="configs/swindocseg/yaml/doclaynet/config_doclay.yaml",
            )

    def detect(self, image):
        return self._model.detect(image)
