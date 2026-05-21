"""DiT (Document Image Transformer) layout backend."""

from __future__ import annotations

from rare.config.paths import WEIGHTS_PATH
from rare.models.registry import register


@register("layout", "dit")
class DiTBackend:
    def __init__(self, config: dict | None = None):
        import layoutparser as lp
        if config is not None:
            self._model = lp.DiTLayoutModel(**config)
        else:
            self._model = lp.DiTLayoutModel(
                WEIGHTS_PATH + "/dit/publaynet_dit-b_cascade.pth",
                yaml_path="configs/dit/yaml/cascade_dit_base.yaml",
            )

    def detect(self, image):
        return self._model.detect(image)
