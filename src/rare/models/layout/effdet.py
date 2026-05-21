"""EfficientDet backend."""

from __future__ import annotations

from rare.models.registry import register


_PUBLAYNET_LABEL_MAP = {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}


@register("layout", "efficientdet")
class EfficientDetBackend:
    def __init__(self, config: dict | None = None):
        import layoutparser as lp
        if config is not None:
            self._model = lp.EfficientDetLayoutModel(**config, label_map=_PUBLAYNET_LABEL_MAP)
        else:
            self._model = lp.EfficientDetLayoutModel(
                config_path="tf_efficientdet_d1",
                model_path="./data/model/efficientdet/publaynet/publaynet-tf_efficientdet_d1.pth.tar",
                label_map=_PUBLAYNET_LABEL_MAP,
            )

    def detect(self, image):
        return self._model.detect(image)
