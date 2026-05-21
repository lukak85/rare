"""RF-DETR backend (Roboflow's DETR variant for document layout)."""

from __future__ import annotations

from rare.models.registry import register


@register("layout", "rf-detr")
class RFDETRBackend:
    def __init__(self, config: dict | None = None):
        import layoutparser as lp
        self._model = lp.RFDETRLayoutModel(**config) if config else lp.RFDETRLayoutModel()

    def detect(self, image):
        return self._model.detect(image)
