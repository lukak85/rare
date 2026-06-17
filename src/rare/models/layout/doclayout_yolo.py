"""DocLayout-YOLO backend."""

from __future__ import annotations

from rare.config.paths import WEIGHTS_PATH
from rare.models.registry import register


OMNIDOCBENCH_PRED_CAT_MAPPING ="""\
title : title
plain text: text
abandon: abandon
figure: figure
figure_caption: figure_caption
"""

@register("layout", "doclayout-yolo")
class DocLayoutYOLOBackend:
    def __init__(self, config: dict | None = None):
        import layoutparser as lp
        if config is not None:
            self._model = lp.DocLayoutYOLOLayoutModel(**config)
        else:
            self._model = lp.DocLayoutYOLOLayoutModel(
                WEIGHTS_PATH + "/model/doclayoutyolo/doclayout_yolo_docstructbench_imgsz1024.pt",
                label_map="Glasana",
            )

    def detect(self, image):
        return self._model.detect(image)
