"""Detectron2-backed layout backends: faster-rcnn, mask-rcnn."""

from __future__ import annotations

from rare.models.registry import register


_PUBLAYNET_LABEL_MAP = {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}


@register("layout", "faster-rcnn")
class FasterRCNNBackend:
    def __init__(self, config: dict | None = None):
        import layoutparser as lp
        if config is not None:
            self._model = lp.Detectron2LayoutModel(
                **config,
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                label_map=_PUBLAYNET_LABEL_MAP,
            )
        else:
            self._model = lp.Detectron2LayoutModel(
                "./data/model/fasterrcnn/publaynet/config.yml",
                model_path="./data/model/fasterrcnn/publaynet/model_final.pth",
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                label_map=_PUBLAYNET_LABEL_MAP,
            )

    def detect(self, image):
        return self._model.detect(image)


@register("layout", "mask-rcnn")
class MaskRCNNBackend:
    def __init__(self, config: dict | None = None):
        import layoutparser as lp
        if config is not None:
            self._model = lp.Detectron2LayoutModel(
                **config,
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                label_map=_PUBLAYNET_LABEL_MAP,
            )
        else:
            self._model = lp.Detectron2LayoutModel(
                "./data/model/maskrcnn/publaynet/50/config.yml",
                model_path="./data/model/maskrcnn/publaynet/50/model_final.pth",
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                label_map=_PUBLAYNET_LABEL_MAP,
            )

    def detect(self, image):
        return self._model.detect(image)
