"""Lazy backend registry shared by layout/order/VLM model adapters.

Each adapter file declares its class with `@register("layout"|"order"|"vlm", "<name>")`.
Adapter files are imported on demand so that:
  - importing the registry is cheap (no heavy deps loaded);
  - the LAYOUTPARSER_BACKEND env var can be set before `import layoutparser`
    happens inside the chosen adapter.
"""

from __future__ import annotations

import importlib
import os
from typing import Callable, TypeVar

T = TypeVar("T")

# Populated by @register decorators when adapter modules are imported.
_REGISTRIES: dict[str, dict[str, type]] = {
    "layout": {},
    "order": {},
    "vlm": {},
}

# Maps backend name → module path that registers it. Loaded lazily.
_DEFERRED: dict[str, dict[str, str]] = {
    "layout": {
        # Detectron2 family
        "faster-rcnn":     "rare.models.layout.detectron2_models",
        "mask-rcnn":       "rare.models.layout.detectron2_models",
        # DiT, LayoutLMv3, VGT, SwinDocSeg — each has its own LP backend
        "dit":             "rare.models.layout.dit",
        "layoutlmv3":      "rare.models.layout.layoutlmv3",
        "vgt":             "rare.models.layout.vgt",
        "swindocseg":      "rare.models.layout.swindocseg",
        # YOLO / DETR family
        "doclayout-yolo":  "rare.models.layout.doclayout_yolo",
        "rf-detr":         "rare.models.layout.rfdetr",
        # Paddle family
        "ppyolo":          "rare.models.layout.paddle_models",
        "pp-doclayoutv3":  "rare.models.layout.paddle_models",
        # Misc
        "efficientdet":    "rare.models.layout.effdet",
        "docstrum":        "rare.models.layout.classical",
        "rlsa":            "rare.models.layout.classical",
        "recursive-xycut": "rare.models.layout.classical",
        "nemotron":        "rare.models.layout.nemotron",
    },
    "order": {
        "top-bottom":             "rare.models.order.builtin",
    },
    "vlm": {
    },
}

# Backend name → LAYOUTPARSER_BACKEND env-var value (must be set before lp import).
_LP_BACKEND_ENV: dict[str, str] = {
    "faster-rcnn":     "detectron2",
    "mask-rcnn":       "detectron2",
    "dit":             "dit",
    "layoutlmv3":      "layoutlmv3",
    "vgt":             "vgt",
    "swindocseg":      "swindocseg",
    "doclayout-yolo":  "doclayout_yolo",
    "rf-detr":         "rfdetr",
    "ppyolo":          "paddle",
    "pp-doclayoutv3":  "ppdoclayoutv3",
    "efficientdet":    "effdet",
    "docstrum":        "docstrum",
    "rlsa":            "rlsa",
    "recursive-xycut": "recursive_xycut",
    "nemotron":        "nemotron",
}


def ensure_layoutparser_backend(name: str) -> None:
    """Set LAYOUTPARSER_BACKEND env var. Must be called before `import layoutparser`."""
    backend = _LP_BACKEND_ENV.get(name)
    if backend:
        os.environ["LAYOUTPARSER_BACKEND"] = backend


def register(kind: str, name: str) -> Callable[[type[T]], type[T]]:
    if kind not in _REGISTRIES:
        raise ValueError(f"Unknown backend kind '{kind}'")

    def decorator(cls: type[T]) -> type[T]:
        _REGISTRIES[kind][name] = cls
        cls.name = name  # type: ignore[attr-defined]
        return cls

    return decorator


def get(kind: str, name: str) -> type:
    """Return the registered backend class for (kind, name), importing its module if needed."""
    if name not in _REGISTRIES[kind]:
        # For layout backends, set LP env var BEFORE the adapter module loads
        # (the adapter may import layoutparser at instantiation time).
        if kind == "layout":
            ensure_layoutparser_backend(name)
        module_path = _DEFERRED.get(kind, {}).get(name)
        if module_path is None:
            raise KeyError(
                f"Unknown {kind} backend '{name}'. Available: {list_backends(kind)}"
            )
        importlib.import_module(module_path)
    if name not in _REGISTRIES[kind]:
        raise RuntimeError(
            f"Adapter module for {kind}/{name} loaded but didn't register the backend"
        )
    return _REGISTRIES[kind][name]


def list_backends(kind: str) -> list[str]:
    return sorted(set(_DEFERRED.get(kind, {}).keys()) | set(_REGISTRIES.get(kind, {}).keys()))
