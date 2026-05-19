import argparse
import os
import sys

# Map CLI method names to LAYOUTPARSER_BACKEND values.
_METHOD_TO_BACKEND = {
    "dit": "dit",
    "docstrum": "docstrum",
    "doclayout-yolo": "doclayout_yolo",
    "dotsocr": "dotsocr",
    "efficientdet": "effdet",
    "faster-rcnn": "detectron2",
    "layoutlmv3": "layoutlmv3",
    "mask-rcnn": "detectron2",
    "nemotron": "nemotron",
    "pp-doclayoutv3" : "ppdoclayoutv3",
    "ppyolo": "paddle",
    "recursive-xycut": "recursive_xycut",
    "rlsa": "rlsa",
    "rfdert": "rfdert",
    "swindocseg": "swindocseg",
    "vgt": "vgt",
}


def _set_backend_from_args():
    """Parse --dla-method early and set LAYOUTPARSER_BACKEND before imports."""
    for i, arg in enumerate(sys.argv):
        if arg in ("-dm", "--dla-method") and i + 1 < len(sys.argv):
            method = sys.argv[i + 1]
            backend = _METHOD_TO_BACKEND.get(method)
            if backend:
                os.environ["LAYOUTPARSER_BACKEND"] = backend
            return


_set_backend_from_args()

from pycocotools.coco import COCO

from config import COCO_ANNO_PATH, WEIGHTS_PATH
from utils.conversionutils import *
from utils.displayutils import *
from utils.evalutils import *
from utils.fileutils import *


# ======================================================================================================================
# Main function
# ==============================================================================


def main(
    img,
    model,
    categories=CATEGORIES,
    evaluation_metric=None,
    image_info=None,
    ground_truth=None,
    visualization=False,
    display_ground=False,
    display_img=None,
    save_coco=None,
    save_image_path=None,
):
    """Run layout analysis on a single image and optionally evaluate/visualize/save.

    Args:
        img: Image data (numpy array) or file path (str), depending on the model.
        image_info: COCO image info dict (id, file_name, width, height).
        ground_truth: Ground-truth layout (lp.Layout) or None.
        model: A layoutparser model instance with a .detect() method.
        evaluation_metric: "f1" or "map", or None to skip evaluation.
        categories: COCO category dict from the annotation file.
        visualization: Whether to display detected bounding boxes.
        display_ground: Whether to also display ground-truth boxes.
        display_img: OpenCV image for visualization (BGR format).
        save_coco: File path (.json) to save COCO-format detection results.
        save_image: Whether to save the visualization image.
    """
    if model is None:
        raise ValueError("No DLA model is provided.")

    # Perform layout detection
    layout = model.detect(img)

    # Visualization
    if visualization:
        draw_layout(display_img, layout, has_score=True)
        if ground_truth is not None and display_ground:
            draw_layout(display_img, ground_truth)

    # Evaluation against ground truth
    metric_scores = {}
    if evaluation_metric is not None:
        if evaluation_metric == "f1":
            print(f1_score(layout, ground_truth))
        else:
            metrics = mean_average_precision(layout, ground_truth)
            map_score = metrics['map'].item()  # mAP@0.50:0.95 (primary metric)
            map_50 = metrics['map_50'].item()  # mAP@IoU=0.50
            map_75 = metrics['map_75'].item()  # mAP@IoU=0.75
            # print(map_score, map_50, map_75)
            metric_scores["map"] = map_score
            metric_scores["map_50"] = map_50
            metric_scores["map_75"] = map_75

    # Save detections in COCO format
    if save_coco is not None:
        if not save_coco.endswith(".json"):
            raise ValueError("The save path for COCO annotations must end with .json")

        save_coco_to_json(
            layout_parser_to_coco(layout, image_info, categories),
            save_coco,
        )

    # Save visualized detections
    if save_image_path is not None:
        draw_layout(display_img, layout, save_path=save_image_path)

    return metric_scores

# ==============================================================================
# Helper functions
# ==============================================================================


def read_picture(path, to_rgb=True):
    """Read an image from disk, optionally converting BGR to RGB."""
    img = cv2.imread(path)
    if to_rgb:
        img = img[..., ::-1]
    return img


def load_coco_annotations(annotations, categories=None):
    """Convert a list of COCO annotation dicts into a layoutparser Layout.

    Args:
        annotations: List of COCO annotation dicts with 'bbox', 'category_id', 'id'.
        coco: Optional COCO object. If provided, category IDs are resolved to names.
    """
    layout = lp.Layout()

    for ann in annotations:
        x, y, w, h = ann["bbox"]
        layout.append(
            lp.TextBlock(
                block=lp.Rectangle(x, y, w + x, h + y),
                type=(
                    categories[ann["category_id"]]["name"]
                    if categories
                    else ann["category_id"]
                ),
                id=ann["id"],
                score=ann.get("score", 1),
            )
        )

    return layout


def check_connection(address="localhost", port=8000, timeout=5):
    """Check whether a TCP connection can be established (e.g. for vLLM server)."""
    import socket

    try:
        socket.create_connection((address, port), timeout=timeout)
        return True
    except OSError:
        return False


def init_model(method, config, verbose=False):
    """Instantiate the appropriate DLA model based on the method name.

    Args:
        method: One of "faster-rcnn", "mask-rcnn", "vgt", "doclayout-yolo",
                "layoutlmv3", "dit", "pp-doclayoutv3" or "rf-detr".
        config: Dict of model kwargs loaded from JSON config, or None.
        verbose: Whether to enable verbose output (only used by some models).

    Returns:
        A layoutparser model instance.
    """
    if method == "faster-rcnn":
        return (
            lp.Detectron2LayoutModel(
                **config,
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
             )
            if config is not None
            else lp.Detectron2LayoutModel(
                "./data/model/fasterrcnn/publaynet/config.yml",
                model_path="./data/model/fasterrcnn/publaynet/model_final.pth",
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
            )
        )
    elif method == "mask-rcnn":
        return (
            lp.Detectron2LayoutModel(
                **config,
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
            )
            if config is not None
            else lp.Detectron2LayoutModel(
                "./data/model/maskrcnn/publaynet/50/config.yml",
                model_path="./data/model/maskrcnn/publaynet/50/model_final.pth",
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
            )
        )
    elif method == "efficientdet":
        return (
            lp.EfficientDetLayoutModel(
                **config,
                label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
            )
            if config is not None
            else lp.EfficientDetLayoutModel(
                config_path="tf_efficientdet_d1",
                model_path="./data/model/efficientdet/publaynet/publaynet-tf_efficientdet_d1.pth.tar",
                label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"}
            )
        )
    elif method == "ppyolo":
        return (
            lp.PaddleDetectionLayoutModel(
                **config,
                label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
            )
            if config is not None
            else lp.PaddleDetectionLayoutModel(
                config_path="./data/model/ppyolo/publaynet/ppyolov2_r50vd_dcn_365e_publaynet/infer_cfg.yml",
                model_path="./data/model/ppyolo/publaynet/ppyolov2_r50vd_dcn_365e_publaynet",
                label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
            )
        )
    elif method == "docstrum":
        return (
            lp.DocstrumLayoutModel(**config, verbose=verbose)
            if config is not None
            else lp.DocstrumLayoutModel(verbose=verbose)
        )
    elif method == "layoutlmv3":
        return (
            lp.LayoutLMv3LayoutModel(**config)
            if config is not None
            else lp.LayoutLMv3LayoutModel(WEIGHTS_PATH + "/layoutlmv3/model_final.pth",
                                          yaml_path="config/layoutlmv3/yaml/cascade_layoutlmv3.yaml")
        )
    elif method == "dit":
        return (
            lp.DiTLayoutModel(**config)
            if config is not None
            else lp.DiTLayoutModel(WEIGHTS_PATH + "/dit/publaynet_dit-b_cascade.pth",
                                   yaml_path="config/dit/yaml/cascade_dit_base.yaml")
        )
    elif method == "doclayout-yolo":
        return (
            lp.DocLayoutYOLOLayoutModel(**config)
            if config is not None
            else lp.DocLayoutYOLOLayoutModel(
                "./data/model/doclayoutyolo/doclayout_yolo_docstructbench_imgsz1024.pt",
                label_map="Glasana"
            )
        )
    elif method == "vgt":
        return (
            lp.VGTLayoutModel(**config)
            if config is not None
            else lp.VGTLayoutModel(
                WEIGHTS_PATH + "/vgt/D4LA_VGT_model.pth",
                grid_root="./data/vgt/grid/",
                yaml_path="config/vgt/yaml/D4LA_VGT_cascade_PTM.yaml")
        )
    elif method == "pp-doclayoutv3":
        return (
            lp.PPDocLayoutV3LayoutModel(**config)
            if config is not None
            else lp.PPDocLayoutV3LayoutModel()
        )
    elif method == "rf-detr":
        return (
            lp.RFDETRLayoutModel(**config)
            if config is not None
            else lp.RFDETRLayoutModel()
        )
    elif method == "swindocseg":
        return (
            lp.SwinDocSegLayoutModel(**config)
            if config is not None
            else lp.SwinDocSegLayoutModel(
                WEIGHTS_PATH + "/swindocseg/model_final_doclay_swindocseg.pth",
                yaml_path="config/swindocseg/yaml/doclaynet/config_doclay.yaml"
            )
        )
    elif method == "nemotron":
        return (
            lp.NemotronLayoutModel(**config)
            if config is not None
            else lp.NemotronLayoutModel()
        )
    elif method == "recursive-xycut":
        return (
            lp.RecursiveXYCutLayoutModel(**config)
            if config is not None
            else lp.RecursiveXYCutLayoutModel()
        )
    elif method == "rlsa":
        return (
            lp.RLSALayoutModel(**config)
            if config is not None
            else lp.RLSALayoutModel()
        )
    else:
        raise ValueError(f"Unknown DLA method: {method}")


def load_images_for_mode(mode, coco, file_path):
    """Collect image paths, annotations, and image info based on the processing mode.

    Args:
        mode: "page" (single image), "pdf" (all pages of a PDF), or "corpus".
        coco: COCO annotation object.
        file_path: Path to the input image or PDF image folder.

    Returns:
        Tuple of (image_list, coco_anns_list, img_info_list).
    """
    image_list = []
    coco_anns_list = []
    img_info_list = []

    if mode == "page":
        filename = file_path.split("/")[-1]
        parent_dir = "/".join(file_path.split("/")[:-1])
        for image_id, image_info in coco.imgs.items():
            if image_info["file_name"] == filename:
                coco_anns = load_coco_annotations(
                    coco.loadAnns(coco.getAnnIds([image_id])),
                    categories=coco.cats
                )
                image_list.append(f"{parent_dir}/{image_info['file_name']}")
                coco_anns_list.append(coco_anns)
                img_info_list.append(image_info)
                break

    elif mode == "pdf":
        pdf_name = file_path.split("/")[-1]
        parent_dir = "/".join(file_path.split("/")[:-1])
        for image_id, image_info in coco.imgs.items():
            if pdf_name in image_info["file_name"]:
                coco_anns = load_coco_annotations(
                    coco.loadAnns(coco.getAnnIds([image_id])),
                    categories = coco.cats
                )
                image_list.append(f"{parent_dir}/{image_info['file_name']}")
                coco_anns_list.append(coco_anns)
                img_info_list.append(image_info)

    elif mode == "corpus":
        NotImplementedError(f"Corpus document layout analysis is not yet implemented.")

    return image_list, coco_anns_list, img_info_list


def build_save_path(save_arg, mode, image_path):
    """Build the output JSON path for saving COCO annotations.

    For 'page' mode, returns save_arg directly.
    For 'pdf' mode, creates a subfolder structure: <save_arg>/<pdf_hash>/<page>.json
    """
    if mode == "page":
        return save_arg

    # Extract PDF hash and page number from the image filename
    filename = image_path.split("/")[-1]
    pdf_hash = filename.split("_")[0]
    page_json = filename.split("_")[1].replace(".jpg", ".json")
    return f"{save_arg}/{pdf_hash}/{page_json}"


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document layout analysis pipeline")
    parser.add_argument(
        "-c", "--config",
        help="JSON configuration file path for the model",
        type=str,
    )
    parser.add_argument(
        "-d", "--dla",
        help="DLA method: detectron2, docstrum, dotsocr, doclayout-yolo, layoutlmv3, dit",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-dd", "--display-detection",
        help="Display image with detected bounding boxes",
        action="store_true",
    )
    parser.add_argument(
        "-dg", "--display-ground",
        help="Display ground-truth bounding boxes",
        action="store_true",
    )
    parser.add_argument(
        "-dt", "--display-text",
        help="Display text corresponding with bounding boxes",
        action="store_true",
    )
    parser.add_argument(
        "-e", "--evaluation-metric",
        help="Evaluation metric: f1 or map",
        type=str,
    )
    parser.add_argument(
        "-f", "--file",
        help="Path to the input image or PDF image folder",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-m", "--mode",
        help="Processing mode: page (single image), pdf (whole PDF), or corpus",
        type=str,
        default="page",
    )
    parser.add_argument(
        "-o", "--order",
        help="Reading order method: xycut",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-s", "--save",
        help="Save COCO annotations to this JSON file path",
        type=str,
    )
    parser.add_argument(
        "-si", "--save-image-path",
        help="Save image with detections displayed in this file",
        type=str,
    )
    parser.add_argument(
        "-v", "--verbose",
        help="Enable verbose output",
        action="store_true",
    )

    args = parser.parse_args()

    image_list, coco_anns_list, img_info_list = None, None, None
    if args.display_ground or args.display_detection:
        coco = COCO(COCO_ANNO_PATH)

            # Collect images and annotations for the chosen mode
        image_list, coco_anns_list, img_info_list = load_images_for_mode(
            args.mode, coco, args.file
        )

        categories = coco.cats
    else:
        image_list = [args.file]
        coco_anns_list, img_info_list = None, None

    config = read_config(args.config)
    show = args.display_detection or args.display_ground

    # Initialize the DLA model
    model = init_model(args.dla_method, config, verbose=args.verbose)

    metric_scores_list = []

    # Process each image
    if not coco_anns_list and not img_info_list:
        for image_path in image_list:
            print(f"Processing image: {image_path}")

            # Load the image: as numpy array for detectron2, as path for other models
            if args.dla_method in ("faster-rcnn", "mask-rcnn"):
                image = read_picture(image_path)
            elif args.dla_method in ("docstrum", "rlsa"):
                image = read_picture(image_path, to_rgb=False)
            else:
                image = image_path

            save_coco_path = (
                build_save_path(args.save, args.mode, image_path)
                if args.save
                else None
            )

            main(
                image,
                model,
                args.evaluation_metric.lower() if args.evaluation_metric else None,
                visualization=show,
                display_img=image if args.dla_method in ("faster-rcnn", "mask-rcnn", "docstrum", "rlsa") else cv2.imread(image),
                save_coco=save_coco_path,
                save_image_path=args.save_image_path,
            )
    else:
        for image_path, coco_anns, img_info in zip(image_list, coco_anns_list, img_info_list):
            print(f"Processing image: {image_path}")

            # Load the image: as numpy array for detectron2, as path for other models
            if args.dla_method in ("faster-rcnn", "mask-rcnn", "ppyolo", "efficientdet"):
                image = read_picture(image_path)
            elif args.dla_method in ("docstrum", "rlsa"):
                image = read_picture(image_path, to_rgb=False)
            else:
                image = image_path

            save_coco_path = (
                build_save_path(args.save, args.mode, image_path)
                if args.save
                else None
            )

            metric_scores = main(
                image,
                model,
                categories=categories,
                evaluation_metric=args.evaluation_metric.lower() if args.evaluation_metric else None,
                image_info=img_info,
                ground_truth=coco_anns,
                visualization=show,
                display_ground=args.display_ground,
                display_img=image if args.dla_method in ("faster-rcnn", "mask-rcnn", "docstrum", "rlsa", "ppyolo", "efficientdet") else cv2.imread(image),
                save_coco=save_coco_path,
                save_image_path=args.save_image_path,
            )

            metric_scores_list.append(metric_scores)

        # Print average evaluation scores across all images
        if args.evaluation_metric:
            from collections import defaultdict
            # Accumulate sums and counts per key
            totals = defaultdict(float)
            counts = defaultdict(int)

            for d in metric_scores_list:
                for key, value in d.items():
                    totals[key] += value
                    counts[key] += 1

            averages = {key: totals[key] / counts[key] for key in totals}

            print(metric_scores_list)
            print(f"Averages: {averages}")