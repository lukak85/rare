import argparse
import json
import os

import cv2
from pycocotools.coco import COCO

from layoutparser.visualization import draw_text
from rare.utils.displayutils import *
from rare.utils.fileutils import save_coco_to_json, read_json

IMAGES_ROOT = "datasets/glasbena_mladina/images"
PDF_ROOT = "annotation/pawls/skiff_files/apps/pawls/papers/"
STATUS_JSON = "annotation/pawls/skiff_files/apps/pawls/papers/status/development_user@example.com.json"

# IoU threshold above which two annotations are considered duplicates
DUPLICATE_IOU_THRESHOLD = 0.95


# ==============================================================================
# Annotation helpers
# ==============================================================================


def load_coco_annotations(annotations, categories=None):
    """Convert COCO annotation dicts to a layoutparser Layout.

    Args:
        annotations: List of COCO annotation dicts.
        categories: Optional COCO categories dict. If provided, resolves
                    category IDs to human-readable names.
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


def join_annotations(path):
    """Read all JSON annotation files in a folder and merge them.

    Reassigns annotation IDs sequentially to avoid conflicts across files.
    """
    coco_anns_list = []
    coco_imgs_list = []
    coco_cats = None
    annotation_id = 1

    for filename in os.listdir(path):
        if not filename.endswith(".json"):
            continue

        coco = COCO(os.path.join(path, filename))
        coco_anns = coco.loadAnns(coco.getAnnIds())

        # Reassign annotation IDs to avoid collisions
        for ann in coco_anns:
            ann["id"] = annotation_id
            annotation_id += 1

        coco_anns_list.extend(coco_anns)
        coco_imgs_list.extend(coco.loadImgs(coco.getImgIds()))

        if coco_cats is None:
            coco_cats = coco.cats

    return {
        "images": coco_imgs_list,
        "annotations": coco_anns_list,
        "categories": [coco_cats[cid] for cid in coco_cats],
    }


def remove_duplicates(coco, annotations_file):
    """Remove near-duplicate annotations based on IoU overlap.

    Two annotations on the same image with IoU > DUPLICATE_IOU_THRESHOLD are
    considered duplicates. When both have scores, the higher-scoring one is kept.

    Args:
        coco: A loaded COCO object.
        annotations_file: Path to the annotation file (used for validation).
    """
    from pycocotools import mask as maskUtils

    if not annotations_file:
        raise ValueError("Please provide an annotation file to remove duplicates from.")

    coco_anns = coco.loadAnns(coco.getAnnIds())

    # Group annotations by image
    anns_by_image = {}
    for ann in coco_anns:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    # For each image, deduplicate by IoU
    unique_anns = []
    for image_id, anns in anns_by_image.items():
        kept = []
        for ann in anns:
            is_duplicate = False
            for existing in kept:
                iou = maskUtils.iou([ann["bbox"]], [existing["bbox"]], [False])
                if iou[0][0] > DUPLICATE_IOU_THRESHOLD:
                    # Keep the annotation with the higher score
                    if ann.get("score") is not None and ann["score"] > existing.get("score", 0):
                        kept.remove(existing)
                        kept.append(ann)
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(ann)
        unique_anns.extend(kept)

    return {
        "images": [coco.imgs[img_id] for img_id in coco.imgs],
        "annotations": unique_anns,
        "categories": [coco.cats[cid] for cid in coco.cats],
    }


def visualize_annotations(coco, image_id, connections=None, save_path=None, visualize_text=False):
    """Load and display annotations for a single image.

    Args:
        coco: A loaded COCO object.
        image_id: The ID of the image to visualize.
        connections: A loaded JSON object containing connections between regions.
        save_path: Optional path to save the visualization.
        visualize_text: Whether to draw extracted text next to the bounding boxes.
    """
    img_info = coco.loadImgs(coco.getImgIds([int(image_id)]))[0]
    img_path = os.path.join(IMAGES_ROOT, img_info["file_name"])
    anns = coco.loadAnns(coco.getAnnIds([int(image_id)]))
    layout = load_coco_annotations(anns, categories=coco.cats)
    display_img = cv2.imread(img_path)
    positions = None
    if visualize_text:
        draw_text(display_img, layout)
    else:
        if connections:
            id_map, tgt_index = build_id_map(anns, connections, img_info["file_name"], (img_info["width"], img_info["height"]))
            coco_id_order = [id_map[i] for i in tgt_index if i in id_map]
            index_map = {id_: i for i, id_ in enumerate(sorted(coco_id_order))}
            positions = [index_map[id_] for id_ in coco_id_order]
        draw_layout(display_img, layout, order=positions, save_path=save_path)


def visualize_all_images(coco, save_path=None, skip_hashes=None):
    """Visualize annotations for all images, optionally skipping some.

    Args:
        coco: A loaded COCO object.
        save_path: Optional path to save the visualizations.
        skip_hashes: Set of document hashes to skip.
    """
    for image_id in coco.imgs:
        img_info = coco.loadImgs(coco.getImgIds([int(image_id)]))[0]
        doc_hash = img_info["file_name"].split("_")[0]

        if skip_hashes and doc_hash in skip_hashes:
            continue

        print(f"Processing image {img_info['file_name']} with id {image_id}")
        img_path = os.path.join(IMAGES_ROOT, img_info["file_name"])
        anns = coco.loadAnns(coco.getAnnIds([int(image_id)]))
        layout = load_coco_annotations(anns, categories=coco.cats)
        display_img = cv2.imread(img_path)
        draw_layout(display_img, layout, save_path=save_path)

def iou(b1, b2):
    # b = [x, y, w, h] in COCO convention
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xa, ya = max(x1, x2), max(y1, y2)
    xb, yb = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0

def build_id_map(layout_coco, reading_json, file_name, shape, iou_threshold=0.9):
    # Group COCO annotations by image_id for fast lookup
    rb_to_coco = {}  # reading-bank region id -> coco annotation id
    tgt_index = None
    for item in reading_json:
        if item["image"] == file_name:
            tgt_index = item["layoutreader"]["text"]["tgt_index"]
            for region in item["regions"]:
                best, best_iou = None, 0.0
                for ann in layout_coco:
                    score = iou(region["bbox_norm_1000"], norm_1000(ann["bbox"], shape))
                    if score > best_iou:
                        best, best_iou = ann, score
                if best and best_iou >= iou_threshold:
                    rb_to_coco[region["index"]] = best["id"]
    return rb_to_coco, tgt_index

def norm_1000(annotation, shape):
    return [
        annotation[0] / shape[0] * 1000,
        annotation[1] / shape[1] * 1000,
        (annotation[0] + annotation[2]) / shape[0] * 1000,
        (annotation[1] + annotation[3]) / shape[1] * 1000,
    ]

def extract_chars_in_boxes(pdf_path: str, page_num: int, bboxes: list[dict]) -> dict:
    """
    bboxes: list of {"id": str, "x0": float, "y0": float, "x1": float, "y1": float}
    Coordinates are in PDF points (origin = bottom-left by default in pdfplumber).
    Returns: {bbox_id: extracted_text}
    """
    results = {}

    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]

        for bbox in bboxes:
            # Crop the page to the bounding box region
            region = page.within_bbox((bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"]))
            text = region.extract_text() or ""
            results[bbox["id"]] = text.strip()

    return results

def load_coco_bboxes(coco_path: str, image_id: int) -> list[dict]:
    """
    Reads COCO annotations for a given image_id.
    Converts COCO [x, y, w, h] → {"id", "x0", "y0", "x1", "y1"}.
    """
    c = COCO(coco_path)
    annotations = c.loadAnns(c.getAnnIds([int(image_id)]))

    bboxes = []
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        bboxes.append({
            "id":  ann["id"],
            "x0": x,
            "y0": y,
            "x1": x + w,
            "y1": y + h,
        })

    return bboxes

# ==============================================================================
# Entry point
# ==============================================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rare tools", description="Annotation management utilities (was helper.py)")

    parser.add_argument(
        "-a", "--annotations-file",
        help="Path to the COCO annotation file",
        type=str,
    )
    parser.add_argument(
        "-b", "--box-id",
        help="Bounding box ID to focus on.",
        type=str,
    )
    parser.add_argument(
        "-c", "--connections-annotations-file",
        help="Path to the JSON annotation file including connections",
        type=str,
    )
    parser.add_argument(
        "-i", "--image-id",
        help="Image ID to visualize",
        type=str,
    )
    parser.add_argument(
        "-m", "--mode",
        help="Action: join-annotations, prepare-annotations, order-images, "
             "remove-scores, review-annotations, count-annotations, text-extraction or visualize (default)",
        type=str,
        default="visualize",
    )
    parser.add_argument(
        "-o", "--output-path",
        help="Output file path",
        type=str,
    )
    parser.add_argument(
        "-p", "--path",
        help="Input folder path (for join-annotations)",
        type=str,
    )
    parser.add_argument(
        "-r", "--remove-duplicates",
        help="Remove duplicate annotations (IoU > 0.95) when joining",
        action="store_true",
    )
    parser.add_argument(
        "-s", "--save-visualization",
        help="Path to save the annotation visualization",
        type=str,
    )
    parser.add_argument(
        "-t", "--visualize-text",
        help="Whether to visualize extracted text next to the bounding boxes",
        action="store_true",
    )

    args = parser.parse_args(argv)

    # ---- Mode dispatch ----

    if args.mode == "prepare-annotations":
        if not args.annotations_file:
            print("Please provide an annotation file to prepare.")
            return 1

        # Join all annotations in the folder
        coco_data = join_annotations(args.annotations_file)
        output_json = os.path.join(args.annotations_file, args.output_path)
        save_coco_to_json(coco_data, output_json)

        # Remove duplicates
        coco = COCO(output_json)
        coco_data = remove_duplicates(coco, args.annotations_file)
        save_coco_to_json(coco_data, output_json)

        # Visualize for review
        coco = COCO(output_json)
        visualize_all_images(coco, save_path=args.save_visualization)

    elif args.mode == "order-images":
        if not args.annotations_file:
            print("Please provide an annotation file.")
            return 1

        coco = COCO(args.annotations_file)
        sorted_images = sorted(coco.dataset["images"], key=lambda x: x["id"])

        # Filter to finished documents only
        with open(STATUS_JSON, "r") as f:
            status_data = json.load(f)
        finished_images = [
            img for img in sorted_images
            if status_data[img["file_name"].split("_")[0]]["finished"]
        ]

        sorted_annotations = sorted(
            coco.dataset["annotations"], key=lambda x: x["image_id"]
        )
        save_coco_to_json(
            {
                "images": finished_images,
                "annotations": sorted_annotations,
                "categories": coco.dataset["categories"],
            },
            args.output_path,
        )

    elif args.mode == "remove-scores":
        coco = COCO(args.annotations_file)
        for ann in coco.dataset["annotations"]:
            ann.pop("score", None)
        save_coco_to_json(
            {
                "images": coco.dataset["images"],
                "annotations": coco.dataset["annotations"],
                "categories": coco.dataset["categories"],
            },
            args.output_path,
        )

    elif args.mode == "review-annotations":
        if not args.annotations_file:
            print("Please provide an annotation file.")
            return 1

        # Documents already reviewed — skip these when reviewing
        already_checked = {
            #"00de9bb518f39464b6b5bb7254d6fdd6e2e2e1fa46710ffe84a6863dca4be950",
            "0166d9b3f20fa5a4f6bd9d6d001f8b81b24665a6368dd0c10ed3d8a9e30dd691",
            "04bb9872050b5a73939ae9734a7a1f6935df7b6623f03dc407f3403d52392aa6",
            "04bc67afae7e1c9113cbbd83e98df59f252ba7757ad90d2c8856f227e5cd8beb",
            "0525ec05617fc357460ca247faa0b0be9b2caedc8b2663680f852b93541831b6",
            "111eba9400e08e9e0a5a257aae5c3d36c3c63dd383005a3ca65cbb4d884d8346",
            "1c3968e8cc47ae26ed907f561ccd55dedbbad3c6645f289fe964582ba864bddd",
            "20ecf1d1b0602973c2449ed90428bc31847ab613749ffc5d7ce92c5e05788f27",
            "230edb119aff067fecd3586eb3ce857f9ce402b0867037c156efaaaa32d0ba4b",
            "2a6e4009dac571c6d4e8b58009acd58a0c0ea1d859f21ac518cf82f2d52a5eda",
            "326f6533357ab6e301abf9731667626678ccfa078497c866e12df4ff1f652e8f",
            "4289acbeebf1a459a5339c0f3ed89268ae9437541e5fcce8cd3fa1862517e19a",
            "53473f43fd47f257cab19acbf24ef1b1b7abe75b4cd643a2387cb10c6c4c44ea",
            "7612369f2c0ac02697feb81598cf9069a94ea21637329c59bf3955ab731860c2",
            "7901803e4e1f43b71379ab2657057fc8545977dc4b5f6cbda225c965c4d1c849",
            "7c43f3e9c7b8ef76798616f47f26cb7e514b7d7216e2e934e366c5eb7266339d",
            "7f2ee648660870a37590aafa87d1d5636bdddea816f5c386770961f6724fb495",
            "8b73208759cd38d30f92e167303c95774902d0554e704b7f64bcbde96ec0d00e",
            "8ec2c4adff08b0297d741164d97068f8f561c18923f28a042382e742be45996c",
            "9057f730adf6c4b43959e687df737ed7c84618b62567853161ee45cbb688ba21",
            "9d4eceb46db57f78273b82f5c7e2139b1386b264a1b18703f3d247b5310886dc",
            "ac30fbcf6678b2b5d3f278a37fb3785adcc1a0791cac4328acd7a86cada649ad",
            "ba0f4987395c485a886948b2d4d527e7a0cf6382feb245bde7ef39ac8cae0435",
            "c95ad1a22c65a26798da6407ccf29373b4ff999b0b4d4d4828f803bff7405529",
            "cd0c26aa8cad0a2c40e96abccad393a2f9a55742c651724081168c2425acd7a2",
            "d393c9ee0d6653bafac4c34990cffbc414f57ee1ae11a01669b0ae0b8fcdb97f",
            "ef01d9a74ff40330527608d5ff5434c22664a7a3f639949b281646ad6bfd28f5",
            "f5753bdada7c6202759859b13c320ce9830aea66fcd49e63721d2b3dca0c45bb",
        }

        coco = COCO(args.annotations_file)
        visualize_all_images(
            coco, save_path=args.save_visualization, skip_hashes=already_checked
        )

    elif args.mode == "join-annotations":
        if not args.path:
            print("Please provide a folder path to join annotations.")
            return 1
        coco_data = join_annotations(args.path)
        save_coco_to_json(coco_data, args.output_path)

    elif args.mode == "count-annotations":
        if not args.annotations_file:
            print("Please provide an annotation file.")
            return 1

        coco = COCO(args.annotations_file)
        counts = {}
        for ann in coco.loadAnns(coco.getAnnIds()):
            cat_name = coco.cats[ann["category_id"]]["name"]
            counts[cat_name] = counts.get(cat_name, 0) + 1

        total = sum(counts.values())
        col_w = max(len(name) for name in counts) + 2
        print(f"\n{'Category':<{col_w}} {'Count':>8}  {'%':>6}")
        print("-" * (col_w + 18))
        for name, count in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"{name:<{col_w}} {count:>8}  {count / total * 100:>5.1f}%")
        print("-" * (col_w + 18))
        print(f"{'TOTAL':<{col_w}} {total:>8}  {'100.0%':>6}\n")

    elif args.mode == "assign-ids":
        if not args.path:
            print("Please provide a folder path to join annotations.")
            return 1
        coco_data = read_json(args.path)
        for anno_id, annotation in enumerate(coco_data["annotations"]):
            annotation["id"] = anno_id
            annotation["segmentation"] = None
        save_coco_to_json(coco_data, args.output_path)

    elif args.mode == "text-extraction":
        if not args.annotations_file:
            print("Please provide an annotation file.")
            return 1
        if not args.image_id:
            print("Please provide an image ID to extract text for.")
            return 1

        coco = COCO(args.annotations_file)
        img_info = coco.loadImgs(coco.getImgIds([int(args.image_id)]))[0]
        file_name = img_info["file_name"].split(".")[0].split("_")
        files_name = file_name[0]
        page = int(file_name[1])

        anns = coco.loadAnns(coco.getAnnIds([int(args.image_id)]))

        boxes = load_coco_bboxes(args.annotations_file, args.image_id)
        res = extract_chars_in_boxes(
            PDF_ROOT + files_name  + "/" + files_name + ".pdf",
            page,
            boxes
        )

        if args.box_id:
            print(res[(int(args.box_id))])
        else:
            if args.connections_annotations_file:
                rec = json.load(open(args.connections_annotations_file))

                id_map, tgt_index = build_id_map(anns, rec, img_info["file_name"],
                                                 (img_info["width"], img_info["height"]))
                coco_id_order = [id_map[i] for i in tgt_index if i in id_map]

                # Sort res based on positions
                #sorted_res = [res[coco_id] for coco_id in sorted(coco_id_order, key=lambda x: positions[index_map[x]])]
                sorted_res = [res[coco_id] for coco_id in coco_id_order]

                print(sorted_res)
            else:
                print(res)

    elif args.remove_duplicates:
        coco = COCO(args.annotations_file)
        coco_data = remove_duplicates(coco, args.annotations_file)
        save_coco_to_json(coco_data, args.output_path)

    else:
        # Default: visualize a single image's annotations
        if not args.annotations_file:
            print("Please provide an annotation file to visualize.")
            return 1
        if not args.image_id:
            print("Please provide an image ID to visualize.")
            return 1

        coco = COCO(args.annotations_file)
        rec = None
        if args.connections_annotations_file:
            rec = json.load(open(args.connections_annotations_file))
        visualize_annotations(coco, args.image_id, connections=rec, save_path=args.save_visualization, visualize_text=args.visualize_text)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
