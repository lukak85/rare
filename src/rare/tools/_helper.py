import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import cv2
from pycocotools.coco import COCO

from typing import List, Optional

from rare.utils.displayutils import *
from rare.utils.fileutils import save_coco_to_json, read_json
from rare.utils.conversionutils import scale_coco_annotations

from ..utils.displayutils import D4LA_COLOR_MAP, GLASANA_COLOR_MAP, PUBLAYNET_COLOR_MAP

COLOR_MAP_DATASETS = {
    "D4LA": D4LA_COLOR_MAP,
    "DocLayNet": DOCLAYNET_COLOR_MAP,
    "PubLayNet": PUBLAYNET_COLOR_MAP,
    "PP-DocLayoutV3": PPDOCLAYOUTV3_COLOR_MAP,
    "Prima": PRIMA_COLOR_MAP,
    "GlasbenaMladina": GLASANA_COLOR_MAP
}

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
    import layoutparser as lp
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
                score=ann.get("score", None),
            )
        )

    return layout


def join_annotations(path):
    """Read all JSON annotation files in a folder and merge them.

    Reassigns annotation and image IDs sequentially to avoid conflicts across
    files. Images that share a ``file_name`` across files are merged into a
    single entry so their annotations point at the same image.
    """
    coco_anns_list = []
    coco_imgs_list = []
    coco_cats = None
    annotation_id = 1
    image_id = 1
    # Map a file_name to its globally-unique reassigned image id, so the same
    # image appearing in multiple files isn't duplicated.
    file_name_to_new_id = {}

    def natural_key(s):
        # split into runs of digits and non-digits, converting digit runs to int
        return [int(part) if part.isdigit() else part
                for part in re.split(r'(\d+)', s)]

    for filename in os.listdir(path):
        if not filename.endswith(".json"):
            continue

        coco = COCO(os.path.join(path, filename))
        coco_anns = coco.loadAnns(coco.getAnnIds())

        # Reassign image IDs to avoid collisions, deduplicating by file_name.
        # Sort by original id first so new IDs follow the original ordering.
        # Build a per-file map from each old image id to its new global id.
        old_to_new_image_id = {}
        for img in sorted(coco.loadImgs(coco.getImgIds()), key=lambda item: natural_key(item['file_name'])):
            old_id = img["id"]
            new_id = file_name_to_new_id.get(img["file_name"])
            if new_id is None:
                new_id = image_id
                image_id += 1
                file_name_to_new_id[img["file_name"]] = new_id
                img["id"] = new_id
                coco_imgs_list.append(img)
            old_to_new_image_id[old_id] = new_id

        # Reassign annotation IDs and remap their image references.
        for ann in coco_anns:
            ann["id"] = annotation_id
            annotation_id += 1
            ann["image_id"] = old_to_new_image_id[ann["image_id"]]

        coco_anns_list.extend(coco_anns)

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


def visualize_annotations(coco, image_id, save_path=None, visualize_text=False, images_root=False, dataset="Glasna"):
    """Load and display annotations for a single image.

    Args:
        coco: A loaded COCO object.
        image_id: The ID of the image to visualize.
        connections: A loaded JSON object containing connections between regions.
        save_path: Optional path to save the visualization.
        visualize_text: Whether to draw extracted text next to the bounding boxes.
        dataset: The dataset type for color mapping.
    """
    img_info = coco.loadImgs(coco.getImgIds([int(image_id)]))[0]
    img_path = os.path.join(IMAGES_ROOT if not images_root else images_root, img_info["file_name"])
    anns = coco.loadAnns(coco.getAnnIds([int(image_id)]))
    display_img = cv2.imread(img_path)
    # The COCO may have been generated at a different DPI than the image we are
    # drawing on; rescale boxes from their stored size to the actual image size
    # (a no-op when they already match).
    dst_h, dst_w = display_img.shape[:2]
    # anns = scale_coco_annotations(anns, (img_info["width"], img_info["height"]), (dst_w, dst_h))
    layout = load_coco_annotations(anns, categories=coco.cats)
    positions = None
    if visualize_text:
        draw_text(display_img, layout)
    else:
        # Prefer the precomputed permutation in `order_id`; fall back to the
        # connections.json IoU match only when annotations lack order_id.
        positions = order_from_order_id(anns)
        draw_layout(display_img, layout, order=positions, save_path=save_path, color_map=COLOR_MAP_DATASETS.get(dataset))


def visualize_all_images(coco, save_path=None, skip_hashes=None, dataset="Glasna"):
    """Visualize annotations for all images, optionally skipping some.

    Args:
        coco: A loaded COCO object.
        save_path: Optional path to save the visualizations.
        skip_hashes: Set of document hashes to skip.
        dataset: The dataset type for color mapping.
    """
    for image_id in coco.imgs:
        img_info = coco.loadImgs(coco.getImgIds([int(image_id)]))[0]
        doc_hash = img_info["file_name"].split("_")[0]

        if skip_hashes and doc_hash in skip_hashes:
            continue

        print(f"Processing image {img_info['file_name']} with id {image_id} of type {img_info['page_type']}")
        img_path = os.path.join(IMAGES_ROOT, img_info["file_name"])
        anns = coco.loadAnns(coco.getAnnIds([int(image_id)]))
        display_img = cv2.imread(img_path)
        dst_h, dst_w = display_img.shape[:2]
        anns = scale_coco_annotations(anns, (img_info["width"], img_info["height"]), (dst_w, dst_h))
        layout = load_coco_annotations(anns, categories=coco.cats)
        positions = order_from_order_id(anns)
        draw_layout(display_img, layout, order=positions, save_path=save_path, color_map=COLOR_MAP_DATASETS.get(dataset))

def iou(b1, b2):
    # b = [x, y, w, h] in COCO convention
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xa, ya = max(x1, x2), max(y1, y2)
    xb, yb = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0

def order_from_order_id(anns):
    """Reading-order permutation from per-annotation `order_id` fields.

    `anns` is the list loaded for one image (layout index i <-> anns[i], since
    load_coco_annotations preserves order). Returns a list of layout indices in
    reading order — the `order` argument draw_layout expects — or None if the
    annotations don't carry `order_id`. This is the precomputed alternative to
    build_id_map (the IoU match is baked in by scripts/join_annotations.py).
    """
    if not anns or any("order_id" not in a for a in anns):
        return None
    return sorted(range(len(anns)), key=lambda i: anns[i]["order_id"])


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

def extract_chars_in_boxes(pdf_path: str, page_num: int, bboxes: List[dict]) -> dict:
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

def load_coco_bboxes(coco_path: str, image_id: int) -> List[dict]:
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
# PDF collection
# ==============================================================================

# A PAWLS paper folder is named after the SHA-256 hex digest of the PDF it holds.
HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_of(path, chunk_size=1 << 20):
    """SHA-256 hex digest of a file's bytes (streamed, so large PDFs are fine)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_pdfs(papers_root, output_dir, original_names=False):
    """Gather every per-folder PDF under a PAWLS papers root into one directory.

    PAWLS stores each ingested document as `<sha256>/<sha256>.pdf`, where the
    folder/file name is the SHA-256 of the original PDF's bytes. The same root
    often still holds the original, human-named PDFs (e.g. `44310785_..._07.pdf`)
    that were ingested. We hash those originals to recover the
    hash -> original-name mapping, copy out each folder's PDF, and write a
    `mapping.json` recording the correspondence.

    Args:
        papers_root: Folder containing the `<hash>/` subfolders (and, ideally,
            the original named PDFs at its top level).
        output_dir: Destination directory for the collected PDFs.
        original_names: If True, name each copy after its recovered original
            filename (falling back to `<hash>.pdf` when no original is found).

    Returns:
        The list of mapping records written to `mapping.json`.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Recover hash -> original filename by hashing the named PDFs at the root.
    hash_to_original = {}
    for entry in sorted(os.listdir(papers_root)):
        full = os.path.join(papers_root, entry)
        if os.path.isfile(full) and entry.lower().endswith(".pdf"):
            hash_to_original[sha256_of(full)] = entry

    mapping = []
    used_names = set()
    copied = skipped = 0
    for entry in sorted(os.listdir(papers_root)):
        if not HASH_RE.match(entry):
            continue
        src = os.path.join(papers_root, entry, f"{entry}.pdf")
        if not os.path.isfile(src):
            print(f"  ! no PDF in {entry}/, skipping")
            skipped += 1
            continue

        original = hash_to_original.get(entry)
        dst_name = original if (original_names and original) else f"{entry}.pdf"

        # Guard against the rare case of two originals colliding on one name.
        base, ext = os.path.splitext(dst_name)
        candidate = dst_name
        n = 1
        while candidate in used_names:
            candidate = f"{base}_{n}{ext}"
            n += 1
        dst_name = candidate
        used_names.add(dst_name)

        shutil.copy2(src, os.path.join(output_dir, dst_name))
        mapping.append({"hash": entry, "original_name": original, "copied_as": dst_name})
        copied += 1

    mapping_path = os.path.join(output_dir, "mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)

    matched = sum(1 for m in mapping if m["original_name"])
    print(f"\nCollected {copied} PDF(s) into {output_dir} ({skipped} skipped)")
    print(f"Recovered original names for {matched}/{copied} via SHA-256 match")
    print(f"Wrote mapping -> {mapping_path}\n")
    return mapping


def load_mapping(source):
    """Return a list of {"hash", "original_name"} records for reverse lookup.

    `source` may be either a `mapping.json` written by collect_pdfs, or a papers
    root directory (in which case the hash -> original-name correspondence is
    rebuilt by hashing the named PDFs at its top level).
    """
    if os.path.isfile(source) and source.lower().endswith(".json"):
        with open(source) as f:
            return json.load(f)

    if os.path.isdir(source):
        records = []
        seen = set()
        # Map original names by hashing the top-level named PDFs.
        hash_to_original = {}
        for entry in sorted(os.listdir(source)):
            full = os.path.join(source, entry)
            if os.path.isfile(full) and entry.lower().endswith(".pdf"):
                hash_to_original[sha256_of(full)] = entry
        for entry in sorted(os.listdir(source)):
            if HASH_RE.match(entry):
                records.append({"hash": entry, "original_name": hash_to_original.get(entry)})
                seen.add(entry)
        # Include originals whose hash folder is absent, so lookups still resolve.
        for h, name in hash_to_original.items():
            if h not in seen:
                records.append({"hash": h, "original_name": name})
        return records

    raise ValueError(f"Source is neither a mapping.json nor a directory: {source}")


def lookup_pdf(query, source):
    """Resolve a hash to its original name, or an original name to its hash.

    Matches in either direction: an exact 64-hex `query` resolves hash ->
    original name; otherwise `query` is matched (exact, then substring) against
    original filenames to resolve name -> hash.

    Returns the list of matching records (possibly empty).
    """
    records = load_mapping(source)

    if HASH_RE.match(query):
        return [r for r in records if r["hash"] == query]

    exact = [r for r in records if r.get("original_name") == query]
    if exact:
        return exact
    return [r for r in records if r.get("original_name") and query in r["original_name"]]


# ==============================================================================
# Entry point
# ==============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="rare tools", description="Annotation management utilities (was helper.py)")

    parser.add_argument(
        "-a", "--annotations-file",
        help="Path to the COCO annotation file",
        type=str,
    )
    parser.add_argument(
        "--annotations-file-to-compare",
        help="Path to the COCO annotation file to compare against",
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
    parser.add_argument("--config", help="JSON config file for the chosen backend.")
    parser.add_argument(
        "-d", "--dataset",
        help="Dataset type (e.g., Glasana, PubLayNet, D4LA)",
        type=str,
    )
    parser.add_argument(
        "-i", "--image-id",
        help="Image ID to visualize",
        type=str,
    )
    parser.add_argument(
        "-ir", "--images-root",
        help="Root folder for images",
        type=str,
    )
    parser.add_argument("--layout", help="Layout backend (pipeline track).")
    parser.add_argument(
        "-m", "--mode",
        help="Action: join-annotations, run-detection, prepare-annotations, order-images, "
             "remove-scores, review-annotations, count-annotations, text-extraction, "
             "collect-pdfs, lookup-pdf, evaluate-layout or visualize (default)",
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
        "--stem",
        help="Stem of image filename (for run-detection)",
        type=str,
    )
    parser.add_argument(
        "-t", "--visualize-text",
        help="Whether to visualize extracted text next to the bounding boxes",
        action="store_true",
    )
    parser.add_argument(
        "--original-names",
        help="For collect-pdfs: name collected PDFs after their recovered "
             "original filename instead of the SHA-256 hash",
        action="store_true",
    )
    parser.add_argument(
        "--query",
        help="For lookup-pdf: a SHA-256 hash (resolves to original name) or an "
             "original filename / substring (resolves to hash)",
        type=str,
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
        visualize_all_images(coco, save_path=args.save_visualization, dataset=args.dataset)

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
        if not args.dataset:
            print("Please provide dataset type (such as GlasbenaMladina, PubLayNet, D4LA).")
            return 1

        # Documents already reviewed — skip these when reviewing
        already_checked = {
            # "00de9bb518f39464b6b5bb7254d6fdd6e2e2e1fa46710ffe84a6863dca4be950",
            # "0166d9b3f20fa5a4f6bd9d6d001f8b81b24665a6368dd0c10ed3d8a9e30dd691",
            # "04bb9872050b5a73939ae9734a7a1f6935df7b6623f03dc407f3403d52392aa6",
            # "04bc67afae7e1c9113cbbd83e98df59f252ba7757ad90d2c8856f227e5cd8beb",
            # "0525ec05617fc357460ca247faa0b0be9b2caedc8b2663680f852b93541831b6",
            # "111eba9400e08e9e0a5a257aae5c3d36c3c63dd383005a3ca65cbb4d884d8346",
            # "1c3968e8cc47ae26ed907f561ccd55dedbbad3c6645f289fe964582ba864bddd",
            # "20ecf1d1b0602973c2449ed90428bc31847ab613749ffc5d7ce92c5e05788f27",
            # "230edb119aff067fecd3586eb3ce857f9ce402b0867037c156efaaaa32d0ba4b",
            # "2a6e4009dac571c6d4e8b58009acd58a0c0ea1d859f21ac518cf82f2d52a5eda",
            # "326f6533357ab6e301abf9731667626678ccfa078497c866e12df4ff1f652e8f",
            # "4289acbeebf1a459a5339c0f3ed89268ae9437541e5fcce8cd3fa1862517e19a",
            # "53473f43fd47f257cab19acbf24ef1b1b7abe75b4cd643a2387cb10c6c4c44ea",
            # "7612369f2c0ac02697feb81598cf9069a94ea21637329c59bf3955ab731860c2",
            # "7901803e4e1f43b71379ab2657057fc8545977dc4b5f6cbda225c965c4d1c849",
            # "7c43f3e9c7b8ef76798616f47f26cb7e514b7d7216e2e934e366c5eb7266339d",
            # "7f2ee648660870a37590aafa87d1d5636bdddea816f5c386770961f6724fb495",
            # "8b73208759cd38d30f92e167303c95774902d0554e704b7f64bcbde96ec0d00e",
            # "8ec2c4adff08b0297d741164d97068f8f561c18923f28a042382e742be45996c",
            # "9057f730adf6c4b43959e687df737ed7c84618b62567853161ee45cbb688ba21",
            # "9d4eceb46db57f78273b82f5c7e2139b1386b264a1b18703f3d247b5310886dc",
            # "ac30fbcf6678b2b5d3f278a37fb3785adcc1a0791cac4328acd7a86cada649ad",
            # "ba0f4987395c485a886948b2d4d527e7a0cf6382feb245bde7ef39ac8cae0435",
            # "c95ad1a22c65a26798da6407ccf29373b4ff999b0b4d4d4828f803bff7405529",
            # "cd0c26aa8cad0a2c40e96abccad393a2f9a55742c651724081168c2425acd7a2",
            # "d393c9ee0d6653bafac4c34990cffbc414f57ee1ae11a01669b0ae0b8fcdb97f",
            # "ef01d9a74ff40330527608d5ff5434c22664a7a3f639949b281646ad6bfd28f5",
            # "f5753bdada7c6202759859b13c320ce9830aea66fcd49e63721d2b3dca0c45bb",
        }

        coco = COCO(args.annotations_file)
        visualize_all_images(
            coco, save_path=args.save_visualization, skip_hashes=already_checked, dataset=args.dataset,
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

    elif args.mode == "evaluate-layout":
        if not args.annotations_file or not args.annotations_file_to_compare:
            print("Please provide both an annotation file and a file to compare against.")
            return 1
        if not args.image_id:
            print("Please provide an image ID to visualize.")
            return 1
        if not args.layout:
            print("Please provide a layout backend.")
            return 1
        if not args.dataset:
            print("Please provide dataset type (such as Glasana, PubLayNet, D4LA).")
            return 1

        coco = COCO(args.annotations_file)
        coco_compare = COCO(args.annotations_file_to_compare)

        from ..evaluate.pipeline_eval import score_layout
        anns = coco.loadAnns(coco.getAnnIds([int(args.image_id)]))
        anns_compare = coco_compare.loadAnns(coco_compare.getAnnIds([int(args.image_id)]))
        layout = load_coco_annotations(anns, categories=coco.cats)
        layout_compare = load_coco_annotations(anns_compare, categories=coco_compare.cats)

        pred_category_map = None

        from ..evaluate.omnidocbench import DEFAULT_CATEGORY_MAP
        if args.layout == 'doclayout-yolo':
            from ..models.layout.doclayout_yolo import PRED_CATEGORY_MAPS
            pred_category_map = PRED_CATEGORY_MAPS[args.dataset]
        elif args.layout == 'rf-detr':
            from ..models.layout.rfdetr import PRED_CATEGORY_MAPS
            pred_category_map = PRED_CATEGORY_MAPS[args.dataset]

        gt_category_map = DEFAULT_CATEGORY_MAP

        results = score_layout(layout, layout_compare, pred_category_map=pred_category_map, gt_category_map=gt_category_map)

        print(results)

    elif args.mode == "run-detection":
        if not args.path or not args.stem:
            print("Please provide a path and stem of the image filename to run detection on.")
            return 1
        if not args.layout:
            print("Please provide a layout backend.")
            return 1
        if not args.output_path:
            print("Please provide an output path to save the COCO predictions.")
            return 1

        from rare.models.registry import ensure_layoutparser_backend, get
        from rare.utils.conversionutils import layout_parser_to_coco

        ensure_layoutparser_backend(args.layout)

        def _read_config(path):
            if path is None:
                return None
            return json.loads(Path(path).read_text())

        layout_cls = get("layout", args.layout)
        layout = layout_cls(config=_read_config(args.config))
        order_cls = get("order", "paddlex-xy-cut")
        order = order_cls()

        # Get all images that start with stem
        folder = Path(args.path)
        files = list(folder.glob(f"{args.stem}*"))

        coco_predictions: list[dict] = []

        def _open_image(path):
            from PIL import Image
            return Image.open(path)

        idx = 0
        for image_path in files:
            image = _open_image(image_path)
            predicted = layout.detect(image_path)
            predicted_order = order.order(
                predicted,
                image=image,
            )

            import numpy as np
            from PIL import Image
            image_pil = Image.open(image_path)
            height, width = np.asarray(image_pil).shape[:2]

            categories = layout.label_map
            image_info = {
                "id": idx,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
            coco_predictions.append(layout_parser_to_coco(
                predicted, image_info, categories,
                predicted_order=predicted_order,
            ))

            idx += 1

        coco_dir = Path(args.output_path) / "per_page"
        coco_dir.mkdir(parents=True, exist_ok=True)
        for prediction in coco_predictions:
            save_coco_to_json(prediction, str(coco_dir / f"{prediction['images'][0]['id']}.json"))

        coco_data = join_annotations(coco_dir)
        save_coco_to_json(coco_data, f"{args.output_path}/output.json")

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

    elif args.mode == "collect-pdfs":
        papers_root = args.path or PDF_ROOT
        if not args.output_path:
            print("Please provide an output directory with -o/--output-path.")
            return 1
        if not os.path.isdir(papers_root):
            print(f"Papers root not found: {papers_root}")
            return 1
        collect_pdfs(papers_root, args.output_path, original_names=args.original_names)

    elif args.mode == "lookup-pdf":
        if not args.query:
            print("Please provide a value to look up with --query.")
            return 1
        # Source: an explicit mapping.json or papers root via -p, else default.
        source = args.path or args.annotations_file or PDF_ROOT
        matches = lookup_pdf(args.query, source)
        if not matches:
            print(f"No match for {args.query!r} in {source}")
            return 1
        for m in matches:
            print(f"{m['hash']}  <->  {m.get('original_name') or '(unknown)'}")

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
        if not args.dataset:
            print("Please provide dataset type (such as Glasana, PubLayNet, D4LA).")
            return 1

        coco = COCO(args.annotations_file)
        visualize_annotations(coco, args.image_id, save_path=args.save_visualization, visualize_text=args.visualize_text, images_root=args.images_root, dataset=args.dataset)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
