import pdfplumber

def extract_chars_in_boxes(pdf_path: str, page_num: int, bboxes: list[dict]) -> dict:
    """
    bboxes: list of {"id": str, "x0": float, "y0": float, "x1": float, "y1": float}
    Coordinates are in PDF points (origin = bottom-left by default in pdfplumber).
    Returns: {bbox_id: extracted_text}
    """
    results = {}

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]

        for bbox in bboxes:
            # Crop the page to the bounding box region
            region = page.within_bbox((bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"]))
            text = region.extract_text() or ""
            results[bbox["id"]] = text.strip()

    return results