"""Visualization utilities for displaying document layouts."""

import cv2
import layoutparser as lp
import numpy as np

# Color maps for layoutparser's draw_box function
COLOR_MAP = {
    "Paragraph": "red",
    "Title": "blue",
    "List": "green",
    "Table": "purple",
    "Figure": "pink",
    "Header": "orange",
}

DOCLAYNET_COLOR_MAP = {
    "Caption": "red",
    "Footnote": "blue",
    "Formula": "green",
    "List-item": "purple",
    "Page-footer": "pink",
    "Page-header": "orange",
    "Picture": "yellow",
    "Section-header": "brown",
    "Table": "cyan",
    "Text": "grey",
    "Title": "magenta"
}

D4LA_COLOR_MAP = {
    "DocTitle": "red",
    "ParaTitle": "blue",
    "ParaText": "green",
    "ListText": "purple",
    "RegionTitle": "pink",
    "Date": "orange",
    "LetterHead": "cyan",
    "LetterDear": "magenta",
    "LetterSign": "yellow",
    "Question": "brown",
    "OtherText": "grey",
    "RegionKV": "olive",
    "RegionList": "teal",
    "Abstract": "cyan",
    "Author": "cyan",
    "TableName": "blue",
    "Table": "blue",
    "Figure": "blue",
    "FigureName": "blue",
    "Equation": "blue",
    "Reference": "blue",
    "Footer": "orange",
    "PageHeader": "orange",
    "PageFooter": "orange",
    "Number": "orange",
    "Catalog": "orange",
    "PageNumber": "orange",
}

# Color map matching our annotation categories
GLASANA_COLOR_MAP = {
    "Header": "#9EA3FF",
    "Footer": "#0D38D4",
    "PageNum": "#69C0FF",
    "Section": "#008BAD",
    "Kicker": "#61F2D3",
    "Headline": "#0D9E38",
    "Deck": "#D3DB5C",
    "Subhead": "#D96D09",
    "Author": "#FFC6AD",
    "Dropcap": "#DE5492",
    "Paragraph": "#AB59F7",
    "Quote": "#9EA3FF",
    "Footnote": "#0D38D4",
    "Figure": "#69C0FF",
    "Caption": "#008BAD",
    "Advertisement": "#61F2D3",
    "Dateline": "#0D9E38",
    "EditNote": "#D3DB5C",
    "MarginNote": "#D96D09",
    "UnorderedList": "#FFC6AD",
    "OrderedList": "#DE5492",
    "Byline": "#833561",
    "Translator": "#9EA3FF",
    "TOC": "#0D38D4",
    "Literary": "#69C0FF",
    "Question": "#A4F08F",
    "Subsubhead": "#0A0A46",
    "Literature": "#311F24",
    "Abandon": "#000000",
    "FigByline": "#9EA3FF"
}


def draw_layout(
    img,
    layout,
    save_path=None,
    has_score=False,
    color_map=GLASANA_COLOR_MAP,
    order=None,
    order_color="blue",
    order_line_width=2,
):
    """Draw labeled bounding boxes on an image and display it.

    Args:
        img: Image (numpy array, BGR or RGB).
        layout: A layoutparser Layout with TextBlocks.
        save_path: Optional path to save the figure.
        order: Optional reading order. A list of layout indices (ints).
        order_color: Line/number color for the reading-order overlay.
        order_line_width: Line width for the reading-order overlay.
    """

    viz = lp.draw_box(
        img,
        [b.set(id=f"{b.score:.2f}/{b.type}" if has_score else f"{b.type}") for b in layout],
        color_map=color_map,
        show_element_id=True,
        id_font_size=10,
        id_text_background_color="grey",
        id_text_color="white",
        order=order,
        order_color=order_color,
        order_line_width=order_line_width,
    )
    draw_pil_image(viz, save_path)

def draw_text(
    img,
    layout,
):
    """Draw labeled bounding boxes on an image and display it.

    Args:
        img: Image (numpy array, BGR or RGB).
        layout: A layoutparser Layout with TextBlocks.
    """

    print()
    viz = lp.draw_text(
        img,
        [b.set(id=f"{b.type}/{b.id}") for b in layout],
    )
    draw_pil_image(viz)


def _resolve_order(blocks, order):
    """Map an `order` argument (ids or indices) to layout indices.

    Accepts either a list of ints (positions in `blocks`) or a list of
    strings (block.id values). Unknown ids / out-of-range indices are
    silently dropped so partial / dirty inputs still render.
    """
    if not order:
        return []

    if all(isinstance(x, str) for x in order):
        id_to_idx = {b.id: i for i, b in enumerate(blocks) if b.id is not None}
        resolved = [id_to_idx[x] for x in order if x in id_to_idx]
    else:
        n = len(blocks)
        resolved = [int(x) for x in order if 0 <= int(x) < n]

    # Drop duplicates while preserving first-seen order.
    seen, unique = set(), []
    for i in resolved:
        if i not in seen:
            seen.add(i)
            unique.append(i)
    return unique


def draw_pil_image(img, save_path=None):
    """Display a PIL/numpy image with matplotlib, optionally saving to file.

    Handles BGR-to-RGB conversion for OpenCV images.
    """
    import matplotlib.pyplot as plt

    if not isinstance(img, np.ndarray):
        img = np.array(img)

    # Convert BGR to RGB if needed (3-channel images from OpenCV)
    if img.ndim == 3 and img.shape[2] == 3:
        img = img[:, :, ::-1]

    plt.imshow(img)
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()


def draw_cv2_image(img):
    """Display an OpenCV BGR image with matplotlib."""
    import matplotlib.pyplot as plt

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    plt.close()
