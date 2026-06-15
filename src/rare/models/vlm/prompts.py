"""Prompt templates and label-mapping tables for VLM backends."""

from __future__ import annotations

from rare.doc.schema import LABEL_TO_CLASS

# The exact label vocabulary used by GlasanaDocument. Prompts list these so
# the VLM emits matching strings.
GLASANA_LABELS: list[str] = sorted(LABEL_TO_CLASS.keys())


OMNIDOCBENCH_PROMPT = r'''You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:

1. Text Processing:
- Accurately recognize all text content in the PDF image without guessing or inferring.
- Convert the recognized text into Markdown format.
- Maintain the original document structure, including headings, paragraphs, lists, etc.

2. Figure Handling:
- Ignore figures content in the PDF image. Do not attempt to describe or convert images.

3. Output Format:
- Ensure the output Markdown document has a clear structure with appropriate line breaks between elements.
- For complex layouts, try to maintain the original document's structure and format as closely as possible.

Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into Markdown format without adding any extra explanations or comments.
'''

GENERAL_STRUCTURED_PROMPT = """You are a document layout parser for Slovene magazines (Glasbena Mladina).

For the given magazine page image, output a single JSON object describing every
visible region in reading order (top-to-bottom, left-to-right within columns).

The JSON object MUST have this exact shape:
{{
  "page_no": <int>,
  "width":   <int, image width in pixels>,
  "height":  <int, image height in pixels>,
  "regions": [
    {{
      "label":          <one of {labels}>,
      "text":           <string; literal text content; empty string for Figure>,
      "bbox_norm_1000": [<x0>, <y0>, <x1>, <y1>]  // 0-1000 normalised, top-left origin; null if unsure
    }},
    ...
  ]
}}

Rules:
- Emit one region per visually distinct block (a headline, a paragraph, a caption, ...).
- Order regions by reading order.
- Use exactly one of the labels listed above; do not invent new ones. If unsure, prefer "Paragraph".
- Preserve original Slovene text and diacritics exactly.
- Output ONLY the JSON object. No prose, no Markdown fence, nothing else.
"""


WHOLE_PDF_PROMPT = """You are a document layout parser for Slovene magazines (Glasbena Mladina).

For the entire PDF, output a single JSON object with the shape:
{{
  "pages": [
    {{
      "page_no": <int, 0-based>,
      "width":   <int, page width in pixels>,
      "height":  <int, page height in pixels>,
      "regions": [
        {{
          "label":          <one of {labels}>,
          "text":           <string; literal text content; empty string for Figure>,
          "bbox_norm_1000": [<x0>, <y0>, <x1>, <y1>]  // 0-1000 normalised, top-left origin; null if unsure
        }},
        ...
      ]
    }},
    ...
  ]
}}

Rules:
- Emit one region per visually distinct block.
- Order regions within each page by reading order.
- Pages must be ordered by page_no.
- Use exactly one of the labels listed above; do not invent new ones. If unsure, prefer "Paragraph".
- Preserve original Slovene text and diacritics exactly.
- Output ONLY the JSON object. No prose, no Markdown fence, nothing else.
"""


# dots.ocr-native label vocabulary → Glasana labels. Adjust as needed.
DOTSOCR_LABEL_MAP: dict[str, str] = {
    "Title":          "Headline",
    "Section-header": "Subhead",
    "Text":           "Paragraph",
    "List-item":      "UnorderedList",
    "Caption":        "Caption",
    "Footnote":       "Footnote",
    "Formula":        "Paragraph",
    "Picture":        "Figure",
    "Table":          "Table",
    "Page-header":    "Header",
    "Page-footer":    "Footer",
}


# MinerU-native block-type vocabulary → Glasana labels.
MINERU_LABEL_MAP: dict[str, str] = {
    "title":              "Headline",
    "text":               "Paragraph",
    "image":              "Figure",
    "image_caption":      "Caption",
    "image_footnote":     "FigByline",
    "table":              "Table",
    "table_caption":      "Caption",
    "table_footnote":     "FigByline",
    "interline_equation": "Paragraph",
    "isolate_formula":    "Paragraph",
    "list":               "UnorderedList",
    "index":              "TOC",
}


def general_prompt() -> str:
    return GENERAL_STRUCTURED_PROMPT.format(labels=GLASANA_LABELS)


def whole_pdf_prompt() -> str:
    return WHOLE_PDF_PROMPT.format(labels=GLASANA_LABELS)


def omnidocbench_pdf_prompt() -> str:
    return OMNIDOCBENCH_PROMPT.format(labels=GLASANA_LABELS)
