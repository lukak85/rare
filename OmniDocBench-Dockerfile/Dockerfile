# OmniDocBench v1_5 — layout / formula DETECTION evaluation
# Detection (COCODet mAP/mAR) only needs Python + pycocotools.
# No TeX Live / ImageMagick / Ghostscript required (those are CDM-only).

FROM python:3.10-slim

# git: to clone the branch.  build-essential: pycocotools compiles a C ext.
# libgl1 + libglib2.0-0: runtime libs for opencv-python (pinned in requirements).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Pin to the v1_5 branch — this is the branch that registers
# "detection_dataset_simple_format".
RUN git clone --branch v1_5 --depth 1 \
        https://github.com/opendatalab/OmniDocBench.git .

# Patch DetectionDatasetSimpleFormat: upstream hard-codes `+'.jpg'` when keying
# predictions (line 163) and looks up GT pages by their full filename (line 175),
# so GT stored as .png never matches and every page reports "No matching
# prediction". Key/look up by stem instead so mixed .png/.jpg datasets match.
# Line-targeted so the base COCO loader's identical lookup (line 133) is untouched.
RUN sed -i \
        -e '163s|.*|            pred_dict[os.path.splitext(pred["image_name"])[0]].append(pred)|' \
        -e '175s|.*|            sample = pred_dict.get(os.path.splitext(sample_name)[0])|' \
        dataset/detection_dataset.py

RUN pip install --no-cache-dir -r requirements.txt

# pdf_validation.py is the v1_5 entrypoint that does `import dataset`,
# which is what registers the detection dataset loader.
ENTRYPOINT ["python", "pdf_validation.py"]
CMD ["--config", "configs/layout_detection.yaml"]
