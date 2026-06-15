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

RUN pip install --no-cache-dir -r requirements.txt

# pdf_validation.py is the v1_5 entrypoint that does `import dataset`,
# which is what registers the detection dataset loader.
ENTRYPOINT ["python", "pdf_validation.py"]
CMD ["--config", "configs/layout_detection.yaml"]
