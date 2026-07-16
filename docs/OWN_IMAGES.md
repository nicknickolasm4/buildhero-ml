# Adding your own photos

The public Roboflow sources in `download_roboflow.py` are only a
pre-training base. Your own photos — matching your actual fixtures,
lighting, and phone-scan angles — are what make the model production-ready.

## 1. Take photos

Outlet, switch, light — vary angle, distance, lighting, partial occlusion.
See `DATASET.md` for the full shooting checklist. MVP target: 150–300 photos
per class.

## 2. Annotate

Draw a box around each object, assign it a class.

- **Roboflow** (easiest, you already have an account): dashboard →
  **Create New Project** → type **Object Detection** → upload photos →
  annotate in-browser. Name classes exactly `outlet`, `switch`, `light` if
  you want them to line up with this repo's canonical ids without needing
  an alias.
- **CVAT** (self-hosted, free): same idea, runs locally — use this if you'd
  rather not upload photos to a third party.

## 3. Export

- **Roboflow**: inside the project, **Generate** a version → **Export
  Dataset** → format **YOLOv8**. This gives you either a downloadable zip or
  a workspace/project/version you can pull automatically (see step 4a).
- **CVAT**: export as **YOLOv8** (or "YOLO 1.1" + convert — check your CVAT
  version's export list).

## 4. Bring it into the pipeline

**4a. Automatic** (source lives on Roboflow) — add it to
`download_roboflow.py`'s `DEFAULT_SOURCES`, same shape as the existing
public sources:

```python
DEFAULT_SOURCES = [
    {"workspace": "yolov5-dtypd", "project": "plug-socket-detect", "version": 1},
    {"workspace": "biiim", "project": "rocker", "version": 1},
    {"workspace": "YOUR-WORKSPACE", "project": "YOUR-PROJECT", "version": 1},
]
```

Re-run `python scripts/download_roboflow.py` — it downloads and merges your
project alongside the public ones.

**4b. Manual** (already have an exported/unzipped folder — from Roboflow,
CVAT, or anywhere else) — use `--local`, no API key needed:

```bash
python scripts/download_roboflow.py --local /path/to/exported/dataset
```

The folder must contain a `data.yaml` (the standard YOLOv8 export layout —
`train/images`, `train/labels`, `valid/images`, `valid/labels`). This runs
through the same class-remapping as the automatic path, so class ids always
line up correctly even if your export's class order doesn't match this
repo's canonical order.

See `ADDING_SOURCES.md` for combining multiple sources (Roboflow + manual)
in one run, and for adding an object class beyond outlet/switch/light.

## 5. Retrain

```bash
python scripts/train.py
# or, if you already merged everything:
python scripts/run_local_pipeline.py --skip-download
```
