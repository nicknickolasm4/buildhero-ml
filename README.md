# BuildHero ML — Electrical Fixture Detection (YOLO)

Training pipeline for detecting **outlets, switches and light fixtures** in
room-scan camera frames. The exported models feed the mobile scanners:

| Target | Format | Consumer |
|--------|--------|----------|
| iOS    | CoreML (`.mlpackage`) | `react-native-room-scanner` (RoomPlan + ARKit paths) |
| Android| TFLite (`.tflite`)    | ARCore scan module |

## Classes

Canonical class ids — **never reorder**, the mobile integrations map by index:

```
0: outlet   # wall power outlet (BR NBR 14136 + US NEMA)
1: switch   # wall light switch
2: light    # ceiling/wall light fixture, lamp
```

## Quickstart

```bash
cd ml
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Bootstrap dataset from public Roboflow Universe projects (needs ROBOFLOW_API_KEY)
export ROBOFLOW_API_KEY=...
python scripts/download_roboflow.py

# 2. Fine-tune (defaults: yolo11n, 100 epochs, 640px)
python scripts/train.py

# 3. Export for mobile
python scripts/export_coreml.py --weights runs/detect/train/weights/best.pt --install
python scripts/export_tflite.py --weights runs/detect/train/weights/best.pt
```

## Dataset strategy

Public datasets are mostly **US-standard (NEMA)** fixtures — visually different
from the Brazilian NBR 14136 standard. They are used only to pre-train; real
accuracy requires own-data collection. Read **`docs/DATASET.md`** before
collecting or annotating photos.

Layout (created by `download_roboflow.py`, gitignored):

```
ml/data/dataset/
├── train/{images,labels}/
└── valid/{images,labels}/
```

`electrical.yaml` is the Ultralytics data config pointing at that layout.
