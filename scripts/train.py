#!/usr/bin/env python3
"""Fine-tune a YOLO detector on the electrical-fixture dataset.

Defaults target mobile inference: yolo11n at 640 px. Weights land in
ml/runs/detect/<name>/weights/best.pt — feed that to the export scripts.

Usage:
    python scripts/train.py
    python scripts/train.py --model yolov8s.pt --epochs 200 --batch 32
"""

from __future__ import annotations

import argparse
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ML_ROOT / "electrical.yaml"))
    parser.add_argument("--model", default="yolo11n.pt",
                        help="base weights to fine-tune (nano fits mobile budgets)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", default="electrical",
                        help="run name under ml/runs/detect/")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        project=str(ML_ROOT / "runs" / "detect"),
        resume=args.resume,
        # Scan frames vary a lot in exposure/blur — keep default augments,
        # add slight rotation to mimic phone sweep.
        degrees=5.0,
    )

    metrics = model.val(data=args.data, imgsz=args.imgsz)
    print(f"mAP50: {metrics.box.map50:.3f}  mAP50-95: {metrics.box.map:.3f}")
    best = ML_ROOT / "runs" / "detect" / args.name / "weights" / "best.pt"
    print(f"best weights: {best}")


if __name__ == "__main__":
    main()
