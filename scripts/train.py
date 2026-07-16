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
    parser.add_argument("--model", default="yolo11s.pt",
                        help="base weights to fine-tune. Small (+7.5 mAP over nano) is "
                             "the sweet spot for far/small outlets and still runs in "
                             "~30-50 ms on the iPhone Pro ANE — far under the app's "
                             "300 ms detection throttle. (YOLO12's attention layers "
                             "convert worse to CoreML/ANE; stay on the 11 family.)")
    parser.add_argument("--epochs", type=int, default=100)
    # Matches the 960 mobile export — train and deploy at the same input size.
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=16,
                        help="lower to ~4-8 for imgsz 960 on a 16 GB Apple Silicon Mac")
    parser.add_argument("--device", default=None,
                        help="'mps' (Apple Silicon GPU), 'cpu', '0' (CUDA). "
                             "Default: let Ultralytics auto-detect.")
    parser.add_argument("--workers", type=int, default=None,
                        help="dataloader workers (default: Ultralytics picks). "
                             "Set 4 on macOS if you hit dataloader stalls.")
    parser.add_argument("--name", default="electrical",
                        help="run name under ml/runs/detect/")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from ultralytics import YOLO

    train_kwargs = dict(
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
    if args.device is not None:
        train_kwargs["device"] = args.device
    if args.workers is not None:
        train_kwargs["workers"] = args.workers

    model = YOLO(args.model)
    model.train(**train_kwargs)

    metrics = model.val(data=args.data, imgsz=args.imgsz)
    print(f"mAP50: {metrics.box.map50:.3f}  mAP50-95: {metrics.box.map:.3f}")
    best = ML_ROOT / "runs" / "detect" / args.name / "weights" / "best.pt"
    print(f"best weights: {best}")


if __name__ == "__main__":
    main()
