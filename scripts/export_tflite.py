#!/usr/bin/env python3
"""Export trained YOLO weights to TFLite for the Android room scanner.

Produces electrical_detector.tflite (fp16 by default, --int8 for full
quantization using the training set as calibration data). --install copies
it into the react-native-room-scanner Android module assets.

Usage:
    python scripts/export_tflite.py --weights runs/detect/electrical/weights/best.pt --install
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent.parent
EXPORTS_DIR = ML_ROOT / "exports"
ANDROID_ASSETS_DIR = (
    ML_ROOT.parent
    / "buildheroios"
    / "local_modules"
    / "react-native-room-scanner"
    / "android"
    / "src"
    / "main"
    / "assets"
)
MODEL_NAME = "electrical_detector.tflite"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--int8", action="store_true",
                        help="full int8 quantization (smaller/faster, needs dataset for calibration)")
    parser.add_argument("--install", action="store_true",
                        help=f"copy the export into {ANDROID_ASSETS_DIR}")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    exported = Path(model.export(
        format="tflite",
        imgsz=args.imgsz,
        int8=args.int8,
        half=not args.int8,
        data=str(ML_ROOT / "electrical.yaml") if args.int8 else None,
    ))

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    staged = EXPORTS_DIR / MODEL_NAME
    shutil.copy2(exported, staged)
    print(f"exported: {staged}")

    if args.install:
        ANDROID_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        dest = ANDROID_ASSETS_DIR / MODEL_NAME
        shutil.copy2(staged, dest)
        print(f"installed: {dest}")


if __name__ == "__main__":
    main()
