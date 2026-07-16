#!/usr/bin/env python3
"""Export trained YOLO weights to CoreML for the iOS room scanner.

Produces ElectricalDetector.mlpackage with embedded NMS, so Swift receives
final boxes (VNRecognizedObjectObservation) instead of raw tensors.
--install copies it into the react-native-room-scanner iOS module.

Usage:
    python scripts/export_coreml.py --weights runs/detect/electrical/weights/best.pt --install
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent.parent
EXPORTS_DIR = ML_ROOT / "exports"
IOS_MODULE_DIR = (
    ML_ROOT.parent / "buildheroios" / "local_modules" / "react-native-room-scanner" / "ios"
)
MODEL_NAME = "ElectricalDetector.mlpackage"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, default=960)  # 960: outlets ~2-4 m away are too few pixels at 640 for reliable detection
    parser.add_argument("--install", action="store_true",
                        help=f"copy the export into {IOS_MODULE_DIR}")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    exported = Path(model.export(format="coreml", nms=True, imgsz=args.imgsz, half=True))

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    staged = EXPORTS_DIR / MODEL_NAME
    if staged.exists():
        shutil.rmtree(staged)
    shutil.copytree(exported, staged)
    print(f"exported: {staged}")

    if args.install:
        dest = IOS_MODULE_DIR / MODEL_NAME
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(staged, dest)
        print(f"installed: {dest}")
        print("Run `pod install` in buildheroios/ios and rebuild the app.")


if __name__ == "__main__":
    main()
