#!/usr/bin/env python3
"""Run the full pipeline locally: download dataset, train, export CoreML + TFLite.

Requires ROBOFLOW_API_KEY in the environment unless --skip-download.

Usage:
    python scripts/run_local_pipeline.py
    python scripts/run_local_pipeline.py --skip-download
    python scripts/run_local_pipeline.py --skip-download --skip-train --weights runs/detect/electrical/weights/best.pt
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ML_ROOT / "scripts"
DEFAULT_WEIGHTS = ML_ROOT / "runs" / "detect" / "electrical" / "weights" / "best.pt"


def run(*args: str) -> None:
    print(f"\n$ {' '.join(args)}")
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS),
                         help="weights to export (default: freshly trained best.pt)")
    # Defaults sized for a 16 GB Apple Silicon Mac: MPS GPU, small batch at
    # the 960 input the mobile export uses.
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--device", default="mps",
                        help="'mps' (Apple Silicon), 'cpu', or a CUDA index")
    parser.add_argument("--skip-tflite", action="store_true",
                        help="skip the Android TFLite export (needs TensorFlow)")
    args = parser.parse_args()

    if not args.skip_download:
        if not os.environ.get("ROBOFLOW_API_KEY"):
            sys.exit("ROBOFLOW_API_KEY is not set — export it or pass --skip-download")
        run(sys.executable, str(SCRIPTS_DIR / "download_roboflow.py"))

    if not args.skip_train:
        run(
            sys.executable, str(SCRIPTS_DIR / "train.py"),
            "--imgsz", str(args.imgsz),
            "--epochs", str(args.epochs),
            "--batch", str(args.batch),
            "--device", args.device,
        )

    weights = Path(args.weights)
    if not weights.exists():
        sys.exit(f"weights not found: {weights} (train first, or pass --weights)")

    run(sys.executable, str(SCRIPTS_DIR / "export_coreml.py"),
        "--weights", str(weights), "--imgsz", str(args.imgsz), "--install")
    if not args.skip_tflite:
        run(sys.executable, str(SCRIPTS_DIR / "export_tflite.py"),
            "--weights", str(weights), "--imgsz", str(args.imgsz), "--install")

    print("\ndone — models installed into buildheroios/local_modules/react-native-room-scanner/")


if __name__ == "__main__":
    main()
