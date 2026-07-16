#!/usr/bin/env python3
"""Build model-manifest.json for an OTA model release.

Hashes the exported mobile models (ml/exports/) and writes the manifest the
API serves via /ml-models/manifest. Copy the three files into
build-hero-api/ml-models/ and deploy — the Docker image ships the directory
as-is (see build-hero-api/ml-models/README.md).

Usage:
    python scripts/make_manifest.py --version model-v2
    cp exports/model-manifest.json exports/ElectricalDetector.mlpackage.zip \
       exports/electrical_detector.tflite ../build-hero-api/ml-models/

Note: the iOS export is zipped here (an .mlpackage is a directory).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent.parent
EXPORTS_DIR = ML_ROOT / "exports"

IOS_PACKAGE = EXPORTS_DIR / "ElectricalDetector.mlpackage"
IOS_ZIP = EXPORTS_DIR / "ElectricalDetector.mlpackage.zip"
ANDROID_FILE = EXPORTS_DIR / "electrical_detector.tflite"


def hashes(path: Path) -> dict:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha256.update(chunk)
            md5.update(chunk)
    return {
        "file": path.name,
        "sha256": sha256.hexdigest(),
        # md5 included because expo-file-system verifies downloads as md5.
        "md5": md5.hexdigest(),
        "size": path.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="release version, e.g. model-v2")
    args = parser.parse_args()

    assets: dict = {}

    if IOS_PACKAGE.is_dir():
        if IOS_ZIP.exists():
            IOS_ZIP.unlink()
        shutil.make_archive(str(IOS_ZIP.with_suffix("")), "zip", root_dir=EXPORTS_DIR,
                            base_dir=IOS_PACKAGE.name)
        assets["ios"] = hashes(IOS_ZIP)
    elif IOS_ZIP.exists():
        assets["ios"] = hashes(IOS_ZIP)

    if ANDROID_FILE.exists():
        assets["android"] = hashes(ANDROID_FILE)

    if not assets:
        print(f"Nothing to publish — run the export scripts first ({EXPORTS_DIR}).")
        return 1

    manifest = {
        "version": args.version,
        "published_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "assets": assets,
    }
    out = EXPORTS_DIR / "model-manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out}")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
