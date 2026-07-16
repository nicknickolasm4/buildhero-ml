#!/usr/bin/env python3
"""Bootstrap the electrical-fixture dataset from public Roboflow Universe projects.

Downloads each source project in YOLOv8 format, remaps its class ids to the
canonical contract (0 outlet, 1 switch, 2 light — see ../electrical.yaml) and
merges everything into ml/data/dataset/{train,valid}/{images,labels}.

Labels whose class has no canonical mapping are dropped; images left with an
empty label file are kept (negatives help). Requires ROBOFLOW_API_KEY.

Usage:
    python scripts/download_roboflow.py [--sources sources.json]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ML_ROOT / "data" / "dataset"
DOWNLOADS_DIR = ML_ROOT / "data" / "downloads"

CANONICAL = {"outlet": 0, "switch": 1, "light": 2}

# Lowercased source-class → canonical name. Extend as new sources are added.
CLASS_ALIASES = {
    "outlet": "outlet",
    "outlets": "outlet",
    "socket": "outlet",
    "sockets": "outlet",
    "power outlet": "outlet",
    "power_outlet": "outlet",
    "electrical outlet": "outlet",
    "wall socket": "outlet",
    "tomada": "outlet",
    "plug_2pin": "outlet",
    "plug_3pin": "outlet",
    "plug_rectangle": "outlet",
    "switch": "switch",
    "switches": "switch",
    "light switch": "switch",
    "light_switch": "switch",
    "light-switch": "switch",
    "wall switch": "switch",
    "interruptor": "switch",
    "light": "light",
    "lights": "light",
    "lamp": "light",
    "light fixture": "light",
    "ceiling light": "light",
    "luminaria": "light",
    "luminária": "light",
}

# Public Roboflow Universe projects used for US-standard pre-training.
# Format: workspace, project, version. Override with --sources <json file>
# containing a list of {"workspace", "project", "version"} objects.
# roboflow-universe-projects/electrical-outlets and /light-switches (the
# original sources) were removed/renamed upstream — replaced 2026-07 with
# these confirmed-live public projects.
DEFAULT_SOURCES = [
    {"workspace": "yolov5-dtypd", "project": "plug-socket-detect", "version": 1},
    {"workspace": "biiim", "project": "rocker", "version": 1},
]

# Roboflow YOLOv8 exports use these split dir names.
SPLIT_MAP = {"train": "train", "valid": "valid", "test": "valid"}


def load_sources(path: str | None) -> list[dict]:
    if not path:
        return DEFAULT_SOURCES
    return json.loads(Path(path).read_text())


def download(source: dict, api_key: str) -> Path:
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(source["workspace"]).project(source["project"])
    dest = DOWNLOADS_DIR / f"{source['workspace']}__{source['project']}__v{source['version']}"
    if dest.exists():
        print(f"  already downloaded: {dest.name}")
        return dest
    dataset = project.version(source["version"]).download("yolov8", location=str(dest))
    return Path(dataset.location)


def source_class_names(export_dir: Path) -> list[str]:
    """Read the class list from the export's data.yaml (names: list or dict)."""
    import yaml

    data = yaml.safe_load((export_dir / "data.yaml").read_text())
    names = data.get("names", [])
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)


def remap_line(line: str, id_map: dict[int, int]) -> str | None:
    parts = line.split()
    if not parts:
        return None
    src_id = int(parts[0])
    if src_id not in id_map:
        return None
    return " ".join([str(id_map[src_id])] + parts[1:])


def merge(export_dir: Path, prefix: str) -> tuple[int, int]:
    names = source_class_names(export_dir)
    id_map: dict[int, int] = {}
    for idx, name in enumerate(names):
        canonical = CLASS_ALIASES.get(name.lower().strip())
        if canonical:
            id_map[idx] = CANONICAL[canonical]
    if not id_map:
        print(f"  WARNING: no class of {names} maps to canonical classes — skipped")
        return 0, 0

    images = labels = 0
    for src_split, dst_split in SPLIT_MAP.items():
        img_dir = export_dir / src_split / "images"
        lbl_dir = export_dir / src_split / "labels"
        if not img_dir.is_dir():
            continue
        dst_img = DATASET_DIR / dst_split / "images"
        dst_lbl = DATASET_DIR / dst_split / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        for img in img_dir.iterdir():
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            new_stem = f"{prefix}__{img.stem}"
            shutil.copy2(img, dst_img / f"{new_stem}{img.suffix.lower()}")
            images += 1

            src_label = lbl_dir / f"{img.stem}.txt"
            kept: list[str] = []
            if src_label.exists():
                for line in src_label.read_text().splitlines():
                    remapped = remap_line(line, id_map)
                    if remapped:
                        kept.append(remapped)
            (dst_lbl / f"{new_stem}.txt").write_text("\n".join(kept) + ("\n" if kept else ""))
            labels += len(kept)
    return images, labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", help="JSON file with a list of {workspace, project, version}")
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("ROBOFLOW_API_KEY is not set", file=sys.stderr)
        return 1

    total_images = total_labels = 0
    for source in load_sources(args.sources):
        slug = f"{source['workspace']}/{source['project']}:v{source['version']}"
        print(f"source {slug}")
        try:
            export_dir = download(source, api_key)
        except Exception as exc:  # noqa: BLE001 — surface and continue with other sources
            print(f"  download failed: {exc}", file=sys.stderr)
            continue
        prefix = f"{source['project']}-v{source['version']}"
        images, labels = merge(export_dir, prefix)
        print(f"  merged {images} images, {labels} boxes")
        total_images += images
        total_labels += labels

    print(f"\ndataset at {DATASET_DIR}: {total_images} images, {total_labels} boxes")
    if total_images == 0:
        print("Nothing merged — check sources/API key.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
