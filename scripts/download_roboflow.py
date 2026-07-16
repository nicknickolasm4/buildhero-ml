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

# Normalized source-class → canonical name. Lookup lowercases the name and
# turns -/_ into spaces first, so one entry covers light-switch/light_switch/
# Light Switch. Extend as new sources are added.
CLASS_ALIASES = {
    "outlet": "outlet",
    "outlets": "outlet",
    "socket": "outlet",
    "sockets": "outlet",
    "power outlet": "outlet",
    "electrical outlet": "outlet",
    "wall socket": "outlet",
    "tomada": "outlet",
    # Home-inspection datasets commonly label wall receptacles "plug(s)".
    "plug": "outlet",
    "plugs": "outlet",
    "power plug": "outlet",
    "wall plug": "outlet",
    "receptacle": "outlet",
    "plug 2pin": "outlet",
    "plug 3pin": "outlet",
    "plug rectangle": "outlet",
    # socket-and-switch / the combined workspace-acira project label the wall
    # receptacle "0 Outlet" — this one class is 6277 of the boxes, so missing
    # it turns thousands of real outlets into empty negatives.
    "0 outlet": "outlet",
    "switch": "switch",
    "switches": "switch",
    "light switch": "switch",
    "wall switch": "switch",
    "interruptor": "switch",
    # biiim/rocker labels the switch by its state.
    "on": "switch",
    "off": "switch",
    # mateo-ojeda/light-switch-mqb8v ships this typo.
    "ligth switch": "switch",
    "light": "light",
    "lights": "light",
    "lamp": "light",
    "light fixture": "light",
    "ceiling light": "light",
    "luminaria": "light",
    "luminária": "light",
}


def canonical_for(name: str) -> str | None:
    normalized = " ".join(name.lower().replace("-", " ").replace("_", " ").split())
    return CLASS_ALIASES.get(normalized)

# Public Roboflow Universe projects used for US-standard pre-training.
# Format: workspace, project, version. Override with --sources <json file>
# containing a list of {"workspace", "project", "version"} objects.
# roboflow-universe-projects/electrical-outlets and /light-switches (the
# original sources) were removed/renamed upstream — replaced 2026-07 with
# these confirmed-live public projects.
DEFAULT_SOURCES = [
    {"workspace": "yolov5-dtypd", "project": "plug-socket-detect", "version": 1},
    {"workspace": "biiim", "project": "rocker", "version": 1},
    # Added 2026-07-16: the user's own combined project (workspace-acira),
    # merging outlet/switch/damage sources — 6823 images. Its dominant class
    # is "0 Outlet" (6277 boxes), aliased above. NOTE: generate a version in
    # the Roboflow UI first (the project has 0 generated versions and the
    # download API can only fetch a generated version).
    {
        "workspace": "workspace-acira",
        "project": "switch-and-sockets-sensors-and-plugs-and-socket-damage",
        "version": "latest",
    },
]

# Images whose labels all got dropped become negatives. A few help precision;
# thousands (socket-and-switch v2: 6515 imgs / 244 boxes) drown the positives
# and teach the model to ignore fixtures, so cap them per source.
NEGATIVE_RATIO_CAP = 0.2
NEGATIVE_MIN_KEEP = 10

# Roboflow YOLOv8 exports use these split dir names.
SPLIT_MAP = {"train": "train", "valid": "valid", "test": "valid"}


def load_sources(path: str | None) -> list[dict]:
    if not path:
        return DEFAULT_SOURCES
    return json.loads(Path(path).read_text())


def resolve_version(project, requested) -> int:
    """A source may pin a version number or say "latest" (also the default
    when the key is omitted) — resolved against the project's generated
    versions at download time."""
    if requested not in (None, "latest"):
        return int(requested)
    versions = project.versions()
    if not versions:
        raise RuntimeError("project has no generated versions to download")
    return max(int(str(v.id).split("/")[-1]) for v in versions)


def download(source: dict, api_key: str) -> Path:
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(source["workspace"]).project(source["project"])
    version = resolve_version(project, source.get("version"))
    # Write back so the caller's merge prefix carries the real number.
    source["version"] = version
    dest = DOWNLOADS_DIR / f"{source['workspace']}__{source['project']}__v{version}"
    if dest.exists():
        print(f"  already downloaded: {dest.name}")
        return dest
    dataset = project.version(version).download("yolov8", location=str(dest))
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
        canonical = canonical_for(name)
        if canonical:
            id_map[idx] = CANONICAL[canonical]
    mapped = [names[i] for i in sorted(id_map)]
    dropped = [n for n in names if n not in mapped]
    print(f"  classes: keeping {mapped or 'none'}; dropping {dropped or 'none'}")
    if not id_map:
        print(f"  WARNING: no class of {names} maps to canonical classes — skipped")
        return 0, 0

    # First pass: collect every image with its remapped labels, so negatives
    # (all labels dropped) can be capped relative to the positives.
    entries: list[tuple[Path, str, list[str]]] = []  # (image, dst_split, kept_lines)
    for src_split, dst_split in SPLIT_MAP.items():
        img_dir = export_dir / src_split / "images"
        lbl_dir = export_dir / src_split / "labels"
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            src_label = lbl_dir / f"{img.stem}.txt"
            kept: list[str] = []
            if src_label.exists():
                for line in src_label.read_text().splitlines():
                    remapped = remap_line(line, id_map)
                    if remapped:
                        kept.append(remapped)
            entries.append((img, dst_split, kept))

    positives = [e for e in entries if e[2]]
    negatives = [e for e in entries if not e[2]]
    negative_cap = max(NEGATIVE_MIN_KEEP, int(len(positives) * NEGATIVE_RATIO_CAP))
    if len(negatives) > negative_cap:
        print(f"  capping negatives: keeping {negative_cap} of {len(negatives)} unlabeled images")
        negatives = negatives[:negative_cap]

    images = labels = 0
    for img, dst_split, kept in positives + negatives:
        dst_img = DATASET_DIR / dst_split / "images"
        dst_lbl = DATASET_DIR / dst_split / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        new_stem = f"{prefix}__{img.stem}"
        shutil.copy2(img, dst_img / f"{new_stem}{img.suffix.lower()}")
        (dst_lbl / f"{new_stem}.txt").write_text("\n".join(kept) + ("\n" if kept else ""))
        images += 1
        labels += len(kept)
    return images, labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", help="JSON file with a list of {workspace, project, version}")
    parser.add_argument(
        "--local", action="append", default=[], metavar="DIR",
        help="path to an already-downloaded/exported YOLOv8 dataset dir (must contain "
             "data.yaml) to merge without calling the Roboflow API — repeatable",
    )
    parser.add_argument(
        "--no-remote", action="store_true",
        help="skip the Roboflow API sources entirely (only merge --local dirs)",
    )
    args = parser.parse_args()

    remote_sources = [] if args.no_remote else load_sources(args.sources)
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if remote_sources and not api_key:
        print("ROBOFLOW_API_KEY is not set", file=sys.stderr)
        return 1

    total_images = total_labels = 0

    for source in remote_sources:
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

    for local_dir in args.local:
        path = Path(local_dir)
        print(f"local {path}")
        if not (path / "data.yaml").exists():
            print(f"  no data.yaml in {path} — skipped", file=sys.stderr)
            continue
        images, labels = merge(path, path.name)
        print(f"  merged {images} images, {labels} boxes")
        total_images += images
        total_labels += labels

    print(f"\ndataset at {DATASET_DIR}: {total_images} images, {total_labels} boxes")
    if total_images == 0:
        print("Nothing merged — check sources/API key/local paths.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
