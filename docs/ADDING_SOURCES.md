# Adding more sources and object classes

## Combining multiple Roboflow sources (existing classes)

`download_roboflow.py` already merges as many sources as you list — nothing
extra to enable. Two ways to list them:

**Edit the default list** — `DEFAULT_SOURCES` in the script:

```python
DEFAULT_SOURCES = [
    {"workspace": "yolov5-dtypd", "project": "plug-socket-detect", "version": 1},
    {"workspace": "biiim", "project": "rocker", "version": 1},
    {"workspace": "another-workspace", "project": "another-project", "version": 2},
]
```

**Or pass a JSON file** (overrides the default list for that run, doesn't
require editing the script):

```json
[
  {"workspace": "yolov5-dtypd", "project": "plug-socket-detect", "version": 1},
  {"workspace": "another-workspace", "project": "another-project", "version": 2}
]
```

```bash
python scripts/download_roboflow.py --sources my_sources.json
```

## Combining Roboflow + manual/local folders in one run

`--local` is repeatable and works alongside remote sources — combine
everything in a single invocation:

```bash
python scripts/download_roboflow.py \
  --sources my_sources.json \
  --local /path/to/own/dataset-1 \
  --local /path/to/own/dataset-2
```

Skip the Roboflow API entirely (no account/key needed) with `--no-remote`:

```bash
python scripts/download_roboflow.py --no-remote --local /path/to/own/dataset
```

Each `--local` dir must be a standard YOLOv8 export (contains `data.yaml` +
`train/valid` `images`/`labels` folders) — same as what `--local` in
`OWN_IMAGES.md` describes. All sources — remote and local — merge into the
same `ml/data/dataset/` before training, so `train.py` sees one combined
dataset regardless of where each part came from.

## Adding a brand-new object class

Outlet/switch/light are the only classes wired end-to-end today (mobile
apps read detections by index — see `README.md`: **never reorder existing
ids**). Adding a 4th class (e.g. a smoke detector, HVAC vent, water
shutoff) touches three places:

1. **`electrical.yaml`** — bump `nc` and append the new name; canonical ids
   0/1/2 (outlet/switch/light) must stay exactly as they are, new class is
   id `3`:
   ```yaml
   nc: 4
   names: ['outlet', 'switch', 'light', 'smoke_detector']
   ```

2. **`scripts/download_roboflow.py`** — add the new id to `CANONICAL`, and
   map every class-name variant you expect from source datasets into it via
   `CLASS_ALIASES`:
   ```python
   CANONICAL = {"outlet": 0, "switch": 1, "light": 2, "smoke_detector": 3}

   CLASS_ALIASES = {
       # ...existing entries...
       "smoke detector": "smoke_detector",
       "smoke_alarm": "smoke_detector",
       "detector": "smoke_detector",
   }
   ```

3. **Find sources for the new class** — same as `OWN_IMAGES.md`: search
   Roboflow Universe for a public project, or annotate your own photos, then
   add it via `DEFAULT_SOURCES`/`--sources` (remote) or `--local` (manual) —
   whichever combination you're already using for the other classes works
   the same way for this one.

**Before shipping a new class**, the mobile side (`buildheroios` — CoreML
output count, Android TFLite consumer) needs updating too so it knows what
to do with detections of the new id. That's outside this script — flag it
as a follow-up, don't assume it "just works" once the model exports.
