# Dataset Guide — BR Electrical Fixtures (NBR 14136)

The bootstrap datasets from Roboflow Universe are dominated by **US NEMA**
outlets and rocker switches. Brazilian fixtures follow **NBR 14136** (round
recessed sockets, 3 round pins) and look substantially different — a model
trained only on US data misses most BR outlets. Own-data collection is what
makes the model production-ready.

## Collection targets

| Stage | Photos per class | Purpose |
|-------|-----------------|---------|
| MVP   | 150–300 | Usable in-app detection, high-confidence cases |
| Production | 1000+ | Robust across brands, lighting, wall colors |

After MVP ships, user scan frames (with consent) become the ongoing source.

## What to photograph

**outlet (0):** NBR 14136 sockets — single, double, in 4x2/4x4 plates,
floor boxes, extension blocks. Include popular brands: **Pial/Legrand,
Tramontina, WEG, Alumbra, Fame**. Include old-standard (pre-2011) NEMA-style
BR outlets still common in older homes.

**switch (1):** single/double/triple rocker switches, dimmers,
switch+outlet combo plates (annotate both classes separately).

**light (2):** ceiling fixtures (plafon), pendants, spots embutidos,
arandelas, bare-socket bulbs, fluorescent battens.

## Variation checklist (every class)

- Wall colors: white, colored, textured (grafiato), tile, exposed brick
- Lighting: daylight, artificial warm/cool, backlit, dim
- Angles: frontal, ~45°, low/high (simulate phone scan sweep)
- Distance: 0.5 m – 4 m (scan frames see fixtures small — include far shots)
- Occlusion: partially behind furniture, cables plugged in, furniture shadows
- Motion blur: a few slightly blurred shots (scan frames are not stills)

Shoot in **landscape at the capture resolution** the scanner uses (1920×1440
or similar). Avoid photos where the fixture exceeds ~40% of the frame — that
never happens during a room scan.

## Annotation

- Tool: [Roboflow](https://roboflow.com) free tier (exports YOLO format
  directly) or [CVAT](https://cvat.ai) self-hosted.
- Tight boxes around the **visible plate/fixture**, not the wall shadow.
- Combo plates (switch+outlet): one box per function, two classes.
- Skip fixtures under ~12 px — they teach the model noise.
- Keep the canonical class order: `outlet, switch, light` (see
  `../electrical.yaml`).

## Export / merge

Export from Roboflow in **YOLOv8** format and drop the split folders into
`ml/data/dataset/` (same layout `download_roboflow.py` creates). Re-run
`scripts/train.py` — it picks up everything under that layout.
