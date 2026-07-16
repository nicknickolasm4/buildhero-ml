# Training on Google Colab (no local GPU needed)

Use this when your machine has no GPU. Dataset download and training run in
the Colab notebook; export (CoreML/TFLite) runs afterward on your own
machine, since `--install` copies the model straight into the monorepo
(`buildheroios/local_modules/...`).

## 1. Roboflow account

1. Sign up at [roboflow.com](https://roboflow.com) (free tier is enough —
   you're only downloading existing public datasets).
2. **Settings → Roboflow API** → copy your **Private API Key**.

## 2. Open a Colab notebook

1. Go to [colab.research.google.com](https://colab.research.google.com) →
   **New notebook**.
2. `Runtime → Change runtime type → T4 GPU` → Save.

## 3. Store the API key as a Colab secret (never paste it into a cell)

1. Click the key icon 🔑 in the left sidebar.
2. **Add new secret** → name `ROBOFLOW_API_KEY`, value = the key from step 1.
3. Enable notebook access for the secret (toggle it on).

## 4. Clone the repo

```python
!git clone https://github.com/nicknickolasm4/buildhero-ml.git
%cd /content/buildhero-ml
!ls
```

Use the **absolute** path (`/content/buildhero-ml`) for `%cd`, not a relative
one — if you re-run this cell later without restarting the runtime, a
relative `%cd` nests into itself (`buildhero-ml/buildhero-ml/...`) and every
path after it breaks.

## 5. Install dependencies

```python
!pip install -r requirements.txt
```

## 6. Load the API key into the environment

```python
from google.colab import userdata
import os
os.environ["ROBOFLOW_API_KEY"] = userdata.get("ROBOFLOW_API_KEY")
```

## 7. Download the dataset

```python
!python scripts/download_roboflow.py
```

Confirm the last line reads `dataset at .../data/dataset: N images, N boxes`
with **N > 0** before continuing. `0 images, 0 boxes` means a source is
missing, private, or the API key didn't load — don't move on until this is
fixed.

## 8. Train

```python
!python scripts/train.py
```

Default: `yolo11n`, 100 epochs, 640px. Weights land at
`runs/detect/electrical/weights/best.pt` (the run name is `electrical`, set
in `train.py` — **not** Ultralytics' generic default of `train`).

## 9. Save the weights before the session disconnects

Colab's VM is ephemeral — everything is wiped on disconnect. Save `best.pt`
somewhere durable before doing anything else:

```python
from google.colab import drive
drive.mount('/content/drive')
!cp runs/detect/electrical/weights/best.pt /content/drive/MyDrive/best.pt
```

## 10. Download the weights to your machine

```python
from google.colab import files
files.download('runs/detect/electrical/weights/best.pt')
```

## 11. Export — run this locally, not in Colab

`--install` copies the exported model into `buildheroios/local_modules/...`,
which only exists in your local monorepo checkout, not in the Colab VM.
Put the downloaded `best.pt` under
`ml/runs/detect/electrical/weights/best.pt` in your local checkout, then:

```bash
cd ml
source .venv/bin/activate
python scripts/run_local_pipeline.py --skip-download --skip-train
```

(see `README.md` for the one-shot local pipeline script, or run the export
scripts individually.)
