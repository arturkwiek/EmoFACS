# Facial Emotion Recognition — FACS-based pipeline

End-to-end pipeline: **image / webcam frame → Action Units → valence + arousal**.

```
image
  └─ FaceDetector  (py-feat / MTCNN)
       └─ AUExtractor  → 20-dim AU vector
            └─ EmotionRegressor (MLP)  → valence, arousal
```

---

## Project structure

```
emotion-facs/
├─ src/
│  ├─ detection/detector.py       # face detection via py-feat
│  ├─ aus/au_extractor.py         # AU vector extraction
│  ├─ emotion_model/
│  │  ├─ model.py                 # EmotionRegressor MLP
│  │  └─ inference.py             # inference wrapper
│  └─ pipeline.py                 # orchestrates all modules
├─ scripts/
│  ├─ run_on_image.py             # single-image demo
│  └─ run_on_webcam.py            # real-time webcam demo
├─ models/                        # place trained .pth weights here
├─ requirements.txt
└─ README.md
```

---

## Setup

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Static image

```bash
python scripts/run_on_image.py path/to/photo.jpg
# with trained weights:
python scripts/run_on_image.py photo.jpg --weights models/emotion_regressor.pth
```

Sample output:
```json
{
  "valence": 0.312,
  "arousal": -0.045,
  "raw_aus": [0.0, 1.2, 0.0, ...]
}
```

### Webcam (real-time)

```bash
python scripts/run_on_webcam.py
# specify camera index and weights:
python scripts/run_on_webcam.py --cam 0 --weights models/emotion_regressor.pth
```

Per-frame overlay (top-left of the video window):

```
Calm          ████████░░  48.2%
Neutral       █████░░░░░  30.1%
Happy         ██░░░░░░░░   9.3%
Sad           █░░░░░░░░░   5.5%
Disgusted     ░░░░░░░░░░   3.7%
V:+0.31  A:-0.04
```

Press **q** to quit.

---

## Training

The `EmotionRegressor` is a small MLP that maps AU vectors to (valence, arousal).
You can fine-tune it on labelled datasets such as **AffectNet** or **RAF-DB**:

1. Extract AU vectors from all images using `FaceDetector` + `AUExtractor`.
2. Pair each AU vector with its ground-truth valence/arousal label.
3. Train with MSE loss and save with `torch.save(model.state_dict(), "models/emotion_regressor.pth")`.

Without a trained weights file the pipeline still runs — it just uses randomly initialised weights.

---

## Swapping the face detector

`FaceDetector` in `src/detection/detector.py` wraps py-feat's `Detector`.
To replace it with RetinaFace or another backend, edit only that file and keep the same `detect(img_bgr) → preds` interface.
