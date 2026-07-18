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
EmoFACS/
├─ src/
│  ├─ detection/detector.py       # face detection via py-feat
│  ├─ aus/au_extractor.py         # AU vector extraction
│  ├─ emotion_model/
│  │  ├─ model.py                 # EmotionRegressor MLP
│  │  └─ inference.py             # inference wrapper
│  ├─ emolog/logger.py            # SQLite logging of measurements
│  └─ pipeline.py                 # orchestrates all modules
├─ scripts/
│  ├─ run_on_image.py             # single-image demo
│  ├─ run_on_webcam.py            # real-time webcam demo
│  ├─ serve_camera_windows.py     # MJPEG camera server (for WSL setups)
│  ├─ analyze_history.py          # analysis of logged sessions
│  └─ log_factors.py              # annotate sessions with context factors
├─ models/                        # place trained .pth weights here
├─ data/emolog.db                 # measurement log (created on first run)
├─ requirements.txt
├─ TROUBLESHOOTING.md             # environment issues (WSL, camera, versions)
└─ README.md
```

---

## Setup

Windows:

```powershell
py -m venv .venv-win
.\.venv-win\Scripts\python.exe -m pip install --upgrade pip
.\.venv-win\Scripts\python.exe -m pip install -r requirements.txt
```

Linux / macOS / WSL:

```bash
python -m venv .venv-wsl
./.venv-wsl/bin/python -m pip install -r requirements.txt
```

Give each platform its own environment directory. A single `.venv` shared
by Windows and WSL half-works and is confusing to debug — see
[TROUBLESHOOTING.md](TROUBLESHOOTING.md) section 8.

The version bounds in `requirements.txt` are not cosmetic: py-feat 0.6.2
uses APIs that were removed from newer `torchvision` and `scipy`, and it
has no newer release to upgrade to. Relaxing those upper bounds
reintroduces two separate `ImportError`s — section 4 has the details.

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

The right command depends on where you run it. **Pick your platform first
— the two are not interchangeable.**

#### On Windows (native)

The camera is a local device, so no stream setup is needed:

```powershell
.\.venv-win\Scripts\python.exe scripts\run_on_webcam.py --every 10
# specify camera index and weights:
.\.venv-win\Scripts\python.exe scripts\run_on_webcam.py --cam 0 --weights models\emotion_regressor.pth
```

**Always call the interpreter by its full path**, never a bare `python`
or `pip`. A venv whose directory was renamed keeps stale absolute paths
compiled into `Scripts\*.exe`, so `pip install` can silently install into
the wrong interpreter — see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
section 9.

This is a verified working configuration: torch 2.6.0+cpu, torchvision
0.21.0+cpu, scipy 1.13.1, py-feat 0.6.2. Detection runs at ~2 s per frame
on CPU, which `--every 10` keeps comfortable.

#### On WSL

`--cam` is **required** and must point at a stream URL — a local device
index cannot work here. See [Running under WSL](#running-under-wsl) below
for the two-process setup and the reason behind it.

```bash
python scripts/run_on_webcam.py --cam http://<HOST_IP>:8080/video
```

> **Known issue:** capture and analysis work under WSL, but the preview
> window does not render the video. See
> [TROUBLESHOOTING.md](TROUBLESHOOTING.md) sections 5 and 7.

Options:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--cam` | `0` | Camera device index, or a stream URL (under WSL a URL is mandatory) |
| `--weights` | *none* | Trained `.pth` file; without it the regressor uses random weights |
| `--every` | `3` | Run detection every N-th frame, reusing the last result in between |
| `--no-log` | off | Do not write measurements to the database |
| `--db` | `data/emolog.db` | Path to the measurement log |

Detection costs roughly **2–5 s per frame on CPU**, so `--every` trades
analysis rate for a fluid preview. The default of `3` is far too low for
CPU-only machines — `--every 10` or higher is a more realistic starting
point. Use `--every 1` to analyse every frame.

#### Running under WSL

**Why a plain `python scripts/run_on_webcam.py` cannot work here.** WSL2
does not expose USB cameras, and attaching one with `usbipd` is not
sufficient: webcam streaming needs isochronous USB transfers, which
USB/IP does not carry. The device opens, `/dev/video0` appears, and then
no frame ever arrives — the symptom is a repeating `select() timeout`.

The fix is to let Windows read the camera and stream it to WSL over HTTP.
The whole ML pipeline stays in WSL; Windows only needs `cv2` — no `torch`,
no `py-feat`.

This needs **two processes running at once, one per system.**

```powershell
# 1. Windows (PowerShell, from the repo root) — leave this running
.\.venv\Scripts\python.exe scripts\serve_camera_windows.py
```

```bash
# 2. WSL — find the Windows host address
ip route show default | awk '{print $3}'

# 3. WSL — run the pipeline against the stream
python scripts/run_on_webcam.py --cam http://<HOST_IP>:8080/video
```

If step 3 cannot connect, the Windows firewall is the usual cause —
see [TROUBLESHOOTING.md](TROUBLESHOOTING.md) section 3 for the firewall
rule and a `curl` check that isolates the stream from the pipeline.

Per-frame overlay (top-left of the video window):

```
Neutral       ████████░░  48.2%
Happy         █████░░░░░  30.1%
Sad           ██░░░░░░░░   9.3%
Surprised     █░░░░░░░░░   5.5%
Angry         ░░░░░░░░░░   3.7%
V:+0.31  A:-0.04  [emotions]
```

The bars come from py-feat's trained classifier, which emits seven
categories: Angry, Disgusted, Fearful, Happy, Sad, Surprised, Neutral.
The tag after the V/A values shows where those two numbers came from:

| Tag | Meaning |
| --- | --- |
| `emotions` | probability-weighted centroid of the emotion centres |
| `regressor` | the trained AU→V/A regressor (requires `--weights`) |
| `untrained!` | regressor with random weights — values are meaningless |

`Calm` and `Excited` exist in the valence-arousal map and can appear in
the fallback path, but never in the classifier output above.

Press **q** to quit.

### Logged data

Every measurement is written to `data/emolog.db` (SQLite) unless
`--no-log` is given. Two helper scripts work on that database:

```bash
python scripts/analyze_history.py     # summarise logged sessions over time
python scripts/log_factors.py         # record daily context (sleep, caffeine, ...)
```

The schema is three tables: `sessions` (one row per run), `measurements`
(one row per successful detection) and `daily_factors` (one row per day,
filled in by hand). A run that never receives a frame still creates its
`sessions` row, so failed attempts show up as sessions with zero
measurements. To remove those:

```bash
sqlite3 data/emolog.db "DELETE FROM sessions WHERE id NOT IN (SELECT DISTINCT session_id FROM measurements);"
```

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
