#!/usr/bin/env python
# scripts/run_on_webcam.py
"""
Real-time facial emotion recognition via webcam.

Usage
-----
    python scripts/run_on_webcam.py [--weights models/emotion_regressor.pth] [--cam 0]

`--cam` also accepts a stream URL, which is how this runs under WSL: the
camera is served from Windows by `scripts/serve_camera_windows.py` and
consumed here over HTTP.

    python scripts/run_on_webcam.py --cam http://172.22.128.1:8080/video

Controls
--------
    q  — quit
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline import EmotionPipeline
from src.emolog.logger import EmotionLogger, default_db_path

# ── Emotion centres in Valence-Arousal space ──────────────────────────────────
_EMOTION_CENTERS: dict[str, tuple[float, float]] = {
    "Neutral":   ( 0.00,  0.00),
    "Happy":     ( 0.60,  0.20),
    "Excited":   ( 0.50,  0.65),
    "Calm":      ( 0.30, -0.45),
    "Sad":       (-0.45, -0.30),
    "Angry":     (-0.55,  0.55),
    "Fearful":   (-0.25,  0.65),
    "Disgusted": (-0.50,  0.10),
    "Surprised": ( 0.10,  0.70),
}

# BGR colours per emotion
_EMOTION_COLORS: dict[str, tuple[int, int, int]] = {
    "Neutral":   (180, 180, 180),
    "Happy":     ( 60, 210,  60),
    "Excited":   (  0, 165, 255),
    "Calm":      (200, 210,  80),
    "Sad":       (210,  80,  30),
    "Angry":     ( 30,  30, 220),
    "Fearful":   (180,  40, 160),
    "Disgusted": ( 30, 130,  30),
    "Surprised": (220, 200,   0),
}

_SIGMA = 0.35  # Gaussian spread in VA space


def _emotions_from_va(valence: float, arousal: float) -> list[tuple[str, float]]:
    """Fuzzy Gaussian membership → normalised probabilities, sorted descending."""
    scores = {
        name: math.exp(-((valence - vc) ** 2 + (arousal - ac) ** 2) / (2 * _SIGMA ** 2))
        for name, (vc, ac) in _EMOTION_CENTERS.items()
    }
    total = sum(scores.values()) or 1.0
    return sorted(((n, s / total) for n, s in scores.items()), key=lambda x: x[1], reverse=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time facial emotion recognition.")
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to a trained EmotionRegressor .pth file (optional).",
    )
    parser.add_argument(
        "--cam",
        default="0",
        help="Camera device index, or a stream URL such as "
             "http://172.22.128.1:8080/video (default: 0).",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=3,
        help="Run detection every N-th frame and reuse the last result "
             "in between (default: 3). Use 1 to analyse every frame.",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable logging of measurements to the SQLite database.",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Path to the emolog SQLite database "
             "(default: <repo>/data/emolog.db).",
    )
    return parser.parse_args()


def draw_overlay(frame, result: dict) -> None:
    """Render fuzzy emotion bars and a VA circumplex on the frame in-place."""
    h, w = frame.shape[:2]

    if "error" in result:
        cv2.putText(frame, f"[{result['error']}]", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return

    valence = result["valence"]
    arousal = result["arousal"]

    if math.isnan(valence) or math.isnan(arousal):
        cv2.putText(frame, "[No face detected]", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return

    emotions_map = result.get("emotions")
    va_trained = result.get("va_trained", False)

    if emotions_map:
        # Real probabilities from py-feat's trained emotion classifier.
        emotions = sorted(emotions_map.items(), key=lambda x: x[1], reverse=True)
        if not va_trained:
            # No trained AU→V/A regressor — derive V/A as the probability-
            # weighted centroid of the emotion centres, so the circumplex
            # reflects the real classifier output instead of random weights.
            valence = sum(p * _EMOTION_CENTERS.get(n, (0.0, 0.0))[0]
                          for n, p in emotions)
            arousal = sum(p * _EMOTION_CENTERS.get(n, (0.0, 0.0))[1]
                          for n, p in emotions)
            va_source = "emotions"
        else:
            va_source = "regressor"
    else:
        # Fallback: fuzzy Gaussian membership derived from raw V/A.
        emotions = _emotions_from_va(valence, arousal)
        va_source = "regressor" if va_trained else "untrained!"

    # ── 1. Semi-transparent panel background ─────────────────────────────────
    top_n = 5
    panel_x, panel_y = 10, 10
    bar_w_max = 180
    row_h = 28
    panel_h = top_n * row_h + 20
    panel_w = bar_w_max + 130

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x - 4, panel_y - 4),
                  (panel_x + panel_w, panel_y + panel_h), (15, 15, 15), -1)
    # Darker panel (0.45 → 0.70) so light text stays readable on bright scenes.
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    # ── 2. Emotion bars ───────────────────────────────────────────────────────
    for i, (name, prob) in enumerate(emotions[:top_n]):
        y = panel_y + 18 + i * row_h
        color = _EMOTION_COLORS.get(name, (200, 200, 200))

        cv2.putText(frame, f"{name:<10}", (panel_x + 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1, cv2.LINE_AA)

        bar_x = panel_x + 95
        cv2.rectangle(frame, (bar_x, y - 13), (bar_x + bar_w_max, y + 4),
                      (60, 60, 60), -1)
        fill = int(bar_w_max * prob)
        cv2.rectangle(frame, (bar_x, y - 13), (bar_x + fill, y + 4), color, -1)

        # White percentage text with a thin dark outline — readable regardless
        # of the emotion colour (grey-on-grey "Neutral" was nearly invisible).
        pct = f"{prob * 100:4.1f}%"
        cv2.putText(frame, pct, (bar_x + bar_w_max + 5, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, pct, (bar_x + bar_w_max + 5, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    # ── 3. Raw V/A values (small, below panel) ───────────────────────────────
    va_text = f"V:{valence:+.2f}  A:{arousal:+.2f}  [{va_source}]"
    cv2.putText(frame, va_text,
                (panel_x + 2, panel_y + panel_h + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, va_text,
                (panel_x + 2, panel_y + panel_h + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

    # ── 4. VA circumplex (bottom-right corner) ────────────────────────────────
    cx, cy, r = w - 75, h - 75, 58

    overlay2 = frame.copy()
    cv2.circle(overlay2, (cx, cy), r, (25, 25, 25), -1)
    cv2.addWeighted(overlay2, 0.5, frame, 0.5, 0, frame)

    cv2.line(frame, (cx - r, cy), (cx + r, cy), (80, 80, 80), 1)
    cv2.line(frame, (cx, cy - r), (cx, cy + r), (80, 80, 80), 1)
    cv2.circle(frame, (cx, cy), r, (80, 80, 80), 1)

    cv2.putText(frame, "+V", (cx + r - 14, cy - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (110, 110, 110), 1)
    cv2.putText(frame, "+A", (cx + 3, cy - r + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (110, 110, 110), 1)

    dot_x = cx + int(valence * (r - 4))
    dot_y = cy - int(arousal * (r - 4))
    dot_color = _EMOTION_COLORS.get(emotions[0][0], (0, 200, 0))
    cv2.circle(frame, (dot_x, dot_y), 7, dot_color, -1)
    cv2.circle(frame, (dot_x, dot_y), 7, (255, 255, 255), 1)


def _is_wsl() -> bool:
    """Return True when running inside Windows Subsystem for Linux."""
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def parse_source(value: str) -> "int | str":
    """Interpret --cam as a device index when numeric, otherwise as a URL."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def open_camera(source: "int | str") -> "cv2.VideoCapture":
    """
    Open the camera, falling back to DirectShow when the default backend
    (MSMF on Windows) fails — MSMF is known to stop delivering frames
    with errors like 0xC00D3EA2 when the device hiccups or another app
    grabs it.

    The DirectShow fallback applies to local device indices only; for a
    stream URL the default backend (FFMPEG) is the only sensible one.
    """
    cap = cv2.VideoCapture(source)
    if cap.isOpened():
        return cap
    cap.release()
    if isinstance(source, int) and hasattr(cv2, "CAP_DSHOW"):
        print("[warning] Default backend failed — trying DirectShow.")
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    return cap


def _find_camera(preferred: "int | str") -> "cv2.VideoCapture | None":
    """
    Try the preferred source first; if it is a device index and fails,
    probe indices 0–4. Returns an opened VideoCapture or None.

    A URL is never substituted with a local index — silently falling back
    to a different camera would hide the fact that the stream is down.
    """
    cap = open_camera(preferred)
    if cap.isOpened():
        return cap
    if not isinstance(preferred, int):
        return None
    for idx in range(5):
        if idx == preferred:
            continue
        cap = open_camera(idx)
        if cap.isOpened():
            print(f"[info] Camera {preferred} unavailable — using index {idx}.")
            return cap
    return None


def main() -> None:
    args = parse_args()

    weights = args.weights
    if weights and not os.path.isfile(weights):
        print(f"[warning] weights file not found: {weights} — running with random weights.")
        weights = None

    pipeline = EmotionPipeline(weights_path=weights)

    source = parse_source(args.cam)

    # A local device index under WSL opens successfully and then never
    # delivers a frame (USB/IP does not carry the isochronous transfers
    # UVC streaming needs). Failing here beats ~10 s of `select() timeout`
    # that looks like a camera warming up.
    if isinstance(source, int) and _is_wsl():
        sys.exit(
            f"[error] --cam {source} is a local device index, which cannot "
            "work under WSL.\n"
            "WSL2 does not expose USB cameras, and usbipd is not enough: UVC "
            "streaming needs\nisochronous USB transfers, which USB/IP does not "
            "carry. The device opens and\nthen never delivers a frame.\n\n"
            "Serve the camera from Windows instead:\n"
            "  # on Windows, leave running:\n"
            "  .\\.venv\\Scripts\\python.exe scripts/serve_camera_windows.py\n"
            "  # then here, with the host IP from `ip route show default`:\n"
            "  python scripts/run_on_webcam.py --cam http://<HOST_IP>:8080/video\n\n"
            "See TROUBLESHOOTING.md section 3."
        )

    cap = _find_camera(source)
    if cap is None or not cap.isOpened():
        if isinstance(source, int):
            msg = f"[error] Cannot open camera index {source}."
        else:
            msg = f"[error] Cannot open stream {source}."
        if _is_wsl():
            msg += (
                "\n\nYou appear to be running inside WSL. "
                "WSL2 does not expose USB cameras as /dev/video* devices by default,\n"
                "and attaching one with usbipd is not enough: UVC streaming needs\n"
                "isochronous USB transfers, which USB/IP does not carry, so the\n"
                "device opens but never delivers a frame (select() timeout).\n"
                "Options:\n"
                "  1. Serve the camera from Windows and consume it here (recommended):\n"
                "       # on Windows:\n"
                "       .\\.venv\\Scripts\\python.exe scripts/serve_camera_windows.py\n"
                "       # then in WSL, with the host IP from `ip route show default`:\n"
                "       python scripts/run_on_webcam.py --cam http://<HOST_IP>:8080/video\n"
                "  2. Run the whole script with Windows-native Python instead of WSL:\n"
                "       py scripts/run_on_webcam.py\n"
            )
        sys.exit(msg)

    logger = None
    if not args.no_log:
        logger = EmotionLogger(
            db_path=args.db,
            source=f"webcam:{args.cam}",
            va_trained=weights is not None,
        )
        print(f"[emolog] Logging to {args.db or default_db_path()} "
              f"(session #{logger.session_id}). Use --no-log to disable.")

    every = max(1, args.every)
    frame_idx = 0
    last_result: dict = {"error": "warming_up"}

    # Camera-failure handling: MSMF on Windows can permanently stop
    # delivering frames (error 0xC00D3EA2). Instead of spinning forever
    # on warnings, back off briefly, try to reconnect a few times, and
    # exit cleanly (the emolog session still gets saved via `finally`).
    MAX_EMPTY_STREAK = 30    # ~3 s of consecutive empty frames
    MAX_RECONNECTS = 3
    empty_streak = 0
    reconnects = 0

    print("Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                empty_streak += 1
                if empty_streak == 1:
                    print("[warning] Empty frame received — waiting for "
                          "the camera to recover...")
                time.sleep(0.1)
                if empty_streak >= MAX_EMPTY_STREAK:
                    if reconnects >= MAX_RECONNECTS:
                        print("[error] Camera is not recovering — exiting.")
                        break
                    reconnects += 1
                    print(f"[warning] Reconnecting to camera "
                          f"({reconnects}/{MAX_RECONNECTS})...")
                    cap.release()
                    time.sleep(1.0)
                    cap = open_camera(source)
                    empty_streak = 0
                continue
            empty_streak = 0
            reconnects = 0

            # Detection is by far the slowest step (~seconds on CPU), so run
            # it only on every N-th frame and redraw the last result in
            # between — the preview stays fluid while the analysis updates
            # periodically.
            if frame_idx % every == 0:
                last_result = pipeline.run_on_frame(frame)
                if logger:
                    logger.log(last_result)  # error results are ignored
            frame_idx += 1

            draw_overlay(frame, last_result)

            cv2.imshow("Emotion Recognition - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if logger:
            logger.close()
            print(f"[emolog] Session #{logger.session_id} saved "
                  f"({logger.count} measurements).")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
