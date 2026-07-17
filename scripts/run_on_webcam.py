#!/usr/bin/env python
# scripts/run_on_webcam.py
"""
Real-time facial emotion recognition via webcam.

Usage
-----
    python scripts/run_on_webcam.py [--weights models/emotion_regressor.pth] [--cam 0]

Controls
--------
    q  — quit
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline import EmotionPipeline

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
        type=int,
        default=0,
        help="Camera device index (default: 0).",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=3,
        help="Run detection every N-th frame and reuse the last result "
             "in between (default: 3). Use 1 to analyse every frame.",
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
    emotions = _emotions_from_va(valence, arousal)

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
    va_text = f"V:{valence:+.2f}  A:{arousal:+.2f}"
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


def main() -> None:
    args = parse_args()

    weights = args.weights
    if weights and not os.path.isfile(weights):
        print(f"[warning] weights file not found: {weights} — running with random weights.")
        weights = None

    pipeline = EmotionPipeline(weights_path=weights)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        sys.exit(f"[error] Cannot open camera index {args.cam}.")

    every = max(1, args.every)
    frame_idx = 0
    last_result: dict = {"error": "warming_up"}

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[warning] Empty frame received, skipping.")
            continue

        # Detection is by far the slowest step (~seconds on CPU), so run it
        # only on every N-th frame and redraw the last result in between —
        # the preview stays fluid while the analysis updates periodically.
        if frame_idx % every == 0:
            last_result = pipeline.run_on_frame(frame)
        frame_idx += 1

        draw_overlay(frame, last_result)

        cv2.imshow("Emotion Recognition — press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
