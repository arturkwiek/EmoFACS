#!/usr/bin/env python
# scripts/run_on_image.py
"""
Run the emotion pipeline on a single image and print the result.

Usage
-----
    python scripts/run_on_image.py <image_path> [--weights models/emotion_regressor.pth]

Examples
--------
    python scripts/run_on_image.py photo.jpg
    python scripts/run_on_image.py photo.jpg --weights models/emotion_regressor.pth
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Allow running from the repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline import EmotionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Facial emotion recognition on a static image.")
    parser.add_argument("image", help="Path to the input image file.")
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to a trained EmotionRegressor .pth file (optional).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    weights = args.weights
    if weights and not os.path.isfile(weights):
        print(f"[warning] weights file not found: {weights} — running with random weights.")
        weights = None

    pipeline = EmotionPipeline(weights_path=weights)
    result = pipeline.run_on_image(args.image)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
