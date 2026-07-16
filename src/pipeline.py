# src/pipeline.py
from __future__ import annotations

import os

import cv2

from .detection.detector import FaceDetector
from .aus.au_extractor import AUExtractor
from .emotion_model.inference import EmotionInference

# py-feat typically exposes 20 AU columns
_DEFAULT_AU_DIM = 20


class EmotionPipeline:
    """
    End-to-end pipeline: image / frame → AU vector → valence + arousal.

    Parameters
    ----------
    weights_path : str | None
        Path to a trained EmotionRegressor state-dict (.pth).
        Pass None (default) to run with random weights for smoke-testing.
    au_dim : int
        Number of AU features expected by the regressor (default 20).
    """

    def __init__(
        self,
        weights_path: str | None = None,
        au_dim: int = _DEFAULT_AU_DIM,
    ) -> None:
        self.detector = FaceDetector()
        self.au_extractor = AUExtractor()
        self.emotion = EmotionInference(au_dim=au_dim, weights_path=weights_path)

    def run_on_image(self, img_path: str) -> dict:
        """
        Run the pipeline on a single image file.

        Returns
        -------
        dict with keys: valence, arousal, raw_aus  — or  {"error": reason}.
        """
        if not os.path.isfile(img_path):
            return {"error": f"file_not_found: {img_path}"}

        img = cv2.imread(img_path)
        if img is None:
            return {"error": f"cannot_read_image: {img_path}"}

        return self._process_frame(img)

    def run_on_frame(self, frame_bgr) -> dict:
        """
        Run the pipeline on a raw BGR numpy array (e.g. from cv2.VideoCapture).
        """
        return self._process_frame(frame_bgr)

    def _process_frame(self, img_bgr) -> dict:
        preds = self.detector.detect(img_bgr)
        au_vec = self.au_extractor.extract(preds)
        if au_vec is None:
            return {"error": "no_face_detected"}
        va = self.emotion.predict(au_vec)
        return {
            "valence": va["valence"],
            "arousal": va["arousal"],
            "raw_aus": au_vec.tolist(),
        }
