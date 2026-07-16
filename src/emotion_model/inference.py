# src/emotion_model/inference.py
from __future__ import annotations

import numpy as np
import torch

from .model import EmotionRegressor


class EmotionInference:
    """
    Wrapper around EmotionRegressor for single-sample inference.

    Parameters
    ----------
    au_dim : int
        Number of AU features (must match the trained model).
    weights_path : str | None
        Path to a saved state-dict (.pth).  If None, the model runs with
        randomly initialised weights (useful for smoke-testing the pipeline).
    """

    def __init__(self, au_dim: int, weights_path: str | None = None) -> None:
        self.model = EmotionRegressor(au_dim)
        if weights_path:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, au_vec: np.ndarray) -> dict[str, float]:
        """
        Parameters
        ----------
        au_vec : np.ndarray
            1-D float32 array of length au_dim.

        Returns
        -------
        dict with keys "valence" and "arousal" (both float).
        """
        x = torch.tensor(au_vec, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            va = self.model(x)[0].numpy()
        return {"valence": float(va[0]), "arousal": float(va[1])}
