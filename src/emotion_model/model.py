# src/emotion_model/model.py
import torch
import torch.nn as nn


class EmotionRegressor(nn.Module):
    """
    Lightweight MLP: AU vector → (valence, arousal).

    Parameters
    ----------
    au_dim : int
        Dimensionality of the input AU feature vector.
        py-feat typically returns 20 AUs.
    """

    def __init__(self, au_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(au_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # [valence, arousal]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
