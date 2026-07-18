# src/aus/au_extractor.py
import numpy as np

# py-feat's emotion column names → display names used across the project.
# Both 'neutrality' (0.6.x) and 'neutral' are handled for version safety.
_EMOTION_NAME_MAP = {
    "anger": "Angry",
    "disgust": "Disgusted",
    "fear": "Fearful",
    "happiness": "Happy",
    "sadness": "Sad",
    "surprise": "Surprised",
    "neutrality": "Neutral",
    "neutral": "Neutral",
}


class AUExtractor:
    def extract(self, preds):
        """
        Extract the AU vector from a py-feat Fex prediction object.

        Returns a 1-D float32 numpy array, or None if no face was detected.
        """
        if preds is None or len(preds.aus) == 0:
            return None
        au_vec = preds.aus.values[0]
        return np.array(au_vec, dtype=np.float32)

    def extract_emotions(self, preds):
        """
        Extract real emotion probabilities from py-feat's built-in,
        trained emotion classifier (preds.emotions).

        Returns
        -------
        dict[str, float] mapping display names ("Happy", "Angry", ...)
        to probabilities, or None if unavailable / no face detected.
        """
        emotions = getattr(preds, "emotions", None)
        if emotions is None or len(emotions) == 0:
            return None

        row = emotions.iloc[0]
        out: dict = {}
        for col, val in row.items():
            name = _EMOTION_NAME_MAP.get(str(col).lower())
            if name is None:
                continue
            val = float(val)
            if not np.isnan(val):
                out[name] = val
        return out or None
