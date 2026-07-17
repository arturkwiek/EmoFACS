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


class AUExtra