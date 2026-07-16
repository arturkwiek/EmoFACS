# src/aus/au_extractor.py
import numpy as np


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
