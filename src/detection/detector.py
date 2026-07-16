# src/detection/detector.py

# ── Compatibility shims for py-feat 0.6.x on newer libraries ──────────────────

# torchvision >=0.20 removed read_video; py-feat 0.6.x still imports it.
import torchvision.io as _tio
if not hasattr(_tio, "read_video"):
    def _read_video_stub(*args, **kwargs):
        raise RuntimeError("read_video is not available in this torchvision version.")
    _tio.read_video = _read_video_stub

# scipy >=1.14 removed simps (renamed to simpson); py-feat 0.6.x still uses it.
import scipy.integrate as _si
if not hasattr(_si, "simps"):
    _si.simps = _si.simpson

# ─────────────────────────────────────────────────────────────────────────────

import cv2
import tempfile
import os
from feat import Detector


class FaceDetector:
    def __init__(self):
        self.detector = Detector()

    def detect(self, img_bgr):
        """
        Run face detection + AU extraction on a BGR numpy array.
        py-feat 0.6.x requires a file path — we write a temp file and
        pass the path, then clean it up immediately.
        Returns a Fex object with .aus, .emotions, .facepose, etc.
        """
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        try:
            cv2.imwrite(tmp.name, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            preds = self.detector.detect_image(tmp.name)
        finally:
            tmp.close()
            os.unlink(tmp.name)
        return preds
