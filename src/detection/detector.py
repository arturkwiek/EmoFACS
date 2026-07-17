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

        Performance notes:
        - The frame is written directly as BGR (cv2.imwrite expects BGR),
          so no colour-space round-trip is needed.
        - PNG is used instead of JPEG: lossless, so AU extraction is not
          affected by compression artifacts. Fast compression level keeps
          the encode time in the low-millisecond range.
        - The temp file handle is closed *before* cv2.imwrite opens the
          path — required for reliable behaviour on Windows.
        """
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            cv2.imwrite(tmp_path, img_bgr,
                        [cv2.IMWRITE_PNG_COMPRESSION, 1])
            preds = self.detector.detect_image(tmp_path)
        finally:
            os.unlink(tmp_path)
        return preds
