#!/usr/bin/env python
# scripts/serve_camera_windows.py
"""
Serve a webcam as an MJPEG stream over HTTP.

Intended to be run with **Windows-native Python** so that the camera is
opened by the OS that actually owns it, while the analysis pipeline keeps
running inside WSL. WSL2 cannot consume USB webcams reliably: usbipd can
attach the device, but UVC streaming relies on isochronous USB transfers,
which USB/IP does not carry — the device opens and then never delivers a
frame (`select() timeout`).

Only `cv2` is required here — no torch, no py-feat.

Usage
-----
    .\\.venv\\Scripts\\python.exe scripts/serve_camera_windows.py

Then, from WSL:

    python scripts/run_on_webcam.py --cam http://<WINDOWS_HOST_IP>:8080/video

Controls
--------
    Ctrl+C — stop the server
"""
from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

_BOUNDARY = "emofacsframe"

# How long to wait for the first frame before giving up on the camera.
CAMERA_TIMEOUT_S = 15.0

# Latest encoded JPEG, refreshed by the capture thread and read by every
# connected client. A single shared frame keeps all viewers on the newest
# image instead of letting a slow client build up a backlog.
_latest: bytes | None = None
_latest_lock = threading.Lock()
_stop = threading.Event()
# Set once the camera is open and the first frame has been encoded, so the
# server only advertises itself when it actually has something to serve.
_ready = threading.Event()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a local webcam as an MJPEG stream over HTTP."
    )
    parser.add_argument("--cam", type=int, default=0,
                        help="Camera device index (default: 0).")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Interface to bind to (default: 0.0.0.0, all).")
    parser.add_argument("--port", type=int, default=8080,
                        help="TCP port to listen on (default: 8080).")
    parser.add_argument("--width", type=int, default=640,
                        help="Capture width (default: 640).")
    parser.add_argument("--height", type=int, default=480,
                        help="Capture height (default: 480).")
    parser.add_argument("--fps", type=int, default=15,
                        help="Target capture frame rate (default: 15).")
    parser.add_argument("--quality", type=int, default=80,
                        help="JPEG quality, 1-100 (default: 80).")
    return parser.parse_args()


def open_camera(index: int) -> "cv2.VideoCapture":
    """
    Open the camera, falling back to DirectShow when the default backend
    (MSMF) fails — MSMF is known to stop delivering frames with errors like
    0xC00D3EA2 when the device hiccups or another app grabs it.
    """
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        return cap
    cap.release()
    if hasattr(cv2, "CAP_DSHOW"):
        print("[warning] Default backend failed — trying DirectShow.")
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    return cap


def capture_loop(args: argparse.Namespace) -> None:
    """Grab frames continuously and publish the newest one as JPEG."""
    global _latest

    cap = open_camera(args.cam)
    if not cap.isOpened():
        print(f"[error] Cannot open camera index {args.cam}.")
        _stop.set()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[info] Camera {args.cam} opened at {actual_w}x{actual_h}.")

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), args.quality]
    interval = 1.0 / max(1, args.fps)
    empty_streak = 0

    try:
        while not _stop.is_set():
            ret, frame = cap.read()
            if not ret:
                empty_streak += 1
                if empty_streak == 1:
                    print("[warning] Empty frame received — waiting for "
                          "the camera to recover...")
                if empty_streak >= 100:
                    print("[error] Camera is not recovering — stopping.")
                    break
                time.sleep(0.1)
                continue
            empty_streak = 0

            ok, buf = cv2.imencode(".jpg", frame, encode_params)
            if ok:
                with _latest_lock:
                    _latest = buf.tobytes()
                _ready.set()
            time.sleep(interval)
    finally:
        cap.release()
        _stop.set()
        print("[info] Capture stopped.")


class MJPEGHandler(BaseHTTPRequestHandler):
    # Quieter than the default, which logs every single frame request.
    def log_message(self, fmt, *fmt_args) -> None:
        return

    def do_GET(self) -> None:
        if self.path.rstrip("/") in ("", "/index.html"):
            self._serve_index()
        elif self.path.rstrip("/") == "/video":
            self._serve_stream()
        else:
            self.send_error(404, "Not found")

    def _serve_index(self) -> None:
        body = (
            b"<html><body style='margin:0;background:#111'>"
            b"<img src='/video' style='width:100%'>"
            b"</body></html>"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_stream(self) -> None:
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header(
            "Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY}"
        )
        self.end_headers()

        print(f"[info] Client connected: {self.client_address[0]}")
        try:
            while not _stop.is_set():
                with _latest_lock:
                    frame = _latest
                if frame is None:
                    time.sleep(0.05)
                    continue
                self.wfile.write(f"--{_BOUNDARY}\r\n".encode())
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                time.sleep(0.02)
        except (BrokenPipeError, ConnectionResetError):
            # Normal when the consumer goes away (e.g. the WSL script exits).
            print(f"[info] Client disconnected: {self.client_address[0]}")


def _local_ips() -> list[str]:
    """Best-effort list of addresses this host is reachable at."""
    ips = []
    try:
        _, _, addrs = socket.gethostbyname_ex(socket.gethostname())
        ips.extend(addrs)
    except OSError:
        pass
    return ips


def main() -> None:
    args = parse_args()

    started_at = time.monotonic()
    thread = threading.Thread(target=capture_loop, args=(args,), daemon=True)
    thread.start()

    # Wait for a real frame rather than a fixed delay: opening a camera can
    # take several seconds, and starting the server before then would serve
    # an empty stream that looks like a network problem from the client side.
    print("[info] Waiting for the first frame...")
    while not _ready.wait(timeout=0.25):
        if _stop.is_set():
            sys.exit("[error] Camera unavailable — not starting the server.")
        if time.monotonic() - started_at > CAMERA_TIMEOUT_S:
            _stop.set()
            sys.exit(
                f"[error] No frame within {CAMERA_TIMEOUT_S:.0f} s — "
                "not starting the server."
            )

    server = ThreadingHTTPServer((args.host, args.port), MJPEGHandler)
    print(f"[info] Serving MJPEG on http://{args.host}:{args.port}/video")
    for ip in _local_ips():
        print(f"[info]   reachable at http://{ip}:{args.port}/video")
    print("[info] Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[info] Shutting down...")
    finally:
        _stop.set()
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
