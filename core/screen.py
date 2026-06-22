"""
ScreenGrabber
=============
Double-buffered, thread-isolated screen capture at 60 FPS.
Always returns the latest frame instantly - zero blocking on the caller side.

Two buffers are maintained:
  _frame       : full-resolution BGR  (used by StateController for pixel checks)
  _small_frame : 224x224 BGR          (used by CombatEngine for model inference)
"""

import threading
import time

import cv2
import mss
import numpy as np

MODEL_SIZE   = (224, 224)
DEFAULT_FPS  = 60


class ScreenGrabber:
    def __init__(self, monitor_idx: int = 1, target_fps: int = DEFAULT_FPS):
        self._monitor_idx = monitor_idx
        self._interval    = 1.0 / target_fps
        self._lock        = threading.Lock()
        self._frame:       np.ndarray | None = None
        self._small_frame: np.ndarray | None = None
        self._running     = False
        self._thread:      threading.Thread | None = None

    # -- Lifecycle ------------------------------------------------------------

    def start(self) -> None:
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, daemon=True, name="ScreenGrabber"
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    # -- Public accessors (non-blocking) --------------------------------------

    def get_frame(self) -> np.ndarray | None:
        """Full-resolution BGR frame, or None if not yet captured."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def get_small_frame(self) -> np.ndarray | None:
        """224x224 BGR frame ready for model inference."""
        with self._lock:
            return None if self._small_frame is None else self._small_frame.copy()

    # -- Capture loop ---------------------------------------------------------

    def _loop(self) -> None:
        with mss.MSS() as sct:
            monitor = sct.monitors[self._monitor_idx]
            while self._running:
                t0  = time.perf_counter()
                raw = np.array(sct.grab(monitor))
                bgr = raw[:, :, :3]                                      # drop alpha
                sml = cv2.resize(bgr, MODEL_SIZE, interpolation=cv2.INTER_LINEAR)

                with self._lock:
                    self._frame       = bgr
                    self._small_frame = sml

                wait = self._interval - (time.perf_counter() - t0)
                if wait > 0:
                    time.sleep(wait)
