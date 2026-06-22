"""
StateController  -  Layer 1: The Brain
=======================================
Runs a dedicated thread that classifies the current game situation.

Boss alive/dead detection uses OCR on the text region below the boss name:
  ALIVE : region contains "100,000,000" (HP numbers with "/")
  DEAD  : region contains "Respawn"

OCR runs at 5 Hz (every 12th frame at 60 FPS) to avoid CPU overhead.
Pixel ops run every frame for loot/engage detection.
"""

import threading
import time
from enum import Enum, auto

import cv2
import mss
import numpy as np

try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False
    print("[State] pytesseract not found - install it for reliable boss detection.")


# -- Timing --------------------------------------------------------------------
POLL_FPS    = 60
OCR_EVERY_N = 12   # run OCR once every N frames (~5 Hz at 60 FPS)

# -- Boss text region ----------------------------------------------------------
# Covers the area where "100,000,000/100,000,000" or "Respawn Xs" appears.
# Centered horizontally, just below the boss name.
# Calibrated from find_bar.py: bar starts at x=173, width=2341 -> center~1343
_TEXT_REGION = {
    "left":   700,    # center of screen minus padding
    "top":    87,     # same Y as bar top - text sits just below bar
    "width":  1300,   # wide enough to capture full text
    "height": 120,    # covers bar + HP/Respawn text line below it
}

# -- Loot / Reward UI ---------------------------------------------------------
_LOOT_PIXEL_XY  = (960, 700)
_LOOT_COLOR_BGR = np.array([0, 200, 255], dtype=np.int32)
_LOOT_TOL       = 40

# -- Engaging reticle (stub) --------------------------------------------------
_ENGAGE_PIXEL_XY  = None
_ENGAGE_COLOR_BGR = None
_ENGAGE_TOL       = 35


class GameState(Enum):
    EXPLORING = auto()
    ENGAGING  = auto()
    COMBAT    = auto()
    LOOTING   = auto()


class StateController:
    def __init__(self, grabber):
        self._grabber     = grabber
        self._state       = GameState.EXPLORING
        self._lock        = threading.Lock()
        self._running     = False
        self._thread: threading.Thread | None = None
        self._interval       = 1.0 / POLL_FPS
        self._frame_count    = 0
        self._boss_alive      = False  # cached OCR result between OCR frames
        self._boss_respawning = False  # True while "Respawn Xs" is visible on screen
        self._no_signal_count = 0     # consecutive ambiguous OCR reads
        # Require this many consecutive ambiguous reads before leaving COMBAT.
        # At 5 Hz OCR, 10 = ~2 seconds of silence before switching to EXPLORING.
        self._NO_SIGNAL_MAX  = 10

    # -- Lifecycle ------------------------------------------------------------

    def start(self) -> None:
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, daemon=True, name="StateController"
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    @property
    def state(self) -> GameState:
        with self._lock:
            return self._state

    @property
    def boss_respawning(self) -> bool:
        """True while the Respawn countdown is visible (boss dead but we're nearby)."""
        with self._lock:
            return self._boss_respawning

    # -- OCR detection --------------------------------------------------------

    def _read_boss_text(self) -> str:
        """Capture the boss info region and return OCR text (lowercased)."""
        with mss.MSS() as sct:
            img = np.array(sct.grab(_TEXT_REGION))[:, :, :3]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2x upscale - OCR accuracy improves significantly on small UI text
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # Light blur to kill background texture noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # High threshold - game text is near-white, background is dark
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Invert: black text on white -> much better Tesseract accuracy
        thresh = cv2.bitwise_not(thresh)

        return pytesseract.image_to_string(thresh, config="--psm 6 --oem 3").lower()

    def _update_boss_alive(self) -> None:
        """Run OCR and update the cached alive/dead/respawning state."""
        if not _OCR_AVAILABLE:
            return
        try:
            text = self._read_boss_text()
            if "resp" in text:               # definitive dead signal - timer visible
                self._boss_alive      = False
                self._no_signal_count = 0
                with self._lock:
                    self._boss_respawning = True
            elif "/" in text:                # definitive alive signal - HP visible
                self._boss_alive      = True
                self._no_signal_count = 0
                with self._lock:
                    self._boss_respawning = False
            else:
                # Ambiguous - can't see boss UI at all (wandered off or brief glitch)
                with self._lock:
                    self._boss_respawning = False
                self._no_signal_count += 1
                if self._no_signal_count >= self._NO_SIGNAL_MAX:
                    self._boss_alive = False
        except Exception as exc:
            print(f"[State] OCR error: {exc}")

    # -- Pixel helpers ---------------------------------------------------------

    @staticmethod
    def _pixel_matches(frame: np.ndarray,
                       xy: tuple[int, int],
                       color: np.ndarray,
                       tol: int) -> bool:
        px = frame[xy[1], xy[0]].astype(np.int32)
        return bool(np.all(np.abs(px - color) < tol))

    # -- State classification --------------------------------------------------

    def _classify(self, frame: np.ndarray) -> GameState:
        # OCR is expensive - only run every N frames
        self._frame_count += 1
        if self._frame_count % OCR_EVERY_N == 0:
            self._update_boss_alive()

        # Priority: LOOTING > COMBAT > ENGAGING > EXPLORING
        if self._pixel_matches(frame, _LOOT_PIXEL_XY, _LOOT_COLOR_BGR, _LOOT_TOL):
            return GameState.LOOTING

        if self._boss_alive:
            return GameState.COMBAT

        if _ENGAGE_PIXEL_XY is not None and _ENGAGE_COLOR_BGR is not None:
            if self._pixel_matches(frame, _ENGAGE_PIXEL_XY,
                                   _ENGAGE_COLOR_BGR, _ENGAGE_TOL):
                return GameState.ENGAGING

        return GameState.EXPLORING

    # -- Poll loop -------------------------------------------------------------

    def _loop(self) -> None:
        while self._running:
            t0    = time.perf_counter()
            frame = self._grabber.get_frame()
            if frame is not None:
                new = self._classify(frame)
                with self._lock:
                    self._state = new
            wait = self._interval - (time.perf_counter() - t0)
            if wait > 0:
                time.sleep(wait)
