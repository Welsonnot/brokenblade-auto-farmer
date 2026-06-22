"""
UI Detector
===========
Detects boss alive/dead state by reading the text that appears
below the boss name - much more reliable than pixel color matching.

  ALIVE : text region contains "/"  (HP display: "100,000,000/100,000,000")
  DEAD  : text region contains "respawn"

Install:  pip install mss opencv-python pytesseract
          + Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki
"""

import time
import sys

import cv2
import mss
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -- Boss text region ----------------------------------------------------------
# The area where HP numbers or "Respawn Xs" appears.
# Calibrated from find_bar.py: bar at x=173, width=2341, center~1343
TEXT_REGION = {
    "left":   700,
    "top":    87,
    "width":  1300,
    "height": 120,
}

CONFIRMS = 2   # consecutive dead reads before confirming


class BossDetector:
    def __init__(self):
        self._sct = mss.MSS()

    # -- Internal --------------------------------------------------------------

    def _read_text(self) -> str:
        img  = np.array(self._sct.grab(TEXT_REGION))[:, :, :3]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2x upscale - OCR accuracy improves significantly on small UI text
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # Light blur to reduce grid/texture noise from the game background
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # High threshold - game text is near-white (~240+), background is dark
        _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Invert to black text on white - Tesseract accuracy is much better this way
        th = cv2.bitwise_not(th)

        return pytesseract.image_to_string(th, config="--psm 6 --oem 3").lower()

    # -- Public API ------------------------------------------------------------

    def is_alive(self) -> bool:
        text = self._read_text()
        if "resp" in text:   # catches "respawn", "respwn", "resp4wn", etc.
            return False
        if "/" in text:      # HP display always has "/" separator
            return True
        return False         # no HP panel visible = not in active combat

    def is_dead_confirmed(self) -> bool:
        """True only after CONFIRMS consecutive dead readings."""
        for _ in range(CONFIRMS):
            if self.is_alive():
                return False
            time.sleep(0.1)
        return True

    def is_boss_dead(self) -> bool:
        return self.is_dead_confirmed()

    def calibrate(self) -> None:
        """Save captured region so you can verify coordinates visually."""
        img = np.array(self._sct.grab(TEXT_REGION))[:, :, :3]
        cv2.imwrite("cal_bar.png", img)
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
        cv2.imwrite("cal_bar_thresh.png", th)
        text = self._read_text()
        print(f"Captured region saved to cal_bar.png / cal_bar_thresh.png")
        print(f"OCR read: '{text.strip()}'")


if __name__ == "__main__":
    det = BossDetector()

    if "--calibrate" in sys.argv:
        det.calibrate()
        sys.exit()

    print("Monitoring boss state.  Ctrl+C to stop.")
    while True:
        alive  = det.is_alive()
        status = "ALIVE" if alive else "DEAD"
        print(f"\r[{time.strftime('%H:%M:%S')}] Boss: {status:<10}", end="", flush=True)
        time.sleep(0.2)
