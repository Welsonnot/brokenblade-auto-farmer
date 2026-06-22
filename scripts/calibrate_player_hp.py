"""
calibrate_player_hp.py - find player HEALTH text region (OCR-based)
=====================================================================
Scans the upper-left of screen for the player HP number "X/Y".
Run while in-game with HP bar visible.

    python calibrate_player_hp.py

Outputs the pixel coords to paste into game_env.py.
Saves cal_player_hp.png showing the detected region.
"""

import mss
import numpy as np
import cv2
import re

try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = \
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except ImportError:
    print("pip install pytesseract"); exit(1)

# Player HP is in the upper-left quadrant
SCAN_REGION = {"left": 0, "top": 100, "width": 700, "height": 200}
HP_RE = re.compile(r'([\d,]+)\s*/\s*([\d,]+)')


def try_ocr(img_bgr: np.ndarray) -> str:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    big  = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, thr = cv2.threshold(big, 160, 255, cv2.THRESH_BINARY)
    return pytesseract.image_to_string(
        thr, config="--psm 7 --oem 3 "
                    "-c tessedit_char_whitelist=0123456789/,").strip()


def main():
    print("Capturing ...")
    with mss.MSS() as sct:
        full = np.ascontiguousarray(
            np.array(sct.grab(sct.monitors[1]))[:, :, :3])
        region = np.ascontiguousarray(
            np.array(sct.grab(SCAN_REGION))[:, :, :3])

    h, w = full.shape[:2]
    print(f"Screen: {w}x{h}")
    print(f"Scanning region: left={SCAN_REGION['left']} "
          f"top={SCAN_REGION['top']} {SCAN_REGION['width']}x{SCAN_REGION['height']}\n")

    # Slide a 40-px tall strip through the region
    best_top, best_match = None, None
    rh = region.shape[0]
    for y in range(0, rh - 40, 5):
        strip = region[y: y + 40, :]
        text  = try_ocr(strip)
        m     = HP_RE.search(text)
        if m:
            abs_y = SCAN_REGION["top"] + y
            cur = int(m.group(1).replace(',', ''))
            mx  = int(m.group(2).replace(',', ''))
            if mx > 1000 and cur <= mx:
                print(f"  row {abs_y:4d}  ->  '{text}'  "
                      f"cur={cur:,}  max={mx:,}  pct={cur/mx:.0%}")
                if best_match is None:
                    best_top = abs_y
                    best_match = m

    if best_match is None:
        print("\n[FAIL]  No player HP text found in scan region.")
        print("    Make sure the HEALTH bar is visible on screen.")
        print("    Try expanding SCAN_REGION at the top of this file.")
        return

    cur = int(best_match.group(1).replace(',', ''))
    mx  = int(best_match.group(2).replace(',', ''))
    print(f"\n[OK]  Player HP text found at top={best_top}")
    print(f"    Current HP : {cur:,}")
    print(f"    Max HP     : {mx:,}")
    print(f"    Fraction   : {cur/mx:.1%}")

    # Build a generous ROI around it
    php_top    = best_top - 8
    php_height = 50
    php_left   = SCAN_REGION["left"]
    php_width  = SCAN_REGION["width"]

    print(f"\n[INFO]  Paste into game_env.py:")
    print(f'    _PHP_TEXT_ROI = {{"left": {php_left}, "top": {php_top}, '
          f'"width": {php_width}, "height": {php_height}}}')

    # Save debug image
    debug = full.copy()
    cv2.rectangle(debug,
                  (php_left, php_top),
                  (php_left + php_width, php_top + php_height),
                  (0, 255, 0), 2)
    crop = debug[max(0, php_top - 30): php_top + php_height + 30,
                 max(0, php_left - 30): php_left + php_width + 30]
    cv2.imwrite("cal_player_hp.png", crop)
    print(f"\n[IMG]  Saved cal_player_hp.png")


if __name__ == "__main__":
    main()
