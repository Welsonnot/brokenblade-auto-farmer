"""
check_player_hp.py - live player HP test (OCR + auto-calibrate)
================================================================
Auto-locates your HEALTH text on startup, then monitors HP live.

  python check_player_hp.py

Ctrl+C to quit.
"""

import os, sys
# Allow running from anywhere: add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import re
import mss
import numpy as np
import cv2

try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = \
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except ImportError:
    print("pip install pytesseract"); exit(1)

from core.hp_tracker import RobustHPReader

# Search zone (upper-left of screen - where HEALTH bar lives)
SCAN_REGION = {"left": 0, "top": 100, "width": 700, "height": 200}
HP_RE       = re.compile(r'([\d,]+)\s*/\s*([\d,]+)')
HOLD_S      = 3.0

# Cache state
_last_hp        = -1.0
_last_read_time = 0.0
_last_cur       = 0
_last_max       = 0
_last_raw       = ""


def _parse_num(s: str) -> int:
    cleaned = s.replace(',', '').replace(' ', '').strip()
    return int(cleaned) if cleaned.isdigit() else 0


def _ocr_attempts(img_bgr: np.ndarray) -> list[str]:
    """Multiple preprocessing methods for robustness."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cfg = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789/,"
    results = []
    for scale in (2, 3):
        big = cv2.resize(gray, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)
        for invert in (True, False):
            _, thr = cv2.threshold(
                big, 160, 255,
                cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY)
            try:
                results.append(
                    pytesseract.image_to_string(thr, config=cfg).strip())
            except Exception:
                pass
    return results


def _try_parse(texts: list[str]) -> tuple[int, int, str] | None:
    for text in texts:
        m = HP_RE.search(text)
        if not m:
            continue
        cur = _parse_num(m.group(1))
        mx  = _parse_num(m.group(2))
        if mx < 1000 or cur > mx:
            continue
        return (cur, mx, text)
    return None


def auto_calibrate(sct) -> dict | None:
    """Scan upper-left for HP text and return a precise ROI."""
    print("Auto-locating HEALTH text ... please make sure HP bar is visible.")
    region = np.ascontiguousarray(
        np.array(sct.grab(SCAN_REGION))[:, :, :3])

    rh = region.shape[0]
    for y in range(0, rh - 40, 5):
        strip = region[y: y + 40, :]
        parsed = _try_parse(_ocr_attempts(strip))
        if parsed is not None:
            cur, mx, raw = parsed
            abs_y = SCAN_REGION["top"] + y
            print(f"\n[OK]  Found at row {abs_y}: "
                  f"HP={cur/mx:.0%} ({cur:,}/{mx:,})")
            return {"left": SCAN_REGION["left"],
                    "top": abs_y - 8,
                    "width": SCAN_REGION["width"],
                    "height": 50}
    return None


def main():
    with mss.MSS() as sct:
        roi = auto_calibrate(sct)
        if roi is None:
            print("\n[FAIL]  Could not find HEALTH text in upper-left.")
            return

        print(f"\nUsing ROI: left={roi['left']} top={roi['top']} "
              f"{roi['width']}x{roi['height']}\n")
        print("Watching player HP via OCR + validator ... Ctrl+C to stop\n")

        tracker = RobustHPReader("player", hold_s=HOLD_S, boss_mode=False)

        while True:
            img = np.ascontiguousarray(np.array(sct.grab(roi))[:, :, :3])
            parsed = _try_parse(_ocr_attempts(img))

            if parsed is not None:
                cur, mx, raw = parsed
                hp = tracker.update(cur, mx)
                accepted = (tracker.current == cur)
            else:
                hp = tracker.update(None, None)
                accepted = False
                raw = "(no parse)"

            if hp < 0:
                status = "[FAIL]  NO READING"
            else:
                if hp > 0.66:
                    icon = "[OK]"
                elif hp > 0.33:
                    icon = "[WARN]"
                elif hp > 0.10:
                    icon = "[LOW]"
                else:
                    icon = "[DEAD]"
                lock = "[LOCK]" if tracker.max else "  "
                cur_str = f"{tracker.current:,}" if tracker.current else "?"
                max_str = f"{tracker.max:,}" if tracker.max else "?"
                flag = "[OK]" if accepted else "[FAIL]"
                status = (f"{icon} {lock} HP = {hp:6.1%}  "
                          f"({cur_str}/{max_str})  {flag} raw='{raw}'")
            print(f"\r{status:<130}", end="", flush=True)
            time.sleep(0.25)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDone.")
