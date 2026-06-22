"""
calibrate_hp_ocr.py - find the HP number text region (debug version)
=====================================================================
Run while boss is ALIVE with BossLock (G) active.

    python calibrate_hp_ocr.py
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

SCAN = {"left": 900, "top": 70, "width": 800, "height": 150}
HP_RE = re.compile(r'([\d,. ]+?)\s*/\s*([\d,. ]+)')


def parse_number(s: str) -> int:
    """Parse '3,000,000,000' or '3 000 000 000' or '3000000000' -> int."""
    cleaned = s.replace(',', '').replace('.', '').replace(' ', '').strip()
    return int(cleaned) if cleaned.isdigit() else 0


def try_methods(img_bgr: np.ndarray) -> list[tuple[str, str, str]]:
    """Try multiple OCR preprocessing methods, return all results."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    results = []

    for scale in [2, 3, 4]:
        big = cv2.resize(gray, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)

        # Method A: binary threshold (dark text on white bg)
        _, thr_a = cv2.threshold(big, 160, 255, cv2.THRESH_BINARY)
        text_a = pytesseract.image_to_string(
            thr_a, config="--psm 7 --oem 3 "
                          "-c tessedit_char_whitelist=0123456789/,").strip()
        results.append((f"scale={scale} BINARY  thr=160", text_a,
                        _check(text_a)))

        # Method B: inverted threshold
        _, thr_b = cv2.threshold(big, 160, 255, cv2.THRESH_BINARY_INV)
        text_b = pytesseract.image_to_string(
            thr_b, config="--psm 7 --oem 3 "
                          "-c tessedit_char_whitelist=0123456789/,").strip()
        results.append((f"scale={scale} INV     thr=160", text_b,
                        _check(text_b)))

        # Method C: OTSU auto threshold
        _, thr_c = cv2.threshold(big, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text_c = pytesseract.image_to_string(
            thr_c, config="--psm 7 --oem 3 "
                          "-c tessedit_char_whitelist=0123456789/,").strip()
        results.append((f"scale={scale} OTSU         ", text_c,
                        _check(text_c)))

        # Method D: OTSU inverted
        _, thr_d = cv2.threshold(big, 0, 255,
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        text_d = pytesseract.image_to_string(
            thr_d, config="--psm 7 --oem 3 "
                          "-c tessedit_char_whitelist=0123456789/,").strip()
        results.append((f"scale={scale} OTSU_INV     ", text_d,
                        _check(text_d)))

        # Method E: adaptive threshold
        ada = cv2.adaptiveThreshold(big, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 10)
        text_e = pytesseract.image_to_string(
            ada, config="--psm 7 --oem 3 "
                        "-c tessedit_char_whitelist=0123456789/,").strip()
        results.append((f"scale={scale} ADAPTIVE     ", text_e,
                        _check(text_e)))

    return results


def _check(text: str) -> str:
    m = HP_RE.search(text)
    if not m:
        return "no match"
    cur = parse_number(m.group(1))
    mx  = parse_number(m.group(2))
    return f"cur={cur:,}  max={mx:,}  pct={cur/mx:.0%}" if mx else "div0"


def main():
    print("Capturing ...")
    with mss.MSS() as sct:
        full = np.ascontiguousarray(
            np.array(sct.grab(sct.monitors[1]))[:, :, :3])

    # Extract the HP text strip at known position (from last calibration: top=105)
    for test_top in [100, 105, 110, 115, 120]:
        strip = full[test_top: test_top + 45,
                     SCAN["left"]: SCAN["left"] + SCAN["width"]]

        print(f"\n{'='*70}")
        print(f"  Testing row {test_top} -> {test_top + 45}")
        print(f"{'='*70}")

        results = try_methods(strip)
        for method, raw_text, parsed in results:
            if raw_text:
                print(f"  {method}  raw='{raw_text}'  ->  {parsed}")

    # Also save debug images of the best strip
    best_top = 105
    strip = full[best_top: best_top + 45,
                 SCAN["left"]: SCAN["left"] + SCAN["width"]]
    cv2.imwrite("cal_strip_raw.png", strip)

    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    big  = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, thr = cv2.threshold(big, 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite("cal_strip_otsu.png", thr)
    _, inv = cv2.threshold(big, 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite("cal_strip_otsu_inv.png", inv)

    print(f"\n[IMG]  Saved debug images:")
    print(f"    cal_strip_raw.png      - raw screenshot strip")
    print(f"    cal_strip_otsu.png     - OTSU threshold")
    print(f"    cal_strip_otsu_inv.png - OTSU inverted")
    print(f"\nOpen these to see what Tesseract is actually reading.")


if __name__ == "__main__":
    main()
