"""
calibrate_hp_bar.py - find & fix the boss HP bar pixel coords
==============================================================
Run while the boss is ALIVE with BossLock (G) active.
Saves  cal_bar.png  so you can see exactly what row is being scanned.

Usage:
    python calibrate_hp_bar.py

After running, open cal_bar.png to inspect the highlighted region.
"""

import mss
import numpy as np
import cv2

# Current coords - we'll scan +/-30 rows around this to find the real bar
_BAR_ROW   = 92
_BAR_LEFT  = 1137
_BAR_RIGHT = 1812
SCAN_RANGE = 30    # rows above/below _BAR_ROW to search


def detect_row(frame: np.ndarray, row: int, left: int, right: int) -> int:
    """Count HP-colored pixels in this row (red OR orange)."""
    seg = frame[row, left:right]
    r   = seg[:, 2].astype(int)
    g   = seg[:, 1].astype(int)
    b   = seg[:, 0].astype(int)
    # Red:    r - g > 60,  r > 130, b < 110
    # Orange: r > 150,  g > 60,  r - g > 30,  b < 80
    red    = (r - g > 60) & (r > 130) & (b < 110)
    orange = (r > 150)    & (g > 60)  & (r - g > 30) & (b < 80)
    return int(np.sum(red | orange))


def main():
    print("Capturing screen ...")
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        frame   = np.ascontiguousarray(
            np.array(sct.grab(monitor))[:, :, :3])   # BGR

    h, w = frame.shape[:2]
    print(f"Screen: {w}x{h}")

    # -- Scan rows around expected position ------------------------------------
    print(f"\nScanning rows {_BAR_ROW - SCAN_RANGE} -> {_BAR_ROW + SCAN_RANGE} "
          f"(cols {_BAR_LEFT}-{_BAR_RIGHT}):\n")

    best_row, best_count = _BAR_ROW, 0
    for row in range(max(0, _BAR_ROW - SCAN_RANGE),
                     min(h, _BAR_ROW + SCAN_RANGE + 1)):
        count = detect_row(frame, row, _BAR_LEFT, _BAR_RIGHT)
        marker = " <- current _BAR_ROW" if row == _BAR_ROW else ""
        if count > 5:
            print(f"  row {row:4d}  ->  {count:4d} HP-colored pixels{marker}")
        if count > best_count:
            best_count = count
            best_row   = row

    print(f"\n[OK]  Best row = {best_row}  ({best_count} pixels)")

    if best_row != _BAR_ROW:
        print(f"\n[WARN]  Update _BAR_ROW in game_env.py and check_boss_hp.py:")
        print(f"    _BAR_ROW = {best_row}")
    else:
        print("\n   _BAR_ROW looks correct.")

    # -- Print pixel sample at best row ----------------------------------------
    mid = (_BAR_LEFT + _BAR_RIGHT) // 2
    sample = frame[best_row, mid - 5: mid + 5]
    print(f"\nPixel sample at row {best_row}, cols {mid-5}-{mid+5} (BGR):")
    for i, px in enumerate(sample):
        print(f"  [{mid - 5 + i}]  B={px[0]:3d}  G={px[1]:3d}  R={px[2]:3d}")

    # -- Save debug image ------------------------------------------------------
    debug = frame.copy()
    # Highlight the scan region
    cv2.rectangle(debug,
                  (_BAR_LEFT, _BAR_ROW - SCAN_RANGE),
                  (_BAR_RIGHT, _BAR_ROW + SCAN_RANGE),
                  (0, 255, 255), 1)   # yellow box
    # Highlight best row
    cv2.line(debug,
             (_BAR_LEFT, best_row), (_BAR_RIGHT, best_row),
             (0, 255, 0), 2)          # green line

    # Crop just the relevant region (+50px padding)
    pad  = 50
    crop = debug[
        max(0, _BAR_ROW - SCAN_RANGE - pad): _BAR_ROW + SCAN_RANGE + pad,
        max(0, _BAR_LEFT - pad): _BAR_RIGHT + pad
    ]
    cv2.imwrite("cal_bar.png", crop)
    print("\n[IMG]  Saved  cal_bar.png  - open it to see the scanned region.")
    print("     Yellow box = scan range.  Green line = best row found.")


if __name__ == "__main__":
    main()
