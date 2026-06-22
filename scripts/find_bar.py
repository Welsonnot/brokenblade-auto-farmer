"""
find_bar.py
===========
Scans the screen row by row to automatically locate the boss health bar.
Run this WHILE the boss is alive and the red bar is visible.

Prints the exact BAR_X, BAR_Y, BAR_W, BAR_H values to paste into your files.
Also saves debug images so you can visually confirm the result.
"""

import time
import mss
import numpy as np
import cv2

R_MIN     = 140
R_G_RATIO = 2.0
R_B_RATIO = 2.0

MIN_RED_PX_IN_ROW  = 50   # a row needs this many red px to count as part of the bar
MIN_RED_PX_IN_COL  = 3    # a column needs this many red px across bar rows

print("Waiting 3 seconds - switch to Roblox and make sure boss health bar is visible ...")
time.sleep(3)

with mss.MSS() as sct:
    monitor = sct.monitors[1]
    raw     = np.array(sct.grab(monitor))
    frame   = raw[:, :, :3]   # BGR

print(f"Screenshot captured: {frame.shape[1]}x{frame.shape[0]} px")

# -- Build red-pixel mask ------------------------------------------------------
r = frame[:, :, 2].astype(np.int32)
g = frame[:, :, 1].astype(np.int32)
b = frame[:, :, 0].astype(np.int32)
red_mask = (r >= R_MIN) & (r > g * R_G_RATIO) & (r > b * R_B_RATIO)

# -- Scan rows (only top half - boss bar is always near the top) ---------------
H, W = frame.shape[:2]
row_counts = red_mask[:H//2, :].sum(axis=1)

bar_rows = np.where(row_counts >= MIN_RED_PX_IN_ROW)[0]

if len(bar_rows) == 0:
    print("\n[FAIL]  No red bar found. Make sure:")
    print("   1. Boss is alive with health bar visible")
    print("   2. Roblox is in focus (not minimised)")
    print("   3. You ran this within 3 seconds of switching to Roblox")
else:
    BAR_Y = int(bar_rows[0])
    BAR_H = int(bar_rows[-1] - bar_rows[0] + 1)

    # -- Find left/right extent using column sums within those rows ------------
    bar_region_mask = red_mask[BAR_Y:BAR_Y + BAR_H, :]
    col_counts      = bar_region_mask.sum(axis=0)
    bar_cols        = np.where(col_counts >= MIN_RED_PX_IN_COL)[0]

    BAR_X = int(bar_cols[0])
    BAR_W = int(bar_cols[-1] - bar_cols[0] + 1)

    total_red = int(red_mask[BAR_Y:BAR_Y+BAR_H, BAR_X:BAR_X+BAR_W].sum())

    print("\n[OK]  Boss health bar found!")
    print(f"   BAR_X = {BAR_X}")
    print(f"   BAR_Y = {BAR_Y}")
    print(f"   BAR_W = {BAR_W}")
    print(f"   BAR_H = {BAR_H}")
    print(f"   Red pixels detected: {total_red}")

    # -- Save debug images -----------------------------------------------------
    # Cropped bar region
    bar_crop = frame[BAR_Y:BAR_Y+BAR_H, BAR_X:BAR_X+BAR_W]
    cv2.imwrite("cal_bar.png", bar_crop)

    # Full screen with rectangle drawn around detected bar
    annotated = frame.copy()
    cv2.rectangle(annotated,
                  (BAR_X, BAR_Y),
                  (BAR_X + BAR_W, BAR_Y + BAR_H),
                  (0, 255, 0), 2)
    cv2.imwrite("cal_full_annotated.png", annotated)

    print("\n   Saved: cal_bar.png           (cropped bar - should look red)")
    print("   Saved: cal_full_annotated.png (full screen with green rectangle)")
    print("\nPaste those 4 values into core/state.py and ui_detector.py")
