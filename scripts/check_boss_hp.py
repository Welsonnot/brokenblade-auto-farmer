"""
check_boss_hp.py - live test of boss-bar visibility
=====================================================
Mirrors the simple pixel scan used by game_env.py.
Tests whether the bot can see the boss HP bar on screen RIGHT NOW.

  python check_boss_hp.py

Ctrl+C to quit.
"""

import time
import mss
import numpy as np

# -- Same coords as game_env.py ------------------------------------------------
_BAR_TOP   = 85
_BAR_BOT   = 116
_BAR_LEFT  = 1137
_BAR_RIGHT = 1812


def boss_visible(frame: np.ndarray) -> tuple[bool, int, int]:
    """Returns (visible, best_row_count, best_row)."""
    region = frame[_BAR_TOP:_BAR_BOT, _BAR_LEFT:_BAR_RIGHT]
    r = region[:, :, 2].astype(np.int16)
    b = region[:, :, 0].astype(np.int16)
    warm = (r > 130) & (b < 120) & ((r - b) > 50)
    counts = warm.sum(axis=1)
    best_row = int(counts.argmax())
    best_count = int(counts[best_row])
    return (best_count >= 30, best_count, _BAR_TOP + best_row)


def main():
    print("Watching boss bar visibility ... Ctrl+C to stop")
    print(f"Scan region: rows {_BAR_TOP}-{_BAR_BOT}, "
          f"cols {_BAR_LEFT}-{_BAR_RIGHT}\n")

    with mss.MSS() as sct:
        monitor = sct.monitors[1]
        while True:
            frame = np.ascontiguousarray(
                np.array(sct.grab(monitor))[:, :, :3])
            visible, count, row = boss_visible(frame)

            if visible:
                status = f"[OK]  BOSS BAR VISIBLE   row={row}  warm_pixels={count}"
            else:
                status = f"[FAIL]  NO BOSS BAR       (best row={row}, pixels={count})"

            print(f"\r{status:<80}", end="", flush=True)
            time.sleep(0.25)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDone.")
