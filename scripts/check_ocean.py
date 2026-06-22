"""
check_ocean.py - live test of OOB ocean detector
=================================================
Mirrors the pixel scan used by game_env.py.
Walk around in-game and verify it triggers ONLY when you're in the ocean.

  python check_ocean.py

Ctrl+C to quit.
"""

import time
import mss
import numpy as np

# Same constants as game_env.py
_OCEAN_SAMPLE_TOP   = 700
_OCEAN_SAMPLE_BOT   = 1280
_OCEAN_SAMPLE_LEFT  = 550
_OCEAN_SAMPLE_RIGHT = 1900
_OCEAN_PCT_THR      = 0.55


def ocean_ratio(frame: np.ndarray) -> float:
    region = frame[_OCEAN_SAMPLE_TOP:_OCEAN_SAMPLE_BOT,
                   _OCEAN_SAMPLE_LEFT:_OCEAN_SAMPLE_RIGHT]
    r = region[:, :, 2].astype(np.int16)
    b = region[:, :, 0].astype(np.int16)
    water = (b > r + 12)
    return float(water.sum()) / water.size


def main():
    print("Watching for ocean OOB ... Ctrl+C to stop")
    print(f"Threshold: {_OCEAN_PCT_THR:.0%}\n")
    with mss.MSS() as sct:
        monitor = sct.monitors[1]
        while True:
            frame = np.ascontiguousarray(
                np.array(sct.grab(monitor))[:, :, :3])
            ratio = ocean_ratio(frame)
            if ratio > _OCEAN_PCT_THR:
                status = (f"[OCEAN]  OCEAN OOB!   blue={ratio:.1%}  "
                          f"(would PENALIZE)")
            elif ratio > 0.50:
                status = (f"[WARN]   high blue   {ratio:.1%}  "
                          f"(near threshold)")
            else:
                status = f"[OK]  on map      blue={ratio:.1%}"
            print(f"\r{status:<70}", end="", flush=True)
            time.sleep(0.25)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDone.")
