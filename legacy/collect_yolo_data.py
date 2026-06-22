"""
YOLO Training Data Collector
=============================
Captures raw screenshots for later labeling.

HOW TO USE
----------
1. Start the game and go near the boss area.
2. Run this script in a terminal: python collect_yolo_data.py
3. Hold F3 while you can see the boss label or boss spawn text on screen.
   Frames are saved at ~2 per second while F3 is held.
4. Press Esc to stop.
5. Go to https://roboflow.com (free account) and upload the yolo_data/raw/ folder.
6. Label with a single class "velik_label" - draw boxes around the floating
   "[Lv.6000] Velik" name tag.  Include the boss HP bar too.
7. Export as "YOLOv8" format -> it gives you a zip with images/labels/data.yaml.
8. Extract the zip into this project's yolo_data/ folder.
9. Run: python train_yolo.py

Tips for good data:
  - Get screenshots from close, medium, and far distance.
  - Get screenshots at different angles (boss slightly left/right/center of screen).
  - Get screenshots with and without snow/ice particles on screen.
  - 80-150 labeled images is plenty for YOLOv8n to learn this single class.
"""

import os
import time
import mss
import numpy as np
import cv2
from pynput import keyboard

# -- Config --------------------------------------------------------------------
SAVE_DIR     = "yolo_data/raw"
CAPTURE_HZ   = 2      # frames per second while F3 is held
# -----------------------------------------------------------------------------

os.makedirs(SAVE_DIR, exist_ok=True)

_collecting = False
_quit       = False
_count      = 0


def _on_press(key):
    global _collecting
    if key == keyboard.Key.f3:
        _collecting = True
    elif key == keyboard.Key.esc:
        global _quit
        _quit = True
        return False


def _on_release(key):
    global _collecting
    if key == keyboard.Key.f3:
        _collecting = False


def main():
    global _count, _quit

    listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
    listener.start()

    print("=" * 50)
    print("  YOLO Data Collector")
    print("  Hold F3 to capture frames")
    print("  Esc = quit")
    print(f"  Saving to: {os.path.abspath(SAVE_DIR)}")
    print("=" * 50)

    interval = 1.0 / CAPTURE_HZ

    with mss.mss() as sct:
        monitor = sct.monitors[1]   # primary monitor (full screen)

        while not _quit:
            if _collecting:
                img = np.array(sct.grab(monitor))[:, :, :3]  # BGR
                fname = os.path.join(SAVE_DIR, f"frame_{_count:05d}.png")
                cv2.imwrite(fname, img)
                _count += 1
                print(f"\r  Captured {_count} frames", end="", flush=True)
                time.sleep(interval)
            else:
                time.sleep(0.05)

    listener.stop()
    print(f"\n\nDone - {_count} frames saved to {os.path.abspath(SAVE_DIR)}/")
    print("Upload them to https://roboflow.com to label, then run train_yolo.py")


if __name__ == "__main__":
    main()
