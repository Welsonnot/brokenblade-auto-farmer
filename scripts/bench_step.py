"""
bench_step.py - Isolate which part of env.step() is eating the time.
Run with Roblox open and in-game. Takes ~5 seconds.

    python bench_step.py
"""

# Run from project root so relative paths resolve.
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)


import time
import numpy as np
import mss
import cv2

try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    _OCR_OK = True
except ImportError:
    _OCR_OK = False

# Mirror game_env.py constants exactly
FRAME_SIZE = (224, 224)
_BAR_TOP, _BAR_BOT   = 85,  116
_BAR_LEFT, _BAR_RIGHT = 1137, 1812
_OCEAN_TOP, _OCEAN_BOT   = 700,  1280
_OCEAN_LEFT, _OCEAN_RIGHT = 550, 1900
_QUEST_ROI = {"left": 60, "top": 280, "width": 240, "height": 50}

N = 60   # sample steps

# -- Simulated shared-grabber warm frame ---------------------------------------
# During actual training the ScreenGrabber holds a frame in RAM and get_frame()
# just does a lock + .copy() (~0.4 ms).  Simulate that here so the bench
# reflects post-fix performance, not the old double-DXGI path.

def ms(t): return t * 1000

with mss.MSS() as sct:
    monitor = sct.monitors[1]

    grab_t, process_t, ocean_t, boss_t, ocr_t = [], [], [], [], []

    print(f"Running {N} iterations ...\n")

    for i in range(N):
        # -- Frame grab --------------------------------------------------------
        t0 = time.perf_counter()
        raw   = np.array(sct.grab(monitor))
        frame = np.ascontiguousarray(raw[:, :, :3])
        grab_t.append(time.perf_counter() - t0)

        # -- Frame resize + RGB convert (obs) ----------------------------------
        t0 = time.perf_counter()
        small = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        _obs  = np.ascontiguousarray(rgb.transpose(2, 0, 1))
        process_t.append(time.perf_counter() - t0)

        # -- Ocean detection ---------------------------------------------------
        t0 = time.perf_counter()
        reg = frame[_OCEAN_TOP:_OCEAN_BOT, _OCEAN_LEFT:_OCEAN_RIGHT]
        r = reg[:, :, 2].astype(np.int16)
        b = reg[:, :, 0].astype(np.int16)
        _ = float((b > r + 12).sum()) / (b > r + 12).size
        ocean_t.append(time.perf_counter() - t0)

        # -- Boss visibility ---------------------------------------------------
        t0 = time.perf_counter()
        bar = frame[_BAR_TOP:_BAR_BOT, _BAR_LEFT:_BAR_RIGHT]
        r = bar[:, :, 2].astype(np.int16)
        b = bar[:, :, 0].astype(np.int16)
        warm = (r > 130) & (b < 120) & ((r - b) > 50)
        _ = bool(warm.sum(axis=1).max() >= 30)
        boss_t.append(time.perf_counter() - t0)

        # -- OCR (pytesseract) - run every 30 steps like game_env -------------
        if _OCR_OK and i % 30 == 0:
            t0 = time.perf_counter()
            img  = np.array(sct.grab(_QUEST_ROI))[:, :, :3]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            _, thr = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            thr = cv2.bitwise_not(thr)
            pytesseract.image_to_string(
                thr, config="--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789/")
            ocr_t.append(time.perf_counter() - t0)

def rep(name, arr, width=18):
    a = np.array(arr) * 1000
    print(f"  {name:{width}s}  avg={a.mean():6.1f} ms   "
          f"min={a.min():5.1f}   max={a.max():6.1f}")

print("-" * 62)
print("Per-component timings (ms):")
rep("frame_grab",    grab_t)
rep("frame_process", process_t)
rep("ocean_detect",  ocean_t)
rep("boss_detect",   boss_t)
if ocr_t:
    rep("OCR (every 30)", ocr_t)
else:
    print("  OCR                   skipped (pytesseract not found)")

total = np.array(grab_t) + np.array(process_t) + np.array(ocean_t) + np.array(boss_t)
print("-" * 62)
print(f"  4-op subtotal      avg={total.mean()*1000:6.1f} ms")

if ocr_t:
    ocr_avg = np.mean(ocr_t)
    ocr_per_step = ocr_avg / 30           # amortised over 30 steps
    print(f"  OCR amortised      avg={ocr_per_step*1000:6.1f} ms/step")
    effective = total.mean() + ocr_per_step
    print(f"  Effective total    avg={effective*1000:6.1f} ms  "
          f"-> ~{1/effective:.0f} it/s  (without tap-hold or GPU inference)")
else:
    effective = total.mean()
    print(f"  Effective total    avg={effective*1000:6.1f} ms  "
          f"-> ~{1/effective:.0f} it/s  (without tap-hold or GPU inference)")

print()
print("Bottleneck guide:")
print("  frame_grab   > 30 ms  -> mss is slow on your system")
print("  OCR          > 200 ms -> pytesseract is the killer; disable it")
print("  ocean_detect > 5 ms   -> numpy on CPU is slow; check CPU load")
