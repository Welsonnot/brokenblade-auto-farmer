"""
ML Bot - Step 3b
================
Loads the trained model, grabs the screen at 15 FPS, predicts which keys
to press, and sends them via pydirectinput (DirectInput level - bypasses
most game input filters).

Stops pressing keys while is_boss_dead() returns True.
Adds small random jitter to each input to avoid robotic timing patterns.

F9 = pause / resume
F1 = quit

Install:  pip install torch torchvision mss opencv-python pynput pydirectinput
"""

import os
import time
import random
import threading

import cv2
import mss
import numpy as np
import torch
import torchvision.transforms as T
import pydirectinput
from pynput import keyboard

from legacy.ui_detector import BossDetector
from legacy.train_model import build_model, MODEL_PATH

INFERENCE_FPS = 15
KEY_THRESHOLD = 0.5   # sigmoid probability to treat as "pressed"

# Human-like jitter per input event (seconds)
JITTER_LO = 0.005
JITTER_HI = 0.025

FRAME_SIZE = (224, 224)
KEYS       = ['z', 'x', 'c', 'g']   # keyboard keys tracked by the model

_transform = T.Compose([
    T.ToPILImage(),
    T.Resize(FRAME_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])

# -- Runtime state ------------------------------------------------------------
_active        = False
_quit          = False
_key_state     = {k: False for k in KEYS}
_click_active  = False
_lock          = threading.Lock()


def _jitter():
    time.sleep(random.uniform(JITTER_LO, JITTER_HI))


def _apply_inputs(probs: list[float]):
    """Diff current key state against model predictions and send only changes."""
    global _click_active

    # Keyboard keys: z x c g  -> indices 0-3
    for i, key in enumerate(KEYS):
        want = probs[i] > KEY_THRESHOLD
        with _lock:
            have = _key_state[key]
        if want and not have:
            _jitter()
            pydirectinput.keyDown(key)
            with _lock:
                _key_state[key] = True
        elif not want and have:
            _jitter()
            pydirectinput.keyUp(key)
            with _lock:
                _key_state[key] = False

    # Left click -> index 4
    want_click = probs[4] > KEY_THRESHOLD
    if want_click and not _click_active:
        _jitter()
        pydirectinput.mouseDown()
        _click_active = True
    elif not want_click and _click_active:
        _jitter()
        pydirectinput.mouseUp()
        _click_active = False


def _release_all():
    global _click_active
    for key in KEYS:
        with _lock:
            if _key_state[key]:
                pydirectinput.keyUp(key)
                _key_state[key] = False
    if _click_active:
        pydirectinput.mouseUp()
        _click_active = False


def _on_press(key):
    global _active, _quit
    if key == keyboard.Key.f1:
        _active = not _active
        print(f"\n[Bot] {'ACTIVE' if _active else 'PAUSED'}")
        if not _active:
            _release_all()
    elif key == keyboard.Key.esc:
        _quit = True
        _release_all()
        print("\n[Bot] Quitting ...")
        return False


def run():
    global _quit, _active

    if not os.path.isfile(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        print("Train first with:  python train_model.py")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Bot] Loading model on {device} ...")

    model = build_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval().to(device)

    detector = BossDetector()
    sct      = mss.MSS()
    monitor  = sct.monitors[1]
    interval = 1.0 / INFERENCE_FPS

    kb_listener = keyboard.Listener(on_press=_on_press)
    kb_listener.start()

    print("[Bot] Ready.  F1 = start/pause  |  Esc = quit")

    while not _quit:
        if not _active:
            time.sleep(0.05)
            continue

        if detector.is_boss_dead():
            _release_all()
            print("[Bot] Boss dead - waiting for respawn ...")
            # Poll until boss reappears, same as autoattack.py
            while not _quit and not detector.is_alive():
                time.sleep(0.4)
            if _quit:
                break
            print(f"[Bot] Boss detected - resuming")
            continue

        t0 = time.perf_counter()

        raw   = np.array(sct.grab(monitor))[:, :, :3]
        rgb   = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        inp   = _transform(rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.sigmoid(model(inp)).squeeze().cpu().tolist()

        _apply_inputs(probs)

        elapsed = time.perf_counter() - t0
        wait    = interval - elapsed
        if wait > 0:
            time.sleep(wait)

    kb_listener.stop()
    print("[Bot] Stopped.")


if __name__ == "__main__":
    run()
