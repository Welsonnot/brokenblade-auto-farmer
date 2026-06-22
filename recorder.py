"""
Input Recorder - Universal Edition
====================================
Listens to the ENTIRE keyboard and both mouse buttons.
No whitelist - if you press it, it gets recorded.

F1 = start / stop a recording session.
Ctrl+C = quit.

Output:
  recordings/
    YYYYMMDD_HHMMSS/
      frame_0000000.jpg  ...
      inputs.csv

Tracked columns
  Letters  : a-z
  Numbers  : 1 2 3 4 5
  Special  : space  shift
  Mouse    : m1 (left click)  m2 (right click)
"""

import ctypes
import cv2
import csv
import mss
import numpy as np
import os
import threading
import time
from datetime  import datetime
from pynput    import keyboard, mouse

# -- Direct Win32 hardware state -----------------------------------------------
# pynput's WH_MOUSE_LL hook can miss hold events when Roblox uses Raw Input
# for camera control (right-click look-around).  GetAsyncKeyState reads the
# actual hardware button state regardless of how the game routes its input.
_user32 = ctypes.windll.user32
_VK_LBUTTON = 0x01
_VK_RBUTTON = 0x02

def _hw_mouse(vk: int) -> bool:
    """True if the given mouse button is physically held right now."""
    return bool(_user32.GetAsyncKeyState(vk) & 0x8000)

# -- Config --------------------------------------------------------------------
TARGET_FPS = 15
FRAME_SIZE = (224, 224)
OUTPUT_DIR = "recordings"
HOTKEY     = keyboard.Key.f1

# -- All tracked keys - order defines CSV column order -------------------------
_LETTERS  = list('abcdefghijklmnopqrstuvwxyz')
_NUMBERS  = ['1', '2', '3', '4', '5']
_SPECIAL  = ['space', 'shift']
_MOUSE    = ['m1', 'm2']

ALL_KEYS    = _LETTERS + _NUMBERS + _SPECIAL + _MOUSE   # 35 columns
CSV_COLUMNS = ['timestamp', 'frame_file'] + ALL_KEYS

# pynput special-key -> our column name
_PYNPUT_MAP = {
    keyboard.Key.space:   'space',
    keyboard.Key.shift:   'shift',
    keyboard.Key.shift_r: 'shift',
}

# -- Session state -------------------------------------------------------------
_recording   = False
_frame_count = 0
_session_dir = ""
_csv_writer  = None
_csv_file    = None
_lock        = threading.Lock()
_key_states  = {k: False for k in ALL_KEYS}   # held-down state per key


# -- Session control -----------------------------------------------------------

def _start() -> None:
    global _recording, _frame_count, _session_dir, _csv_writer, _csv_file
    session_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    _session_dir = os.path.join(OUTPUT_DIR, session_id)
    os.makedirs(_session_dir, exist_ok=True)

    _csv_file   = open(os.path.join(_session_dir, "inputs.csv"), 'w', newline='')
    _csv_writer = csv.writer(_csv_file)
    _csv_writer.writerow(CSV_COLUMNS)

    _frame_count = 0
    _recording   = True
    print(f"[REC] [PLAY]  Started  ->  {_session_dir}")


def _stop() -> None:
    global _recording, _csv_file
    _recording = False
    if _csv_file:
        _csv_file.close()
        _csv_file = None
    print(f"[REC] [STOP]  Stopped  ({_frame_count} frames saved)")


# -- Key name resolver ---------------------------------------------------------

def _key_name(key) -> str | None:
    """Return the column name for this pynput key, or None to ignore."""
    # Special keys (space, shift)
    name = _PYNPUT_MAP.get(key)
    if name:
        return name
    # Regular printable keys (letters, digits)
    try:
        ch = key.char
        if ch:
            ch = ch.lower()
            if ch in _key_states:
                return ch
    except AttributeError:
        pass
    return None


# -- Input listeners -----------------------------------------------------------

def _on_press(key) -> None:
    if key == HOTKEY:
        (_stop if _recording else _start)()
        return
    name = _key_name(key)
    if name:
        with _lock:
            _key_states[name] = True


def _on_release(key) -> None:
    name = _key_name(key)
    if name:
        with _lock:
            _key_states[name] = False


def _on_click(x, y, button, pressed) -> None:
    col = ('m1' if button == mouse.Button.left  else
           'm2' if button == mouse.Button.right else None)
    if col:
        with _lock:
            _key_states[col] = pressed


# -- Capture loop --------------------------------------------------------------

def _capture_loop() -> None:
    global _frame_count
    sct      = mss.MSS()
    monitor  = sct.monitors[1]
    interval = 1.0 / TARGET_FPS

    while True:
        if not _recording:
            time.sleep(0.01)
            continue

        t0        = time.perf_counter()
        timestamp = time.time()

        raw   = np.array(sct.grab(monitor))
        frame = cv2.resize(raw[:, :, :3], FRAME_SIZE,
                           interpolation=cv2.INTER_AREA)
        fname = f"frame_{_frame_count:07d}.jpg"
        cv2.imwrite(os.path.join(_session_dir, fname), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])

        with _lock:
            snap = _key_states.copy()

        # Override mouse buttons with direct hardware state.
        # Roblox uses Raw Input for camera control (right-click hold), which
        # bypasses the WH_MOUSE_LL hook that pynput relies on - so the pynput
        # state can get stuck at False while the button is physically held.
        snap['m1'] = _hw_mouse(_VK_LBUTTON)
        snap['m2'] = _hw_mouse(_VK_RBUTTON)

        if _csv_writer:
            _csv_writer.writerow(
                [f"{timestamp:.6f}", fname] + [int(snap[k]) for k in ALL_KEYS]
            )

        _frame_count += 1

        wait = interval - (time.perf_counter() - t0)
        if wait > 0:
            time.sleep(wait)


# -- Entry point ---------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    threading.Thread(target=_capture_loop, daemon=True).start()

    kb_listener    = keyboard.Listener(on_press=_on_press, on_release=_on_release)
    mouse_listener = mouse.Listener(on_click=_on_click)
    kb_listener.start()
    mouse_listener.start()

    print("Recorder ready - listening to ENTIRE keyboard + mouse.")
    print("F1 = start / stop    Ctrl+C = quit")
    try:
        kb_listener.join()
    except KeyboardInterrupt:
        if _recording:
            _stop()
        print("Exited.")
