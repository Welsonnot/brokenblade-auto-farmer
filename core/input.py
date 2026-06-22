"""
InputEmulator
=============
Hardware-level input via pydirectinput (DirectInput) and win32api.
- Tracks key/mouse state to avoid redundant down/up calls.
- Adds human-like per-event jitter to avoid robotic timing signatures.
- Thread-safe: all public methods acquire the internal lock.

Label order expected by apply_predictions():
  [m1, z, x, f, w, a, s, d]
"""

import random
import threading
import time

import pydirectinput

pydirectinput.PAUSE = 0   # disable pydirectinput's built-in sleep - we handle timing

JITTER_LO = 0.0005   # 0.5 ms - was 4 ms
JITTER_HI = 0.002    # 2 ms   - was 18 ms

_KEYBOARD_LABELS = ['z', 'x', 'f', 'w', 'a', 's', 'd']
_ALL_LABELS      = ['m1'] + _KEYBOARD_LABELS


class InputEmulator:
    def __init__(self):
        self._lock         = threading.Lock()
        self._key_state:   dict[str, bool] = {k: False for k in _ALL_LABELS}
        self._click_active = False

    # -- Internal -------------------------------------------------------------

    @staticmethod
    def _jitter() -> None:
        time.sleep(random.uniform(JITTER_LO, JITTER_HI))

    # -- Keyboard -------------------------------------------------------------

    def key_down(self, key: str) -> None:
        with self._lock:
            if self._key_state.get(key):
                return
            self._key_state[key] = True
        self._jitter()
        pydirectinput.keyDown(key)

    def key_up(self, key: str) -> None:
        with self._lock:
            if not self._key_state.get(key):
                return
            self._key_state[key] = False
        self._jitter()
        pydirectinput.keyUp(key)

    def tap(self, key: str, hold: float = 0.05) -> None:
        """Press and release a key with an optional hold duration."""
        self.key_down(key)
        time.sleep(hold)
        self.key_up(key)

    # -- Mouse -----------------------------------------------------------------

    def mouse_down(self) -> None:
        with self._lock:
            if self._click_active:
                return
            self._click_active = True
        self._jitter()
        pydirectinput.mouseDown()

    def mouse_up(self) -> None:
        with self._lock:
            if not self._click_active:
                return
            self._click_active = False
        self._jitter()
        pydirectinput.mouseUp()

    def click(self, hold: float | None = None) -> None:
        """Single left click with optional custom hold duration."""
        dur = hold if hold is not None else random.uniform(0.015, 0.025)
        self.mouse_down()
        time.sleep(dur)
        self.mouse_up()

    def right_click(self, hold: float | None = None) -> None:
        """Single right click with optional custom hold duration."""
        dur = hold if hold is not None else random.uniform(0.015, 0.025)
        self._jitter()
        pydirectinput.mouseDown(button='right')
        time.sleep(dur)
        pydirectinput.mouseUp(button='right')

    # -- Bulk operations -------------------------------------------------------

    def release_all(self) -> None:
        """Emergency release - drops all keys and mouse buttons immediately."""
        with self._lock:
            keys_held = [k for k, v in self._key_state.items() if v]
            click_was = self._click_active
            for k in keys_held:
                self._key_state[k] = False
            self._click_active = False

        for k in keys_held:
            pydirectinput.keyUp(k)
        if click_was:
            pydirectinput.mouseUp()

    def apply_predictions(self, probs: list[float], threshold: float = 0.5) -> None:
        """
        Diff current input state against model output and send only deltas.
        Label index order: [m1, z, x, f, w, a, s, d]
        """
        for i, label in enumerate(_ALL_LABELS):
            want = probs[i] > threshold
            if label == 'm1':
                if want:
                    self.mouse_down()
                else:
                    self.mouse_up()
            else:
                if want:
                    self.key_down(label)
                else:
                    self.key_up(label)
