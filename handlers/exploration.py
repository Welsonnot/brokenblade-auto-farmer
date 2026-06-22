"""
ExplorationManager  -  Layer 2: EXPLORING / ENGAGING handler
=============================================================
No camera rotation (right-click drag causes angle issues in Roblox).

Detection method (automatic):
  * If models/boss_detector.pt exists -> uses YOLO to detect the boss label.
  * Otherwise -> simple G-spam fallback (still works, just less precise).

WAITING   - OCR sees "Respawn Xs" -> boss is dead and character is at spawn.
            Stand perfectly still.  Tap G every 0.8 s.
            StateController will auto-switch to COMBAT when boss spawns.

NAVIGATING - Boss UI not on screen -> character wandered.
             Walk W + Ctrl (sprint), tap G every 0.8 s.
             YOLO: if boss label detected, G auto-faces + walks toward it.
             Every 6 s tap D (0.3 s) to alter heading if walking wrong way.
             StateController OCR detects HP bar and flips to COMBAT.
"""

from __future__ import annotations
import time
import mss
import numpy as np

from core.yolo_detect import BossLabelDetector

_LOCK_INTERVAL = 0.80   # G tap cadence while navigating / waiting
_DASH_INTERVAL = 3.50   # Q dash every N seconds while navigating
_TURN_INTERVAL = 6.00   # tap D every N seconds to vary heading

# Screen center X - label to the left -> boss is left, etc.
# (Used for logging; G BossLock handles the actual steering.)
_CENTER_X = 728


class ExplorationManager:

    def __init__(self, input_emu, state_ctl):
        self._input   = input_emu
        self._state   = state_ctl
        self._t_lock  = 0.0
        self._t_dash  = 0.0
        self._t_turn  = 0.0
        self._sct     = mss.mss()

        # Try loading YOLO model - silently falls back if not trained yet
        self._yolo = BossLabelDetector()

    # -- Lifecycle -------------------------------------------------------------

    def on_enter(self) -> None:
        self._release_all()

    def on_exit(self) -> None:
        self._release_all()

    def tick(self) -> None:
        if self._state.boss_respawning:
            self._do_waiting()
        else:
            self._do_navigating()

    def tick_engaging(self) -> None:
        self.tick()

    # -- WAITING - stand still, boss will respawn ------------------------------

    def _do_waiting(self) -> None:
        self._release_all()
        now = time.time()
        if now - self._t_lock >= _LOCK_INTERVAL:
            self._input.tap('g', 0.08)
            self._t_lock = now

    # -- NAVIGATING - walk toward boss, spam G to BossLock when in range -------

    def _do_navigating(self) -> None:
        now = time.time()

        # Check YOLO for boss label
        boss_visible, label_x = self._yolo_check()

        if boss_visible:
            # Boss label is on screen - G will auto-face + lock it
            self._input.key_down('w')
            self._input.key_down('ctrl')
            if now - self._t_lock >= _LOCK_INTERVAL:
                self._input.tap('g', 0.08)
                self._t_lock = now
            print(f"[Explore] YOLO: boss visible at x={label_x:.0f}", end='\r')
        else:
            # Boss not visible yet - walk and spam G hoping to enter range
            self._input.key_down('w')
            self._input.key_down('ctrl')

            if now - self._t_lock >= _LOCK_INTERVAL:
                self._input.tap('g', 0.08)
                self._t_lock = now

            # Q dash to cover ground
            if now - self._t_dash >= _DASH_INTERVAL:
                self._input.key_up('w')
                self._input.tap('q', 0.05)
                self._input.key_down('w')
                self._t_dash = now

            # Brief D tap to vary heading every 6 s
            if now - self._t_turn >= _TURN_INTERVAL:
                self._input.tap('d', 0.3)
                self._t_turn = now

    # -- YOLO check ------------------------------------------------------------

    def _yolo_check(self) -> tuple[bool, float]:
        """
        Returns (boss_visible, center_x_of_best_detection).
        center_x = 0.0 if not visible.
        Uses pixel grab -> numpy array -> YOLO inference.
        Falls back to (False, 0.0) if YOLO not available.
        """
        if not self._yolo.available:
            return False, 0.0

        try:
            monitor = self._sct.monitors[1]
            frame   = np.array(self._sct.grab(monitor))[:, :, :3]
            dets    = self._yolo.run(frame)
            if dets:
                best = dets[0]
                return True, best.center_x
        except Exception as exc:
            print(f"[Explore] YOLO check error: {exc}")

        return False, 0.0

    # -- Helpers ---------------------------------------------------------------

    def _release_all(self) -> None:
        for key in ('w', 'a', 'd', 'ctrl'):
            self._input.key_up(key)
