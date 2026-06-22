"""
RecoveryGuard - OOB & Stuck Protection for Overnight RL Runs
=============================================================
Completely decoupled from the RL step rate via time.time() differentials.
Call guard.check() once per env.step() - it is O(1) on most calls.

Two protection layers:

  Layer 1 - Visual Stuck Detector (15 s)
      Compares downscaled frames every 2 s.
      If the scene pixel diff stays below threshold for 15 s while
      navigating -> the character is walking into a wall.
      Recovery: tap Space (jump) then Q (dash) to break free.

  Layer 2 - Navigation Timeout (240 s)
      Tracks how long the bot stays in EXPLORING/ENGAGING without
      ever reaching COMBAT.  At 240 s the bot is genuinely lost.
      Recovery: Esc -> R -> Enter  (Roblox character reset menu).
      Returns 'reset' so the caller can terminate the episode cleanly.
"""

from __future__ import annotations
import json
import os
import time
import numpy as np
import cv2

from core.state import GameState
from core.input import InputEmulator

# -- Load tunable params from config (same file Qwen advisor writes to) ---------
def _load_params() -> dict:
    # __file__ is in rl/, so go up one level to reach the project root
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(_root, "config", "rl_params.json")
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}

_P = _load_params()

# -- Tuneable constants ---------------------------------------------------------
STUCK_WINDOW_S     = 20.0                              # seconds visually frozen -> nudge (was 10)
LOST_WINDOW_S      = _P.get("LOST_WINDOW_S",  120.0)  # seconds without combat -> reset (was 30)
FRAME_SAMPLE_EVERY = 2.0                               # wall-clock seconds between frame comparisons
STUCK_PIX_THR      = 8.0                               # mean abs pixel diff = "frozen" (was 3.0)
RESET_SETTLE_S     = _P.get("RESET_SETTLE_S",  2.0)   # seconds after Roblox reset

# Downscale resolution for comparison (fast, ~0.3 ms)
_CMP_W, _CMP_H = 112, 63


class RecoveryGuard:
    """
    Stateful guard - one instance per BrokenBladeEnv.
    Thread-safe: all state is written only from the env step thread.
    """

    def __init__(self, input_emu: InputEmulator) -> None:
        self._input            = input_emu
        self._prev_small: np.ndarray | None = None
        self._t_last_sample    = 0.0
        self._t_stuck_since    = time.time()
        self._t_search_since   = time.time()

    # -- Public API ------------------------------------------------------------

    def check(self,
              game_state: GameState,
              native_frame: np.ndarray | None) -> str:
        """
        Call once per env.step().
        Returns:
            'ok'     - nothing to do
            'nudged' - jump+dash executed, timer reset (episode continues)
            'reset'  - character reset executed, caller should terminate episode
        """
        now = time.time()

        # Boss found -> clear both timers so a hard fight doesn't look stuck
        if game_state == GameState.COMBAT:
            self._t_search_since = now
            self._t_stuck_since  = now
            self._prev_small     = None
            return 'ok'

        searching = game_state in (GameState.EXPLORING, GameState.ENGAGING)

        # -- Layer 2: Navigation timeout ---------------------------------------
        if searching and (now - self._t_search_since) >= LOST_WINDOW_S:
            elapsed = now - self._t_search_since
            print(f"\n[Recovery] [WARN]  Lost for {elapsed:.0f}s - "
                  "character reset ...")
            self._character_reset()
            self.reset_timers()
            return 'reset'

        # -- Layer 1: Visual stuck detector -----------------------------------
        # Only run the comparison every FRAME_SAMPLE_EVERY seconds
        if (searching
                and native_frame is not None
                and (now - self._t_last_sample) >= FRAME_SAMPLE_EVERY):

            small = cv2.resize(
                native_frame, (_CMP_W, _CMP_H),
                interpolation=cv2.INTER_AREA
            ).astype(np.float32)
            self._t_last_sample = now

            if self._prev_small is not None:
                diff = float(np.mean(np.abs(small - self._prev_small)))

                if diff < STUCK_PIX_THR:
                    # Scene barely changed
                    stuck_for = now - self._t_stuck_since
                    if stuck_for >= STUCK_WINDOW_S:
                        print(f"\n[Recovery] [RECOVERY]  Visually stuck "
                              f"(diff={diff:.2f}) for {stuck_for:.0f}s "
                              "- nudging ...")
                        self._nudge()
                        self._t_stuck_since = now
                        self._prev_small    = small
                        return 'nudged'
                else:
                    # Scene is changing - character is moving
                    self._t_stuck_since = now

            self._prev_small = small

        return 'ok'

    def reset_timers(self) -> None:
        """Call on every episode reset so timers start fresh."""
        now = time.time()
        self._t_search_since = now
        self._t_stuck_since  = now
        self._t_last_sample  = 0.0
        self._prev_small     = None

    # -- Recovery macros -------------------------------------------------------

    def _nudge(self) -> None:
        """Jump + Dash to break out of wall collision."""
        self._input.tap('space', 0.10)
        time.sleep(0.15)
        self._input.tap('q', 0.10)

    def _refocus_roblox(self) -> None:
        """
        Bring the Roblox window back to foreground after the Esc menu closes.

        The Esc -> R -> Enter reset sequence can pull focus away from the game
        window.  If focus lands on the desktop or another app, the next
        episode's key inputs (pydirectinput) go to the wrong target.

        Uses win32gui from pywin32 (standard on Windows game setups).
        The try/except makes this non-fatal if pywin32 is not installed.
        """
        try:
            import win32gui
            found: list[int] = []

            def _handler(hwnd: int, _: object) -> None:
                if (win32gui.IsWindowVisible(hwnd)
                        and 'Roblox' in win32gui.GetWindowText(hwnd)):
                    found.append(hwnd)

            win32gui.EnumWindows(_handler, None)
            if found:
                win32gui.SetForegroundWindow(found[0])
                time.sleep(0.10)   # let the window manager settle
        except Exception:
            pass   # non-fatal - worst case keys go to wrong window for one step

    def _character_reset(self) -> None:
        """
        Hard Roblox character reset via the in-game menu.
        Sequence: Esc -> R -> Enter
        Blocks for RESET_SETTLE_S while the respawn animation plays,
        then re-focuses the Roblox window before returning.
        """
        self._input.release_all()
        time.sleep(0.20)

        self._input.tap('escape', 0.10)
        time.sleep(0.30)
        self._input.tap('r', 0.10)
        time.sleep(0.30)
        self._input.tap('return', 0.10)

        # Wait for the respawn animation so the next observation is valid
        time.sleep(RESET_SETTLE_S)

        # Re-focus Roblox: ESC menu may have stolen focus from the game window
        self._refocus_roblox()
        print("[Recovery] [OK]  Character reset complete - new episode starting.")
