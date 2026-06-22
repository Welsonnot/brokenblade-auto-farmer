"""
LootHandler  -  Layer 2: LOOTING handler
=========================================
Handles chest interactions and reward UI popups after a boss kill.

Phase flow:
  INTERACT -> press E to open the chest / interact prompt
  CLAIM    -> click the Claim / OK button (hardcoded screen coord)
  CLOSE    -> press F or Esc to dismiss any remaining popup
  IDLE     -> wait for StateController to transition back to EXPLORING

Adjust CLAIM_XY to match the "Claim" or "OK" button in Broken Blade's UI.
"""

import random
import time


_INTERACT_KEY  = 'e'
_CLOSE_KEY     = 'f'
CLAIM_XY       = (960, 600)   # <- set to center of your "Claim/OK" button

_PHASE_DELAYS  = {
    'INTERACT': 0.40,
    'CLAIM':    0.60,
    'CLOSE':    0.30,
}


class LootHandler:
    def __init__(self, input_emu):
        self._input  = input_emu
        self._phase  = 'INTERACT'
        self._t_next = 0.0

    # -- Lifecycle -------------------------------------------------------------

    def on_enter(self) -> None:
        self._phase  = 'INTERACT'
        self._t_next = time.time()   # act immediately on first tick

    def on_exit(self) -> None:
        self._input.release_all()

    # -- Tick -----------------------------------------------------------------

    def tick(self) -> None:
        now = time.time()
        if now < self._t_next:
            return

        if self._phase == 'INTERACT':
            self._input.tap(_INTERACT_KEY, 0.08)
            self._phase  = 'CLAIM'
            self._t_next = now + _PHASE_DELAYS['INTERACT']

        elif self._phase == 'CLAIM':
            # Move mouse to claim button and click
            import pydirectinput
            pydirectinput.moveTo(CLAIM_XY[0], CLAIM_XY[1])
            time.sleep(random.uniform(0.05, 0.10))
            self._input.click()
            self._phase  = 'CLOSE'
            self._t_next = now + _PHASE_DELAYS['CLAIM']

        elif self._phase == 'CLOSE':
            self._input.tap(_CLOSE_KEY, 0.06)
            self._phase  = 'IDLE'
            self._t_next = now + _PHASE_DELAYS['CLOSE']

        # IDLE: do nothing - StateController will detect UI is gone
        # and transition to EXPLORING naturally.
