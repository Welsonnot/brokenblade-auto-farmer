"""
Broken Blade Bot  -  Master Execution Loop
==========================================
Two-Layer Architecture
  Layer 1  StateController  (Brain)   - dedicated 60 FPS thread
  Layer 2  Action handlers  (Muscle)  - dispatched from this 30 Hz main loop

Hotkeys:
  F1  = pause / resume
  Esc = quit cleanly

Usage:
  python main.py
"""

import sys
import time
import threading

from pynput import keyboard

from core.screen import ScreenGrabber
from core.input  import InputEmulator
from core.state  import StateController, GameState
from handlers.combat      import CombatEngine
from handlers.exploration import ExplorationManager
from handlers.loot        import LootHandler

MAIN_HZ = 30   # handler dispatch rate (StateController runs independently at 60 Hz)

# -- Global control flags ------------------------------------------------------
_active = False
_quit   = False
_lock   = threading.Lock()


def _on_press(key) -> bool | None:
    global _active, _quit
    if key == keyboard.Key.f1:
        with _lock:
            _active = not _active
        print(f"\n[Bot] {'[PLAY]  ACTIVE' if _active else '[PAUSE]  PAUSED'}")
    elif key == keyboard.Key.esc:
        with _lock:
            _quit = True
        print("\n[Bot] Shutting down ...")
        return False   # stop pynput listener
    return None


# -- Entry point ---------------------------------------------------------------
def main() -> None:
    print("=" * 52)
    print("  Broken Blade Bot  |  Two-Layer Architecture")
    print("  F1 = start/pause     Esc = quit")
    print("=" * 52)

    # -- Subsystem init --------------------------------------------------------
    grabber   = ScreenGrabber(monitor_idx=2, target_fps=60)
    input_emu = InputEmulator()
    state_ctl = StateController(grabber)

    combat  = CombatEngine(grabber, input_emu)
    explore = ExplorationManager(input_emu, state_ctl)
    loot    = LootHandler(input_emu)

    handlers = {
        GameState.EXPLORING: explore,   # holds W+Ctrl toward boss, taps G to lock
        GameState.ENGAGING:  explore,
        GameState.COMBAT:    combat,
        GameState.LOOTING:   loot,
    }

    # -- Start background threads -----------------------------------------------
    grabber.start()
    state_ctl.start()

    kb = keyboard.Listener(on_press=_on_press)
    kb.start()

    print("[Bot] Ready.\n")

    # -- Main dispatch loop -----------------------------------------------------
    interval   = 1.0 / MAIN_HZ
    prev_state: GameState | None = None

    try:
        while True:
            with _lock:
                should_quit   = _quit
                should_active = _active

            if should_quit:
                break

            if not should_active:
                time.sleep(0.05)
                continue

            t0    = time.perf_counter()
            state = state_ctl.state

            # -- State transition -----------------------------------------------
            if state != prev_state:
                # on_exit for old state
                if prev_state is not None and prev_state in handlers:
                    try:
                        handlers[prev_state].on_exit()
                    except Exception as exc:
                        print(f"[Bot] on_exit error ({prev_state.name}): {exc}")

                # on_enter for new state
                if state in handlers:
                    try:
                        handlers[state].on_enter()
                    except Exception as exc:
                        print(f"[Bot] on_enter error ({state.name}): {exc}")

                print(f"[Bot] -- {state.name}")
                prev_state = state

            # -- Dispatch tick --------------------------------------------------
            try:
                if state == GameState.ENGAGING:
                    explore.tick_engaging()
                elif state in handlers:
                    handlers[state].tick()
            except Exception as exc:
                print(f"[Bot] tick error ({state.name}): {exc}")

            # -- Rate limiting --------------------------------------------------
            wait = interval - (time.perf_counter() - t0)
            if wait > 0:
                time.sleep(wait)

    except KeyboardInterrupt:
        pass

    finally:
        print("[Bot] Releasing inputs ...")
        input_emu.release_all()
        state_ctl.stop()
        grabber.stop()
        kb.stop()
        print("[Bot] Stopped cleanly.")


if __name__ == "__main__":
    main()
