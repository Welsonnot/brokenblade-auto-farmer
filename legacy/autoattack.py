"""
Auto-Attack - Infinite Loop (definitive fix)
=============================================
Based on direct screen observation:
  - HP bar lives at y=65, x=415-1045
  - ALIVE:  right half (x=730-1045) is solid orange-red
  - DEAD:   right half is completely dark - only a tiny left-edge sliver remains
  So we only ever check the right half. Zero false positives.

Loop:
  1. Wait until right-half of HP bar turns red  (boss alive/respawned)
  2. Press G  (BossLock)
  3. Attack Z/X/C, checking HP after every key
  4. Right half goes dark  ->  boss dead
  5. Wait minimum 5 s, then back to step 1 (no hardcoded timer - we detect the respawn)

F1 = stop

Install:  pip install pynput pyautogui Pillow
"""

import time
import threading
import pyautogui
from pynput import keyboard
from pynput.keyboard import Key, Controller

pyautogui.FAILSAFE = False

# ---------------------------------------------
#  ATTACK TIMING
# ---------------------------------------------
Z_HOLD      = 0.05
Z_DELAY     = 0.30
PARRY_HOLD  = 0.05
PARRY_DELAY = 0.45

# ---------------------------------------------
#  HP BAR - RIGHT HALF ONLY
#  Alive: 200-400+ red pixels here
#  Dead:  0-5 red pixels here (sliver is on far left, not here)
# ---------------------------------------------
BAR_X      = 730    # start of right half
BAR_Y      = 62     # top of bar strip
BAR_W      = 315    # width  (730 -> 1045)
BAR_H      = 12     # height of strip

# Colour: orange-red HP bar.  R dominant, G can be up to ~120.
# Background when dead is dark blue/purple (very low R).
R_MIN      = 120
R_G_RATIO  = 1.15   # R must be > G * this
R_B_RATIO  = 1.15   # R must be > B * this

ALIVE_MIN  = 25     # px needed to call it "alive"  (dead gives ~0 px)
DEAD_MAX   = 8      # px to call it "dead"
CONFIRMS   = 3      # consecutive dead reads before we believe it

MIN_DEAD_WAIT = 5   # seconds to wait after death before checking respawn

# ---------------------------------------------
#  STATE
# ---------------------------------------------
active = True
kb     = Controller()


# ---------------------------------------------
#  HELPERS
# ---------------------------------------------
def press(key, hold=0.05):
    kb.press(key)
    time.sleep(hold)
    kb.release(key)


def count_red():
    """Count orange-red pixels in the right half of the HP bar."""
    try:
        img = pyautogui.screenshot(region=(BAR_X, BAR_Y, BAR_W, BAR_H))
        pix = img.load()
        n = 0
        for py in range(img.height):
            for px in range(img.width):
                r, g, b = pix[px, py]
                if r >= R_MIN and r > g * R_G_RATIO and r > b * R_B_RATIO:
                    n += 1
        return n
    except Exception:
        return 0


def is_alive():
    return count_red() >= ALIVE_MIN


def is_dead_confirmed():
    """True only if CONFIRMS consecutive reads are all below DEAD_MAX."""
    for _ in range(CONFIRMS):
        if count_red() > DEAD_MAX:
            return False
        time.sleep(0.05)
    return True


# ---------------------------------------------
#  PRESS KEY + INSTANT DEATH CHECK
# ---------------------------------------------
def attack_key(key, hold, delay):
    """Press key, wait, return True if boss confirmed dead."""
    press(key, hold)
    time.sleep(delay)
    return active and is_dead_confirmed()


# ---------------------------------------------
#  MAIN LOOP
# ---------------------------------------------
def main_loop():
    global active

    while active:

        # -- 1. Wait for boss HP bar to appear --------
        print("[Loop] Waiting for boss ...")
        while active and not is_alive():
            time.sleep(0.4)
        if not active:
            break
        print(f"[Loop] Boss detected ({count_red()} red px) [OK]")

        # -- 2. BossLock -------------------------------
        print("[Loop] G - BossLock")
        press('g', 0.1)
        time.sleep(0.3)

        # -- 3. Attack until dead ----------------------
        print("[Loop] Attacking ...")
        dead = False
        while active and not dead:
            if attack_key('z', Z_HOLD, Z_DELAY):       dead = True; break
            if attack_key('z', Z_HOLD, Z_DELAY):       dead = True; break
            if attack_key('x', PARRY_HOLD, PARRY_DELAY): dead = True; break
            if attack_key('z', Z_HOLD, Z_DELAY):       dead = True; break
            if attack_key('z', Z_HOLD, Z_DELAY):       dead = True; break
            if attack_key('c', PARRY_HOLD, PARRY_DELAY): dead = True; break
        if not active:
            break

        print("[Loop] [DEAD]  Boss dead - waiting for respawn ...")

        # -- 4. Short cooldown then detect respawn -----
        time.sleep(MIN_DEAD_WAIT)
        # Back to step 1 - loop will detect bar coming back

    print("[Loop] Stopped.")


# ---------------------------------------------
#  F1 = STOP
# ---------------------------------------------
def on_press(key):
    global active
    if key == Key.f1:
        active = False
        print("\n[F1] Stopping ...")
        return False


if __name__ == "__main__":
    print("=" * 46)
    print("  Auto-Attack  |  Infinite Loop")
    print("=" * 46)
    print("  Checking right-half of HP bar:")
    print(f"  x={BAR_X}-{BAR_X+BAR_W}, y={BAR_Y}-{BAR_Y+BAR_H}")
    print("  Running. F1 to stop.\n")

    # Quick sanity check on startup
    n = count_red()
    print(f"[Init] Current red px in bar zone: {n}")
    if n >= ALIVE_MIN:
        print("[Init] Boss is alive right now [OK]")
    else:
        print("[Init] Boss not detected yet - will wait ...")

    threading.Thread(target=main_loop, daemon=True).start()

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
