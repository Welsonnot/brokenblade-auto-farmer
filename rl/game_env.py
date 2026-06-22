"""
BrokenBladeEnv - Custom Gymnasium Environment
==============================================
Wraps the live Broken Blade game window for PPO fine-tuning.

Observation space : Box(0, 255, (3, 224, 224), uint8)
Action space      : MultiDiscrete([2] * N_ACTIONS)

Reward function - clean, sparse, no flaky OCR percentages:
    Kill counter +1        +5.0   (OCR on quest panel, every 30 steps - monotonic)
    Boss killed            +10.0  (StateController sees Respawn timer)
    Attack with boss bar   +0.05  per step pressing M1 while bar is visible
    Attack with NO bar     -0.10 to -0.50 escalating (5 steps -> -0.50/step)
    Time penalty           -0.005 per step
    Player death           -3.0  (episode terminated)

Episode termination:
    * Player dies (EXPLORING state persists >8 s after COMBAT was active)
    * OOB reset triggered
    * MAX_STEPS reached (truncation, ~60 s at 60 Hz)
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

# -- Load tunable params from config (Qwen advisor writes here) ----------------
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

import cv2
import mss
import numpy as np
import pydirectinput
import gymnasium as gym
from gymnasium import spaces

try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    _OCR_OK = True
except ImportError:
    _OCR_OK = False

from core.screen     import ScreenGrabber
from core.state      import StateController, GameState
from core.input      import InputEmulator
from models.policy_net import ACTION_KEYS, N_ACTIONS, HOLD_KEYS
from rl.recovery     import RecoveryGuard

# -- Environment constants ------------------------------------------------------
FRAME_SIZE = (224, 224)    # (W, H) for cv2.resize
MAX_STEPS  = 3600          # ~60 s at 60 Hz -> episode truncation
STEP_HZ    = 60            # action rate
STEP_WAIT  = 1.0 / STEP_HZ   # 16.67 ms budget per step
TAP_HOLD   = 0.020            # 20 ms - one Roblox tick at 60 fps; min to register

# Derived from HOLD_KEYS in policy_net - no hardcoded indices needed
_HOLD_IDX = frozenset(i for i, k in enumerate(ACTION_KEYS) if k in HOLD_KEYS)
_TAP_IDX  = frozenset(i for i, k in enumerate(ACTION_KEYS) if k not in HOLD_KEYS)

# Action indices for reward shaping
_M1_IDX    = ACTION_KEYS.index('m1')    # left click
_W_IDX     = ACTION_KEYS.index('w')     # forward - movement reward
_SKILL_IDX = frozenset(                  # Z X C V F
    i for i, k in enumerate(ACTION_KEYS) if k in {'z', 'x', 'c', 'v', 'f'}
)
_AIR_ATTACK_BASE     = _P.get("AIR_ATTACK_BASE",    0.05)
_AIR_ATTACK_GRACE    = _P.get("AIR_ATTACK_GRACE",   8)
_AIR_ATTACK_MAX_MUL  = _P.get("AIR_ATTACK_MAX_MUL", 4)

# -- RND intrinsic reward scaling ----------------------------------------------
# rnd_beta scales the normalised curiosity signal (~N(0,1)) to the same
# order of magnitude as the extrinsic rewards.  0.3 -> curiosity can add
# +/-0.3/step, competing with the time penalty (-0.005) without dominating
# the kill reward (+5.0).  Zero disables curiosity without restarting.
_RND_BETA            = float(_P.get("rnd_beta", 0.3))

# -- Out-of-bounds / ocean detection - TWO-BRANCH OR -------------------------
#
# Branch A  (daytime / bright ocean)   - ALL four must pass:
#   bri  > ocean_bright_bri_min  (85)   <- land+monster (77) and ice arena (88)
#   std  < ocean_bright_var_max  (35)   <- ocean is smooth (~10-28)
#   B-R  > ocean_bright_bias_min (38)   <- ocean sky-reflection (~40-51)
#   G-R  > ocean_bright_gr_min   (15)   <- water elevates green above red
#                                          ice crystals G-R~8, ocean G-R>15
#
# Branch B  (night / dark ocean)       - both must pass:
#   bri  < ocean_dark_bri_max    (60)   <- very dark
#   std  < ocean_dark_var_max    (8)    <- extremely uniform (night water ~3.6)
#                                          boss arena (std~22) safely rejected
#                                          even without the game-state gate
#
# Side strips (cols 56-90 LEFT, cols 134-168 RIGHT) skip character glow.
# Gate: only runs when game_state == EXPLORING and boss bar not visible.
_OCEAN_BRIGHT_BRI_MIN  = float(_P.get("ocean_bright_bri_min",  85.0))
_OCEAN_BRIGHT_VAR_MAX  = float(_P.get("ocean_bright_var_max",  35.0))
_OCEAN_BRIGHT_BIAS_MIN = float(_P.get("ocean_bright_bias_min", 38.0))
_OCEAN_BRIGHT_GR_MIN   = float(_P.get("ocean_bright_gr_min",   15.0))
_OCEAN_DARK_BRI_MAX    = float(_P.get("ocean_dark_bri_max",    60.0))
_OCEAN_DARK_VAR_MAX    = float(_P.get("ocean_dark_var_max",     8.0))
_OCEAN_PENALTY         = _P.get("OCEAN_PENALTY",      5.0)
_OCEAN_FAIL_STEPS      = _P.get("OCEAN_FAIL_STEPS",   1)
_OCEAN_FAIL_PENALTY    = _P.get("OCEAN_FAIL_PENALTY", 50.0)

# -- Reward sensor - relative coordinates (resolution-independent) -------------
# Boss HP bar region
_BAR_TOP_F   = 85   / 1440   # 0.059
_BAR_BOT_F   = 116  / 1440   # 0.081
_BAR_LEFT_F  = 1137 / 2560   # 0.444
_BAR_RIGHT_F = 1812 / 2560   # 0.707

# Quest kill counter OCR - fractions for mss grab (absolute screen coords)
_QUEST_LEFT_F = 60  / 2560   # 0.023
_QUEST_TOP_F  = 280 / 1440   # 0.194
_QUEST_W_F    = 240 / 2560   # 0.094
_QUEST_H_F    = 50  / 1440   # 0.035

_KILL_CHECK_EVERY = 30

# Death heuristic
_DEATH_TIMEOUT_S = 8.0


# -- Reward Sensor -------------------------------------------------------------

class _RewardSensor:
    """
    Lean reward signals - no flaky HP percentages.

      boss_visible(frame)  -> bool   Is a HP bar on screen?  (~0.2 ms pixel scan)
      kill_delta()         -> int    +1 each kill, 0 otherwise (OCR every 30 steps)
    """

    def __init__(self) -> None:
        self._sct = mss.MSS()
        # Scale quest ROI to actual monitor resolution at startup
        mon = self._sct.monitors[2]
        mw, mh = mon['width'], mon['height']
        self._quest_roi = {
            "left":   int(mw * _QUEST_LEFT_F),
            "top":    int(mh * _QUEST_TOP_F),
            "width":  int(mw * _QUEST_W_F),
            "height": int(mh * _QUEST_H_F),
        }
        self._prev_kills: int | None = None
        self._steps                  = 0

    def boss_visible(self, frame_native: np.ndarray) -> bool:
        """
        Boolean: is ANY warm-colored horizontal bar (red/orange/yellow) visible
        at the boss HP region?  Coordinates scale to any monitor resolution.
        Fast (~0.2 ms vectorized numpy) - safe to call every step at 60 Hz.
        """
        try:
            h, w = frame_native.shape[:2]
            top   = int(h * _BAR_TOP_F)
            bot   = int(h * _BAR_BOT_F)
            left  = int(w * _BAR_LEFT_F)
            right = int(w * _BAR_RIGHT_F)
            region = frame_native[top:bot, left:right]
            r = region[:, :, 2].astype(np.int16)
            b = region[:, :, 0].astype(np.int16)
            warm = (r > 130) & (b < 120) & ((r - b) > 50)
            return bool(warm.sum(axis=1).max() >= 30)
        except Exception:
            return False

    def in_ocean(self, obs_chw: np.ndarray) -> bool:
        """
        Boolean: is the bot in the ocean (OOB)?

        TWO-BRANCH detection - Branch A OR Branch B must pass.

        Branch A - Daytime / bright ocean  (all four required):
          bri  > 85   Land+monster reads 77  -> excluded by minimum brightness.
          std  < 35   Ocean is smooth (waves ~ 28).
          B-R  > 38   Ocean sky reflection.
          G-R  > 15   Water elevates green above red. Ice crystals read 8.6.

        Branch B - Night / dark ocean  (both required):
          bri  < 60   Very dark.
          std  < 8    Extremely uniform (night water ~ 3.6).
                      Boss arena (std ~ 22) safely rejected even without gate.

        Strips: LEFT (cols 56-90) + RIGHT (cols 134-168), rows 85-160.
        Gate (in step()): only runs when EXPLORING and no boss bar visible.

        ~0.05 ms (two 75x34 float32 slices, one std call).
        """
        try:
            r_l = obs_chw[0, 85:160,  56: 90].astype(np.float32)
            g_l = obs_chw[1, 85:160,  56: 90].astype(np.float32)
            b_l = obs_chw[2, 85:160,  56: 90].astype(np.float32)
            r_r = obs_chw[0, 85:160, 134:168].astype(np.float32)
            g_r = obs_chw[1, 85:160, 134:168].astype(np.float32)
            b_r = obs_chw[2, 85:160, 134:168].astype(np.float32)
            r   = np.concatenate([r_l.ravel(), r_r.ravel()])
            g   = np.concatenate([g_l.ravel(), g_r.ravel()])
            b   = np.concatenate([b_l.ravel(), b_r.ravel()])
            bri       = (r + b) * 0.5
            mean_bri  = float(np.mean(bri))
            std_bri   = float(np.std(bri))
            blue_bias = float(np.mean(b) - np.mean(r))
            green_red = float(np.mean(g) - np.mean(r))   # water elevates G; ice does not

            # Branch A: daytime/bright ocean - bright, smooth, blue, green-elevated
            bright_ocean = (mean_bri  > _OCEAN_BRIGHT_BRI_MIN
                            and std_bri   < _OCEAN_BRIGHT_VAR_MAX
                            and blue_bias > _OCEAN_BRIGHT_BIAS_MIN
                            and green_red > _OCEAN_BRIGHT_GR_MIN)

            # Branch B: night/dark ocean - very dark and extremely uniform
            dark_ocean   = (mean_bri  < _OCEAN_DARK_BRI_MAX
                            and std_bri < _OCEAN_DARK_VAR_MAX)

            return bright_ocean or dark_ocean
        except Exception:
            return False

    def kill_delta(self) -> int:
        """
        Reads the quest kill counter ("X/500") and returns the increase since
        the last call.  Runs OCR every _KILL_CHECK_EVERY steps to limit cost.
        Returns 0 normally, 1 when a kill registers.
        """
        self._steps += 1
        if not _OCR_OK or self._steps % _KILL_CHECK_EVERY != 0:
            return 0
        try:
            img  = np.array(self._sct.grab(self._quest_roi))[:, :, :3]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=3, fy=3,
                              interpolation=cv2.INTER_CUBIC)
            _, thr = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            thr  = cv2.bitwise_not(thr)
            text = pytesseract.image_to_string(
                thr,
                config="--psm 7 --oem 3 "
                       "-c tessedit_char_whitelist=0123456789/")
            m = re.search(r'(\d+)\s*/\s*\d+', text)
            if m is None:
                return 0
            count = int(m.group(1))
            if self._prev_kills is None:
                self._prev_kills = count
                return 0
            delta = max(0, count - self._prev_kills)
            self._prev_kills = count
            return delta
        except Exception:
            return 0

    def reset(self) -> None:
        self._prev_kills = None
        self._steps      = 0


# -- Gymnasium Environment -----------------------------------------------------

class BrokenBladeEnv(gym.Env):
    """Custom Gymnasium environment wrapping a live Broken Blade session."""

    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 input_emu:   InputEmulator,
                 state_ctl:   StateController,
                 grabber:     ScreenGrabber,
                 rnd_model:   object | None = None,
                 render_mode: str    | None = None) -> None:
        super().__init__()
        self._input      = input_emu
        self._state      = state_ctl
        self._grabber    = grabber    # shared - avoids double DXGI grab
        self._rnd        = rnd_model  # None = RND disabled
        self._sensor     = _RewardSensor()
        self.render_mode = render_mode

        # Spaces
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(3, *FRAME_SIZE), dtype=np.uint8)
        self.action_space = spaces.MultiDiscrete([2] * N_ACTIONS)

        # OOB / stuck protection - timing-decoupled from RL step rate
        self._recovery = RecoveryGuard(input_emu)

        # Episode bookkeeping
        self._step_count:        int            = 0
        self._t_last_combat:     float          = 0.0
        self._held:              set[int]       = set()
        self._air_attack_streak: int            = 0
        self._ocean_streak:      int            = 0   # consecutive steps in ocean

    # -- Gymnasium API ---------------------------------------------------------

    def reset(self, *, seed: int | None = None,
              options: dict | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._release_all()
        self._step_count        = 0
        self._t_last_combat     = 0.0
        self._air_attack_streak = 0
        self._ocean_streak      = 0
        self._sensor.reset()
        self._recovery.reset_timers()
        time.sleep(0.4)   # let the game settle
        return self._obs(), {}

    def step(self, action: np.ndarray
             ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        t0 = time.perf_counter()

        # -- Phase 1: fire hold-key deltas + ALL tap keyDowns at once (~1 ms) --
        taps = self._apply_start(action)

        # -- Phase 2: grab frame WHILE keys are held (overlaps tap hold window) -
        # mss screen capture takes ~8 ms.  Doing it here means the 20 ms tap
        # hold costs us only max(capture, TAP_HOLD) instead of capture+TAP_HOLD.
        native = self._grab_native()

        # -- Phase 3: wait out the remaining hold time, then release taps -------
        # Only sleep when there are actual tap keys to hold - no-tap steps get
        # the full 16.67 ms budget and can genuinely hit 60 Hz.
        if taps:
            held_for = time.perf_counter() - t0
            if held_for < TAP_HOLD:
                time.sleep(TAP_HOLD - held_for)
            self._apply_end(taps)

        obs        = self._frame_to_obs(native)
        reward     = self._reward()
        game_state = self._state.state

        # -- Boss visibility - computed early to gate ocean detection ----------
        # boss_alive is also reused by the attack-gating block below.
        boss_alive = self._sensor.boss_visible(native)

        # -- Ocean detection - gated by game state + boss bar ------------------
        # Three-factor pixel check runs ONLY during EXPLORING with no boss bar.
        # If the game state is COMBAT or ENGAGING, or the boss HP bar is on
        # screen, we are guaranteed to be on land - skip the pixel check.
        #
        # This prevents false positives in dark, blue-ambient arenas (e.g.
        # the Velik boss arena) where floor colours match all three thresholds.
        if game_state == GameState.EXPLORING and not boss_alive:
            in_ocean = self._sensor.in_ocean(obs)
        else:
            in_ocean = False

        # -- Movement reward - counteracts all-negative collapse during navigation
        # +0.003/step for pressing W while exploring/engaging.  Small enough not
        # to dominate combat rewards (+5 kill / +10 boss) but enough to beat the
        # time penalty (-0.005) and keep the policy moving toward the boss.
        if action[_W_IDX] and game_state in (GameState.EXPLORING,
                                              GameState.ENGAGING):
            reward += 0.003

        # -- RND intrinsic curiosity reward ------------------------------------
        # Scoped to navigation phases only.  Crucially, ZERO when in_ocean:
        # water is a "novel" area the bot hasn't explored, so without this gate
        # the curiosity bonus partially offsets the ocean penalty and the bot
        # keeps wandering into the sea.
        #   EXPLORING (on land) -> full curiosity signal
        #   ENGAGING            -> 20 % signal (boss nearby)
        #   COMBAT / in_ocean   -> zero
        if self._rnd is not None and not in_ocean:
            r_i = float(max(-1.0, min(1.0, self._rnd.get_reward(obs))))
            if game_state == GameState.EXPLORING:
                reward += r_i * _RND_BETA
            elif game_state == GameState.ENGAGING:
                reward += r_i * _RND_BETA * 0.2

        # -- Attack gating - pixel-based "is a bar on screen?" check ------------
        # Air attack penalty is ONLY applied in ENGAGING/COMBAT (near boss).
        # During EXPLORING the bot is navigating - pressing attack keys by habit
        # from IL pretraining is harmless and should not be penalised (otherwise
        # PPO learns "never attack" before it ever reaches the boss).
        # boss_alive already computed above.
        if action[_M1_IDX] or any(action[i] for i in _SKILL_IDX):
            if boss_alive:
                self._air_attack_streak = 0
                if action[_M1_IDX]:
                    reward += 0.15
            elif game_state != GameState.EXPLORING:
                # Near boss but missing - penalise after grace period
                self._air_attack_streak += 1
                over_grace = self._air_attack_streak - _AIR_ATTACK_GRACE
                if over_grace > 0:
                    multiplier = min(over_grace, _AIR_ATTACK_MAX_MUL)
                    reward -= _AIR_ATTACK_BASE * multiplier
            else:
                # Exploring - reset streak, no penalty
                self._air_attack_streak = 0
        else:
            self._air_attack_streak = 0

        # -- Ocean / OOB detection - catastrophic penalty -----------------------
        # in_ocean already computed above (needed earlier for RND gate).
        ocean_terminate = False
        if in_ocean:
            if self._ocean_streak == 0:
                print(f"[Env] Ocean entered at step {self._step_count}")
            self._ocean_streak += 1
            reward -= _OCEAN_PENALTY
            if self._ocean_streak >= _OCEAN_FAIL_STEPS:
                reward -= _OCEAN_FAIL_PENALTY
                ocean_terminate = True
                print(f"\n[Env] [OCEAN]  Ocean OOB - terminating episode "
                      f"(streak={self._ocean_streak} steps)")
        else:
            self._ocean_streak = 0

        if game_state == GameState.COMBAT:
            self._t_last_combat = time.time()

        # -- Recovery guard (decoupled from step rate via time.time()) ---------
        recovery = self._recovery.check(game_state, native)
        recovery_reset = (recovery == 'reset')

        self._step_count += 1
        terminated = self._is_dead() or recovery_reset or ocean_terminate
        truncated  = (self._step_count >= MAX_STEPS) and not terminated

        if ocean_terminate:
            # Force character reset on ocean termination so next episode
            # starts at spawn, not deeper in the ocean
            self._recovery._character_reset()
        if terminated:
            if not recovery_reset and not ocean_terminate:
                reward -= 3.0   # death penalty
            self._release_all()

        # -- Phase 5: pace to 60 Hz - sleep only what's left -------------------
        total = time.perf_counter() - t0
        if total < STEP_WAIT:
            time.sleep(STEP_WAIT - total)

        return obs, reward, terminated, truncated, {
            "step":         self._step_count,
            "game_state":   game_state.name,
            "recovery":     recovery,
            "in_ocean":     in_ocean,
            "ocean_streak": self._ocean_streak,
        }

    def close(self) -> None:
        self._release_all()

    # -- Reward ----------------------------------------------------------------

    def _reward(self) -> float:
        """
        Clean, sparse reward signal - no flaky HP percentages.
          -0.005  per step (time penalty)
          +5.0    per quest kill (OCR every 30 steps, monotonic)
          +10.0   on boss respawn-timer detected (one-time per kill)
        Attack/no-attack shaping is applied in step() based on bar visibility.
        """
        r = -0.005   # per-step time penalty

        # Quest kill counter - main reward signal
        r += self._sensor.kill_delta() * 5.0

        # Boss respawn detected - bonus on top of kill counter
        if self._state.boss_respawning:
            r += 10.0

        return r

    # -- Death detection -------------------------------------------------------

    def _is_dead(self) -> bool:
        """
        Heuristic: if we've been in EXPLORING state for >8 s since the last
        COMBAT contact, the player almost certainly died and respawned.
        """
        if self._t_last_combat == 0.0:
            return False
        if self._state.state == GameState.EXPLORING:
            return (time.time() - self._t_last_combat) > _DEATH_TIMEOUT_S
        return False

    # -- Action application (split into two phases for speed) -----------------
    #
    # Old approach  (sequential taps):  N_taps x 22 ms + frame_grab
    # New approach  (batched + overlap): max(20 ms, frame_grab) + ~1 ms
    #
    # For 3 taps:  3 x 22 ms + 8 ms = 74 ms -> 13 Hz
    #              max(20, 8) + 1   = 21 ms -> 48 Hz   <- 3.5x faster
    #
    # pydirectinput.PAUSE is already 0 (set in core/input.py import).

    def _apply_start(self, action: np.ndarray) -> list[tuple[str, str | None]]:
        """
        Phase 1: apply hold-key state deltas and fire all tap keyDowns at once.
        Returns a list of tap descriptors for _apply_end() to release.

        Hold keys go through InputEmulator (state tracking, jitter acceptable
        because they only fire on transitions). Tap keys call pydirectinput
        directly - no jitter overhead, released after TAP_HOLD in _apply_end.
        """
        taps: list[tuple[str, str | None]] = []
        for idx, pressed in enumerate(action):
            key = ACTION_KEYS[idx]
            if idx in _HOLD_IDX:
                # Hold keys - keep down while active, release when inactive
                if pressed:
                    if idx not in self._held:
                        self._input.key_down(key)
                        self._held.add(idx)
                else:
                    if idx in self._held:
                        self._input.key_up(key)
                        self._held.discard(idx)
            elif pressed:
                # Tap keys - all keyDowns fired simultaneously here
                if key == 'm1':
                    pydirectinput.mouseDown()
                    taps.append(('m1', None))
                elif key == 'm2':
                    pydirectinput.mouseDown(button='right')
                    taps.append(('m2', None))
                else:
                    pydirectinput.keyDown(key)
                    taps.append(('key', key))
        return taps

    def _apply_end(self, taps: list[tuple[str, str | None]]) -> None:
        """Phase 2: release all tap keys/buttons from _apply_start."""
        for kind, key in taps:
            if kind == 'key':
                pydirectinput.keyUp(key)
            elif kind == 'm1':
                pydirectinput.mouseUp()
            else:   # m2
                pydirectinput.mouseUp(button='right')

    def _release_all(self) -> None:
        for idx in list(self._held):
            try:
                self._input.key_up(ACTION_KEYS[idx])
            except Exception:
                pass
        self._held.clear()
        self._input.release_all()

    # -- Observation -----------------------------------------------------------

    def _obs(self) -> np.ndarray:
        return self._frame_to_obs(self._grab_native())

    def _grab_native(self) -> np.ndarray:
        """
        Read the latest full-resolution BGR frame from the shared ScreenGrabber.
        ~0.4 ms (RAM copy) vs ~30-60 ms for a fresh DXGI grab.
        The ScreenGrabber background thread keeps the frame continuously updated
        at 60 fps - no double-grab contention on the DXGI interface.
        """
        frame = self._grabber.get_frame()
        if frame is not None:
            return frame
        # Fallback only on the very first step before the grabber has a frame
        with mss.MSS() as sct:
            monitor = sct.monitors[2]
            return np.ascontiguousarray(
                np.array(sct.grab(monitor))[:, :, :3])

    def _frame_to_obs(self, native: np.ndarray) -> np.ndarray:
        """Resize to 224x224, convert to RGB, return (3, H, W) uint8."""
        small = cv2.resize(native, FRAME_SIZE, interpolation=cv2.INTER_AREA)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb.transpose(2, 0, 1))   # (3, H, W)
