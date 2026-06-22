"""
RobustHPReader
===============
Stateful filter that turns noisy OCR HP readings into reliable signals.

Strategy:
  1. CALIBRATE  - collect max-HP votes; lock the value after 3 matches
                  (handles OCR noise on max number)
  2. VALIDATE   - for each new (cur, max) read:
                     * max must be within 2 % of the locked value
                     * cur must not exceed max
                     * cur must not rise faster than 50 % HP/s
                     * cur must not drop more than current HP in 0.5 s
  3. HOLD       - on rejection, return cached value for hold_s seconds
  4. RECALIBRATE - if 5+ consecutive reads have a NEW consistent max,
                  accept the change (handles boss switching, level-up)
"""

from __future__ import annotations
import time


class RobustHPReader:
    """Filter noisy OCR readings into a stable HP fraction."""

    def __init__(self, name: str = "hp", hold_s: float = 3.0,
                 boss_mode: bool = False) -> None:
        self.name        = name
        self.hold_s      = hold_s
        self.boss_mode   = boss_mode      # True -> -1 when stale; False -> hold

        self._stable_max:   int | None  = None
        self._candidate_max: dict[int, int] = {}

        self._last_cur:       int | None  = None
        self._last_hp:        float       = -1.0
        self._last_read_time: float       = 0.0
        self._rejects:        int         = 0

    # -- Public API ------------------------------------------------------------

    @property
    def hp(self) -> float:
        """Most recently accepted HP fraction in [0, 1], or -1 if unknown."""
        return self._last_hp

    @property
    def current(self) -> int | None:
        return self._last_cur

    @property
    def max(self) -> int | None:
        return self._stable_max

    def update(self, cur: int | None, mx: int | None) -> float:
        """
        Submit a raw OCR (cur, max) reading.
        Returns the current HP fraction (filtered) - last cached value if
        the read was rejected, or -1.0 if stale & boss_mode.
        """
        now = time.time()

        # No OCR result this round
        if cur is None or mx is None or mx <= 0 or cur < 0:
            return self._cached_or_stale(now)

        # -- Stage 1: Calibrate max HP ----------------------------------------
        if self._stable_max is None:
            self._candidate_max[mx] = self._candidate_max.get(mx, 0) + 1
            best_val, best_count = max(
                self._candidate_max.items(), key=lambda kv: kv[1])
            if best_count >= 3:
                self._stable_max = best_val
                self._candidate_max.clear()
                if 0 <= cur <= self._stable_max:
                    self._accept(cur, now)
                return self._last_hp if self._last_hp >= 0 else cur / mx
            # During calibration, return raw reading for visualization
            return cur / mx if mx > 0 else -1.0

        # -- Stage 2: Validate against locked max -----------------------------
        diff_pct = abs(mx - self._stable_max) / self._stable_max
        if diff_pct > 0.02:
            # Possible boss switch / level-up - require 5 consistent reads
            self._candidate_max[mx] = self._candidate_max.get(mx, 0) + 1
            best_val, best_count = max(
                self._candidate_max.items(), key=lambda kv: kv[1])
            if best_count >= 5:
                self._stable_max = best_val
                self._candidate_max.clear()
                self._last_cur = None  # reset cur history after max changes
            else:
                self._rejects += 1
                return self._cached_or_stale(now)
        else:
            self._candidate_max.clear()

        # -- Stage 3: Validate current ---------------------------------------
        if cur > self._stable_max:
            self._rejects += 1
            return self._cached_or_stale(now)

        if self._last_cur is not None and self._last_read_time > 0:
            elapsed = max(0.01, now - self._last_read_time)
            change  = cur - self._last_cur
            # Bound rapid HP rises (heals) - max 50 % HP/s + 5 % floor
            max_inc = self._stable_max * 0.50 * elapsed \
                    + self._stable_max * 0.05
            if change > max_inc:
                self._rejects += 1
                return self._cached_or_stale(now)

        # All checks passed
        self._rejects = 0
        self._accept(cur, now)
        return self._last_hp

    def reset(self) -> None:
        self._stable_max      = None
        self._candidate_max   = {}
        self._last_cur        = None
        self._last_hp         = -1.0
        self._last_read_time  = 0.0
        self._rejects         = 0

    # -- Internals -------------------------------------------------------------

    def _accept(self, cur: int, now: float) -> None:
        self._last_cur       = cur
        self._last_hp        = cur / self._stable_max if self._stable_max else -1.0
        self._last_read_time = now

    def _cached_or_stale(self, now: float) -> float:
        if self._last_read_time == 0.0:
            return -1.0
        age = now - self._last_read_time
        if age < self.hold_s:
            return self._last_hp
        # Stale
        if self.boss_mode:
            return -1.0
        # Player mode: hold last known value indefinitely
        return self._last_hp
