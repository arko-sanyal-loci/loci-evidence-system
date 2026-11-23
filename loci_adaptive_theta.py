"""
loci_adaptive_theta.py — Adaptive θ_HOT for the LOCI heat-routing system.

Fixed scalar thresholds assume a stationary score distribution, but SLP scores
shift as context matures (more scored items → tighter separation between MILD and
HOT).  This module tracks context maturity and adjusts θ_HOT accordingly.

Design
──────

  Maturity  m = tanh(n / N_REF)      ∈ [0, 1]
                 n = scored items seen by this instance
                 N_REF = 40 → 63% maturity at 40 items; near-full at ~120

  Adaptive  θ(m) = FLOOR + m × (θ_target − FLOOR)
                 FLOOR  = 0.640  immature start: inclusive, high recall
                 θ_target ∈ [ANCHOR, CEIL]
                           → ANCHOR (0.7250) until enough data
                           → valley midpoint once the distribution is stable

  Valley    Histogram recent scores over [0.60, 0.90] in 15 buckets (Δ = 0.02).
            Locate the minimum-density bucket.  Its midpoint is the natural
            MILD–HOT boundary; clamped to [ANCHOR − 0.05, CEIL].

            Falls back to ANCHOR when < MIN_ITEMS_FOR_VALLEY scored items
            are available or no clear valley exists.

Singleton
─────────
  ADAPTIVE_THETA_HOT  — module-level instance, shared by all importers.
  Call  .update(slp_score)  each time a pair is scored.
  Read  .theta              to get the current threshold.

Thread safety: .update() and .theta use a simple lock — safe for asyncio + threads.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from typing import Optional


# ── Constants ─────────────────────────────────────────────────────────────────

THETA_FLOOR  = 0.640   # immature sessions: low, inclusive, high recall
THETA_ANCHOR = 0.7250  # defended analytical value; also the convergence target
THETA_CEIL   = 0.825   # hard cap (observed natural boundary on nomic v2 corpus)

N_REF                 = 40    # tanh normaliser: 63% maturity at N_REF items
MIN_ITEMS_FOR_VALLEY  = 15    # need at least this many items to fit a valley
WINDOW_SIZE           = 150   # sliding window (oldest scores discarded)

# Histogram parameters for valley detection
HIST_LO     = 0.60
HIST_HI     = 0.90
HIST_BINS   = 15                            # each bin spans 0.02
_BIN_WIDTH  = (HIST_HI - HIST_LO) / HIST_BINS


# ── Core adaptive class ───────────────────────────────────────────────────────

class AdaptiveThetaHot:
    """
    Tracks scored SLP values and returns a maturity-calibrated θ_HOT.

    Usage
    ─────
        from loci_adaptive_theta import ADAPTIVE_THETA_HOT

        router = SymPrevRouter()
        for query, entity in pairs:
            result = router.score(query, entity)
            slp = result.slp_score(heat=arc_heat)
            ADAPTIVE_THETA_HOT.update(slp)
            if slp >= ADAPTIVE_THETA_HOT.theta:
                # HOT path
    """

    def __init__(
        self,
        floor:   float = THETA_FLOOR,
        anchor:  float = THETA_ANCHOR,
        ceil:    float = THETA_CEIL,
        n_ref:   int   = N_REF,
        window:  int   = WINDOW_SIZE,
    ) -> None:
        self._floor  = floor
        self._anchor = anchor
        self._ceil   = ceil
        self._n_ref  = n_ref
        self._scores: deque[float] = deque(maxlen=window)
        self._total:  int = 0
        self._lock = threading.Lock()

    # ── Public interface ───────────────────────────────────────────────────────

    def update(self, slp_score: float) -> None:
        """Record a new SLP observation; call once per scored pair."""
        with self._lock:
            self._scores.append(slp_score)
            self._total += 1

    @property
    def theta(self) -> float:
        """Current adaptive θ_HOT.  Thread-safe read."""
        with self._lock:
            return self._compute_theta()

    @property
    def maturity(self) -> float:
        """Context maturity ∈ [0, 1].  Thread-safe read."""
        with self._lock:
            return self._maturity()

    def snapshot(self) -> dict:
        """Diagnostic snapshot — not for routing decisions."""
        with self._lock:
            m = self._maturity()
            return {
                "theta":           round(self._compute_theta(), 4),
                "theta_floor":     self._floor,
                "theta_anchor":    self._anchor,
                "theta_ceil":      self._ceil,
                "theta_target":    round(self._theta_target(), 4),
                "maturity":        round(m, 4),
                "n_window":        len(self._scores),
                "n_total":         self._total,
                "valley_midpoint": round(self._valley(), 4) if len(self._scores) >= MIN_ITEMS_FOR_VALLEY else None,
            }

    def reset(self) -> None:
        """Hard reset — clears all observations, returns to immature state."""
        with self._lock:
            self._scores.clear()
            self._total = 0

    # ── Private computation ────────────────────────────────────────────────────

    def _maturity(self) -> float:
        return math.tanh(len(self._scores) / self._n_ref)

    def _valley(self) -> float:
        """
        Find the minimum-density point in [HIST_LO, HIST_HI].

        Builds a histogram over scores in [0.60, 0.90], finds the bucket
        with the lowest count, and returns its midpoint.  Returns ANCHOR
        when the minimum is not well-defined (flat distribution or too few
        items).
        """
        in_range = [s for s in self._scores if HIST_LO <= s <= HIST_HI]
        if len(in_range) < MIN_ITEMS_FOR_VALLEY:
            return self._anchor

        # Build histogram
        counts = [0] * HIST_BINS
        for s in in_range:
            idx = min(int((s - HIST_LO) / _BIN_WIDTH), HIST_BINS - 1)
            counts[idx] += 1

        # Locate min-count bucket (prefer the one in the middle of the range
        # to avoid picking a tail artifact)
        min_count = min(counts)
        candidates = [
            i for i, c in enumerate(counts) if c == min_count
        ]
        # Pick the candidate closest to the middle of the MILD–HOT overlap zone
        center_target = (self._anchor - HIST_LO) / _BIN_WIDTH
        best_idx = min(candidates, key=lambda i: abs(i - center_target))
        valley_mid = HIST_LO + (best_idx + 0.5) * _BIN_WIDTH

        # Sanity clamp: don't stray far from anchor
        return max(self._anchor - 0.05, min(self._ceil, valley_mid))

    def _theta_target(self) -> float:
        """Data-driven target; falls back to anchor when immature."""
        if len(self._scores) < MIN_ITEMS_FOR_VALLEY:
            return self._anchor
        return self._valley()

    def _compute_theta(self) -> float:
        m       = self._maturity()
        target  = self._theta_target()
        blended = self._floor + m * (target - self._floor)
        return max(self._floor, min(self._ceil, blended))

    # ── repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        s = self.snapshot()
        return (
            f"AdaptiveThetaHot("
            f"θ={s['theta']}, maturity={s['maturity']:.2f}, "
            f"n={s['n_window']}/{s['n_total']}, "
            f"target={s['theta_target']})"
        )


# ── Module-level singleton (shared across all importers) ──────────────────────

ADAPTIVE_THETA_HOT = AdaptiveThetaHot()
"""
Shared adaptive θ_HOT instance.  Import and call `.update(slp)` from every
scoring site, then read `.theta` for the current routing threshold.

Each LOCI process keeps one instance per process; it is NOT persisted across
restarts.  On restart θ starts at FLOOR and ramps to ANCHOR over the first
~40 scored items, then tracks the live distribution.
"""


# ── Convenience accessor used by legacy code ─────────────────────────────────

def current_theta() -> float:
    """Drop-in replacement for a fixed THETA_HOT constant."""
    return ADAPTIVE_THETA_HOT.theta


# ── CLI smoke-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random

    print("AdaptiveThetaHot — maturity ramp simulation")
    print(f"  FLOOR={THETA_FLOOR}  ANCHOR={THETA_ANCHOR}  CEIL={THETA_CEIL}  N_REF={N_REF}\n")

    rng = random.Random(42)
    inst = AdaptiveThetaHot()

    # Simulate: HOT items N(0.95, 0.03), MILD items N(0.70, 0.04)
    # 70% MILD, 30% HOT — typical production distribution
    for step in [1, 5, 10, 15, 20, 30, 50, 75, 100, 150]:
        while inst._total < step:
            if rng.random() < 0.30:
                score = min(1.0, rng.gauss(0.95, 0.03))   # HOT
            else:
                score = max(0.0, rng.gauss(0.70, 0.04))   # MILD
            inst.update(score)

        s = inst.snapshot()
        bar_len = int(s["maturity"] * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  n={s['n_window']:4d}  [{bar}] {s['maturity']:.2f}  "
              f"θ={s['theta']:.4f}  target={s['theta_target']:.4f}  "
              f"valley={s['valley_midpoint'] or '—':>6}")
