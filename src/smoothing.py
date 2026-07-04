"""One-Euro landmark smoothing to de-jitter MediaPipe hand tracking.

Raw ``HandLandmarker`` output jitters frame-to-frame (a few thousandths in
normalized space), and that jitter is what makes pinch/grab feel flickery. A
One-Euro filter (Casiez et al. 2012) is the standard fix: it low-pass-filters
each coordinate with a cutoff that ADAPTS to speed -- heavy smoothing when the
hand is still (kills jitter), light smoothing when it moves fast (kills lag). So
a resting hand is rock-steady while a fast drag stays responsive.

Pure math (no MediaPipe/threads), so it's unit-testable headlessly. The tracking
thread owns a ``LandmarkSmoother`` (single owner, like the detectors) and runs
raw hands through it before gesture detection; a slot with no hand this frame has
its filters reset so a reappearing hand doesn't smear from stale state.
"""

import math
from typing import List, Optional, Tuple

from src.hand_tracker import Hand, Landmark

# Filter tuning (normalized coords, ~20-30 fps). ``min_cutoff`` sets the
# at-rest smoothing (lower = steadier but laggier); ``beta`` sets how much fast
# motion loosens it (higher = less lag on a quick drag). ``d_cutoff`` smooths the
# speed estimate itself. Defaults chosen to visibly de-jitter a still hand
# without a perceptible drag lag; retune here if tracking feels sticky/floaty.
DEFAULT_MIN_CUTOFF = 1.2
DEFAULT_BETA = 0.03
DEFAULT_D_CUTOFF = 1.0

# Fallback dt (seconds) when timestamps are equal/non-increasing, so the filter
# can't divide by zero on a repeated timestamp.
_MIN_DT = 1e-3


def _alpha(cutoff: float, dt: float) -> float:
    """Low-pass smoothing factor for a given cutoff frequency and time step."""
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """One-Euro filter for a single scalar signal (one landmark coordinate)."""

    def __init__(
        self,
        min_cutoff: float = DEFAULT_MIN_CUTOFF,
        beta: float = DEFAULT_BETA,
        d_cutoff: float = DEFAULT_D_CUTOFF,
    ) -> None:
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        """Forget all history (call when the signal's source disappears)."""
        self._x_prev: Optional[float] = None
        self._dx_prev = 0.0
        self._t_prev: Optional[float] = None

    def filter(self, x: float, t: float) -> float:
        """Return the smoothed value of ``x`` sampled at time ``t`` (seconds)."""
        if self._x_prev is None or self._t_prev is None:
            self._x_prev, self._t_prev = x, t
            return x
        dt = t - self._t_prev
        if dt <= 0.0:
            dt = _MIN_DT
        dx = (x - self._x_prev) / dt
        edx = self._dx_prev + _alpha(self._d_cutoff, dt) * (dx - self._dx_prev)
        cutoff = self._min_cutoff + self._beta * abs(edx)
        x_hat = self._x_prev + _alpha(cutoff, dt) * (x - self._x_prev)
        self._x_prev, self._dx_prev, self._t_prev = x_hat, edx, t
        return x_hat


class LandmarkSmoother:
    """Per-slot One-Euro smoothing of whole hands (21 landmarks x 3 coords each).

    One filter bank per hand slot; a slot with no hand this frame is reset so a
    reappearing hand starts clean rather than smearing across the gap. Single
    owner (the tracking thread) -- not thread-safe by design.
    """

    def __init__(
        self,
        num_hands: int,
        landmarks_per_hand: int = 21,
        min_cutoff: float = DEFAULT_MIN_CUTOFF,
        beta: float = DEFAULT_BETA,
        d_cutoff: float = DEFAULT_D_CUTOFF,
    ) -> None:
        self._banks = [
            [
                tuple(OneEuroFilter(min_cutoff, beta, d_cutoff) for _ in range(3))
                for _ in range(landmarks_per_hand)
            ]
            for _ in range(num_hands)
        ]

    def smooth(self, hands: List[Hand], t: float) -> List[Hand]:
        """Return ``hands`` with every coordinate One-Euro-smoothed at time ``t``.

        Hands beyond the configured slot count pass through unsmoothed (never
        happens in practice -- num_hands matches the tracker); empty slots have
        their filter banks reset. Never raises.
        """
        out: List[Hand] = []
        for slot, bank in enumerate(self._banks):
            if slot < len(hands) and hands[slot]:
                out.append(self._smooth_hand(bank, hands[slot], t))
            else:
                for landmark_filters in bank:
                    for f in landmark_filters:
                        f.reset()
        # Pass through any hands past our slot count (defensive; shouldn't occur).
        out.extend(hands[len(self._banks):])
        return out

    @staticmethod
    def _smooth_hand(bank, hand: Hand, t: float) -> Hand:
        smoothed: List[Landmark] = []
        for i, (x, y, z) in enumerate(hand):
            if i >= len(bank):  # more landmarks than expected -> pass through
                smoothed.append((x, y, z))
                continue
            fx, fy, fz = bank[i]
            smoothed.append((fx.filter(x, t), fy.filter(y, t), fz.filter(z, t)))
        return smoothed
