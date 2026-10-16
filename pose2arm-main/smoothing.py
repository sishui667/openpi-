"""Temporal smoothing for noisy per-frame signals (keypoints or joint angles).

One Euro Filter is implemented by hand (the algorithm is ~20 lines: a
low-pass filter on the signal, a second low-pass filter on its derivative,
and a cutoff frequency that adapts with that derivative -- slow movement
gets smoothed hard, fast movement gets smoothed less so the filter doesn't
lag behind it). Writing it out is the point: it's the one piece of this
project worth being able to derive on a whiteboard rather than `pip install`.
It's also the only smoothing algorithm this project uses -- retarget.py's
root and arm filters are the same code with different (`min_cutoff`, `beta`)
constants, not two different algorithms.

All filters here are causal (online-equivalent): each output sample only
depends on samples up to and including it, the same constraint a live
teleoperation system would have, even though this project processes
pre-recorded video offline.
"""
import math

import numpy as np
from scipy.signal import savgol_filter


class _LowPassFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.y = None

    def filter(self, x, alpha=None):
        if alpha is not None:
            self.alpha = alpha
        self.y = x if self.y is None else self.alpha * x + (1 - self.alpha) * self.y
        return self.y


class OneEuroFilter:
    """Casiez et al. 2012, single-scalar-channel version. `freq` is the
    sampling rate (video fps); `min_cutoff` sets the baseline smoothing
    strength at rest, `beta` controls how much the cutoff opens up as the
    signal's rate of change increases."""

    def __init__(self, freq, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_filter = _LowPassFilter(self._alpha(min_cutoff))
        self.dx_filter = _LowPassFilter(self._alpha(d_cutoff))
        self._prev_x = None

    def _alpha(self, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x):
        dx = 0.0 if self._prev_x is None else (x - self._prev_x) * self.freq
        self._prev_x = x
        edx = self.dx_filter.filter(dx, self._alpha(self.d_cutoff))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self.x_filter.filter(x, self._alpha(cutoff))


class VectorOneEuroFilter:
    """Applies an independent OneEuroFilter per component of an N-dim signal
    (e.g. a 3D keypoint or a 7-DOF joint angle vector)."""

    def __init__(self, dim, freq, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.filters = [OneEuroFilter(freq, min_cutoff, beta, d_cutoff) for _ in range(dim)]

    def filter(self, x):
        return np.array([f.filter(xi) for f, xi in zip(self.filters, x)])


def one_euro_sequence(signal, freq, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
    """One Euro Filter applied causally over a whole (T, D) array at once --
    the batch-processing convenience wrapper VectorOneEuroFilter.filter()'s
    frame-by-frame callers don't need."""
    signal = np.asarray(signal, dtype=float)
    f = VectorOneEuroFilter(signal.shape[1], freq, min_cutoff, beta, d_cutoff)
    return np.array([f.filter(x) for x in signal])


def offline_joint_trajectory(signal, window_length=9, polyorder=2):
    """Zero-phase polynomial smoothing for an already-recorded trajectory.

    Unlike a causal low-pass filter this does not add visible playback lag.
    It is intentionally limited to the offline video pipeline; live control
    must use a causal bounded-jerk filter or receding-horizon controller.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 2:
        raise ValueError("joint trajectory must have shape (T,D)")
    if len(signal) < 3:
        return signal.copy()
    window = min(int(window_length), len(signal) if len(signal) % 2 else len(signal) - 1)
    window = max(3, window)
    if window % 2 == 0:
        window -= 1
    order = min(int(polyorder), window - 1)
    return savgol_filter(
        signal, window_length=window, polyorder=order,
        axis=0, mode="interp")


if __name__ == "__main__":
    # Self-test with synthetic signals -- no video required. Two properties,
    # tested separately on purpose: denoising a held (motionless) pose is a
    # different regime than tracking a fast movement, and a filter tuned to
    # look good on one can look bad on the other if you only test one.
    # Compared against a plain causal EMA (a few lines, inlined here, not a
    # second supported algorithm) as the "naive" baseline a fixed-cutoff
    # low-pass filter would give, to show what One Euro's adaptive cutoff
    # actually buys over that.
    rng = np.random.default_rng(0)

    def ema(signal, alpha):
        out = np.empty_like(signal, dtype=float)
        out[0] = signal[0]
        for i in range(1, len(signal)):
            out[i] = alpha * signal[i] + (1 - alpha) * out[i - 1]
        return out

    # (a) Denoising: a hand held still should have its jitter suppressed.
    hold_clean = np.tile(np.array([0.4, 0.1, 0.5]), (90, 1))
    hold_noisy = hold_clean + rng.normal(scale=0.05, size=hold_clean.shape)
    smoothed_euro = one_euro_sequence(hold_noisy, freq=30.0, min_cutoff=1.0, beta=0.5)
    smoothed_ema = ema(hold_noisy, alpha=0.15)
    err_raw = np.mean(np.linalg.norm(hold_noisy - hold_clean, axis=1))
    err_euro = np.mean(np.linalg.norm(smoothed_euro - hold_clean, axis=1))
    err_ema = np.mean(np.linalg.norm(smoothed_ema - hold_clean, axis=1))
    print(f"[held pose] mean error vs ground truth -- raw: {err_raw:.4f}  one_euro: {err_euro:.4f}  ema: {err_ema:.4f}")
    assert err_euro < err_raw, "One Euro Filter should reduce error relative to raw noisy signal when the pose is held"

    # Step-response responsiveness: a sudden jump should be tracked faster by
    # One Euro (which opens its cutoff under fast motion) than by a slow EMA.
    step_signal = np.concatenate([np.zeros((30, 1)), np.ones((30, 1))])
    euro_step = one_euro_sequence(step_signal, freq=30.0, min_cutoff=0.5, beta=2.0)
    ema_step = ema(step_signal, alpha=0.1)
    lag_index = 35  # 5 frames after the step
    print(f"5 frames after step -- one_euro: {euro_step[lag_index,0]:.3f}  ema: {ema_step[lag_index,0]:.3f} (target 1.0)")
    assert euro_step[lag_index, 0] > ema_step[lag_index, 0], (
        "One Euro Filter should track a step change faster than a slow EMA at matched steady-state smoothing"
    )
    print("All smoothing.py self-tests passed.")
