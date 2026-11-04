"""Smoothing and lookup of time series, over numpy. Whole-sequence filters are scipy.signal and scipy.ndimage;
np.interp and scipy.interpolate.interp1d interpolate."""
import math

import numpy as np

__all__ = ["moving_average", "trailing_average", "window_mean", "index_at", "loglog_interp", "smoothstep"]


def moving_average(x, k, edges="nearest"):
    """Centred mean over k samples, the same length as x. edges="nearest" repeats the end values beyond the ends;
    edges="shrink" averages over the samples the window still has there. For an even k the window takes one sample more
    before than after."""
    x = np.asarray(x, float)
    if k <= 1:
        return x
    if edges == "nearest":
        return np.convolve(np.r_[np.full(k // 2, x[0]), x, np.full(k - 1 - k // 2, x[-1])], np.ones(k) / k, mode="valid")
    if edges == "shrink":
        return np.convolve(x, np.ones(k), mode="same") / np.convolve(np.ones(len(x)), np.ones(k), mode="same")
    raise ValueError(f"moving_average: edges must be nearest or shrink, not {edges!r}")


def trailing_average(x, k):
    """Mean over the last k samples up to and including each sample; fewer at the start."""
    x = np.asarray(x, float)
    c = np.r_[0.0, np.cumsum(x)]
    i1 = np.arange(1, len(x) + 1)
    i0 = np.maximum(i1 - k, 0)
    return (c[i1] - c[i0]) / (i1 - i0)


def window_mean(ts, xs, t, half):
    """Mean of xs over the samples with |ts - t| <= half; NaN when there is none."""
    m = np.abs(np.asarray(ts) - t) <= half
    return float(np.asarray(xs)[m].mean()) if m.any() else float("nan")


def index_at(ts, t, side="next"):
    """Index into the sorted times ts of the first sample at or after t (side="next") or the last at or before t
    (side="previous"), clamped to the series."""
    n = len(ts)
    if side == "next":
        return int(np.clip(np.searchsorted(ts, t), 0, n - 1))
    if side == "previous":
        return max(0, int(np.searchsorted(ts, t, side="right")) - 1)
    raise ValueError(f"index_at: side must be next or previous, not {side!r}")


def loglog_interp(x, xs, ys):
    """Piecewise power law through the sorted points (xs, ys), extended beyond the ends with the nearest segment's
    exponent. One x."""
    lx, lxs, lys = math.log(x), np.log(xs), np.log(ys)
    k = int(np.clip(np.searchsorted(lxs, lx) - 1, 0, len(lxs) - 2))
    return float(np.exp(lys[k] + (lys[k + 1] - lys[k]) * (lx - lxs[k]) / (lxs[k + 1] - lxs[k])))


def smoothstep(a, b, x):
    """0 below a, 1 above b, and 3 t^2 - 2 t^3 of t = (x - a) / (b - a) between them."""
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)
