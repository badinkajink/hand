"""Rank statistics and intervals, over scipy.stats.

Binomial tail and intervals need no helper: scipy.stats.binom.sf(k - 1, n, p) and
scipy.stats.binomtest(k, n).proportion_ci(method="wilson" or "exact"). A line with standard errors is
scipy.stats.linregress.
"""
import numpy as np
from scipy import stats

__all__ = ["spearman", "auc", "bootstrap_ci"]


def spearman(a, b):
    """Spearman's rho with tied values at their average rank, over the pairs where both values are finite; NaN for fewer
    than 3 pairs or a constant input."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return float("nan")
    return float(stats.spearmanr(a[ok], b[ok])[0])


def auc(pos, neg):
    """Area under the ROC curve: the share of (pos, neg) pairs with pos above neg, a tie counting one half (the
    Mann-Whitney U statistic over the number of pairs)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float(stats.mannwhitneyu(pos, neg, method="asymptotic").statistic / (len(pos) * len(neg)))


def bootstrap_ci(x, statistic=np.mean, n=20000, seed=0, level=0.95):
    """Percentile interval (low, high) of a statistic of x over n resamples of its elements, by scipy.stats.bootstrap
    with the generator default_rng(seed). None for fewer than 2 elements."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return None
    res = stats.bootstrap((x,), statistic, n_resamples=n, confidence_level=level, method="percentile",
                          random_state=np.random.default_rng(seed), vectorized=True, axis=-1)
    return float(res.confidence_interval.low), float(res.confidence_interval.high)
