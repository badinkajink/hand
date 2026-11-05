"""Wrapping of angles. np.unwrap unwraps a whole sequence; these work on one value at a time."""
import numpy as np

__all__ = ["wrap", "nearest", "unwrap_step"]


def wrap(a, period=2.0 * np.pi):
    """a in [-period / 2, period / 2): radians by default, period=360.0 for degrees."""
    return (a + period / 2) % period - period / 2


def nearest(q, q_ref, period=2.0 * np.pi):
    """The equivalent of the angle q nearest to q_ref (a joint angle against the joint's start, say)."""
    return q_ref + wrap(q - q_ref, period)


def unwrap_step(prev, new, period=2.0 * np.pi):
    """The new reading `new` of a turning angle, taken within half a period of the unwrapped previous value prev."""
    return prev + wrap(new - prev, period)
