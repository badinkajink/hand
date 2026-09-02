"""Bench-side instrumentation: what the hardware can be MEASURED to do.

Nothing in here imports mujoco or torch. The point of the separation is that a
measurement of the bench must be checkable without the simulator that made the
prediction.
"""
