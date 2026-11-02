"""Feetech SCS0009 actuator and real-pendulum testbench for BAM (Better Actuator Models).

Plugs into the unmodified BAM checkout in ../bam_upstream: `register()` adds the actuator
name "scs0009" to bam.actuators.actuators, which bam.fit and bam.model.load_model both read.

Firmware control law, measured from the load register (= PWM duty x 1000) over all
81 single-pendulum logs of 2026-10-29:
    duty = 0.189 * kp * err + 0.036 * sign(err)    for |err| > dead zone (1 step = 0.29 deg)
    duty = 0                                       inside the dead zone
    duty clipped to [-1, 1]                        (load reads 1000 at saturation)
The 0.036 offset is the servo's "minimum startup force" setting (45). Both terms are fixed,
not fitted. The rate limiter on the internal target (as in BAM's STS3215 model) is kept with
a fitted max_velocity; if the SCS0009 firmware has none, the fit drives it to its upper bound.

Testbench: the identification arm runs past the weight, so BAM's point-mass-on-uniform-rod
Pendulum does not fit. RealPendulum uses the measured first moment and inertia that
prepare_logs.py writes into each processed log.
"""
from __future__ import annotations

import numpy as np

from bam.actuator import VoltageControlledActuator
from bam.parameter import Parameter
from bam.testbench import Testbench

G = 9.80665
ERROR_GAIN = 0.189          # duty per (kp * rad)
MIN_PWM = 0.036             # minimum startup force 45 -> 3.6% duty
DEAD_ZONE_RAD = np.radians(300.0 / 1024)   # 1 encoder step
# Firmware D term acts on the joint velocity: duty -= D_GAIN * kd * dq (rad/s). From the load
# register in the double-pendulum logs at D 15-150 (d_term_check.py): 0.0011-0.0022, ~0.0015.
D_GAIN = 0.0015


class RealPendulum(Testbench):
    """Pendulum with measured mass properties: gravity torque = -g * first_moment * sin(q)."""

    def __init__(self, log: dict):
        self.first_moment = log["first_moment"]   # kg*m
        self.inertia = log["inertia"]             # kg*m^2 about the servo axis

    def compute_mass(self, q, dq):
        return self.inertia

    def compute_bias(self, q, dq):
        return -G * self.first_moment * np.sin(q)


class SCS0009Actuator(VoltageControlledActuator):
    """Feetech SCS0009, run at the hand's 5 V supply (servos read 4.6-4.8 V under load)."""

    stateful = True

    def __init__(self, testbench_class=RealPendulum):
        super().__init__(testbench_class, vin=4.8, kp=15, error_gain=ERROR_GAIN, max_pwm=1.0)
        self.kd = 0             # firmware D gain, set from the log (BAM fits: 0; hand: 15)
        # Encoder quantization seen by the firmware (rad per count), None = continuous.
        # The real servo reads whole counts (300 deg / 1024), so after each slip the error drops
        # into the dead zone and PWM/load fall to 0: the load flicker seen on the hand.
        self.quantize_step = None
        self.q_target_smooth = None
        self._to_reset_env_ids: list = []

    def load_log(self, log: dict):
        super().load_log(log)
        self.kd = log.get("kd", 0)

    def initialize(self):
        # Output-side torque constant [Nm/A] = back-EMF constant [V/(rad/s)].
        # Peak measured speed 12 rad/s at ~4.8 V caps it near 0.4.
        self.model.kt = Parameter(0.35, 0.05, 1.5)
        # Winding resistance [ohm]; stall ~0.18 Nm at 4.8 V suggests ~10 ohm.
        self.model.R = Parameter(10.0, 1.0, 40.0)
        # Motor + gearbox apparent inertia at the output [kg m^2]
        self.model.armature = Parameter(1e-4, 1e-6, 5e-3)
        # Firmware rate limit on its internal target [rad/s]
        self.model.max_velocity = Parameter(30.0, 2.0, 60.0)
        # D-term scale; not identifiable from D = 0 logs, so fixed unless fitted on purpose
        self.model.d_gain = Parameter(D_GAIN, 0.0, 0.006, optimize=False)

    def get_extra_inertia(self) -> float:
        return self.model.armature.value

    # --- internal target state (same handling as BAM's STS3215Actuator) ---
    def reset(self, env_ids=...) -> None:
        if env_ids is ...:
            self.q_target_smooth = None
            self._to_reset_env_ids.clear()
        else:
            self._to_reset_env_ids.append(env_ids)

    def get_state(self):
        return self.q_target_smooth, list(self._to_reset_env_ids)

    def set_state(self, state) -> None:
        q_target_smooth, ids = (None, []) if state is None else state
        self.q_target_smooth = q_target_smooth
        self._to_reset_env_ids = list(ids)

    def compute_control(self, q_target, q, dq, dt):
        qs = self.q_target_smooth
        if qs is None:
            qs = q
            self._to_reset_env_ids.clear()
        elif self._to_reset_env_ids:
            for ids in self._to_reset_env_ids:
                qs[ids] = q[ids]
            self._to_reset_env_ids.clear()
        step = self.model.max_velocity.value * dt
        self.q_target_smooth = self.backend.clamp(q_target, qs - step, qs + step)

        if self.quantize_step:
            s = self.quantize_step
            err = np.round(self.q_target_smooth / s) * s - np.round(q / s) * s
        else:
            err = self.q_target_smooth - q
        duty = err * self.kp * self.error_gain + MIN_PWM * self.backend.sign(err)
        duty = duty - self.model.d_gain.value * self.kd * dq
        duty = duty * (abs(err) > DEAD_ZONE_RAD)
        duty = self.backend.clamp(duty, -self.max_pwm, self.max_pwm)
        self.duty_cycle = duty
        return self.vin * duty


def register() -> None:
    from bam.actuators import actuators
    actuators["scs0009"] = lambda: SCS0009Actuator(RealPendulum)
