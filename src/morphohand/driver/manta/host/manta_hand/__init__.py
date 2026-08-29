from .driver import MantaHandDriver
from .joint import Joint, JointStatus
from .kinematics import Gantry, GantryFinger

__all__ = ["MantaHandDriver", "Joint", "JointStatus", "Gantry", "GantryFinger"]

try:
    from .servos import ServoBus, Servo, ServoStatus, Finger
    from .hand import Hand, HandFinger

    __all__ += ["ServoBus", "Servo", "ServoStatus", "Finger", "Hand", "HandFinger"]
except ImportError:
    # rustypot (the "servo" extra) isn't installed -- fine, the stepper
    # side doesn't need it. `from manta_hand.servos import ServoBus` will
    # raise its own clear error if someone actually tries to use it.
    pass
