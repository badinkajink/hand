from .joint import Joint, JointStatus
from .kinematics import Gantry, GantryFinger
from .plan import HandPlan, Pose
from .runtime import HandRuntime, MockHardwareBackend

__all__ = ["Joint", "JointStatus", "Gantry", "GantryFinger", "HandPlan", "Pose",
           "HandRuntime", "MockHardwareBackend"]

try:
    from .driver import MantaHandDriver

    __all__ += ["MantaHandDriver"]
except ImportError:
    # pyserial isn't installed -- fine off the CB1. Everything that actually
    # talks to the M8P needs it; the calibration tables in kinematics.py do
    # not, and an offline planner validating a trajectory against them should
    # not have to install a serial stack to read them. `from manta_hand.driver
    # import MantaHandDriver` raises its own clear error if someone tries to
    # open a real link. Same pattern as the servo extra below.
    pass

try:
    from .servos import ServoBus, Servo, ServoStatus, Finger
    from .hand import Hand, HandFinger

    __all__ += ["ServoBus", "Servo", "ServoStatus", "Finger", "Hand", "HandFinger"]
except ImportError:
    # rustypot (the "servo" extra) isn't installed -- fine, the stepper
    # side doesn't need it. `from manta_hand.servos import ServoBus` will
    # raise its own clear error if someone actually tries to use it.
    pass
