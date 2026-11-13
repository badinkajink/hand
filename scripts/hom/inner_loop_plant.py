"""The inner-loop benchmark's plant: the hand-design talk's, as its grid grid2dff_v4b_sv25r was run (the SCS0009 servo
plant, 0.72 N m/rad and 0.18 N m, and the 2.5 MPa pads of the scene cache logs/20261102-hom_servo_plant/scenes).

Import it before hom_gpu or hand_object_scale_sim, which read these variables when they are imported. Until 2026-11-13
10:30 the benchmark's scripts set none of them, and every run of 11-12 and 11-13 (HOM, UHAS training and evaluation, the
usability scorecard, the sampling MPC) used the default work plant (4 N m/rad, 1 N m) and the 10 MPa scene cache of
2026-10-30; those rows carry the suffix _work10.
"""
import os

os.environ.setdefault("SCALE_HAND", "cad_v2")
os.environ.setdefault("HOS_PLANT", "servo")
os.environ.setdefault("HOS_SCENES", "logs/20261102-hom_servo_plant/scenes")
os.environ.setdefault("HOM_ROWS_TAG", "_sv25r")


def describe():
    """The plant and the scene cache in force, for the rows and the logs."""
    import hand_object_scale_sim as S
    return {"plant": S.PLANT, "scenes": str(S.SCENES.relative_to(S.ROOT)) if hasattr(S, "ROOT") else str(S.SCENES)}
