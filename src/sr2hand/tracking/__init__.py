"""Object tracking with AprilTags for the SR2 hand on the UR5 (plan: docs/notes/20261105-object_seating_and_tracking/).

tags   tag families' cells, the tag maps of the hexagonal tag block and of flat plates, in the detector's corner order;
pose   detection, one pose solve over every corner of a body, the joint solve over several cameras, and the bundle
       adjustment of a tag map.

The modules import numpy, scipy and OpenCV (pose only), never MuJoCo, so they run on corvus and under the workstation's
camera Python (~/miniconda3/bin/python).
"""
