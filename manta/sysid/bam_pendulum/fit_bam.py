#!/usr/bin/env python3
"""Run BAM's fitter (bam.fit, unmodified) with the SCS0009 actuator registered.

    ../.venv-bam/bin/python fit_bam.py --logdir data_processed/single --actuator scs0009 \
        --model m6 --output fits/m6.json --trials 3000 --validation_kp 4
All arguments go straight to bam.fit.
"""
import runpy
import sys

import bam_scs0009

bam_scs0009.register()
sys.argv[0] = "bam.fit"
runpy.run_module("bam.fit", run_name="__main__")
