#!/usr/bin/env python3
"""Measure scalar mapping, duplicate constraint refresh and native solver stages."""

import time
import numpy as np
import mujoco
from contact_surface.records import ROOT, write_json, versions
from contact_surface.runtime import translate_contacts, step1
from distributed_contact_transfer import build, wrench


def main():
    out = ROOT / "results/20261004-distributed-contact/mapping_profile"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for spacing in (1.0, 0.5):
        for policy in ("runtime", "runtime_approx", "compiled"):
            config = dict(spacing_mm=spacing, policy=policy, timestep=5e-5)
            model, data, parameters, meta = build(config)
            for _ in range(6000):
                wrench(model, data, meta, 0.0)
                if policy.startswith("runtime"):
                    step1(model, data, parameters)
                    mujoco.mj_step2(model, data)
                else:
                    mujoco.mj_step(model, data)
            samples = []
            for _ in range(1000):
                timings = {}
                start = time.perf_counter()
                wrench(model, data, meta, 0.0)
                tick = time.perf_counter()
                timings["wrench"] = tick - start
                if policy.startswith("runtime"):
                    mujoco.mj_step1(model, data)
                    timings["step1"] = time.perf_counter() - tick
                    translate_contacts(model, data, parameters, timings)
                    tick = time.perf_counter()
                    mujoco.mj_step2(model, data)
                    timings["step2"] = time.perf_counter() - tick
                else:
                    mujoco.mj_step(model, data)
                    timings["native_step"] = time.perf_counter() - tick
                timings["total"] = time.perf_counter() - start
                samples.append(timings)
            rows.append(
                dict(
                    config=config,
                    ncon=data.ncon,
                    samples=1000,
                    mean_us={k: float(np.mean([s[k] for s in samples]) * 1e6) for k in samples[0]},
                    median_us={
                        k: float(np.median([s[k] for s in samples]) * 1e6) for k in samples[0]
                    },
                )
            )
            print(spacing, policy, rows[-1]["mean_us"], flush=True)
    write_json(out / "summary.json", rows)
    write_json(out / "versions.json", versions())


if __name__ == "__main__":
    main()
