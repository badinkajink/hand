#!/usr/bin/env python3
"""Does each axis's DRV_STATUS really come from its own driver?

    python3 /tmp/drv_crosstalk.py <mover> <watched> [<watched> ...] [--mm 10]

Moves <mover> out by --mm and back (homing speed), the others staying disabled,
while reading DRV_STATUS of the mover and every watched axis. For each, reports
the fraction of samples with stst (standstill, bit 31) clear and the sg range.
Only the mover should ever show stst=0 / a moving sg. Needs the mover's zero
valid (moves 0 -> mm -> 0 from the current count).
"""

import sys
import time

sys.path.insert(1, "/tmp")
import gantry_rom_sweep as g  # noqa: E402
from manta_hand import MantaHandDriver  # noqa: E402


def main():
    args = sys.argv[1:]
    mm = float(args[args.index("--mm") + 1]) if "--mm" in args else 10.0
    js = [int(a) for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] != "--mm")]
    mover, watched = js[0], js[1:]
    g.active_joints.append(mover)
    g.driver = d = MantaHandDriver(g.PORT)
    jt = d.joints[mover]
    try:
        jt.set_scale(g.STEPS_PER_MM)
        p0 = jt.status.position / g.STEPS_PER_MM
        samples = {j: [] for j in [mover] + watched}
        jt.enable()
        for target in (p0 + mm, p0):
            jt.move_to_mm(target, g.HOME_VELOCITY, g.SWEEP_ACCEL)
            t0 = time.monotonic()
            while jt.status.moving:
                if time.monotonic() - t0 > 15:
                    raise RuntimeError("move exceeded cap")
                for j in samples:
                    s = int(d.send("RREG5160", f"J{j}", "6F").split()[0], 16)
                    samples[j].append(s)
        jt.disable()
        for j, ss in samples.items():
            moving = [s for s in ss if not (s >> 31) & 1]
            sg = sorted(s & 0x3FF for s in ss)
            print(f"  J{mover} moving -> read J{j}: stst clear in {len(moving) / len(ss):4.0%} of {len(ss)} reads, "
                  f"sg p10 {sg[len(sg) // 10]} median {sg[len(sg) // 2]} p90 {sg[len(sg) * 9 // 10]}"
                  + ("   <-- the mover" if j == mover else ""), flush=True)
    finally:
        g.disable_all()
        d.close()


if __name__ == "__main__":
    main()
