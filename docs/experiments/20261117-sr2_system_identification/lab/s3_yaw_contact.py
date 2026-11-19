"""Read only: where a planned yaw press first meets the S3 block (2026-11-19, for the PLA-side drag). Puts the hand at an
s3_approach plan's end pose and sweeps the pressing finger's yaw from its present reading in the press's direction,
0.25 deg at a time. For each yaw, every cad_v3 collision sphere of every finger (and hom_setup's tip pad, tip end and PIP
balls) gets its clearance to the block's boxes at nominal size (s3_approach.model's boxes without their noise, or grown
by --noise-*). At the first contact it prints the touching part and its height against the plate's top, the finger's
spheres within 1 mm of the plate, the DIP link's gap to the face's plane in 2 mm bands of height (flush or tilted), the
lip's clearance, and every box's clearance after a drag of --up mm.

On 2026-11-19 (index, sign -1, the taped plate): at the plan of 11-18 (gap 6, dz 13) the DIP link's flat side (10.5 mm
from the finger's axis, from 4 to 32 mm above its end) met the plate nearly flush, its top 0.1 mm off the lip. At gap 14
and dz 15 the yaw meets the tape 9 deg further on, with the link's lower end 15 mm below the plate's top first.

    /usr/bin/python3 s3_yaw_contact.py PLAN [--to 20] [--up 5] [--dz 0] [--dgap 0] [--noise-z 0]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("plan")
    ap.add_argument("--to", type=float, default=20.0, help="deg of yaw travel in the press's direction")
    ap.add_argument("--up", type=float, default=5.0, help="mm: the drag up to check after the contact")
    ap.add_argument("--dz", type=float, default=0.0, help="mm: shift the plan's end pose up")
    ap.add_argument("--dgap", type=float, default=0.0, help="mm: move the plan's end pose away from its plate")
    ap.add_argument("--noise-z", type=float, default=0.0)
    ap.add_argument("--noise-n", type=float, default=0.0)
    ap.add_argument("--noise-l", type=float, default=0.0)
    a = ap.parse_args()
    sys.argv = sys.argv[:1]
    import s3_approach as SA
    import s3_press as S3
    FT, HS, SL = S3.imports()
    rec = json.loads(Path(a.plan).read_text())
    f, sign, side = rec["finger"], rec["sign"], rec["plate"]
    if sign is None:
        raise SystemExit("not a yaw press plan")
    p, nh, w = np.array(rec["tag_B"]), np.array(rec["nh_B"]), np.array(rec["w_B"])
    T = np.array(rec["T_BP_end"], float)
    T[2, 3] += a.dz
    T[:3, 3] += a.dgap * (nh if side == "tag" else -nh)
    top, hw, tape = SA.TABLE_Z + SA.TOP, SA.PLATE / 2, rec["tape_mm"]
    nz, nn, nl = a.noise_z, a.noise_n, a.noise_l
    zp = (top - SA.PLATE - nz, top + nz)
    boxes = {"tag plate": ((-SA.PLATE_T - nn, nn), (-hw - nl, hw + nl), zp),
             "post": ((-S3.BEHIND + SA.PLATE_T - nn, -SA.PLATE_T + nn), (-SA.POST / 2 - nl, SA.POST / 2 + nl),
                      (SA.TABLE_Z, top + nz)),
             "pad plate": ((-S3.BEHIND - tape - nn, -S3.BEHIND + SA.PLATE_T + nn), (-hw - nl, hw + nl), zp),
             "lip": ((-S3.BEHIND - tape - nn, nn), (-hw - nl, hw + nl), (top, top + SA.LIP + nz))}
    plate = f"{side} plate"
    face = 0.0 if side == "tag" else -S3.BEHIND - tape          # the pressed face's s
    M, joints0 = rec["mounts_P"], {k: dict(v) for k, v in rec["joints"].items()}
    y0 = joints0[f]["yaw"]
    print(f"{f} yaw sign {sign:+d} on the {plate}: present yaw {y0:+.2f}; end pose palm B "
          f"{np.round(T[:3, 3], 1).tolist()} (dz {a.dz:+g}, dgap {a.dgap:+g}); the plate's top z_B {top:.1f}, the lip "
          f"to {top + SA.LIP:.1f}; noise z {nz:g} n {nn:g} l {nl:g}")

    def clear(T, joints):
        X_P, R, lab = SA.hand_points(FT, HS, M, joints)
        X = X_P @ T[:3, :3].T + T[:3, 3]
        return X, R, lab, {name: SA.box_gaps(X, R, p, nh, w, box) for name, box in boxes.items()}

    for step in np.arange(0.0, abs(a.to) + 1e-9, 0.25):
        yaw = y0 + sign * step
        J = {k: dict(v) for k, v in joints0.items()}
        J[f]["yaw"] = float(yaw)
        X, R, lab, out = clear(T, J)
        name, (g, front) = min(out.items(), key=lambda kv: float(np.min(kv[1][0])))
        i = int(np.argmin(g))
        if abs(step % 2.0) < 1e-9:
            print(f"  yaw {yaw:+6.2f}: least {g[i]:+6.2f} mm ({name}: {lab[i][0]} {lab[i][1]} at "
                  f"{X[i, 2] - top:+.1f} mm from the top)")
        if g[i] > 0:
            continue
        print(f"FIRST CONTACT at yaw {yaw:+.2f} ({yaw - y0:+.2f} from the present): {name} with {lab[i][0]} "
              f"{lab[i][1]}, {X[i, 2] - top:+.1f} mm from the plate's top")
        gp, fp = out[plate]
        near = [(lab[j][1], round(float(X[j, 2] - top), 1), round(float(gp[j]), 2), bool(fp[j]))
                for j in np.argsort(gp)[:12] if lab[j][0] == f and gp[j] < 1.0]
        print("  the finger's spheres within 1 mm of the plate (part, mm from the top, gap, in front of the face):")
        for row in near:
            print("   ", row)
        dip = np.array([lb[0] == f and lb[1] == "pip_frame" for lb in lab])
        gap_face = np.abs((X - p) @ nh - face) - R                  # each sphere's gap to the face's plane
        zz = X[:, 2] - top
        print("  the DIP link's least gap to the face's plane by height from the plate's top (2 mm bands):")
        for z0 in np.arange(-30.0, 12.0, 2.0):
            m = dip & (zz >= z0) & (zz < z0 + 2.0)
            if m.any():
                print(f"    {z0:+5.0f}..{z0 + 2:+3.0f} mm: {gap_face[m].min():+6.2f}")
        gl = out["lip"][0]
        j = int(np.argmin(gl))
        print(f"  the lip's least {gl[j]:+.2f} mm ({lab[j][0]} {lab[j][1]} at {X[j, 2] - top:+.1f} mm from the top)")
        T2 = T.copy()
        T2[2, 3] += a.up
        X2, R2, lab2, out2 = clear(T2, J)
        for name2, (g2, fr2) in out2.items():
            j = int(np.argmin(g2))
            print(f"  after {a.up:+g} mm: {name2:9s} least {g2[j]:+6.2f} ({lab2[j][0]} {lab2[j][1]} at "
                  f"{X2[j, 2] - top:+.1f} mm from the top, in front {bool(fr2[j])})")
        return 0
    print("no contact in the sweep")
    return 1


if __name__ == "__main__":
    sys.exit(main())
