"""Where the arm presses of 2026-11-19 touched the pad, four ways (the owner, 2026-11-19 18:1x: "again the forces are much
lower priority than contact pose").

The arm pushed a pad into a flat plate along press_dir_B while the finger's servos held (lab/arm_press.py). For each
rep, in the pad frame of the fingertip calibrations (x out of the palmar face, z from the proximal mount face; apex at
(8.5, 0, 10.86)), from the hand model's tool chain at the press's joints (tip_contacts.T_tip_pad, SR2Hand.T_P_contact)
and the arm's pose:
  geometry   the pad's point whose outward normal is the press direction: where a flat plate square to the press
             touches a convex pad, whatever the force
  force      the pad's point whose outward normal is opposite the HEX-E's force in the hold (F_hold_B, the force on the
             hand): where a frictionless contact would have to be to push that way
  tilt       the angle between the HEX-E's force and the press direction, and its direction along the pad: the
             plate's friction (or a plate not square to the press)
  GP         the trials' estimator's place (tip_contacts' board GP, median over the press's loaded bins;
             sim/20261119-arm_press_boards.py)
  pattern    the best-matching near-normal calibration press (cosine of the field per newton; same file)
If the boards' near-normal patterns read the force's direction more than its place, the pattern's place follows the
force's place, not the geometry's; the GP, trained on presses and slides, can tell them apart.
  tags       the geometry again with each finger's DIP link turned about its flexion axis by what the DIP tags measured
             on 2026-11-18 (NOTES.md, S1(b): index 9.2 deg and middle 12.2 deg more flexed than the joint readings; the
             thumb within 2 deg, taken as 0), added to the PIP

The pad: the v4b half-ellipsoid (semi-axes 7.4 across, 8.5 deep, 22 proximal and 11 distal of the apex), its support
point for a normal m is (A^2 m_x, B^2 m_y, C^2 m_z) / sqrt(A^2 m_x^2 + B^2 m_y^2 + C^2 m_z^2) with C the half's axis.

    /usr/bin/python3 sim/20261119-arm_press_contact_place.py [--out data/s3/20261119-arm_press_contact_place.json]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src/sr2hand/driver/reskin5x/host")]
from sr2hand.robot.tool_chain import SR2Hand  # noqa: E402
import tip_contacts as TC  # noqa: E402

LOGS = ROOT / "logs/20261119-sr2_s3_press"
AY, AX, AZD, AZP, ZA, ZEND = 7.4, 8.5, 11.0, 22.0, 10.86, 21.41
TAG_FLEX_DEG = {"index": 9.2, "middle": 12.2, "thumb": 0.0}      # 11-18 DIP tags, more flexed than read
RUNS = {"20261119-170343_index_arm_press": "index, PIP 0", "20261119-170846_middle_arm_press": "middle, PIP 0",
        "20261119-171311_thumb_arm_press": "thumb, PIP 0", "20261119-171923_index_arm_press": "index, PIP 10",
        "20261119-172508_index_arm_press": "index, PIP 20"}


def support(m):
    """The pad's surface point (pad frame, mm) whose outward normal is m; z below 0 is off the pad's proximal end."""
    m = np.asarray(m, float) / np.linalg.norm(m)
    C = AZD if m[2] >= 0 else AZP
    ax = np.array([AX, AY, C])
    p = ax ** 2 * m / math.sqrt(float(np.sum(ax ** 2 * m ** 2)))
    return np.array([p[0], p[1], ZA + p[2]])


def normal(p):
    """The pad's outward unit normal at the surface point p (pad frame)."""
    u = p[2] - ZA
    C = AZD if u >= 0 else AZP
    g = np.array([p[0] / AX ** 2, p[1] / AY ** 2, u / C ** 2])
    return g / np.linalg.norm(g)


def on_surface(y, z):
    u = z - ZA
    C = AZD if u >= 0 else AZP
    s = 1.0 - (y / AY) ** 2 - (u / C) ** 2
    return np.array([AX * math.sqrt(max(s, 0.0)), y, z])


def R_P_pad(hand, finger, q_deg):
    """Pad frame -> P rotation at the joints (yaw, mcp, pip; simulator degrees), the gantry at its box centre (the
    gantry only translates)."""
    T = hand.T_P_contact(finger, 0.0, 0.0, *(math.radians(v) for v in q_deg))
    return T[:3, :3] @ TC.T_tip_pad(finger, hand)[:3, :3]


def deg(a, b):
    return math.degrees(math.acos(float(np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", default=str(STUDY / "data/s3/20261119-arm_press_boards.json"))
    ap.add_argument("--out", default=str(STUDY / "data/s3/20261119-arm_press_contact_place.json"))
    a = ap.parse_args()
    hand = SR2Hand.load()
    boards = {r["run"].replace(".jsonl", ""): r for r in json.loads(Path(a.boards).read_text())["runs"]}
    rows = []
    for stem, label in RUNS.items():
        L = [json.loads(x) for x in (LOGS / f"{stem}.jsonl").read_text().splitlines() if x.strip()]
        h = L[0]
        finger = h["finger"]
        q = [h["joints"][finger][j] for j in ("yaw", "mcp", "pip")]
        R_BP = np.array(h["T_BP_end"], float)[:3, :3]
        R = R_BP @ R_P_pad(hand, finger, q)                      # pad frame -> B
        u = R.T @ np.array(h["press_dir_B"], float)              # the press direction in the pad frame
        p_geo = support(u)
        qt = list(q)
        qt[2] += TAG_FLEX_DEG[finger]
        p_tag = support((R_BP @ R_P_pad(hand, finger, qt)).T @ np.array(h["press_dir_B"], float))
        bd = {pr["rep"]: pr for pr in boards.get(stem, {}).get("presses", [])}
        for r in L[1:]:
            if "F_hold_B" not in r or r.get("F_hold_B") is None:
                continue
            fB = np.array(r["F_hold_B"], float)
            f = R.T @ fB                                          # force on the pad, pad frame
            p_force = support(-f)
            tilt = deg(-fB, np.array(h["press_dir_B"], float))
            t_pad = -f / np.linalg.norm(f) - u * float(np.dot(-f / np.linalg.norm(f), u))   # the lean, pad frame
            b = bd.get(r["rep"], {})
            gp = (b.get("estimators", {}).get("gp", {}) or {}).get("p_med")
            pat = b.get("best_p")
            row = dict(run=stem, group=label, rep=r["rep"], finger=finger, joints_sim_deg=np.round(q, 2).tolist(),
                       F_hold_B_N=np.round(fB, 3).tolist(), f_pad_N=np.round(f, 3).tolist(),
                       press_dir_pad=np.round(u, 4).tolist(), tilt_deg=round(tilt, 1),
                       lean_pad=np.round(t_pad, 3).tolist(),
                       z_geometry=round(float(p_geo[2]), 2), y_geometry=round(float(p_geo[1]), 2),
                       z_geometry_tags=round(float(p_tag[2]), 2),
                       z_force=round(float(p_force[2]), 2), y_force=round(float(p_force[1]), 2),
                       z_gp=None if gp is None else round(float(gp[2]), 2),
                       y_gp=None if gp is None else round(float(gp[1]), 2),
                       z_pattern=None if pat is None else round(float(pat[2]), 2), best_cos=b.get("best_cos"),
                       normal_tilt_geometry_deg=round(deg(normal(p_geo), [1, 0, 0]), 1),
                       normal_tilt_force_deg=round(deg(normal(p_force), [1, 0, 0]), 1))
            if gp is not None:
                pg = on_surface(float(gp[1]), float(gp[2]))
                row["normal_gp_vs_geometry_deg"] = round(deg(normal(pg), normal(p_geo)), 1)
            rows.append(row)
            print(f"{label:14s} rep {r['rep']}: F_hold_B {np.round(fB, 2)} N, tilt {tilt:4.1f} deg; z geometry "
                  f"{p_geo[2]:5.1f}, force {p_force[2]:5.1f}, GP {row['z_gp']}, pattern {row['z_pattern']} "
                  f"(cos {row['best_cos']}); y geometry {p_geo[1]:4.1f}, force {p_force[1]:4.1f}, GP {row['y_gp']}; "
                  f"lean (pad) {row['lean_pad']}", flush=True)
    Path(a.out).write_text(json.dumps({"what": __doc__.split("\n\n")[0].replace("\n", " "),
                                       "script": str(Path(__file__).resolve().relative_to(ROOT)), "rows": rows},
                                      indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
