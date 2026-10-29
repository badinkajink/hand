#!/usr/bin/env python3
"""v3 ID 1's calibration suite on the UR5 cell (owner, 2026-10-27 21:5x, to the lab: "sure, we can run the v3 id 1
registration and suite ... the rod has changed position, yes, but everything else is the same"): v2 ID 1's actions as
they ran, cut as scripts/ur5_tip_calibration_suite_value.SUITE says, in the order they ran, as plan files the lab
program runs after v3 ID 1's sphere pass 1 (lab/superseded/20261027-plan_sphere_lab_pass1_v3id1.json).

For a stage, every action of the v2 ID 1 session that ended ok is labelled by family (suite_value.coarse_label) and
kept, reduced (the first n of a seed-0 permutation of the family's ids, as suite_value.suite_mask) or cut by its SUITE
row. Joint 6 as v2 ID 1's robot.bin measured it during the action decides whether it runs unattended: an action whose
joint 6 went past +44 deg (or under -61) goes to the morning list (owner present; the unattended limit is +45 deg),
written as its own plan. The plan keeps the structure of the session's last plan file (geometry, travel, safety,
speed_slider 0.30) with checks.other set to tonight's cell: the tall indenter, the rod at each of its three seatings and
the fingertip holder with both held tips (kind tiphold). Each plan's expected robot time is the sum of its actions'
durations in v2 ID 1's run (the program's speed changes of the suite-value analysis are not in the lab program yet).

The rod (owner via the lab, 21:5x: the rod's three hand seatings of 19:36 to 19:42 differ by up to 4 deg, mostly roll
about the tool axis, and the holder's seating turned out 4.3 deg off). rod1 is its pass 1 as a search: v2 ID 1's 21 pad
touch-offs on the rod, starting 8 mm above the expected contact (3.5 before), reaching 5 mm past it (1.2) with the slow
approach over the whole distance (travel.slow_mm_touch 13 mm), then the 5 pivot presses; no gravity poses (owner,
10-23). The program's registration of the rod from those onsets replaces the seating's frame for rod2, the rest of v2 ID
1's rod session with the suite's cuts. seeds writes one registration folder per rod seating (v2 ID 1's sphere
registration with that seating's T_BI, flange pose and joints) for the path checks; --base-reg and --seating LABEL=FOLDER
make one for a seating recorded later on v3 ID 1's own tool frames (the lab, 2026-10-28 04:2x: a fresh seating after the
rod's fixture goes back on).

The tall indenter (stage 4): tall1 is v2 ID 1's pass 1 on it as it ran (143529-tallv2id1, 21 touch-offs to 2.0 mm past).
tall2 is R29 (lab/superseded/20261027-plan_tall_lab_r29_v2id1.json: its presses, 50 slides, 4 of its 12 twists by the seed-0
permutation, the final reference) with six shared points first after the opening reference: tall presses at targets
that the sphere stage presses too (three on the face, three on the edges, the sphere's force levels), each at the first
twist of 0, 10, 20, 30, 40, -10 and -20 deg whose replay (ur5_tip_calibration_holder_paths: travel from 60 mm above,
approach, depth) passes every check with joint 6 at +44 deg or less. A replay of the whole plan then moves every action
outside joint 6's unattended window to the tall's morning plan.

    .venv/bin/python scripts/ur5_tip_calibration_v3_suite.py sphere2|rod1|rod2|seeds|tall1|tall2 [--created ...]
"""
from __future__ import annotations

import argparse
import collections
import copy
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("UR5_MOUNT", "v7")            # the mount on the robot since R25 (2026-10-26)
import ur5_tip_calibration_suite_value as SV  # noqa: E402

LAB = SV.LAB
J6_MAX, J6_MIN = 44.0, -61.0


def cell_other():
    T = lambda f: json.loads((LAB / f / "registration.json").read_text())["T_BI"]  # noqa: E731
    out = [dict(kind="tall", T_BI=T("20261027-143529-tallv2id1"))]
    out += [dict(kind="rod", T_BI=T(f)) for f in ("20261027-rod-seed", "20261027-rod-seed_2", "20261027-rod-seed_3")]
    out += [dict(kind="tiphold", T_BI=T("20261027-holder-apex-both"))]
    return out


def joint6(session, recs):
    """(min, max) joint 6 (deg) of each action during its run, from the session's robot.bin."""
    R = np.fromfile(LAB / session / "robot.bin", dtype="<f8").reshape(-1, 15)
    t, q6 = R[:, 0], np.degrees(R[:, 7])
    out = {}
    for s, e, _ in recs:
        m = (t >= s["t"]) & (t <= e["t"])
        if m.any():
            out[e["id"]] = (float(q6[m].min()), float(q6[m].max()))
    return out


def suite_rows(stage):
    return [r for r in SV.SUITE if r[0] == stage]


def row_for(rows, ses, label):
    kind = label.split("/")[0]
    for r in rows:
        if r[1] == ses and (r[2] == label or r[2] == f"{kind}/*"):
            return r
    return None


def select(stage, ses_key, skip_kinds=("touch",)):
    session = SV.SESSIONS[ses_key]
    recs, bad = SV.records(session)
    rows = suite_rows(stage)
    lab = [SV.coarse_label(s.get("kind"), e["id"]) for s, e, _ in recs]
    keep_ids = {}
    for r in rows:
        if r[1] != ses_key or r[3] == "cut":
            continue
        kind = r[2].split("/")[0]
        ids = sorted({e["id"] for (s, e, _), lb in zip(recs, lab) if lb.split("/")[0] == kind
                      and (r[2].endswith("/*") or lb == r[2])})
        n = r[4]
        n_keep = int(round(n * len(ids))) if isinstance(n, float) else int(n)
        keep_ids[r[2]] = set(np.random.default_rng(0).permutation(ids)[:n_keep].tolist()) if n_keep < len(ids) else set(ids)
    j6 = joint6(session, recs)
    kept, morning, cut, seen = [], [], collections.Counter(), set()
    for (s, e, act), lb in zip(recs, lab):
        aid = e["id"]
        if aid in seen or s.get("kind") in skip_kinds:
            continue
        seen.add(aid)
        r = row_for(rows, ses_key, lb)
        if r is None or r[3] == "cut" or aid not in keep_ids.get(r[2], set()):
            cut[lb] += 1
            continue
        a = {k: v for k, v in act.items() if not k.startswith("_")}
        a["_dur_s"] = e["t"] - s["t"]
        a["_j6"] = j6.get(aid)
        lo, hi = j6.get(aid, (0.0, 0.0))
        (morning if (hi > J6_MAX or lo < J6_MIN) else kept).append((lb, a))
    return session, kept, morning, cut, bad


def write_plan(session, acts, out, created, note, stage_name):
    files = sorted(glob.glob(str(LAB / session / "plan*.json")))
    base = json.loads(Path(files[-1]).read_text())
    p = {k: copy.deepcopy(v) for k, v in base.items() if k != "actions"}
    p["created"] = created
    p["checks"]["other"] = cell_other()
    p["notes"] = list(p.get("notes", [])) + [note]
    p["actions"] = [{k: v for k, v in a.items() if not k.startswith("_")} for _, a in acts]
    p["suite"] = dict(stage=stage_name, tip="v3 ID 1", source_session=session, script="scripts/ur5_tip_calibration_v3_suite.py",
                      expected_h=round(sum(a["_dur_s"] for _, a in acts) / 3600, 2),
                      families=dict(collections.Counter(lb for lb, _ in acts)))
    Path(out).write_text(json.dumps(p, indent=1) + "\n")
    return p


def cmd_sphere2(a):
    session, kept, morning, cut, bad = select("2 sphere", "sphere")
    h = sum(x["_dur_s"] for _, x in kept) / 3600
    hm = sum(x["_dur_s"] for _, x in morning) / 3600
    fam = collections.Counter(lb for lb, _ in kept)
    print(f"sphere stage 2: {len(kept)} actions, {h:.2f} h as v2 ID 1 ran them; morning (joint 6 past +44): {len(morning)}, "
          f"{hm:.2f} h; cut: {dict(cut)}")
    print("kept by family:", dict(fam))
    note = (f"{a.created}, workstation: v3 ID 1's suite, stage 2 on the low sphere (scripts/ur5_tip_calibration_v3_suite.py "
            f"sphere2): the {len(kept)} actions of v2 ID 1's sphere session 20261026-192158-v2id1 after pass 1 that ended ok, in "
            f"the order they ran, with the suite-value cuts (twist and roll rates cut, the rate set reduced to 6); "
            f"{len(morning)} actions whose joint 6 went past +44 deg in v2 ID 1's run are in the morning plan. About {h:.1f} h "
            "at the program's present speeds. Runs after pass 1 on its registration (resume in the same session).")
    out = LAB / "superseded/20261027-plan_sphere_lab_stage2_v3id1.json"
    write_plan(session, kept, out, a.created, note, "2 sphere")
    outm = LAB / "superseded/20261027-plan_sphere_lab_morning_v3id1.json"
    write_plan(session, morning, outm, a.created, note.replace("stage 2 on the low sphere", "stage 2's morning part on the "
               "low sphere, joint 6 past +44 deg (owner present)"), "2 sphere, morning")
    print("->", out.relative_to(ROOT), "and", outm.relative_to(ROOT))


ROD_SEATINGS = {"rod_indenter_seated": "20261027-rod-seed", "rod_indenter_seated_2": "20261027-rod-seed_2",
                "rod_indenter_seated_3": "20261027-rod-seed_3"}


def cmd_seeds(a):
    base = Path(a.base_reg) if a.base_reg else LAB / SV.SESSIONS["sphere"] / "registration.json"
    reg = json.loads(base.read_text())
    poses = {json.loads(x)["label"]: json.loads(x) for x in (LAB / "poses.jsonl").read_text().splitlines() if x.strip()}
    seatings = dict(x.split("=", 1) for x in a.seating) if a.seating else ROD_SEATINGS
    for label, folder in seatings.items():
        r = copy.deepcopy(reg)
        r["T_BI"] = json.loads((LAB / folder / "registration.json").read_text())["T_BI"]
        r["T_BE_home"] = poses[label]["T_tcp_mm"]
        r["q_home"] = poses[label]["q_rad"]
        r["notes"] = dict(r.get("notes", {}), seed=f"{a.created}, workstation: {base.parent.name}'s registration (T_EM, T_MP) "
                          f"with the rod at {folder}'s T_BI and {label}'s flange pose and joints, for the v3 suite's rod path "
                          "checks")
        out = LAB / f"{folder}-reg"
        out.mkdir(exist_ok=True)
        (out / "registration.json").write_text(json.dumps(r, indent=1) + "\n")
        print("->", (out / "registration.json").relative_to(ROOT))


def cmd_rod1(a):
    session = SV.SESSIONS["rod"]
    base = json.loads((LAB / session / "plan.json").read_text())
    p = {k: copy.deepcopy(v) for k, v in base.items() if k != "actions"}
    acts = []
    for x in base["actions"]:
        if x["stage"] == "pad" and x["kind"] == "touch":
            y = {k: v for k, v in x.items() if k != "check"}
            y.update(standoff=8.0, beyond_mm=5.0)
            acts.append(y)
        elif x["stage"] == "pivot":
            acts.append({k: v for k, v in x.items() if k != "check"})
    p["actions"] = acts
    p["travel"] = dict(p.get("travel", {}), slow_mm_touch=13.0)
    p["created"] = a.created
    other = [o for o in cell_other() if o["kind"] != "rod"]
    other.append(dict(kind="sphere", T_BI=json.loads((LAB / SV.SESSIONS["sphere"] / "registration.json").read_text())["T_BI"]))
    p["checks"]["other"] = other
    p["checks"].pop("placement_mm_deg", None)      # the rod moved 96 mm along -x_B: the old placement would stop it
    p["notes"] = list(p.get("notes", [])) + [
        f"{a.created}, workstation: v3 ID 1's rod pass 1 as a search (scripts/ur5_tip_calibration_v3_suite.py rod1): v2 ID "
        "1's 21 pad touch-offs from 8 mm above the expected contact to 5 mm past it, slow over the whole approach "
        "(travel.slow_mm_touch 13), then the 5 pivot presses; stages preflight, homing, pad, pivot, register. Home on the "
        "rod's seating at its new place (lab/poses.jsonl rod_indenter_seated_3 or the lab's own). checks.other: the low "
        "sphere (v2 ID 1's registration), the tall indenter and the holder with both held tips (tiphold); "
        "checks.placement_mm_deg is left out because the rod moved."]
    p["suite"] = dict(stage="3 rod, pass 1 (search)", tip="v3 ID 1", source_session=session,
                      script="scripts/ur5_tip_calibration_v3_suite.py")
    out = LAB / "superseded/20261027-plan_rod_lab_pass1_search_v3id1.json"
    out.write_text(json.dumps(p, indent=1) + "\n")
    print(f"rod pass 1: {len(acts)} actions ->", out.relative_to(ROOT))


def cmd_rod2(a):
    session, kept, morning, cut, bad = select("3 rod", "rod", skip_kinds=("touch",))
    kept = [(lb, x) for lb, x in kept if lb != "press/pivot"]
    morning = [(lb, x) for lb, x in morning if lb != "press/pivot"]
    h = sum(x["_dur_s"] for _, x in kept) / 3600
    print(f"rod stage 2: {len(kept)} actions, {h:.2f} h as v2 ID 1 ran them; morning: {len(morning)}; cut: {dict(cut)}")
    print("kept by family:", dict(collections.Counter(lb for lb, _ in kept)))
    note = (f"{a.created}, workstation: v3 ID 1's suite, the rod after its pass 1 (scripts/ur5_tip_calibration_v3_suite.py "
            f"rod2): the {len(kept)} actions of v2 ID 1's rod session 20261027-095303-rodv2id1 after the touch-offs and "
            "pivots that ended ok, in the order they ran, with the suite-value cuts (fast grid, rate set, fast slides, roll "
            f"and twist rates); {len(morning)} with joint 6 past +44 deg wait for the owner. About {h:.1f} h at the program's "
            "present speeds. Runs on the rod's registration from pass 1 (resume in the same session).")
    out = LAB / "superseded/20261027-plan_rod_lab_stage2_v3id1.json"
    p = write_plan(session, kept, out, a.created, note, "3 rod")
    p["checks"].pop("placement_mm_deg", None)
    p["checks"]["other"] = [o for o in p["checks"]["other"] if o["kind"] != "rod"] + [
        dict(kind="sphere", T_BI=json.loads((LAB / SV.SESSIONS["sphere"] / "registration.json").read_text())["T_BI"])]
    out.write_text(json.dumps(p, indent=1) + "\n")
    if morning:
        write_plan(session, morning, LAB / "superseded/20261027-plan_rod_lab_morning_v3id1.json", a.created,
                   note.replace("the rod after its pass 1", "the rod's morning part, joint 6 past +44 deg"), "3 rod, morning")
    print("->", out.relative_to(ROOT))


def cmd_tall1(a):
    session = SV.SESSIONS["tall"]
    base = json.loads((LAB / session / "plan.json").read_text())
    p = copy.deepcopy(base)
    p["created"] = a.created
    p["checks"]["other"] = [o for o in cell_other() if o["kind"] != "tall"] + [
        dict(kind="sphere", T_BI=json.loads((LAB / SV.SESSIONS["sphere"] / "registration.json").read_text())["T_BI"])]
    p["notes"] = list(p.get("notes", [])) + [
        f"{a.created}, workstation: v3 ID 1's pass 1 on the tall indenter (scripts/ur5_tip_calibration_v3_suite.py tall1): "
        "v2 ID 1's pass 1 there as it ran (143529-tallv2id1), its touch-offs to 2.0 mm past the expected surface. "
        "checks.other: the low sphere, the rod at its three seatings and the holder with both held tips (tiphold)."]
    p["suite"] = dict(stage="4 tall, pass 1", tip="v3 ID 1", source_session=session, script="scripts/ur5_tip_calibration_v3_suite.py")
    out = LAB / "superseded/20261027-plan_tall_lab_pass1_v3id1.json"
    out.write_text(json.dumps(p, indent=1) + "\n")
    print(f"tall pass 1: {len(p['actions'])} actions ->", out.relative_to(ROOT))


def replay_ok(plan, session, act, j6_max=J6_MAX):
    """(ok, flags, clearances) of one action's travel from 60 mm above its rest, its approach, its depth and the turns
    of a slide, roll or twist, by the holder path checker's replay (ur5_tip_calibration_holder_paths)."""
    import ur5_tip_calibration_holder_paths as HP
    HP.KIND = plan.get("indenter", "tall")
    HP.J6_WIN = (J6_MIN, j6_max)
    mv = HP.Moves(plan, session)
    reg = mv.reg
    segs, _, cP, i_own = mv.action_moves(act, np.asarray(reg.T_BE_home, float))
    T_rest = segs[i_own][1]
    T_top = mv.up(T_rest, 60.0)
    segs = [("L", T_top, T_rest)] + segs[i_own:]
    q = HP.SIM.ik(T_top, np.asarray(reg.q_home, float))
    if q is None:
        return False, ["no IK above"], {}
    m = dict(sin5=9.0, q3=999.0, shoulder=1e9, lim=999.0, j5_margin=999.0, j6_margin=999.0)
    cl = {}
    for si, seg in enumerate(segs):
        sm = HP.samples(seg, q)
        if sm is None:
            return False, [f"no IK on segment {si}"], {}
        for k, (T, qq) in enumerate(sm):
            mk = HP.metrics(qq)
            for key in m:
                m[key] = min(m[key], mk[key])
            if k % 5 == 0 or k == len(sm) - 1:
                g = HP.clear_groups(qq, T if T is not None else HP.SIM.fk(qq), reg, cP if si >= 1 else None)
                for key, v in g.items():
                    cl[key] = min(cl.get(key, 1e9), v)
        q = sm[-1][1]
    flags = [k for k, v in (("sin5", m["sin5"] < 0.5), ("q3", m["q3"] < 20), ("shoulder", m["shoulder"] < 150),
                            ("lim", m["lim"] < 10), ("j5", m["j5_margin"] < 0), ("j6", m["j6_margin"] < 0)) if v]
    flags += [f"{k} {cl[k]:.2f}" for k, need in HP.NEED.items() if cl.get(k, 1e9) < need]
    return not flags, flags, {k: round(v, 2) for k, v in cl.items()}


def cmd_tall2(a):
    import ur5_tip_calibration_holder_paths as HP
    r29 = json.loads((LAB / "superseded/20261027-plan_tall_lab_r29_v2id1.json").read_text())
    p = {k: copy.deepcopy(v) for k, v in r29.items() if k != "actions"}
    p["created"] = a.created
    p["checks"]["other"] = [o for o in cell_other() if o["kind"] != "tall"] + [
        dict(kind="sphere", T_BI=json.loads((LAB / SV.SESSIONS["sphere"] / "registration.json").read_text())["T_BI"])]
    tw_ids = sorted(x["id"] for x in r29["actions"] if x["kind"] == "twist")
    keep_tw = set(np.random.default_rng(0).permutation(tw_ids)[:4].tolist())
    acts = [x for x in r29["actions"] if x["kind"] != "twist" or x["id"] in keep_tw]
    # six shared points: sphere stage 2's grid presses on the face and the edges that the tall reaches
    s2 = json.loads((LAB / "superseded/20261027-plan_sphere_lab_stage2_v3id1.json").read_text())
    have = {x["target"]["index"] for x in acts if x["kind"] == "press"}
    cand = [x for x in s2["actions"] if x["kind"] == "press" and re.match(r"^(grid|dense)-", x["id"])
            and x["target"]["region"] in ("face", "edge") and x["target"]["index"] not in have]
    session = str(LAB / SV.SESSIONS["tall"])
    p_check = dict(p, actions=[])
    shared, tried = [], 0
    for region, n_want in (("face", 3), ("edge", 3)):
        got = 0
        seen = set()
        for x in cand:
            if got >= n_want or x["target"]["region"] != region or x["target"]["index"] in seen:
                continue
            seen.add(x["target"]["index"])
            for tw in (0.0, 10.0, 20.0, 30.0, 40.0, -10.0, -20.0):
                tried += 1
                y = {k: v for k, v in x.items() if k not in ("check",)}
                y.update(id=f"tall-shared-{x['target']['index']}", stage="grid", twist=tw, standoff=1.5,
                         max_depth_mm=x.get("max_depth_mm", 2.5))
                ok, flags, cl = replay_ok(p_check, session, y)
                if ok:
                    y["shared_with"] = x["id"]
                    shared.append(y)
                    got += 1
                    print(f"shared point {y['id']} ({region}, as {x['id']}, levels {x.get('levels')}) at twist {tw:+.0f}: {cl}")
                    break
    i_ref = next((i for i, x in enumerate(acts) if x["id"] == "tall-ref-a"), 0)
    acts = acts[:i_ref + 1] + shared + acts[i_ref + 1:]
    p["actions"] = acts
    p["notes"] = list(p.get("notes", [])) + [
        f"{a.created}, workstation: v3 ID 1's tall stage (scripts/ur5_tip_calibration_v3_suite.py tall2): R29 with 4 of its "
        f"12 twists and {len(shared)} shared points after tall-ref-a (tall presses at sphere stage 2's targets, the sphere's "
        "levels; ids tall-shared-<index>, field shared_with). checks.other: the low sphere, the rod at its three seatings "
        "and the holder with both held tips. Actions outside joint 6's unattended window are in the tall's morning plan."]
    p["suite"] = dict(stage="4 tall", tip="v3 ID 1", source_plan="lab/superseded/20261027-plan_tall_lab_r29_v2id1.json",
                      shared_points=[y["id"] for y in shared], script="scripts/ur5_tip_calibration_v3_suite.py")
    out = LAB / "superseded/20261027-plan_tall_lab_stage2_v3id1.json"
    out.write_text(json.dumps(p, indent=1) + "\n")
    print(f"tall stage: {len(acts)} actions ({len(shared)} shared points, {tried} replays) ->", out.relative_to(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("sphere2", "rod1", "rod2", "seeds", "tall1", "tall2"):
        s = sub.add_parser(name)
        s.add_argument("--created", default=time.strftime("%Y-%m-%d %H:%M"))
        if name == "seeds":                                  # a seating recorded later, on v3 ID 1's own tool frames
            s.add_argument("--base-reg", default=None, help="the registration.json whose T_EM and T_MP the seeds take "
                           "(default v2 ID 1's sphere session's)")
            s.add_argument("--seating", action="append", default=[], help="LABEL=FOLDER: only this seating, its rod frame "
                           "from lab/FOLDER (scripts/ur5_fixture_from_seating.py)")
    a = ap.parse_args()
    dict(sphere2=cmd_sphere2, rod1=cmd_rod1, rod2=cmd_rod2, seeds=cmd_seeds, tall1=cmd_tall1, tall2=cmd_tall2)[a.cmd](a)


if __name__ == "__main__":
    main()
