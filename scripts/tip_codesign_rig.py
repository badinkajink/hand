#!/usr/bin/env python3
"""Hand-level runs of the fingertip co-design study (docs/notes/20261014-tip_codesign_plan.md): plan replays of the
deployed hands D1-D8 and HOM closed-loop turns with a chosen pad modulus, friction and pad-model variant. It wraps
pad_usability_backends (PUB), reorient_backends (RB) and hom_turn3 (HT) without editing them.

  replay   PUB.cmd_run with RB.replace_tips bound to the tip options; scenes under
           logs/20261014-tip_codesign/scenes/<tag>/, rows with E_Pa, pf_s and tag added
  hom      PUB.cmd_hom's loop with HT's scenes built under the same options, resumable by
           (hand, sim, plant, mu, tag, seed); the controller's kp_model and other parameters through --set

Tip options:
  --E <Pa>     pad modulus (stiffness E/h A_s per pad, h 8.5 mm); default 10 MPa
  --pf <s>     the pad-tool contacts as explicit <pair> elements with solreffriction (<s>, 1): the friction rows'
               damping 2 / (dmax s) in place of the normal rows' 2 / (dmax tc), which removes the normal force that
               MuJoCo's elliptic cone adds to a sliding pad (tip_codesign_bed.py); body excludes keep the
               tip-tool dynamic contacts out

    PY=logs/20261001-hom_contact/venv/bin/python
    $PY scripts/tip_codesign_rig.py replay --hands D7 --mu 1.0 --seeds 0 --out <rows.jsonl>
    $PY scripts/tip_codesign_rig.py hom --hands D7 --seeds 0 --mu 1.0 --set kp_model=0.72 --out <rows.jsonl>
"""
from __future__ import annotations

import argparse
import functools
import gc
import json
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pad_usability_backends as PUB  # noqa: E402
import reorient_backends as RB  # noqa: E402

SCRIPT = "scripts/tip_codesign_rig.py"
SCN = ROOT / "logs/20261014-tip_codesign/scenes"
SERVO = PUB.SERVO
_REPLACE = RB.replace_tips


def add_pairs(root, tool: str, pf: float, MU: float):
    """Explicit pad-tool pairs with solreffriction (pf, 1); excludes for the tip-tool body pairs. Returns the count."""
    tb = root.find(f".//body[@name='{tool}']")
    tgeoms = []
    for k, g in enumerate(tb.findall("geom")):
        if g.get("contype", "1") == "0" and g.get("conaffinity", "1") == "0":
            continue
        if not g.get("name"):
            g.set("name", f"{tool}_g{k}")
        tgeoms.append(g.get("name"))
    con = root.find("contact")
    if con is None:
        con = ET.SubElement(root, "contact")
    n = 0
    for f in RB.FINGERS:
        ET.SubElement(con, "exclude", body1=f"{f}_tip", body2=tool)
        for g in root.find(f".//body[@name='{f}_tip']").findall("geom"):
            if not g.get("name", "").startswith(f"{f}_pad"):
                continue
            for tg in tgeoms:
                ET.SubElement(con, "pair", geom1=g.get("name"), geom2=tg, condim="3",
                              friction=f"{MU:g} {MU:g} 0.005 0.0001 0.0001", solref=g.get("solref"),
                              solreffriction=f"{pf:g} 1", solimp=g.get("solimp"))
                n += 1
    return n


def replace_tips(root, tip, model, MU, out_dir, tool=RB.OBJ, s=RB.PAD_S, rs=RB.PAD_RS, tr=RB.PAD_TR, E=RB.E_TPU,
                 region=None, tool_only=False, pf=None):
    """RB.replace_tips, then the explicit pairs of --pf on pad models."""
    meshes, meta = _REPLACE(root, tip, model, MU, out_dir, tool=tool, s=s, rs=rs, tr=tr, E=E, region=region,
                            tool_only=tool_only)
    if pf and model in ("pads", "padsT"):
        meta["n_pairs"] = add_pairs(root, tool, pf, MU)
        meta["pad_pf_s"] = pf
    meta["pad_E_Pa"] = E
    return meshes, meta


def tag_of(a):
    return a.tag or (f"E{a.E:g}" + (f"_pf{a.pf:g}" if a.pf else ""))


def install(a):
    """Bind RB.replace_tips to the tip options and point the scene folders at this study's."""
    RB.replace_tips = functools.partial(replace_tips, E=a.E, pf=a.pf)
    PUB.SCN = SCN / tag_of(a)


def _row_extra(a):
    return {"E_Pa": a.E, "pf_s": a.pf, "tag": tag_of(a), "study": "20261014-tip_codesign"}


def cmd_replay(a):
    """PUB.cmd_run; the resume check counts only rows of this tag, so conditions can share an output file."""
    install(a)
    extra = _row_extra(a)
    orig, orig_done = PUB.append_row, RB.done

    def append(path, row):
        row.update(extra)
        row["script_wrapper"] = SCRIPT
        orig(path, row)

    def done(path):
        have = set()
        if Path(path).exists():
            for line in open(path):
                try:
                    r = json.loads(line)
                    if r.get("tag") == extra["tag"]:
                        have.add(r["case_id"])
                except Exception:
                    pass
        return have
    PUB.append_row, RB.done = append, done
    try:
        PUB.cmd_run(a)
    finally:
        PUB.append_row, RB.done = orig, orig_done


def cmd_hom(a):
    """PUB.cmd_hom's loop; the resume key carries mu and the tag, so conditions can share an output file."""
    import hom_turn3 as HT
    install(a)
    HT.SCENES = PUB.SCN / "hom"
    prm = dict(HT.DEFAULTS)
    for kv in a.set:
        k, v = kv.split("=")
        if k not in prm:
            raise SystemExit(f"unknown controller parameter {k}")
        prm[k] = float(v)
    mu = a.mu if a.mu == "scene" else float(a.mu)
    tag = tag_of(a)
    have = set()
    if a.out.exists():
        for line in open(a.out):
            try:
                r = json.loads(line)
                have.add((r["hand"], r["sim"], r["plant"], str(r["mu"]), r.get("tag"), int(r["seed"])))
            except Exception:
                pass
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for hand in a.hands:
        todo = {sim: [s for s in a.seeds if (hand, sim, a.plant, str(a.mu), tag, s) not in have] for sim in a.sims}
        if not any(todo.values()):
            continue
        p_pads, p_mesh, meta, meta_mesh = HT.scenes(hand, a.plant, mu)
        mirror = HT.Mirror(p_mesh)
        for sim in a.sims:
            if not todo[sim]:
                continue
            plant = HT.MjPlant(p_pads, meta) if sim == "mujoco" else HT.DrakePlant(p_mesh, meta_mesh)
            for seed in todo[sim]:
                row = {"case_id": f"hom|{hand}|{sim}|{a.plant}|{a.mu}|{tag}|{seed}", "hand": hand, "sim": sim,
                       "seed": seed, "plant": a.plant, "mu": a.mu, "tip": "tpu2.7", "controller": "hom_turn3",
                       **_row_extra(a), **{f"prm_{k}": v for k, v in prm.items()},
                       "when": time.strftime("%Y-%m-%d %H:%M")}
                t0 = time.time()
                try:
                    res, trace = HT.rollout(plant, mirror, meta, seed, prm)
                    row.update(res, status="ok")
                    row["trace_coarse"] = [[s["t"], s["th_deg"], s["z_mm"], round(sum(s["F"].values()), 3),
                                            sum(1 for f in RB.FINGERS if s["n"][f] > 0)] for s in trace[::5]]
                except Exception as e:
                    row.update(status="error", error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-1500:])
                row.update(script=SCRIPT, git_rev=RB.git_rev(), wall_total_s=round(time.time() - t0, 2),
                           n_pads=meta.get("n_pads"), pad_d0=meta.get("pad_d0"))
                PUB.append_row(a.out, row)
                print(f"{hand} {sim:<6} {tag} mu {a.mu} seed {seed}: " + (
                    f"turn {row['turn_end_deg']:+.1f} deg, frozen {row['frozen_deg']}, held {row['held']}, "
                    f"fingers {row['fingers_end']}, F {row['F_end']}, {row['wall_s']} s wall"
                    if row["status"] == "ok" else row["error"]), flush=True)
            del plant
            gc.collect()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("replay", "hom"):
        p = sub.add_parser(name)
        p.add_argument("--hands", nargs="+", default=list(RB.HANDS))
        p.add_argument("--seeds", nargs="+", type=int, default=[0])
        p.add_argument("--E", type=float, default=RB.E_TPU)
        p.add_argument("--pf", type=float, default=None)
        p.add_argument("--tag", default=None, help="scene folder and row tag (default from E and pf)")
        p.add_argument("--out", type=Path, required=True)
    r = sub.choices["replay"]
    r.add_argument("--sim", default="mujoco", choices=["mujoco", "drake"])
    r.add_argument("--tips", nargs="+", default=["tpu2.7"])
    r.add_argument("--models", nargs="+", default=["pads"], choices=PUB.MODELS)
    r.add_argument("--plants", nargs="+", default=[SERVO])
    r.add_argument("--numerics", nargs="+", default=["bed"], choices=["bed", "rl"])
    r.add_argument("--ir", nargs="+", type=float, default=[100.0])
    r.add_argument("--mu", nargs="+", default=["1.0"])
    r.add_argument("--dt", nargs="+", type=float, default=None)
    r.add_argument("--spin", type=float, default=None)
    r.add_argument("--jitter", default="j2", choices=["j2", "j0916"])
    r.add_argument("--traces", action="store_true")
    h = sub.choices["hom"]
    h.add_argument("--sims", nargs="+", default=["mujoco"])
    h.add_argument("--plant", default=SERVO)
    h.add_argument("--mu", default="1.0")
    h.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    {"replay": cmd_replay, "hom": cmd_hom}[a.cmd](a)


if __name__ == "__main__":
    main()
