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
  --tr <s>     pad relaxation time (solref tc = tr / 2, d0 recomputed); default 0.02 s
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
MESH_EPS = None          # --mesh-eps: terrace outlines simplified to this deviation (m) in the tip meshes


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


def replace_tips_tcd(root, tip, model, MU, out_dir, tool=RB.OBJ, s=RB.PAD_S, rs=RB.PAD_RS, tr=RB.PAD_TR,
                     E=RB.E_TPU, region=None, tool_only=False, h="fixed", K="median"):
    """RB.replace_tips for a tip_codesign_geometry key: the shape's convex mesh ('pt') or its sphere pads ('pads',
    'padsT') on each fingertip, the distal capsule trimmed to the shape's mount face. Pad stiffness K_i = E a_i / h_i
    through solimp d0_i = 1 - 1/(tc^2 K_i diag) per pad (K='each') or the finger's median K (K='median', as RB)."""
    import mujoco
    import fingertip_geometry as G
    import numpy as np
    import tip_codesign_geometry as TG
    sh = TG.shape(tip)
    meshes = sh.mesh_paths(Path(out_dir) / "meshes" / sh.key, eps=MESH_EPS)
    asset = root.find("asset")
    for sgn, p in meshes.items():
        ET.SubElement(asset, "mesh", name=f"tpu_{'pos' if sgn > 0 else 'neg'}", file=str(p))
    pads = {}
    for f in RB.FINGERS:
        sgn = G.FACE_SIGN[f]
        tipb = root.find(f".//body[@name='{f}_tip']")
        pipb = root.find(f".//body[@name='{f}_pip_frame']")
        for g in tipb.findall("geom"):
            tipb.remove(g)
        for g in pipb.findall("geom"):
            if g.get("type") == "capsule":
                z_end = -(G.PIP_TO_TIP - sh.top_z) + G.CAPSULE_R
                g.set("fromto", f"0 0 0 0 0 {min(z_end, -1e-4):.6f}")
                g.set("name", f"{f}_distal")
        mesh = f"tpu_{'pos' if sgn > 0 else 'neg'}"
        if model == "pt":
            ET.SubElement(tipb, "geom", name=f"{f}_tipgeom", type="mesh", mesh=mesh, material="finger_mat",
                          friction=f"{MU:g} 0.005 0.0001")
            continue
        vis = ET.SubElement(tipb, "geom", name=f"{f}_tipvis", type="mesh", mesh=mesh, contype="0", conaffinity="0",
                            group="1", rgba="0.85 0.55 0.35 0.35")
        if tool_only:
            vis.set("contype", "4")
            vis.set("conaffinity", "1")
            vis.set("friction", f"{MU:g} 0.005 0.0001")
        C, Nn, A, Ks = sh.pads(E, s, rs, sgn, h=h, K=K, full=True)
        if region is not None:
            keep = region(C, Nn, sgn)
            C, Ks = C[keep], Ks[keep]
        pads[f] = (C, Ks)
        for i, c in enumerate(C):
            g = ET.SubElement(tipb, "geom", name=f"{f}_pad{i}", type="sphere", size=f"{rs}",
                              pos=f"{c[0]:.7f} {c[1]:.7f} {c[2]:.7f}", condim="3", priority="1",
                              friction=f"{MU:g} 0 0", rgba="0.75 0.4 0.25 1", mass="0")
            if tool_only:
                g.set("contype", "2")
                g.set("conaffinity", "0")
    meta = {"tip_key": sh.key}
    if pads and tool_only:
        for g in root.find(f".//body[@name='{tool}']").iter("geom"):
            if g.get("contype", "1") != "0" or g.get("conaffinity", "1") != "0":
                g.set("contype", "2")
                g.set("conaffinity", "3")
    if pads:
        m0 = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        tc = tr / 2.0
        d0s = {}
        for f, (C, Ks) in pads.items():
            diag = m0.body_invweight0[m0.body(f"{f}_tip").id, 0] + m0.body_invweight0[m0.body(tool).id, 0]
            d0 = 1.0 - 1.0 / (tc ** 2 * Ks * diag)
            if d0.min() < 0.05:
                raise ValueError(f"{f}: pad d0 {d0.min():.3f} below 0.05 at relaxation {tr} s")
            d0s[f] = [float(d0.min()), float(d0.max())]
            tipb = root.find(f".//body[@name='{f}_tip']")
            for g in tipb.findall("geom"):
                nm = g.get("name", "")
                if nm.startswith(f"{f}_pad"):
                    di = d0[int(nm[len(f) + 4:])]
                    g.set("solref", f"{tc:.6g} 1")
                    g.set("solimp", f"{di:.6g} {di:.6g} 0.001 0.5 2")
        meta.update(pad_d0=d0s, pad_K=float(np.median(np.concatenate([k for _, k in pads.values()]))),
                    n_pads={f: len(c) for f, (c, _) in pads.items()})
    return meshes, meta


def replace_tips(root, tip, model, MU, out_dir, tool=RB.OBJ, s=RB.PAD_S, rs=RB.PAD_RS, tr=RB.PAD_TR, E=RB.E_TPU,
                 region=None, tool_only=False, pf=None, force_tip=None, h="fixed", K="median"):
    """RB.replace_tips (or replace_tips_tcd for a tcd key), then the explicit pairs of --pf on pad models.
    force_tip replaces the caller's tip (the scale study's cell scenes always ask for tpu2.7)."""
    tip = force_tip or tip
    if tip.startswith("tcd_"):
        meshes, meta = replace_tips_tcd(root, tip, model, MU, out_dir, tool=tool, s=s, rs=rs, tr=tr, E=E,
                                        region=region, tool_only=tool_only, h=h, K=K)
    else:
        meshes, meta = _REPLACE(root, tip, model, MU, out_dir, tool=tool, s=s, rs=rs, tr=tr, E=E, region=region,
                                tool_only=tool_only)
    if pf and model in ("pads", "padsT"):
        meta["n_pairs"] = add_pairs(root, tool, pf, MU)
        meta["pad_pf_s"] = pf
    meta["pad_E_Pa"] = E
    return meshes, meta


def tag_of(a):
    return a.tag or (f"E{a.E:g}" + (f"_tr{a.tr:g}" if a.tr != RB.PAD_TR else "") + (f"_pf{a.pf:g}" if a.pf else "")
                     + (f"_h{a.h}" if getattr(a, "h", "fixed") != "fixed" else "")
                     + (f"_K{a.K}" if getattr(a, "K", "median") != "median" else ""))


def install(a, force_tip=None):
    """Bind RB.replace_tips to the tip options and point the scene folders at this study's."""
    global MESH_EPS
    MESH_EPS = getattr(a, "mesh_eps", None)
    RB.replace_tips = functools.partial(replace_tips, E=a.E, pf=a.pf, tr=a.tr, force_tip=force_tip,
                                        h=getattr(a, "h", "fixed"), K=getattr(a, "K", "median"))
    PUB.SCN = SCN / tag_of(a)


def _row_extra(a):
    return {"E_Pa": a.E, "pf_s": a.pf, "tr_s": a.tr, "tag": tag_of(a), "study": "20261014-tip_codesign",
            "mesh_eps_m": getattr(a, "mesh_eps", None)}


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
    HT.SCENES = PUB.SCN / f"hom_{a.tip}_ir{a.ir:g}"

    def scenes(hand, plant, mu_):                    # HT.scenes with the tip and the pads' impratio from --ir
        p_pads, meta_ = RB.build_scene(hand, a.tip, "pads", plant, "bed", a.ir, mu_, out_dir=HT.SCENES)
        p_mesh, meta_mesh_ = RB.build_scene(hand, a.tip, "pt", plant, "bed", 100.0, mu_, out_dir=HT.SCENES)
        return p_pads, p_mesh, meta_, meta_mesh_
    HT.scenes = scenes
    prm = dict(HT.DEFAULTS)
    for kv in a.set:
        k, v = kv.split("=")
        if k not in prm:
            raise SystemExit(f"unknown controller parameter {k}")
        prm[k] = float(v)
    mu = a.mu if a.mu == "scene" else float(a.mu)
    tag = tag_of(a) + (f"_ir{a.ir:g}" if a.ir != 100.0 else "")
    have = set()
    if a.out.exists():
        for line in open(a.out):
            try:
                r = json.loads(line)
                if r.get("tip", "tpu2.7") == a.tip:
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
                row = {"case_id": f"hom|{hand}|{sim}|{a.tip}|{a.plant}|{a.mu}|{tag}|{seed}", "hand": hand, "sim": sim,
                       "seed": seed, "plant": a.plant, "mu": a.mu, "tip": a.tip, "controller": "hom_turn3",
                       **_row_extra(a), "tag": tag, "impratio": a.ir, **{f"prm_{k}": v for k, v in prm.items()},
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


def cmd_scale(a):
    """hand_object_scale_sim's run on the CAD hand (SCALE_HAND=cad_v1 must be set before the call) with the tip,
    modulus and friction of the options: cell scenes under logs/20261014-tip_codesign/scale/<tip>_<tag>_mu<mu>/,
    rows in --out-dir/{hold,turn}.jsonl with tip and tip_tag added (the rows' own 'tag' is the cell's)."""
    import os
    if os.environ.get("SCALE_HAND") != "cad_v1":
        raise SystemExit("set SCALE_HAND=cad_v1")
    import hand_object_scale_sim as S
    install(a, force_tip=a.tip)
    tag = tag_of(a)
    S.SCENES = SCN.parent / "scale" / f"{a.tip}_{tag}_mu{a.mu:g}"
    S.MU = a.mu
    orig_append = S._append
    extra = {"tip": a.tip, "tip_tag": tag, "mu": a.mu, "E_Pa": a.E, "pf_s": a.pf, "study": "20261014-tip_codesign"}

    def append(path, row):
        row.update(extra)
        orig_append(path, row)
    S._append = append
    ns = argparse.Namespace(cells=a.cells, what=a.what, model="pads", seeds=a.seeds, set=a.set, shard="0/1",
                            plant=a.plant, out_dir=str(a.out_dir), only=a.only, traces=False)
    return S.cmd_run(ns)


ASSETS = ROOT / "assets/mjcf/experimental/20261014-tip_codesign"


def _portable(xml_path, out_xml):
    """Copy a scene with its absolute-path meshes into out_xml's folder (meshes/<file>), mesh paths made relative."""
    import shutil
    root = ET.parse(xml_path).getroot()
    (out_xml.parent / "meshes").mkdir(parents=True, exist_ok=True)
    for m in root.iter("mesh"):
        f = m.get("file")
        if f and Path(f).is_absolute():
            dst = out_xml.parent / "meshes" / Path(f).name
            shutil.copyfile(f, dst)
            m.set("file", f"meshes/{dst.name}")
    ET.indent(root, "  ")
    out_xml.write_text(ET.tostring(root, encoding="unicode") + "\n")


def cmd_assets(a):
    """Ledger step 21: pad scenes of each tip under assets/mjcf/experimental/20261014-tip_codesign/<tip>/: the D6 bench
    replay scene on the servo-limited plant at mu 1.0 (D6_<tip>_pads_servo.xml) and the CAD hand
    (build_cad_v1_finger.py --out <tip>/cad_v1 with the tips forced to <tip>: finger.xml, hand.xml,
    scene_screwdriver_medium.xml). Run in .venv."""
    import build_cad_v1_finger as BC
    for tip in a.tips:
        out = ASSETS / tip
        out.mkdir(parents=True, exist_ok=True)
        ns = argparse.Namespace(E=a.E, pf=None, tr=a.tr, tag=None, h=a.h, K=a.K)
        install(ns)
        PUB.SCN = SCN.parent / "assets_build" / tag_of(ns)
        path, meta = PUB.build("D6", tip, "pads", SERVO, "bed", 100.0, "1.0")
        _portable(Path(path), out / f"D6_{tip}_pads_servo.xml")
        install(ns, force_tip=tip)
        argv = sys.argv
        sys.argv = ["build_cad_v1_finger.py", "--out", str(out / "cad_v1")]
        try:
            BC.main()
        finally:
            sys.argv = argv
        (out / "README.json").write_text(json.dumps({
            "tip": tip, "E_Pa": a.E, "pad_builder": {"h": a.h, "K": a.K}, "pad_spacing_m": RB.PAD_S, "pad_radius_m": RB.PAD_RS,
            "D6_scene": f"D6_{tip}_pads_servo.xml", "plant": SERVO, "mu": 1.0, "cad_scene": "cad_v1/scene_screwdriver_medium.xml",
            "n_pads": meta.get("n_pads"), "pad_d0": meta.get("pad_d0"),
            "rl": f"PY scripts/tip_codesign_rig.py rlscene --tip {tip} --src <run dir with frozen_scene.xml> --out-dir <new run dir>",
            "script": SCRIPT, "when": time.strftime("%Y-%m-%d %H:%M")}, indent=1))
        print(f"{tip}: {out.relative_to(ROOT)} (D6 scene, cad_v1/)", flush=True)


def cmd_icem(a):
    """Ledger step 22: hand_object_scale_planners.py run (default config icem_pin, pinned iCEM) on the CAD hand with the tip
    forced to --tip: cell scenes of the stage runs (logs/20261014-tip_codesign/scale/<tip>_<tag>_mu<mu>), each cell's grasp the
    one with the best median held HOM turn in --hom-rows (the tip's own rows); rows carry variant = the tip. Run in .venv with
    SCALE_HAND=cad_v1."""
    import os
    if os.environ.get("SCALE_HAND") != "cad_v1":
        raise SystemExit("set SCALE_HAND=cad_v1")
    import hand_object_scale_planners as HP
    import hand_object_scale_sim as S
    ns = argparse.Namespace(E=a.E, pf=None, tr=a.tr, tag=None, h=a.h, K=a.K)
    install(ns, force_tip=a.tip)
    S.SCENES = SCN.parent / "scale" / f"{a.tip}_{tag_of(ns)}_mu{a.mu:g}"
    S.MU = a.mu
    argv = sys.argv
    sys.argv = (["hand_object_scale_planners.py", "run", "--configs", *a.configs, "--cells", *a.cells, "--seeds",
                 *map(str, a.seeds), "--nthread", str(a.nthread), "--out", str(a.out), "--hom-rows", str(a.hom_rows),
                 "--variant", a.tip])
    try:
        return HP.main()
    finally:
        sys.argv = argv


def cmd_rlscene(a):
    """make_pad_morphology_run.py with the dispatcher installed, so --tip may be a tcd key: the source run's
    frozen_scene.xml with the tip's 1 mm pads, written into a new run directory for the RL queue."""
    import make_pad_morphology_run as MPR
    install(argparse.Namespace(E=a.E, pf=None, tr=a.tr, tag=None, h=a.h, K=a.K))
    argv = sys.argv
    sys.argv = ["make_pad_morphology_run.py", "--src", str(a.src), "--out", str(a.out_dir), "--tip", a.tip, "--E", str(a.E)]
    try:
        return MPR.main()
    finally:
        sys.argv = argv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("replay", "hom"):
        p = sub.add_parser(name)
        p.add_argument("--hands", nargs="+", default=list(RB.HANDS))
        p.add_argument("--seeds", nargs="+", type=int, default=[0])
        p.add_argument("--E", type=float, default=RB.E_TPU)
        p.add_argument("--pf", type=float, default=None)
        p.add_argument("--tr", type=float, default=RB.PAD_TR, help="pad relaxation time 2 tc, s")
        p.add_argument("--tag", default=None, help="scene folder and row tag (default from E and pf)")
        p.add_argument("--h", default="fixed", choices=["fixed", "local"], help="tcd pads' foundation depth")
        p.add_argument("--K", default="median", choices=["median", "each"], help="tcd pads' stiffness")
        p.add_argument("--mesh-eps", type=float, default=None,
                       help="tcd meshes with each terrace outline simplified (Douglas-Peucker) to this deviation, m: "
                            "Drake rejects the near-zero facets of collinear outline points")
        p.add_argument("--out", type=Path, required=True)
    sc = sub.add_parser("scale", help="the scale study's hold and HOM turn on CAD-hand cells with the chosen tip")
    sc.add_argument("--tip", required=True)
    sc.add_argument("--E", type=float, default=RB.E_TPU)
    sc.add_argument("--pf", type=float, default=None)
    sc.add_argument("--tr", type=float, default=RB.PAD_TR)
    sc.add_argument("--h", default="fixed", choices=["fixed", "local"])
    sc.add_argument("--K", default="median", choices=["median", "each"])
    sc.add_argument("--tag", default=None)
    sc.add_argument("--mu", type=float, default=1.0)
    sc.add_argument("--cells", default="diag")
    sc.add_argument("--only", nargs="*", default=[])
    sc.add_argument("--what", default="hold,turn")
    sc.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    sc.add_argument("--set", nargs="*", default=[])
    sc.add_argument("--plant", default=SERVO)
    sc.add_argument("--out-dir", type=Path, required=True)
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
    h.add_argument("--tip", default="tpu2.7")
    h.add_argument("--sims", nargs="+", default=["mujoco"])
    h.add_argument("--plant", default=SERVO)
    h.add_argument("--mu", default="1.0")
    h.add_argument("--set", nargs="*", default=[])
    h.add_argument("--ir", type=float, default=100.0, help="impratio of the pads scene")
    ic = sub.add_parser("icem")
    ic.add_argument("--tip", required=True)
    ic.add_argument("--configs", nargs="+", default=["icem_pin"])
    ic.add_argument("--cells", nargs="+", default=["cylinder:16:0.625", "cylinder:25:0.125"])
    ic.add_argument("--seeds", nargs="+", type=int, default=[0])
    ic.add_argument("--mu", type=float, default=1.0)
    ic.add_argument("--nthread", type=int, default=6)
    ic.add_argument("--hom-rows", type=Path, required=True)
    ic.add_argument("--out", type=Path, required=True)
    for opt, kw in (("--E", dict(type=float, default=RB.E_TPU)), ("--tr", dict(type=float, default=RB.PAD_TR)),
                    ("--h", dict(default="fixed")), ("--K", dict(default="median"))):
        ic.add_argument(opt, **kw)
    for name in ("assets", "rlscene"):
        q = sub.add_parser(name)
        q.add_argument("--E", type=float, default=RB.E_TPU)
        q.add_argument("--tr", type=float, default=RB.PAD_TR)
        q.add_argument("--h", default="fixed", choices=["fixed", "local"])
        q.add_argument("--K", default="median", choices=["median", "each"])
    sub.choices["assets"].add_argument("--tips", nargs="+", required=True)
    rl = sub.choices["rlscene"]
    rl.add_argument("--tip", required=True)
    rl.add_argument("--src", type=Path, required=True, help="run directory with frozen_scene.xml")
    rl.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    {"replay": cmd_replay, "hom": cmd_hom, "scale": cmd_scale, "assets": cmd_assets, "rlscene": cmd_rlscene,
     "icem": cmd_icem}[a.cmd](a)


if __name__ == "__main__":
    main()
