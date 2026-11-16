"""The replay's sphere scene with the real object (2026-11-16, sim-real reversal study): the 40 mm ball (21.0 g) on its
stem (4 mm across, 36.9 mm from the ball's bottom to the block's top, 0.6 g) and the hex tag block (17 mm tall, ~30 mm
across flats, 6.5 g; a cylinder of radius --block-r), 28.1 g in all as weighed; the support post replaced by a table
under the block (its top at the block's bottom when the object sits at the scene's start), removed at the lift as the
post is. Writes the scene and runs rev_sweep's replays on it.
    .venv/bin/python rev_stem.py OUT.json [--block-r 15] [--stem-r 2] 'mu=1.0' 'mu=0.6' ...
"""
import sys, re, json, argparse
from pathlib import Path
ROOT = Path.home() / "Programs/hand"
D = ROOT / "logs/20261116-sim_real_reversal"
sys.path.insert(0, str(D)); sys.path.insert(0, str(ROOT / "scripts/hardware")); sys.path.insert(0, str(ROOT / "scripts"))

def build(block_r=0.015, stem_r=0.002, ball_g=21.0, stem_g=0.6, block_g=6.5, no_table=False, tag=""):
    import sr2_hom_replay as RP
    src, _ = RP.ref_scene("sphere", 40.0)
    x = Path(src).read_text()
    old_geom = '<geom type="sphere" size="0.020000" mass="0.025" material="object_mat" friction="1 0.005 0.0001" />'
    assert x.count(old_geom) == 1, "object geom"
    r, stem, bh = 0.020, 0.03689, 0.017
    y_s0, y_s1 = -r, -(r + stem)                       # the stem from the ball's bottom to the block's top (body y up)
    y_b = -(r + stem + bh / 2)
    new = (f'<geom type="sphere" size="0.020000" mass="{ball_g/1e3:.5f}" material="object_mat" friction="1 0.005 0.0001" />'
           f'<geom name="obj_stem" type="cylinder" fromto="0 {y_s0 + 0.001:.5f} 0 0 {y_s1:.5f} 0" size="{stem_r:.4f}" '
           f'mass="{stem_g/1e3:.5f}" material="object_mat" friction="1 0.005 0.0001" />'
           f'<geom name="obj_block" type="cylinder" fromto="0 {y_s1:.5f} 0 0 {y_s1 - bh:.5f} 0" size="{block_r:.4f}" '
           f'mass="{block_g/1e3:.5f}" material="object_mat" friction="1 0.005 0.0001" />')
    x = x.replace(old_geom, new)
    old_post = re.search(r'<body name="post" pos="[^"]*"><geom name="post"[^/]*/></body>', x)
    assert old_post, "post"
    z_top = 0.12 - (r + stem + bh)                     # the block's bottom at the scene's start pose
    if no_table:
        post = '<body name="post" pos="0 0 -1"><geom name="post" type="box" size="0.01 0.01 0.01" /></body>'
    else:
        post = (f'<body name="post" pos="0 0 {z_top - 0.004:.5f}"><geom name="post" type="box" size="0.06 0.06 0.004" '
                f'rgba="0.4 0.4 0.45 1" friction="1 0.005 0.0001" /></body>')
    x = x[:old_post.start()] + post + x[old_post.end():]
    out = D / "scenes" / f"sph040_stem_block{tag}.xml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(x)
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("sets", nargs="*")
    ap.add_argument("--block-r", type=float, default=0.015); ap.add_argument("--stem-r", type=float, default=0.002)
    ap.add_argument("--block-g", type=float, default=6.5); ap.add_argument("--ball-g", type=float, default=21.0)
    ap.add_argument("--no-table", action="store_true"); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    path = build(a.block_r, a.stem_r, a.ball_g, 0.6, a.block_g, a.no_table, a.tag)
    print("scene", path, flush=True)
    import rev_sweep as RS
    import sr2_hom_replay as RP
    RP.ref_scene = lambda shape, d: (path, path)       # every Replayer in the pool loads the stem-and-block scene
    sys.argv = [sys.argv[0], a.out] + (a.sets or [""])
    from multiprocessing import Pool
    sets = [RS.parse(s) for s in (a.sets or [""])]
    jobs = [(p.name, prm) for prm in sets for p in RS.TRIALS]
    def init():
        RP.ref_scene = lambda shape, d: (path, path)
        RS._init()
    with Pool(min(4, len(jobs)), initializer=init) as pool:
        rows = pool.map(RS.job, jobs)
    Path(a.out).write_text(json.dumps(rows, indent=1))
    for prm in sets:
        print(json.dumps(prm))
        for r in [r for r in rows if r["params"] == prm]:
            if r["diverged"]:
                print(f"   {r['trial'][7:-6]}: DIVERGED"); continue
            s, q = r["sim"], r["real"]
            print(f"   {r['trial'][7:-6]:22s} sim rise {s['rise_max']:5.1f} twist {s['twist_end']:6.1f} tilt {s['tilt_end']:5.1f}"
                  + (" DROP" if s["dropped"] else "     ") + f" | real rise {q['rise_max']:5.1f} twist {q['twist_end']:6.1f} "
                  f"tilt {q['tilt_end']:5.1f} | q {r['joint_rms_deg']:.1f} p {r['pos_rms_mm']:.1f} r {r['rot_rms_deg']:.1f}", flush=True)
