#!/usr/bin/env python3
"""Result page of 2026-11-16: HOM's sphere twist on the real SR2 hand against simulation. The film grasps, whose
contacts sit 18-20 deg below the sphere's equator, scored best in simulation and worst on the hand; the ground-clear
grasps, 2-5 deg below, the reverse (owner, 13:25: "we should figure out why there is a massive discrepancy between sim
and real on these ground-clear vs low-equator trajectories").

    .venv/bin/python scripts/hom/hom_twist_sim_real_page.py        (DOCS_REFRESH=0 on corvus: no docs_index there)

Data (docs/experiments/20261115-hom_bench_trials/data/20261116-sweep/): 20261116-all_twists_table.json (lab/sweep_table.py:
the 15 real twists, corvus and HOM server records merged and scored by sr2hand.control.trial_score), sim/closed_*.json
(closed-loop HOM in simulation with the real object, the workstation's logs/20261116-sim_real_reversal/rev_closed.py),
sim/lat_g10.jsonl (the latitude series in simulation, rev_lat.py), sim/rescore_grids_talk2_summary.json (both grids at
the two talk layouts, rev_rescore.py). Media: media/20261116-low_vs_ground_clear_frames.png, the films
media/20261116-real_hand_twist_ground_clear.mp4 and media/20261116-hom_afternoon_timelapse.mp4.
"""
from __future__ import annotations

import base64
import html
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "docs/experiments/20261115-hom_bench_trials"
DATA, MEDIA, SIM = D / "data/20261116-sweep", D / "media", D / "data/20261116-sweep/sim"
PAGE = D / "20261116-hom_twist_sim_real_reversal.html"
TEMPLATE = ROOT / "scripts/ur5/ur5_tip_calibration_page.template.html"
GR = {"film_g10_x105_y70": 18.2, "film_g11_x70_y110": 20.0, "gr_g13_x105_y70": 2.4, "gr_g0_x70_y110": 5.4}
NAMES = {"film_g10_x105_y70": "film g10, (105, 70)", "film_g11_x70_y110": "film g11, (70, 110)",
         "gr_g13_x105_y70": "ground-clear g13, (105, 70)", "gr_g0_x70_y110": "ground-clear g0, (70, 110)"}
FAMILY = {"gen": "generalist S3 (4.0 N)", "g10": "film g10", "g11": "film g11", "g0gr": "ground-clear g0",
          "g13gr": "ground-clear g13", "lat4": "g10 raised to 4°", "lat8": "g10 raised to 8°",
          "lat12": "g10 raised to 12°"}


def j(p):
    return json.loads(Path(p).read_text())


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def agg(rows):
    """Mean tracking, J, rise and tilt over the seeds that ran, with held and failed counts."""
    ok = [r for r in rows if "error" not in r]
    for r in ok:                                   # the first runs' files carry err/ref and tilt_end only
        if "track" not in r:
            r["track"] = max(0.0, 1.0 - r["err"] / r["ref"]) if r.get("ref") else 0.0
        r.setdefault("tilt_max", r.get("tilt_end", float("nan")))
    m = lambda k: float(np.mean([r[k] for r in ok])) if ok else float("nan")  # noqa: E731
    lo = lambda k: float(np.min([r[k] for r in ok])) if ok else float("nan")  # noqa: E731
    hi = lambda k: float(np.max([r[k] for r in ok])) if ok else float("nan")  # noqa: E731
    return {"track": m("track"), "J": m("J"), "rise": m("rise_max"), "tilt": m("tilt_max"),
            "track_lo": lo("track"), "track_hi": hi("track"), "rise_lo": lo("rise_max"), "rise_hi": hi("rise_max"),
            "held": sum(bool(r.get("held")) for r in ok), "n": len(rows), "failed": len(rows) - len(ok)}


def closed(name):
    rows = j(SIM / name)
    return {g: agg([r for r in rows if r["grasp"] == g]) for g in GR}


def lat_series():
    """{model: {lat: agg}} of the simulated g10 latitude series."""
    out = {}
    for r in jl(SIM / "lat_g10.jsonl"):
        p = r.get("params", {})
        key = f"μ {p.get('mu', 1.0):g}" + (", soft tangential contact" if "imp1" in r.get("scene", "") else "")
        out.setdefault(key, {}).setdefault(float(r["grasp"].split("lat")[1]), []).append(r)
    return {k: {lat: agg(v) for lat, v in sorted(d.items())} for k, d in out.items()}


def real():
    rows = j(DATA / "20261116-all_twists_table.json")
    for r in rows:
        r["family"] = r["label"].split("_")[0]
        r["lat"] = -float(r["latitude_deg"])
    return rows


# ------------------------------------------------------------------ figure
def fig_latitude(R, LS, C):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 14, "font.family": "serif"})
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.6))
    col = {"gen": "0.55", "g10": "tab:red", "g11": "tab:orange", "lat4": "tab:purple", "lat8": "tab:purple",
           "lat12": "tab:purple", "g0gr": "tab:green", "g13gr": "tab:green"}
    mk = {"gen": "s", "g10": "o", "g11": "o", "lat4": "D", "lat8": "D", "lat12": "D", "g0gr": "^", "g13gr": "^"}
    seen = set()
    for r in R:
        f = r["family"]
        lab = None if f in seen else FAMILY[f] if not f.startswith("lat") else "g10 raised (4, 8, 12°)"
        seen.add(f if not f.startswith("lat") else "lat")
        if f.startswith("lat") and "lat_l" in seen:
            lab = None
        if f.startswith("lat"):
            seen.add("lat_l")
        tr = r.get("track_ratio")
        axs[0].plot(r["lat"], np.nan if tr is None else tr, mk[f], color=col[f], ms=10, label=lab, alpha=0.9)
        axs[1].plot(r["lat"], r["rise_max_mm"], mk[f], color=col[f], ms=10, alpha=0.9)
    sty = {"μ 1": ("k", -0.45), "μ 0.5": ("tab:blue", 0.0), "μ 1, soft tangential contact": ("tab:brown", 0.45)}
    for key, d in LS.items():
        c, dx = sty[key]
        for k, (lat, a) in enumerate(sorted(d.items())):
            for ax, f, lo, hi in ((axs[0], "track", "track_lo", "track_hi"), (axs[1], "rise", "rise_lo", "rise_hi")):
                if np.isfinite(a[f]):
                    ax.errorbar(lat + dx, a[f], yerr=[[a[f] - a[lo]], [a[hi] - a[f]]], fmt="s", color=c, ms=7,
                                mfc="white", mew=2, capsize=3, label=(f"simulation, {key}" if k == 0 and ax is axs[0]
                                                                       else None))
                if a["held"] < a["n"] - a["failed"] or a["failed"]:
                    ax.annotate(f"held {a['held']}/{a['n']}", (lat + dx, a[f] if np.isfinite(a[f]) else 0),
                                xytext=(4, -14), textcoords="offset points", fontsize=9, color=c)
    for g, lat in GR.items():
        if g.startswith("gr"):
            for name, c in (("closed_stem_mu-1.0.json", "k"), ("closed_stem2_mu-0.5_kp-0.55_damping-0.011.json", "tab:blue")):
                a = C[name][g]
                axs[0].plot(lat, a["track"], "x", color=c, ms=11, mew=2.5)
                axs[1].plot(lat, a["rise"], "x", color=c, ms=11, mew=2.5)
    axs[0].plot([], [], "x", color="0.3", ms=11, mew=2.5, label="simulation, ground-clear grasps")
    axs[0].set_ylabel("tracking, 1 − err/ref")
    axs[1].set_ylabel("rise into the hand (mm)")
    for ax in axs:
        ax.set_xlabel("contacts below the sphere's equator (°)")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 22)
    axs[0].set_ylim(-0.03, 0.9)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=11, frameon=False)
    axs[0].set_title("Tracking of the commanded twist")
    axs[1].set_title("The sphere's largest rise toward the palm")
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    out = MEDIA / "20261116-twist_latitude_real_sim.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


# ------------------------------------------------------------------ page helpers
def esc(s):
    return html.escape(str(s), quote=True)


def img_uri(p, width=1150, quality=86):
    from PIL import Image
    im = Image.open(p).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def fig(n, p, caption, width=1150):
    return (f'<figure id="fig{n}"><img src="{img_uri(p, width)}" alt="{esc(caption.split(".")[0])}">'
            f"<figcaption>Figure {n}. {caption}</figcaption></figure>")


def film(n, rel, caption):
    return (f'<figure id="fig{n}"><video controls muted loop playsinline preload="metadata" src="{rel}" '
            f'style="width:100%"></video><figcaption>Figure {n}. {caption} (<a href="{rel}">{rel}</a>)</figcaption>'
            f"</figure>")


def table(head, body, num=(), caption=None, n=None):
    cap = f"<caption>Table {n}. {caption}</caption>" if caption else ""
    th = "".join(f'<th{" class=n" if k in num else ""}>{h}</th>' for k, h in enumerate(head))
    tr = "".join("<tr>" + "".join(f'<td{" class=n" if k in num else ""}>{c}</td>' for k, c in enumerate(r)) + "</tr>"
                 for r in body)
    return f'<div class="tw"><table>{cap}<thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def f1(x, nd=1):
    return "&ndash;" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


# ------------------------------------------------------------------ page
def page():
    sys.path.insert(0, str(ROOT / "scripts/docs"))
    import docs_index
    import retro_style
    R = real()
    LS = lat_series()
    names = ["closed_default_s012.json", "closed_stem_mu-1.0.json", "closed_stem_mu-0.6.json", "closed_stem_mu-0.5.json",
             "closed_stem2_mu-0.5_kp-0.55_damping-0.011.json", "closed_bare_mu-0.5.json"]
    C = {n: closed(n) for n in names}
    RS = j(SIM / "rescore_grids_talk2_summary.json") if (SIM / "rescore_grids_talk2_summary.json").exists() else None
    fig1 = fig_latitude(R, LS, C)

    near = [r for r in R if r["lat"] <= 8.5]
    low = [r for r in R if r["lat"] >= 18 and r["family"] in ("g10", "g11")]
    tr = lambda rs: [r["track_ratio"] for r in rs if r.get("track_ratio") is not None]  # noqa: E731
    rise = lambda rs: [r["rise_max_mm"] for r in rs]  # noqa: E731
    l12 = next(r for r in R if r["family"] == "lat12")
    s05 = C["closed_stem_mu-0.5.json"]
    s10 = C["closed_stem_mu-1.0.json"]

    lede = (f"On the real SR2 hand, HOM twisted the 40 mm sphere best when the fingertips held it near its equator, the "
            f"reverse of simulation's ranking. In {len(near)} trials with the contacts 2 to 8° below the equator, the "
            f"sphere followed the commanded twist with a tracking of {min(tr(near)):.2f} to {max(tr(near)):.2f} and rose "
            f"at most {max(rise(near)):.1f} mm toward the palm. In {len(low)} trials with the film grasps, 18 to 20° below, "
            f"tracking fell to {min(tr(low)):.2f} to {max(tr(low)):.2f}, the sphere climbed {min(rise(low)):.1f} to "
            f"{max(rise(low)):.1f} mm, and the whole object tipped. Simulation had scored the film grasps 0.64 to 0.76 and "
            f"the ground-clear ones 0.15 to 0.17. The simulation was rebuilt with the real object, its stem and 6.5 g block, and a "
            f"friction of 0.5 in place of the scenes' 1.0. Then simulated HOM fails on the film grasps and holds the "
            f"ground-clear ones, as the hand did. The latitude sweep shows that model is still incomplete.")
    summary = (f"Real SR2 twists: contacts 2-8° below the sphere's equator tracked {min(tr(near)):.2f}-{max(tr(near)):.2f} "
               f"with ≤{max(rise(near)):.1f} mm rise; the film grasps (18-20° below) {min(tr(low)):.2f}-{max(tr(low)):.2f} "
               f"with {min(rise(low)):.0f}-{max(rise(low)):.0f} mm climb, the reverse of simulation (0.64-0.76 vs 0.15-0.17). "
               f"Simulation with the real object at friction 0.5 (or soft tangential contact at 1.0) reproduces the reversal.")

    S = []
    S.append(("terms", "Terms", """
<dl class="terms">
<dt>Contact latitude</dt><dd>How far below the sphere's equator the plan puts the fingertip pads, in degrees, seen from
the sphere's centre. All three pads share it in these grasps.</dd>
<dt>Film grasps</dt><dd>The grasps of the talk's abstract film: grid2dff grasp 10 at the (105, 70) mm layout and grasp
11 at (70, 110). Their contacts sit 18.2 and 20.0° below the equator.</dd>
<dt>Ground-clear grasps</dt><dd>grid2dgr grasp 13 at (105, 70) and grasp 0 at (70, 110), from the grid whose grasps clear
a floor at the sphere's lowest point. Their contacts sit 2.4 and 5.4° below the equator.</dd>
<dt>Raised variants</dt><dd>Grasp 10 at (105, 70) with its contacts moved up to 15, 12, 8 or 4° below the equator. The
palm rises so the pads' targets keep their height in the palm frame (<code>film_plan.py --lat-deg</code>).</dd>
<dt>Tracking</dt><dd>One minus the RMS angle between the object's orientation and the commanded one, over the RMS
commanded rotation, at least 0 (<code>sr2hand.control.trial_score</code>). 1 follows the command; 0 is no better than
holding still. The commanded twist turns at 30°/s from 2.0 s, an RMS of 62.7° over a full trial.</dd>
<dt>J</dt><dd>Simulation's grasp score (<code>hom_landscape.track_score</code>): tracking times the share of the task
with all three pads loaded, 0 when the object drops.</dd>
<dt>Rise</dt><dd>The object's largest move along the palm normal toward the palm during a trial, in mm.</dd>
<dt>Tilt</dt><dd>The angle of the object's axis from the palm normal.</dd>
</dl>"""))

    S.append(("setup", "Trials on the hand", f"""
<p>The SR2 hand hung from the UR5 with its palm down. The object was a printed 40 mm sphere in rubbery tape about
0.75 mm thick, on a 36.9 mm stem over a 6.5 g hex block of AprilTags, 28 g in all. It stood on its block on the table.
Two D435 cameras tracked the block. HOM ran on the workstation and took the tracked pose and the servo angles every
20 ms. Each trial closed the fingers at 0.8 s, lifted the object at 1.6 s and commanded the twist from 2.0 s to 8.3 s.
The servo plant of HOM's server was <code>kp0.55_kv0_fr0.18_fl0_dp0.011</code>, the forces HOM's defaults
(f_tripod 2.5 N). In the four morning runs of the generalist S3 grasp, the grip force was 4.0 N.</p>
<p>The afternoon ran unattended after the owner left at 13:25. <code>lab/hom_series.py</code> chained the recentre, the
trial and the putdown, which levels the object before it touches down and releases it slowly. Two releases toppled the
sphere, held 21.6° and 32° from upright; a person stood it up again once. Table 1 lists every twist of the day.</p>
{table(["Trial", "Grasp", "Below equator (°)", "Twist (°)", "Rise (mm)", "Drift (mm)", "Tracking", "End"],
       [[r["label"].split("_")[1], FAMILY[r["family"]], f1(r["lat"]), f"{f1(r['twist_min_deg'])} to {f1(r['twist_max_deg'])}",
         f1(r["rise_max_mm"]), f1(r["drift_end_mm"]), f1(r.get("track_ratio"), 2),
         "full" if r["stop"] in (None, "controller") else "drift guard" if "drift" in (r["stop"] or "") else r["stop"]]
        for r in sorted(R, key=lambda r: (r["lat"], r["label"]))], num=(2, 3, 4, 5, 6),
       caption="The real twists of 2026-11-16. Twist: the object's turn about the palm normal, least and most. "
               "Tracking is blank where the run was too short to score. The drift guard stops a trial when the object "
               "moves 20 mm (30 from 12:49) from its start in the palm frame.", n=1)}
{fig(1, fig1, "Tracking and rise against contact latitude. Filled points: the real trials of Table 1. Open squares: "
     "simulated HOM on grasp 10 and its raised variants with the real object and the server's servo plant, the mean "
     "and range of seeds 0 to 2: friction 1.0 (black), 0.5 (blue), and 1.0 with a soft tangential contact (brown). "
     "Crosses: the ground-clear grasps in simulation at friction 1.0 (black) and 0.5 (blue).")}
{fig(2, MEDIA / "20261116-low_vs_ground_clear_frames.png", "The film grasp (top, 13:40) and a ground-clear grasp "
     "(bottom, 13:03) at the same times in their trials, camera 832412070344. With the film grasp the whole object tips by "
     "3.5 s: the stem and block swing out while the sphere climbs into the hand. With the ground-clear grasp the stem stays "
     "near vertical as the sphere turns.")}
{film(3, "media/20261116-real_hand_twist_ground_clear.mp4", "The two ground-clear twists in one continuous recording: "
      "the controls in real time, the putdown, reconfiguration and recentre at 10x with still stretches cut")}
{film(4, "media/20261116-hom_afternoon_timelapse.mp4", "Every step from 12:09 to 13:56: trials at 3x, the rest at 20x, "
      "the right edge blurred where people pass")}"""))

    rows2 = []
    for lab, n in (("bare ball, μ 1.0 (the scenes)", "closed_default_s012.json"), ("real object, μ 1.0", "closed_stem_mu-1.0.json"),
                   ("real object, μ 0.6", "closed_stem_mu-0.6.json"), ("real object, μ 0.5", "closed_stem_mu-0.5.json"),
                   ("real object, μ 0.5, server's servo plant", "closed_stem2_mu-0.5_kp-0.55_damping-0.011.json"),
                   ("bare ball, μ 0.5", "closed_bare_mu-0.5.json")):
        cells = []
        for g in GR:
            a = C[n][g]
            extra = (f", held {a['held']}/{a['n'] - a['failed']}" if a["held"] < a["n"] - a["failed"] else "") + \
                    (f", {a['failed']} failed" if a["failed"] else "")
            cells.append(f"{f1(a['track'], 2)} / {f1(a['rise'])} / {f1(a['tilt'], 0)}{extra}")
        rows2.append([lab] + cells)
    S.append(("sim", "Simulation with the real object", f"""
<p>HOM's scenes model the object as a bare 40 mm ball of 25 g on a post, with a friction of 1.0 between every pair of
bodies. The real object hangs its stem and block below the ball. That puts its centre of mass about 15 mm below the
ball's centre and adds about eight times the ball's own inertia about a horizontal axis. The workstation rebuilt the
object with its stem and block, standing on its block until the lift, and ran HOM closed loop on the four grasps at
several frictions. HOM itself kept its bare-ball model, as on the hand. Table 2 has the results.</p>
{table(["Model", "Film g10, 18.2°", "Film g11, 20.0°", "Ground-clear g13, 2.4°", "Ground-clear g0, 5.4°"], rows2,
       caption="Closed-loop HOM in simulation, mean of seeds 0 to 2: tracking / rise (mm) / largest tilt (°). A failed "
               "run is one where HOM's least-squares solve did not converge. Files: data/20261116-sweep/sim/closed_*.json.",
       n=2)}
<p>At a friction of 0.5 the real object reproduces the hand. The film grasps climb {f1(s05['film_g10_x105_y70']['rise'])}
and {f1(s05['film_g11_x70_y110']['rise'])} mm, tip over and fail to track. The ground-clear grasps rise
{f1(s05['gr_g13_x105_y70']['rise'])} and {f1(s05['gr_g0_x70_y110']['rise'])} mm and track
{f1(s05['gr_g13_x105_y70']['track'], 2)} and {f1(s05['gr_g0_x70_y110']['track'], 2)}. At 1.0 the film grasps track
{f1(s10['film_g10_x105_y70']['track'], 2)} and {f1(s10['film_g11_x70_y110']['track'], 2)}, simulation's original ranking.
On the bare ball at 0.5 the film grasps fail too, but the ground-clear g13 rolls over. Low friction explains the film
grasps' failure; the stem and block keep the ground-clear grasps upright.</p>
<p>A second model fits as well. At a friction of 1.0 with a soft tangential contact (MuJoCo's impratio 1 in place of
the scenes' 100), the film grasps climbed 18.4 and 23.8 mm and tracked 0.36 and 0.04, while the ground-clear grasps held.
The tape's 0.75 mm added to the ball's radius did not reproduce the climb. The sphere's tape is rubbery, so its friction
on the TPU pads is likely above 0.5; the value 0.5 is a fit, and the soft contact may be closer to what happens.</p>
<p>The grid scored the ground-clear grasps low mostly through J's contact factor. Their simulated tracking is 0.62 to
0.74 at every friction, but the index pad is loaded in only 20 to 39% of the task. Replaying the real servo commands
open loop matched no single plant: the fit ran to its bounds, a sluggish, sticky hand, because closed-loop commands
replayed open loop lose the feedback that shaped them.</p>"""))

    S.append(("why", "Why the contacts' height matters", """
<p>A pad pressing on the sphere below its equator pushes partly upward. At 18° below, about 0.31 of each pad's push
points up; at 8°, 0.14; at 2.4°, 0.04. The pads' friction must hold that share down, less the object's weight, which is
small against HOM's 2.5 N grip. At a friction of 0.5 the film grasps use most of the friction available just to keep
the sphere from climbing, and little is left for the twist. Once a pad slips, the sphere rides up, the contacts sink
further below the equator and the push grows. HOM squeezes harder as the fingers lag, so the climb runs on until the
sphere reaches the palm. Near the equator almost nothing pushes up, and all the friction is free to drive the twist.</p>"""))

    lat_rows = []
    for key, d in LS.items():
        for lat, a in d.items():
            lat_rows.append([key, f1(lat), f1(a["track"], 2), f1(a["rise"]), f"{a['held']}/{a['n'] - a['failed']}"
                             + (f", {a['failed']} failed" if a["failed"] else "")])
    S.append(("open", "What does not fit yet", f"""
<p>The latitude sweep separates the two models, and neither matches it. On the hand, grasp 10 raised to 8 and 4° below the
equator behaved like the ground-clear grasps. The sphere rose 3.8 and 2.4 mm and turned 38 to 39°. At 12° the climb
began: {f1(l12['rise_max_mm'])} mm, tracking {f1(l12['track_ratio'], 2)}. In simulation at a friction of 0.5, grasp 10 at
8° dropped the sphere in all three seeds. With the soft contact it climbed 10 to 14 mm. Every model turned the sphere 69 to
82°, about twice as far as the hand. The replay found the real joints stalling well short of HOM's targets, the thumb's
MCP by 25° and the middle's yaw by 20°.</p>
<p>A weaker servo does not close the gap. With the servos' torque capped at 0.09 or 0.12 N m in place of 0.18, the
film grasps stopped climbing under stiff friction, and the 4 and 8° grasps tipped over instead of twisting less; each
was held in 1 of 3 runs at 0.09 N m. With the soft contact, grasp 10 at 4° rose 3.7 to 3.8 mm, close to the hand's
2.4, but it tilted 63 to 95° and twisted 26 to 28°. Where simulated tracking falls, it falls with tilt and drops, not
with the shorter, upright twist of the hand (data/20261116-sweep/sim/tau_*.jsonl).</p>
<p>The joint records show where the hand differs (<code>data/20261116-trial_tracking.json</code>, from the workstation's
<code>scripts/hardware/hom_trial_tracking.py</code>). No trial turned the sphere past about 44° of the commanded
90.6°, whatever the grasp, where simulation reached 69 to 82°. The middle finger's yaw stayed 16 to 25° from its target
in every period of every twist, and the index's yaw 7 to 21° the other way. Both gaps were already that large during the
squeeze, so they are HOM's force offsets, and the yaw joints never closed them. In the deep grasps, at 18.2 and 12°, the
sphere pushed the thumb's distal joint back to its stop near -20° for 12 to 27% of the task, while HOM commanded +21 to
+52°. That never happened at 4 or 8° or with the ground-clear grasps. The thumb giving way under the sphere is one route
for the climb; whether the simulated thumb gives way the same way is the next check.</p>
{table(["Model", "Below equator (°)", "Tracking", "Rise (mm)", "Held"], lat_rows, num=(1, 2, 3),
       caption="Simulated HOM on grasp 10 and its raised variants with the real object and the server's servo plant, "
               "seeds 0 to 2 (data/20261116-sweep/sim/lat_g10.jsonl).", n=3)}"""))

    rs = ""
    if RS:
        rs = "<p>The workstation also re-scored both grids at the two talk layouts on the real object.</p>"
    S.append(("next", "Consequences and next steps", f"""
<p>Every HOM sphere result so far was scored on the bare ball at a friction of 1.0. Re-scored on the real object, the
64 grasps of both grids at the two talk layouts kept a Spearman correlation of only 0.54 between frictions 0.5 and 1.0.
The 24 grasps 15 to 20° below the equator scored a mean J of 0.056 at 0.5, against 0.329 at 1.0. At (105, 70) grid2dff
grasp 7, 12.5° below, ranks first under both models (J 0.603 at 0.5, 0.547 at 1.0); its plan and HOM server are ready.
{rs}</p>
<ul>
<li>Measure the pads' friction on the taped sphere: a fingertip pressed and dragged across it on the UR5, with the
HEX-E reading the forces and the sphere clamped.</li>
<li>Run grasp 7 at (105, 70) on the hand, and repeat the raised variants for their spread.</li>
<li>Compare the yaw joints' travel during the twist with simulation's: on the hand they stay about 20° from their
targets, which may set the ceiling near 44°. Check whether the simulated thumb's distal joint gives way under the sphere
as the real one does at 12 to 18°. No friction, contact softness or torque limit tried so far reproduces the latitude
sweep.</li>
<li>Record the fingertip boards in every trial; only the last press of the day did.</li>
<li>Re-score the twist landscape on the real object before the next grasp or layout choice.</li>
</ul>"""))

    S.append(("records", "Records", """
<p>Corvus's trial records: <code>docs/experiments/20261115-hom_bench_trials/runs_corvus/</code>; HOM's server records
on the workstation: <code>runs/</code> beside them. The table of the day's twists:
<code>data/20261116-sweep/20261116-all_twists_table.json</code> (<code>lab/sweep_table.py</code>). Simulation:
<code>logs/20261116-sim_real_reversal/</code> on the workstation (<code>rev_closed.py</code>, <code>rev_lat.py</code>,
<code>rev_rescore.py</code>, the scene <code>scenes/sph040_stem_block.xml</code>), copied in part to
<code>data/20261116-sweep/sim/</code>. Unattended series: <code>lab/hom_series.py</code>; set-up, putdown and release:
<code>lab/hom_setup.py</code>; films: <code>lab/film_session.py</code>. Videos of every step:
<code>logs/20261116-hom_trials/</code> on corvus.</p>"""))

    toc = "".join(f'<li><a href="#{k}">{t}</a></li>' for k, t, _ in S)
    body = "".join(f'<section id="{k}"><h2>{t}</h2>{s}</section>' for k, t, s in S)
    out = (TEMPLATE.read_text()
           .replace("{{TITLE}}", "Sim-to-real reversal of HOM's sphere-twist grasps on the SR2 hand")
           .replace("{{TEXCSS}}", "")
           .replace("{{EYEBROW}}", "HOM on the SR2 hand · 2026-11-16")
           .replace("{{H1}}", "Sim-to-real reversal of HOM's sphere-twist grasps on the SR2 hand")
           .replace("{{LEDE}}", lede)
           .replace("{{BYLINE}}", "Corvus lab session with the workstation's HOM session (hand-4a); the owner supervised "
                                  "until 13:25")
           .replace("{{FILM}}", "")
           .replace("{{TOC}}", toc)
           .replace("{{SECTIONS}}", body)
           .replace("{{FOOTER}}", "Built by scripts/hom/hom_twist_sim_real_page.py from data/20261116-sweep/."))
    out = retro_style.apply(out)
    out = docs_index.tag(out, topic="mechanisms", summary=summary)
    PAGE.write_text(out)
    print(PAGE, f"{PAGE.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    page()
