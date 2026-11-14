"""Text of the Steel comparison page (scripts/contact/steel_solver_page.py page): every number from its data file."""
from __future__ import annotations

import re
import sys

from steel_solver_page import (DJ, LABEL, MEDIA, NW, PAGE, PLANTS, ROOT, SPEED, TEMPLATE, TEX_CACHE, fig, film_html,
                               table)

TEX_RE = re.compile(r"\\\[(.+?)\\\]|\\\((.+?)\\\)", re.S)


def f2(x):
    return f"{x:.2f}".replace("-", "−")


def f3(x):
    return f"{x:.3f}".replace("-", "−")


def page():
    import json
    sys.path.insert(0, str(ROOT / "scripts/docs"))
    import docs_index
    import retro_style
    import texsvg
    d = json.loads(DJ.read_text())
    H, S = d["hom"], d["speed"]

    def us(k, n=1024):
        return S[k][str(n)]["us_per_world_step"]
    pads, sh, sp = H["mjc_pads"], H["steel_hydro"], H["steel_pt"]
    nh = H["nt_mjc_hydro"]
    ref = us("mjw_pads")
    r_sh, r_sp, r_spt = us("steel_hydro") / ref, ref / us("steel_pt"), us("steel_pt") / us("mjw_pt")
    lo, hi = sh["diff_ci"]
    rss = max(S[k]["1024"]["host_rss_mb"] for k in ("steel_hydro", "steel_pt", "nt_mjc_hydro")) / 1024

    lede = (f"Steel, the solver Skild AI released in Newton, did not meet the rule for replacing the sphere pads on the "
            f"inner-loop benchmark's sphere twist. With compliant hydroelastic contact, HOM's mean score J was {f2(sh['mean_J'])} "
            f"against {f2(pads['mean_J'])} on the pads. A step of that plant took {us('steel_hydro'):.2f} µs per world "
            f"at 1024 worlds, {r_sh:.1f} times MuJoCo-Warp's {ref:.2f} µs with the pads. A Steel step with point contact "
            f"took {us('steel_pt'):.2f} µs, {r_sp:.2f} times faster than the pads' step. HOM's mean J on that plant was "
            f"{f2(sp['mean_J'])}. MuJoCo-Warp with point contact was faster still, at {us('mjw_pt'):.2f} µs.")
    summary = (f"Steel did not replace the sphere pads on the sphere twist: with compliant hydroelastic contact HOM's mean "
               f"J was {f2(sh['mean_J'])} (pads {f2(pads['mean_J'])}) at {r_sh:.1f} times the pads' GPU time at 1024 "
               f"worlds; with point contact Steel was {r_sp:.2f} times faster than the pads, {r_spt:.1f} times slower "
               f"than MuJoCo-Warp point contact, and J fell to {f2(sp['mean_J'])}.")

    S_ = []
    S_.append(("terms", "Terms and metrics", r"""
<dl class="terms">
<dt>Plant</dt><dd>The simulated hand and sphere that HOM drives: one solver with one fingertip contact model
(Table 1). Hand, sphere, servos and scene friction are the inner-loop benchmark's
(<code>scripts/hom/inner_loop_plant.py</code>). The hand is the cad_v2 SR2 hand with the v4b tips, and the object is a
40 mm sphere. The servo model is the SCS0009's: 0.72 N m/rad, at most 0.18 N m. The physics step is 1 ms.</dd>
<dt>Pair</dt><dd>One finger layout with one grasp, from the benchmark's twenty pairs of the twist
(<code>docs/experiments/20261112-inner_loop_benchmark/rows/mpc_pairs_twist.json</code>). Every plant ran the same pairs at
seeds 0 and 1, 40 rollouts in all.</dd>
<dt>HOM</dt><dd>The hand-object controller of the benchmark, run on the CPU (<code>hom_tasks.rollout</code>, the
controller of <code>hom_gpu.cpu_rollout</code>). It closes the fingers, lifts the sphere off its post at 2.4 s and then
turns it about the commanded axis.</dd>
<dt>Score J</dt><dd>HOM's tracking score of a rollout (<code>hom_landscape.track_score</code>):
\[ J = \max\!\left(0,\; 1 - \frac{\theta_\mathrm{err}}{\theta_\mathrm{ref}}\right) c. \]
Here \(\theta_\mathrm{err}\) is the RMS angle between the sphere's orientation and the commanded one over the task.
\(\theta_\mathrm{ref}\) is the RMS commanded rotation, 59.4° on these pairs. \(c\) is the share of the task with all
three fingertips loaded. J is 1 when the sphere follows the commanded turn and 0 when it turns no closer than a sphere
kept still. A drop scores 0.</dd>
<dt>Held</dt><dd>At the end of the rollout the sphere is less than 15 mm below its start height. At least two fingers
also press on it, with 0.12 N or more each.</dd>
<dt>Stopped rollout</dt><dd>A rollout in which HOM's least-squares solve of its Eq. (2) received a NaN and raised
"SVD did not converge". In all 11 that were recorded, the sphere had already fallen out of the hand. Each is scored as a
drop, with J = 0 and held false.</dd>
<dt>Paired difference ΔJ</dt><dd>Per pair, the mean over seeds 0 and 1 of J on a plant minus J on the sphere pads.
Its 95 % interval is the percentile bootstrap over the 20 pairs (20,000 resamples,
<code>sr2hand.numerics.stats.bootstrap_ci</code>).</dd>
<dt>Worlds</dt><dd>Copies of the scene stepped together in one GPU batch. The benchmark's planners step 1024 to 2048
worlds at a time.</dd>
<dt>GPU time per world and step</dt><dd>The wall time of 500 physics steps (0.5 s simulated) divided by 500 and by the
number of worlds, in µs. The worlds start from the plant's own grasp at 2.4 s, with the post removed and the servo
targets fixed, and run 0.2 s before the timed steps. The steps replay CUDA graphs of 10 steps each.</dd>
<dt>Contacts per world</dt><dd>The contact points reported by the collision pipeline at the end of the timed steps,
averaged over the worlds.</dd>
<dt>Held share</dt><dd>In a throughput run, the share of worlds in which the sphere moved less than 20 mm.</dd>
<dt>Hydroelastic contact</dt><dd>A contact model in which each pair of overlapping bodies presses with
\(p = k_h\,\delta\), \(\delta\) the depth inside the other body's signed-distance field. Here
\(k_h\) = 2.5 MPa / 8.5 mm on the tips (the pads' modulus over the pad's depth) and 100 \(k_h\) on the sphere.</dd>
<dt>Object-only contact pairs</dt><dd>The Newton plants collide the sphere with every other body and nothing else, and
every mesh except the hydroelastic tips collides as its convex hull, as MuJoCo collides meshes. The MuJoCo-Warp throughput
rows were run with the same pairs. The fingers cannot touch each other in these runs.</dd>
</dl>"""))

    t1 = table(["Plant", "Solver", "Fingertip contact"], [[lab, s, c] for _k, lab, s, c in PLANTS],
               caption="The plants. All carry the benchmark's hand, sphere, servos and friction.", n=1)
    S_.append(("plants", "Plants and solver settings", t1 + f"""
<p>The two MuJoCo plants are the benchmark's CPU references. The five Newton plants ran Skild AI's Newton fork at
the commit its Steel examples pin (skild-ai/newton f6fc82a8, Newton 1.9.0.dev0, Warp 1.19.0). Each imported the
benchmark's point-contact scene of the pair from MJCF. The servos became joint drives with Steel's
<code>drive_mode="rows"</code>, which bounds the drive's whole reaction at 0.18 N m, as MuJoCo clamps the actuator
force.</p>
<p>Steel ran 16 iterations with compliant rows and 8 otherwise. Its row capacity per world was raised from 32 to 1,536.
With 32 rows, 148 of a grasp's 180 rows were dropped and the sphere fell. Newton's SolverMuJoCo ran the benchmark's MuJoCo-Warp
settings: 20 iterations, 20 line-search steps, elliptic friction cones and impratio 100.</p>
<p>The hydroelastic fields were sampled at 0.5 mm with a ±6 mm band, as on the contact bed of 2026-10-06.
\\(k_h\\) was set from the pads' modulus and was not fitted to the UR5 presses of the v3 tips.</p>"""))

    order = ["mjc_pads", "mjc_pt", "steel_hydro", "steel_pt", "nt_mjc_hydro", "steel_hydro_hard", "steel_pt_pf"]
    body = []
    for k in order:
        h = H[k]
        h1 = H.get(f"{k}_sap")
        ci = "" if h["diff_ci"] is None else f"{f3(h['diff_ci'][0])} to {f3(h['diff_ci'][1])}"
        body.append([LABEL[k], f"{h['held']}", f"{h['stopped']}", f3(h["mean_J"]),
                     "" if k == "mjc_pads" else f3(h["diff"]), ci, f3(h1["mean_J"]) if h1 else ""])
    t2 = table(["Plant", "Held of 40", "Stopped", "Mean J", "ΔJ", "ΔJ, 95 % interval", "Mean J, first run"], body,
               num=(1, 2, 3, 4, 6), n=2,
               caption="HOM through the twist on each plant, 20 pairs at seeds 0 and 1. Stopped rollouts count as drops "
                       "(J 0). The last column is the first run of the Newton plants, with every pair of shapes "
                       "allowed to collide and the meshes as triangle meshes.")
    f2p = fig(2, MEDIA / "20261113-paired_J_difference.png",
              "J on each plant minus J on the sphere pads, per pair (dots, mean of the two seeds). White circle: the "
              "mean over the 20 pairs. Black bar: its 95 % bootstrap interval.")
    films = d.get("films", {})
    f4 = films.get("film_twist_p4_s0")
    film4 = ""
    if f4 and (MEDIA / "20261113-film_twist_p4_s0.mp4").exists():
        film4 = film_html(3, MEDIA / "20261113-film_twist_p4_s0.mp4", MEDIA / "20261113-film_twist_p4_s0.jpg",
                          "HOM through the twist on pair 4 (x 100 mm, y 70 mm, grasp 4), seed 0, one world per plant. "
                          + film_caption(f4))
    S_.append(("hom", "HOM through the twist on each plant", t2 + f"""
<p>On the sphere pads HOM's mean J was {f3(pads['mean_J'])}, with {f3(pads['seed_means'][0])} at seed 0 and
{f3(pads['seed_means'][1])} at seed 1. Steel with compliant hydroelastic contact came closest. Its mean J was {f3(sh['mean_J'])},
and it held the sphere in {sh['held']} of 40 rollouts. The paired difference was {f3(sh['diff'])}, with an interval
from {f3(lo)} to {f3(hi)}. With MuJoCo's own point contact the difference was smaller, {f3(H['mjc_pt']['diff'])}.</p>
<p>Steel with point contact reached a mean J of {f3(sp['mean_J'])} and held {sp['held']} of 40. With point friction in
place of Steel's friction patches, it held {H['steel_pt_pf']['held']}. Hard rows at the hydroelastic contact points held
{H['steel_hydro_hard']['held']}. Newton's MuJoCo solver with the same hydroelastic contacts held {nh['held']}; its mean J
was {f3(nh['mean_J'])}.</p>
<p>The stopped rollouts were drops. On the CPU, the eight stopped rollouts of Steel with point friction and the three of
MuJoCo point contact were rerun with HOM's inputs recorded at the failure. Each time the sphere lay on the floor, 0.31 to
0.34 m from where it started, with no fingertip in contact. HOM's frame from the witness points then had no finite normal for
one finger.</p>
<p>The first run of the Newton plants differed in two ways: every pair of shapes could collide, and the finger meshes
were triangle meshes. For Steel's compliant hydroelastic plant the mean J changed from
{f3(H['steel_hydro_sap']['mean_J'])} to {f3(sh['mean_J'])} (Table 2, last column). Steel's point contact changed from
{f3(H['steel_pt_sap']['mean_J'])} to {f3(sp['mean_J'])}.</p>
""" + f2p + film4))

    sb = []
    for k, lab in SPEED:
        s = S.get(k, {})
        row = [lab] + [f"{s[str(n)]['us_per_world_step']:.2f}" if str(n) in s else "–" for n in NW]
        s1 = s.get("1024", {})
        row += [f"{s1.get('contacts_per_world', 0):.1f}" if s1 else "–",
                f"{min(s[str(n)]['held_frac'] for n in NW if str(n) in s):.2f}" if s else "–"]
        sb.append(row)
    names = json.loads((ROOT / "docs/experiments/20261112-inner_loop_benchmark/rows/mpc_pairs_twist.json").read_text())
    idx = {f"x{q['x']:g}_y{q['y']:g}_g{q['grasp']}": i for i, q in enumerate(names["pairs"])}
    other = [f"pair {idx[S[k]['1024']['pair']]} for {LABEL.get(k, k)}" for k, _l in SPEED
             if "1024" in S.get(k, {}) and idx[S[k]["1024"]["pair"]] != 0]
    t3 = table(["Plant"] + [f"{n} worlds" for n in NW] + ["Contacts per world, 1024", "Lowest held share"], sb,
               num=tuple(range(1, 8)), n=3,
               caption="GPU time per world and step (µs) on the RTX 4070 Ti SUPER, with object-only contact pairs. "
                       "The worlds start from the plant's own grasp of pair 0 at 2.4 s, or of the first pair whose "
                       "seed-0 rollout finished (" + ", ".join(other) + "). MuJoCo-Warp ran the benchmark's solver "
                       "settings.")
    bf = d.get("speed_bench_filter", {}).get("mjw_pads", {}).get("1024", {}).get("us_per_world_step")
    f3p = fig(4, MEDIA / "20261113-throughput_vs_worlds.png",
              "GPU time per world and step against the number of worlds. Solid: MuJoCo-Warp. Dashed: the Newton plants.")
    S_.append(("speed", "GPU time per world and step", t3 + f"""
<p>At 1024 worlds a MuJoCo-Warp step took {ref:.2f} µs per world with the pads and {us('mjw_pt'):.2f} µs with point
contact. A Steel step took {us('steel_pt'):.2f} µs with point contact and {us('steel_pt_pf'):.2f} µs with point friction.
With compliant hydroelastic rows it took {us('steel_hydro'):.2f} µs, and with hard hydroelastic rows
{us('steel_hydro_hard'):.2f} µs. A step of Newton's MuJoCo solver with hydroelastic contact took
{us('nt_mjc_hydro'):.2f} µs.</p>
<p>Steel's point contact was thus {r_sp:.2f} times faster than the pads and {r_spt:.1f} times slower than MuJoCo-Warp's
own point contact. Steel's compliant hydroelastic contact was {r_sh:.1f} times slower than the pads. At 16 worlds every
GPU plant was bound by launch latency, from {min(S[k]['16']['us_per_world_step'] for k in S):.0f} to
{max(S[k]['16']['us_per_world_step'] for k in S):.0f} µs.</p>
<p>Under the benchmark's own filter, the fingers' link spheres also collide with each other. A MuJoCo-Warp step with the
pads then took {bf:.2f} µs per world at 1024 worlds. Table 3 uses object-only pairs for every plant.</p>
<p>Up to {rss:.1f} GB of host memory was used by a Newton plant of 1024 worlds, almost all of it in the model builder
(about 10 MB per world for 1,348 shapes). The workstation's 31 GB, shared with the desktop, left no room for 2048
worlds. At 2048 worlds a MuJoCo-Warp step took {us('mjw_pads', 2048):.2f} µs per world with the pads and
{us('mjw_pt', 2048):.2f} µs with point contact.</p>""" + f3p))

    S_.append(("rule", "The replacement rule", f"""
<p>The rule was written into the runbook before the runs. A Steel plant would replace the pads in the planners only
if it met three conditions. Its step had to be at least 1.3 times as fast as MuJoCo-Warp's with the pads, at 1024 to 2048
worlds. It had to hold the sphere in every world. HOM's mean J on it had to lie within the pads' seed spread,
{f3(pads['seed_means'][1])} to {f3(pads['seed_means'][0])}.</p>
<p>Steel's compliant hydroelastic plant held the sphere in every world of the throughput runs. It was {r_sh:.1f} times
slower than the pads, and HOM's mean J on it was {f3(sh['mean_J'])}. Steel's point contact was {r_sp:.2f} times faster,
below the 1.3 of the rule, and HOM's mean J on it was {f3(sp['mean_J'])}. Neither plant met the rule, so the press
calibration of \\(k_h\\) and the pinch-torque test of the runbook's stage 3 were not run.</p>"""))

    S_.append(("failures", "Runs that failed", """
<ol>
<li><b>Sweep and prune over all shapes (18:29 to 19:01).</b> In the first run every shape of a world could be paired
with every other, 1,348 shapes, most of them the tips' small rigid spheres. Newton's narrow-phase buffer was sized for all
those pairs, 9.3 GB at 256 worlds, so every hydroelastic throughput run above 16 worlds failed. Its HOM rows
are the last column of Table 2.</li>
<li><b>Two watchdog kills (18:48 and 19:01).</b> About 11 GB of host memory is needed to build 1024 worlds, and the lane
line had an 8 GB limit. The box paged, and the memory watchdog stopped the line twice. Each step ran under
<code>timeout</code> in its own process group, so the last step outlived the kill until 19:03, beside the lane's next
line. The later lines ran with a 12 GB limit and <code>timeout --foreground</code>.</li>
<li><b>Triangle meshes.</b> The finger meshes were imported from MJCF as triangle meshes. Among them the 40 mm sphere met
about 17,000 triangles per world, which overflowed the triangle buffer and dropped the sphere in a CPU test. The convex
hulls of the meshes were used in the later runs.</li>
<li><b>Hydroelastic broad phase (19:39 to 19:44).</b> At 256 and 1024 worlds the hydroelastic broad-phase buffer had room
for 1,382 pairs per world, and the grasp needed about 1,390. Contacts were dropped, and the sphere stayed in the hand in
44 to 73 % of the worlds. These rows were rerun from 19:56 to 20:00 with a buffer 1.5 times larger. Every world then held
the sphere, and Table 3 gives the rerun.</li>
</ol>"""))

    S_.append(("next", "Future work", """
<ol>
<li><b>Fewer Steel iterations.</b> <code>speed --plant steel_hydro --iters 8</code> and
<code>hom --plant steel_hydro --iters 8</code> would show whether half the iterations halve the 9.2 µs. Half of 9.2 µs
is still above the pads' 2.8 µs, so the speed condition of the rule would still fail.</li>
<li><b>Tip stiffness from the presses.</b> Fitting \\(k_h\\) to the v3 tips' UR5 presses
(<code>docs/experiments/20261030-v3_tip_contact_model/</code>) and rerunning <code>hom --plant steel_hydro</code> would
test whether the tip stiffness accounts for the gap in J. A ΔJ interval containing 0 would mean that it does.</li>
<li><b>MuJoCo-Warp 4.0.</b> The port of 2026-11-08 measured MuJoCo-Warp 4.0 17 to 28 % faster than 3.6. Rerunning
<code>speed --plant mjw_pads</code> under 4.0 would move the reference that Steel has to beat.</li>
</ol>"""))

    S_.append(("files", "Data, rerun and files", """
<p>The rows are in this folder:</p>
<ul>
<li><code>rows/hom_&lt;plant&gt;.jsonl</code>: HOM, one row per pair and seed. <code>_sap</code> marks the first run,
with every pair of shapes.</li>
<li><code>rows/speed.jsonl</code>: throughput. The rows of Table 3 have <code>pairs</code> set to <code>obj</code>.</li>
<li><code>rows/film_twist_p*_s0.json</code>: the films' clips.</li>
<li><code>rows/hom_stopped_inputs.jsonl</code>: HOM's inputs at the 11 stopped rollouts rerun on the CPU
(<code>scripts/contact/steel_hom_stop_inputs.py</code>), with their rows in <code>rows/hom_*_diag.jsonl</code>.</li>
</ul>
<p>The compact data of this page are <code>data/20261113-steel_solver.json</code>. Trajectories and the grasps at 2.4 s are in
<code>logs/20261113-steel/traj/</code> on the workstation.</p>
<pre class="cmd">PY=.venv/bin/python
PYS=logs/20261113-steel/venv/bin/python     # skild-ai/newton f6fc82a8, Warp 1.19 (README.md: how to rebuild it)
$PY scripts/contact/steel_hand_compare.py hom --plant mjc_pads --seeds 0 1
WARP_CACHE_PATH=$(mktemp -d) $PYS scripts/contact/steel_hand_compare.py hom --plant steel_hydro --seeds 0 1
WARP_CACHE_PATH=$(mktemp -d) $PYS scripts/contact/steel_hand_compare.py speed --plant steel_hydro --nworld 1024
$PY scripts/contact/steel_hand_compare.py speed --plant mjw_pads --nworld 1024
WARP_CACHE_PATH=$(mktemp -d) $PYS scripts/contact/steel_hand_film.py --pairs 0 4
$PY scripts/contact/steel_solver_page.py data figs page</pre>
<p>The GPU runs were lines of the inner-loop benchmark's lane (<code>logs/20261112-mpc/lane.txt</code>):
<code>logs/20261113-steel/stage1.sh</code>, <code>stage2.sh</code>, <code>stage3.sh</code> and <code>film.sh</code>. The
runbook is this folder's <code>README.md</code>.</p>"""))

    f1 = films.get("film_twist_p0_s0")
    film1 = ""
    if f1 and (MEDIA / "20261113-film_twist_p0_s0.mp4").exists():
        film1 = film_html(1, MEDIA / "20261113-film_twist_p0_s0.mp4", MEDIA / "20261113-film_twist_p0_s0.jpg",
                          "HOM through the twist on pair 0 (x 105 mm, y 70 mm, grasp 10), seed 0, one world per plant. "
                          + film_caption(f1))
    toc = "\n".join(f'<li><a href="#{k}">{i + 1}. {t}</a></li>' for i, (k, t, _) in enumerate(S_))
    secs = "\n".join(f'<section id="{k}"><h2>{i + 1}. {t}</h2>{b}</section>' for i, (k, t, b) in enumerate(S_))
    title = "Steel and Newton hydroelastic contact against the sphere pads on the sphere twist"
    h = TEMPLATE.read_text()
    for k, v in dict(TITLE=title, TEXCSS=texsvg.CSS + texsvg.BLOCK_CSS,
                     EYEBROW="Fingertip contact model &middot; inner-loop benchmark",
                     H1=title, LEDE=lede,
                     BYLINE="2026-11-13, workstation (RTX 4070 Ti SUPER, 16 GB), in the benchmark's GPU lane. Owner: "
                            "&ldquo;if steel has hydroelastic contact built in, and it is that much faster than warp, "
                            "then i think we do have to compare to them&rdquo;.",
                     FILM=film1, TOC=toc, SECTIONS=secs,
                     FOOTER="Built by <code>scripts/contact/steel_solver_page.py</code> from <code>data/</code>.").items():
        h = h.replace("{{" + k + "}}", v)
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in TEX_RE.finditer(h)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE))
    h = TEX_RE.sub(lambda m: next(svgs), h)
    h = docs_index.tag(h, topic="contact", summary=summary, date="2026-11-13")
    h = retro_style.apply(h)
    PAGE.write_text(h)
    print(PAGE, f"{PAGE.stat().st_size / 1e6:.1f} MB")


def film_caption(f):
    c = f["clips"]
    parts = []
    for k in f["plants"]:
        x = c.get(k, {})
        parts.append(f"{LABEL[k]}, J {x.get('J', 0):.2f}" + ("" if x.get("held") else ", dropped"))
    return ("Top row, then bottom row: " + "; ".join(parts) + ". Finger links are drawn see-through and the sphere's "
            "axes as rods. Below each clip: the time, the sphere's angular velocity and each finger's force on it. A "
            "single world follows a different trajectory from the same world in the 40-world batches of Table 2, so the "
            "clips show behaviour and Table 2 gives the scores.")
