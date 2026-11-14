"""Text of the throughput page (scripts/hardware/tag_tracking_throughput_page.py page): every number from its data file."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from tag_tracking_throughput_page import (DJ, MEDIA, PAGE, ROOT, RUNS, SNS, TEMPLATE, code, esc, fig, film_html, j,
                                          table)


def _p(x, k="p50", f="{:.1f}"):
    return "–" if not x or k not in x else f.format(x[k])


def page():
    sys.path.insert(0, str(ROOT / "scripts/docs"))
    import docs_index
    import retro_style
    d = j(DJ)
    R = d["runs"]

    def S(n):
        return R[n]["summary"]
    r30, r60, r30e5, r60e5 = S("1080p30_e166"), S("540p60_e100"), S("1080p30_e50"), S("540p60_e50")
    w30, w60 = S("1080p30_noroi"), S("540p60_noroi")
    lng = S("540p60_e100_long") if "540p60_e100_long" in R else None
    o30, o60 = r30["outputs"], r60["outputs"]
    a30, a60 = o30["age_at_publish_ms"]["p50"], o60["age_at_publish_ms"]["p50"]
    cpu = lambda s: s["cpu"]["process_percent_of_one_core"]  # noqa: E731
    jit = lambda s: (s["static"]["position_sd_norm_mm"], s["static"]["rotation_sd_deg"])  # noqa: E731
    L = d["links"]
    lw, lw100, ll = L.get("link_ws", {}).get("summary"), L.get("link_ws100", {}).get("summary"), L.get("link_local2", {}).get("summary")
    dist = d.get("block_distance_mm", {})
    hom = d.get("hom") or {}
    homt = {t: v for f in hom.values() for t, v in f.items() if "median_ms" in v}
    hom_side = {t: v for f, ff in hom.items() if "tracker" in f for t, v in ff.items() if "median_ms" in v}
    hom_alone = {t: v for f, ff in hom.items() if "tracker" not in f for t, v in ff.items() if "median_ms" in v}

    lede = (f"The two D435s on corvus track the hex block at the cameras' full rate, from every frame: {o30['rate_hz']:.0f} "
            f"poses a second at 1920 × 1080 and {o60['rate_hz']:.0f} at 960 × 540. Both cameras contribute to "
            f"{100 * min(o30['both_share'], o60['both_share']):.1f} % of them or more. A pose is {a30:.0f} ms old when it is "
            f"published at 30 fps and {a60:.0f} ms at 60 fps; the plan's target is 50 ms. Most of that age is the readout "
            f"of one frame. The still block's pose varies by {jit(r30)[0]:.3f} mm and {jit(r30)[1]:.2f}° (one standard "
            f"deviation) at 1920 × 1080. The tracker uses {cpu(r30):.0f} to {cpu(r60):.0f} % of one of corvus's 12 threads.")
    if lw:
        lede += (f" Over a UDP link, a stand-in controller on the workstation answers each state of the hand in "
                 f"{lw['round_trip_ms']['p50']:.1f} ms at 50 Hz. The link stops the hand's stream after three missed periods.")
    lede += " Closed-loop control on the hand still needs the station to read the servos while a stream runs."
    summary = (f"Two-camera tag tracker on corvus: {o30['rate_hz']:.0f} Hz at 1080p ({a30:.0f} ms frame age) and "
               f"{o60['rate_hz']:.0f} Hz at 540p ({a60:.0f} ms), 100 % of frames tracked, still-block jitter "
               f"{jit(r30)[0]:.3f} mm / {jit(r30)[1]:.2f}°, {cpu(r30):.0f}-{cpu(r60):.0f} % of a core; UDP link to the "
               f"workstation {lw['round_trip_ms']['p50']:.1f} ms round trip" if lw else "") + \
              "; closed loop needs servo reads during the station's stream."

    S_ = []
    S_.append(("terms", "Terms and metrics", """
<dl class="terms">
<dt>Frame time</dt><dd>The time librealsense stamps on a colour frame, from the camera's hardware clock mapped to
corvus's clock (its global time). Both cameras' frames come with hardware metadata.</dd>
<dt>Frame age</dt><dd>The time from a frame's frame time until the pose solved from it is published (sent over UDP).
For a pose from both cameras the frame time is the mean of the two.</dd>
<dt>Tracked share</dt><dd>The share of outputs with a pose. Each output is either from both cameras, from one camera,
or lost.</dd>
<dt>Skew</dt><dd>The time between the two cameras' frames in one output. The cameras are not synchronised.</dd>
<dt>Jitter</dt><dd>The standard deviation of the block's position (the norm over x, y and z) and of its rotation, while
the block lies still.</dd>
<dt>Detection window</dt><dd>The part of the image searched for tags: the block's corners in that camera's last frame,
widened by 40 px or half the box. When the block was lost, the whole frame is searched at half resolution.</dd>
<dt>CPU</dt><dd>The tracker process's CPU time per second, in percent of one hardware thread. corvus has an i7-8700K
with 6 cores and 12 threads.</dd>
<dt>Round trip</dt><dd>On the controller link, the time from sending a state of the hand until the target that answers
it is applied.</dd>
</dl>"""))

    S_.append(("setup", "Setup", f"""
<p>The two D435s stand where the calibration of 11-13 left them, with {SNS[1]} at its new place after the move
(<a href="../20261113-camera_arm_calibration/20261113-camera_arm_calibration.html">calibration page</a>, Section 7). The
tracker uses that page's fitted lenses, its camera poses after the move and the block's fitted map. The hex block lay
still on the table, {dist.get(SNS[0], float('nan')):.0f} mm from {SNS[0]} and {dist.get(SNS[1], float('nan')):.0f} mm
from {SNS[1]}. The plan has the block at 0.3 m in the task. The arm stood still, and the hand's station was off. The
arm's TCP pose came by read-only RTDE, to give the block's pose in the palm frame as well. Nothing moved during these
runs, so they measure rate, latency, CPU and noise, not tracking in motion.</p>
<p>Each run lasted 60 s at one camera mode, after 3 s of start-up. Exposure, gain and white balance were fixed. The
16.6 ms exposure is the calibration's; the plan asks for 5 ms or less against motion blur.</p>"""))

    S_.append(("tracker", "The tracker", f"""
<p>{code('src/sr2hand/tracking/live.py')} runs one thread per camera and one that fuses them.</p>
<ul>
<li>Each camera thread takes the colour stream as YUYV and uses its luma plane, with no colour conversion. It searches
the detection window at full resolution with <code>pupil_apriltags</code>. When the block was lost there, it searches
the whole frame at half resolution with two threads.</li>
<li>The fusing thread takes each frame of the first camera with the other camera's frame nearest in time. It waits up to
one period for that camera's next frame when the newest lies more than half a period away.</li>
<li>One joint solve over all corners of both cameras gives the block's pose in the arm's base. It is Gauss-Newton with
the projection derivatives of OpenCV, started at the last pose. It drops tags seen more than 65° from their normal and
a tag that reprojects worse than 1 px. On the same corners it agrees with <code>sr2hand.tracking.pose.solve_multi</code>
to 0.01 µm and 0.1 arcsec, in 1.6 ms instead of 9.6 ms.</li>
<li>Each pose goes out as a 186-byte UDP message ({code('src/sr2hand/tracking/stream.py')}). It carries the frame time,
the publish time, the skew, the pose in the base and in the palm frame, the cameras and tags used, and the reprojection
RMS.</li>
</ul>
<p>The pose in the palm frame comes from the arm's TCP pose and the palm's place on the flange in CAD
(<code>sr2hand.robot.tool_chain.SR2Hand</code>). The still block lay at
({', '.join(f'{x:.0f}' for x in r30.get('static', {}).get('p_PO_mm', [float('nan')] * 3))}) mm in the palm frame,
below and beside the hanging hand.</p>"""))

    t1 = []
    for n, lab in RUNS:
        if n not in R:
            continue
        s = S(n)
        o = s["outputs"]
        det = [c["det_ms"]["roi"] if c["det_ms"]["roi"].get("n") else c["det_ms"]["full"] for c in s["per_camera"].values()]
        t1.append([esc(lab), f"{o['rate_hz']:.1f}", f"{100 * o['tracked_share']:.1f} / {100 * o['both_share']:.1f}",
                   f"{o['age_at_publish_ms']['p50']:.1f} / {o['age_at_publish_ms']['p95']:.1f}",
                   f"{s['static']['position_sd_norm_mm']:.3f} / {s['static']['rotation_sd_deg']:.3f}",
                   f"{np.mean([x['p50'] for x in det]):.1f}", f"{cpu(s):.0f}"])
    drops = {n: sum(c["dropped"] for c in S(n)["per_camera"].values()) / (S(n)["seconds"] / 60.0) for n in R}
    px_cell = {sn: 0.5 * 1370.0 * (13.0 / 8.0) / dist[sn] for sn in dist}         # 960 x 540: fx about 685 px
    S_.append(("modes", "Throughput by camera mode", f"""
<p>Every mode tracked every frame (Table 1). The output rate is the camera's frame rate, less a few dropped frames:
{min(drops.values()):.0f} to {max(drops.values()):.0f} a minute over both cameras. Detection in the window takes 1.5 to 3 ms
per camera, and the joint solve about 2 ms.</p>
<p>The frame age is about one frame period plus 6 ms (Figure 3). The colour sensor reads its rows one after another
over the whole period, and the frame reaches corvus only after that. At 30 fps that is
{a30:.0f} ms, within the plan's 50 ms. At 60 fps it is {a60:.0f} ms, and at 15 fps
{S('1080p15_e166')['outputs']['age_at_publish_ms']['p50']:.0f} ms. The 5 ms exposure changes the age by under 2 ms and
raises the jitter by about half.</p>
<p>At 960 × 540 a cell of a 13 mm tag spans {min(px_cell.values()):.1f} to {max(px_cell.values()):.1f} px at the
block's distance, and the tags decode in every frame. That mode's lenses are the 1920 × 1080 fits scaled by one half.
Its mean pose of the still block lies
{np.linalg.norm(np.subtract(r60['static']['position_mm'], r30['static']['position_mm'])):.2f} mm from that at
1920 × 1080.</p>
{table(["camera mode, exposure", "output (Hz)", "tracked / both cameras (%)", "frame age p50 / p95 (ms)",
        "jitter, position (mm) / rotation (°)", "detection p50 (ms)", "CPU (% of a thread)"], t1, num=(1, 2, 3, 4, 5, 6),
       caption="The tracker at each camera mode, 60 s each with the block still (the whole-frame runs 30 s, the long run 10 min). Detection is per camera: in the window, or over the whole frame for the whole-frame runs.", n=1)}
{fig(2, MEDIA / "20261113-frame_age_by_mode.png", "The frame age when the pose is published, per camera mode, with the output rate and the tracker's CPU. The dashed line is the plan's target for the median.")}
{fig(3, MEDIA / "20261113-latency_budget.png", "Where the median frame age goes: one frame period of readout, the transfer to corvus, detection, and the wait for the other camera's frame with the joint solve.")}"""))

    S_.append(("worst", "Worst case: the whole frame every time", f"""
<p>If the detection window never found the block, every frame would be searched whole. Forced to do that, the tracker
still kept the full rate with every frame tracked. At 1920 × 1080 detection took
{np.mean([c['det_ms']['full']['p50'] for c in w30['per_camera'].values()]):.0f} ms per camera, the frame age rose to
{w30['outputs']['age_at_publish_ms']['p50']:.0f} ms and the CPU to {cpu(w30):.0f} % of a thread. At 960 × 540 these were
{np.mean([c['det_ms']['full']['p50'] for c in w60['per_camera'].values()]):.1f} ms,
{w60['outputs']['age_at_publish_ms']['p50']:.0f} ms and {cpu(w60):.0f} %. A lost block therefore costs latency, not
frames, while the window finds it again.</p>"""))

    long_txt = ""
    if lng:
        lo = lng["outputs"]
        long_txt = (f"<p>Over 10 minutes at 960 × 540 and 60 fps, the tracker gave {lo['rate_hz']:.1f} poses a second. "
                    f"Of them, {100 * lo['tracked_share']:.2f} % were tracked and {100 * lo['both_share']:.2f} % came from "
                    f"both cameras. The frame age was {lo['age_at_publish_ms']['p50']:.1f} ms (p95 {lo['age_at_publish_ms']['p95']:.1f}). "
                    f"The jitter was {lng['static']['position_sd_norm_mm']:.3f} mm and {lng['static']['rotation_sd_deg']:.3f}°, "
                    f"and the CPU {cpu(lng):.0f} % of a thread. The cameras dropped "
                    f"{sum(c['dropped'] for c in lng['per_camera'].values())} frames in all.</p>")
    S_.append(("still", "The still block's pose", f"""
<p>Figure 4 shows the block's pose over a minute at the two main modes. The noise is a few micrometres and under
0.1°, well below the plan's 0.3 mm and 0.3°. It says nothing about accuracy, which the calibration page measured, nor
about a moving block.</p>
{long_txt}
{fig(4, MEDIA / "20261113-still_block_traces.png", "The still block's position (µm) and rotation (arcmin) about their means over a minute, in the base frame. Top: 1920 × 1080 at 30 fps. Bottom: 960 × 540 at 60 fps.")}"""))

    link_rows = []
    for n, lk in L.items():
        sm = lk["summary"]
        if sm:
            link_rows.append([esc(lk["label"]), f"{sm['period_ms']['p50']:.2f} / {sm['period_ms']['p99']:.2f}",
                              f"{sm['round_trip_ms']['p50']:.2f} / {sm['round_trip_ms']['p95']:.2f} / {sm['round_trip_ms']['p99']:.2f}",
                              f"{sm['round_trip_ms']['max']:.1f}", f"{sm['targets_applied']} of {sm['periods']}"])
    stop_txt = (f"In the test on corvus the stand-in went quiet after 12 s, and the link stopped "
                f"{ll['stop_after_last_target_ms']:.0f} ms after the last target." if ll and ll.get("stop_after_last_target_ms") else "")
    S_.append(("link", "The controller link", f"""
<p>{code('src/sr2hand/control/link.py')} defines two UDP messages. corvus sends a state of the hand every control
period (186 bytes):</p>
<ul>
<li>the nine servo angles with their time;</li>
<li>the block's pose in the palm frame, with its frame time, flags and tag count;</li>
<li>each finger's centre of pressure and normal force;</li>
<li>the last target applied and a status.</li>
</ul>
<p>The controller answers with a target (66 bytes). It holds the nine joint targets in the station's convention (sim
degrees; yaw, mcp and pip per finger), an optional servo speed and a stop flag. It also names the state it answers.</p>
<p>{code('lab/hand_link.py')} is corvus's side. After sending a state it waits up to 8 ms for the answer, then applies
the newest target once. In a dry run, the default, it only logs the target. With <code>--live</code> it sends the
target to the station as a frame of a policy stream (<code>POST /api/v1/stream/frame</code>) under a lease of 0.25 s.
When no new target has come for three periods, it stops: no more frames, the stream ends, and the station holds the
last goal. The station's own lease does the same if corvus stops. {stop_txt}</p>
<p>{code('lab/echo_controller.py')} stands in for the controller. It answers each state after 2 ms, about HOM's time
per step. Over Tailscale between corvus and the workstation, the round trip took
{lw['round_trip_ms']['p50']:.1f} ms at 50 Hz and {lw100['round_trip_ms']['p50']:.1f} ms at 100 Hz. That is about 1 ms
more than on corvus alone (Table 2, Figure 5). The workstation sits on the campus network and corvus behind the lab's router, so
Tailscale's direct path is the only one.</p>
{table(["link", "period p50 / p99 (ms)", "round trip p50 / p95 / p99 (ms)", "round trip max (ms)", "targets applied"],
       link_rows, num=(1, 2, 3, 4), caption="The controller link with the stand-in controller (2 ms per answer). The run on corvus alone also carried the tracker's poses.", n=2)}
{fig(5, MEDIA / "20261113-link_round_trip.png", "The round trip from a state to the target that answers it, as a cumulative share of periods.")}"""))

    if homt:
        hrows = [[t, f"{v['median_ms']:.2f}", f"{v['p95_ms']:.2f}", f"{v['max_ms']:.2f}", str(v["steps"])]
                 for t, v in hom_alone.items()]
        hrows += [[f"{t}, beside the tracker", f"{v['median_ms']:.2f}", f"{v['p95_ms']:.2f}", f"{v['max_ms']:.2f}",
                   str(v["steps"])] for t, v in hom_side.items()]
        S_.append(("hom", "HOM on corvus", f"""
<p>The HOM controller ({code('src/sr2hand/control/hom_bench.py')}, the bench adapter at 50 Hz) was timed on corvus's
CPU with {code('lab/hom_timing.py')}. A short simulation of each task with the 40 mm object on the default hand gives
realistic inputs. A fresh controller then steps over them, and each step is timed (Table 3). The period on the bench is
20 ms.</p>
<p>With the tracker running beside it at 1920 × 1080, HOM took {np.mean([v['median_ms'] for v in hom_side.values()]):.1f} ms
per step against {np.mean([v['median_ms'] for v in hom_alone.values()]):.1f} ms alone. The tracker in turn gave
{S('1080p30_withhom')['outputs']['rate_hz']:.1f} poses a second at
{S('1080p30_withhom')['outputs']['age_at_publish_ms']['p50']:.0f} ms (Table 1, last row). HOM, the tracker and the link
fit on corvus together. HOM can therefore run there with no network in its loop. The link is for a controller that
needs the workstation's GPU. On the workstation HOM took 2.0 ms per step (loop benchmarks).</p>
{table(["task", "median (ms)", "p95 (ms)", "max (ms)", "steps"], hrows, num=(1, 2, 3, 4),
       caption="HOM's time per step on corvus, alone and with the 1920 × 1080 tracker running beside it.", n=3)}"""))

    S_.append(("ready", "What sim2real control still needs", f"""
<ol>
<li><b>Joint angles during a stream.</b> The station reads no servos while a policy stream runs (its telemetry loop
in <code>manta_hand/runtime.py</code>), so the state's joint angles go stale. HOM's measurement needs them. One way
is to read the nine positions right after each frame is written and return them in the answer to
<code>/stream/frame</code>. Position reads run at up to 111 Hz, about 9 ms for all nine, which fits a 20 ms period.
This changes the driver on the CB1, whose working copy has uncommitted edits, so it waits for the owner.</li>
<li><b>The controller's inputs on corvus.</b> HOM's study files were LFS pointers there. design.json is now an
ordinary file, and reach.json with the 40 mm scenes was copied from the workstation for the timing. The hand as built
needs its own model and plan before a live run.</li>
<li><b>Fingertip contacts.</b> The state has room for each finger's centre of pressure. <code>wrench_viewer.py</code>
gives the boards' contact estimate over a websocket; it is not wired into the link yet. It also needs the filter that
the loop benchmarks asked for.</li>
<li><b>A moving block.</b> These runs had the block still. Tracking through motion and the rolling shutter's shear are
untested, and so is finding the block again after a loss. They need a run with the owner turning the block by hand,
ideally with the 5 ms exposure and the plan's lights.</li>
<li><b>Camera mode.</b> 960 × 540 at 60 fps halves the frame age and still decodes the tags at the task's distance;
1920 × 1080 at 30 fps keeps twice the pixels per tag for steep views. The choice can wait for the moving-block run.</li>
<li><b>The palm frame.</b> The pose in the palm frame rests on the palm's CAD place on the flange. The hand tags of
the plan would check it.</li>
</ol>"""))

    S_.append(("files", "Data, rerun and files", f"""
<p>The compact data are in this folder's <code>data/20261113-tracking_throughput.json</code>: each run's summary, a
sample of its frame ages, the two still-block traces, the link runs and the HOM timing. Every sample of every run stays
on corvus in <code>logs/20261113-tag_tracking_throughput/</code>.</p>
<pre class="cmd">L=docs/experiments/20261113-tag_tracking_throughput/lab
/usr/bin/python3 $L/tracker_bench.py OUT --seconds 60 --width 960 --height 540 --fps 60 --exposure 100 --gain 96 --arm
bash $L/run_matrix.sh OUTDIR                       # the five modes of Table 1
python3 $L/echo_controller.py --listen 47002 --reply CORVUS:47001 --delay-ms 2   # on the controller's host
/usr/bin/python3 $L/hand_link.py --controller HOST:47002 --hz 50 --seconds 60 --log link.jsonl [--no-tracker]
python3 $L/link_summary.py link.jsonl</pre>
<p>Code: <code>src/sr2hand/tracking/live.py</code> (the tracker), <code>src/sr2hand/tracking/stream.py</code> (its
UDP message), <code>src/sr2hand/control/link.py</code> (the link's messages). Lab: <code>tracker_bench.py</code>,
<code>run_matrix.sh</code>, <code>tracker_film.py</code> (Figure 1), <code>hand_link.py</code>,
<code>echo_controller.py</code>, <code>link_summary.py</code>, <code>hom_timing.py</code>. This page:
<code>scripts/hardware/tag_tracking_throughput_page.py</code>.</p>"""))

    film = MEDIA / "20261113-tracker_film.mp4"
    filmfig = film_html(1, film, MEDIA / "20261113-tracker_film_poster.jpg",
                        f"The tracker at 1920 × 1080 and 30 fps, both views cropped around the still hex block "
                        f"({SNS[0]} left, {SNS[1]} right). Yellow: the detection window. Green: the tags found. The "
                        f"joint pose's axes are drawn into each view (x red, y green, z blue, 15 mm). The bar below gives "
                        f"the output rate, the frame age, the cameras and tags used, and the reprojection RMS.") if film.exists() else ""
    toc = "\n".join(f'<li><a href="#{k}">{i + 1}. {t}</a></li>' for i, (k, t, _) in enumerate(S_))
    secs = "\n".join(f'<section id="{k}"><h2>{i + 1}. {t}</h2>{b}</section>' for i, (k, t, b) in enumerate(S_))
    title = "Throughput of the two-camera tag tracker on corvus and the controller link"
    h = TEMPLATE.read_text()
    for k, v in dict(TITLE=title, TEXCSS="", EYEBROW="SR2 hand on the UR5 &middot; object tracking, toward sim2real control",
                     H1=title, LEDE=lede, BYLINE="2026-11-13, corvus. Measured with the owner away; nothing moved.",
                     FILM=filmfig, TOC=toc, SECTIONS=secs,
                     FOOTER=f"Built by <code>scripts/hardware/tag_tracking_throughput_page.py</code> from <code>data/</code>.").items():
        h = h.replace("{{" + k + "}}", v)
    h = docs_index.tag(h, topic="hardware", summary=summary, date="2026-11-13")
    h = retro_style.apply(h)
    PAGE.write_text(h)
    print(PAGE, f"{PAGE.stat().st_size / 1e6:.1f} MB")
