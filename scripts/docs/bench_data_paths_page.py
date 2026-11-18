#!/usr/bin/env python3
"""Build docs/overviews/bench_data_paths.html: the SR2 hand bench's data paths, rates and clocks on one page, with each
recorded failure placed on the map (owner, 2026-11-17: "yes, do that and put it in one of the top level docpage htmls,
or even on the very main index itself"; the question before it: "do you think this shuld be in ROS? i admittedly have no
oversight over hwo you have implemented this. but it seems like we have run up against issues with servo telemetry, tag
detection, fingertip sensing streaming more than a few times now").

The facts come from the bench runbook (docs/experiments/20261115-hom_bench_trials/README.md, its "State on ..."
sections of 2026-11-15 to 11-17) and the trial records of 2026-11-17. The page is an ordinary git file that opens from a
clone: one inline SVG, no scripts, no external files.

    DOCS_REFRESH=0 python3 scripts/docs/bench_data_paths_page.py      (corvus: no index refresh there)
"""
from __future__ import annotations

import html as H
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401
import docs_index  # noqa: E402
import retro_style  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/overviews/bench_data_paths.html"
RUNBOOK = "../experiments/20261115-hom_bench_trials/README.md"

# ------------------------------------------------------------------------------------------------ the failures
# (number, date, what happened, where on the map, cause, what changed); the numbers are the figure's markers
FAILURES = [
    (1, "11-16 11:37", "corvus froze for 1.5 s when the trial opened the arm's control link at the lift; the station's "
     "stream lapsed.", "trial process, arm link", "opening RTDE control blocks the Python interpreter for about 1.5 s",
     "the link opens before t = 0"),
    (2, "11-16 11:57", "the stream's lease (0.25 s) lapsed after 33 frames, and the station kept the grip's goals until "
     "the servos' overload protection turned them off.", "HTTP to the station", "a frame waited behind the station's "
     "lock", "a 1 s lease; no new stream after a lapse"),
    (3, "11-16 12:00", "a servo read timed out under the grip's load and was retried for 1.5 s; the grip loosened and the "
     "sphere dropped at the lift.", "servo bus", "retries on a half-duplex bus", "one read attempt per servo in a frame"),
    (4, "11-16", "each request to the station took about 40 ms longer than it should; a frame with read-back took 55 ms.",
     "HTTP to the station", "Nagle's algorithm against the client's delayed acknowledgements",
     "Nagle off on the CB1: 14 ms a frame"),
    (5, "11-15, 11-16", "servo readings went stale during a stream.", "servo bus",
     "one 1 Mbps half-duplex bus; the station stops polling while a stream writes, and nine reads cost about 9 ms",
     "each reading has its age; each frame returns a read-back"),
    (6, "11-16 16:12, 16:47", "all nine servos reset in the middle of a lift, and the sphere fell.", "servo supply (5 V)",
     "4.7-4.9 V while the hand held a grasp", "torque budget 0.84 N m; a servo diagnostics call on the station"),
    (7, "11-16", "the tracked pose jumped where the cameras saw two of the block's tags.", "cameras and tracker",
     "a pose from two tags", "objects are put down at the table's centre, where both cameras see four"),
    (8, "11-16", "MCP readings jumped by about -110 deg past 62-68 deg of flexion.", "servo potentiometers",
     "the potentiometer leaves its track near the end of its 300 deg", "torque on only with the fingers near straight"),
    (9, "11-17 09:17", "the 50 Hz loop stalled for 40-70 ms, and its 60 ms guard stopped a trial.", "trial process",
     "the fingertip estimator ran as a thread beside the tracker and two video writers (68-73 ms an estimate)",
     "the estimator in a process of its own: 17-19 ms, longest period 21-39 ms"),
    (10, "11-17 09:21", "HOM drove the index PIP to -2.9 deg; its reading jumped to +184-189 deg, and the station refused "
     "every pose after.", "servo potentiometers, joint limits", "a limit 0.3 deg from the potentiometer's end",
     "HOM targets kept 5 deg from the ends"),
    (11, "11-17 16:49", "none of the HOM server's 106 replies reached corvus, and the hand lifted empty.",
     "UDP over Tailscale", "not found; four dry runs a minute later lost no reply",
     "a trial stops without a target in 0.5 s, and its lift is cancelled"),
    (12, "11-17 16:55", "at 7.5 V, with offsets sized for 5 V, all nine servos reset in the middle of a squeeze.",
     "servo supply (7.5 V)", "1.5 times the current of the 5 V trials", "torque budget 0.56 N m (the 5 V current)"),
    (13, "11-17 17:06", "the tracker fitted the sphere's map to extra printed blocks 460 mm away.", "cameras and tracker",
     "the blocks repeated the sphere's tag ids 10-15", "blocks with the same ids stay out of view"),
    (14, "11-17 16:4x", "the station closed an idle connection, and the next request failed.", "HTTP to the station",
     "the server's idle timeout on keep-alive connections", "the step test's client reconnects once"),
]
DEVICE = {3, 5, 6, 7, 8, 10, 12, 13}        # in the devices, the power or the sensing; the rest in the plumbing

# ------------------------------------------------------------------------------------------------ the streams
STREAMS = [
    ("HOM state and target", "corvus's loop and the HOM server", "UDP over Tailscale", "50 Hz",
     "the server's step 3-7 ms", "corvus: each state's time"),
    ("Finger goals", "corvus, the station, the nine servos", "HTTP (keep-alive), then the 1 Mbps servo bus", "50 Hz",
     "14 ms a frame with the read-back", "none needed"),
    ("Joint angles", "the servos, the station, corvus", "the read-back of each frame; a 20 Hz poll between streams",
     "50 Hz in a stream, 8.6 Hz idle", "about 20 ms", "the CB1's, hours off: each reading has its own age"),
    ("Object pose", "two D435 cameras, the tracker in the trial process", "USB 3, 1080p, AprilTags", "30 Hz a camera",
     "65-82 ms (medians, 11-17)", "corvus: each frame's capture time"),
    ("Fingertip fields", "three ReSkin boards, the I2C mux, the splitter, the estimator process, the loop",
     "USB serial; pseudo-terminals with one reader each; a pipe", "160.6 records/s a board, in 50 ms batches",
     "data about 41 ms old; an estimate 17-19 ms", "corvus: each batch's arrival"),
    ("Wrench at the wrist", "the HEX-E, corvus", "Ethernet from its compute box", "100-200 Hz", "", "corvus: arrival"),
    ("Arm pose and moves", "the UR5, corvus", "RTDE", "125 Hz", "", "corvus: arrival"),
    ("Films", "two D435 cameras, two video writers in the trial process", "USB 3", "30 Hz", "", "corvus"),
    ("Gantry moves", "the station, the steppers", "the CB1's second serial link", "per move", "", "none needed"),
]

# ------------------------------------------------------------------------------------------------ the figure
W, Hh = 980, 640


def box(x, y, w, h, title=None, sub=None, fill="#ffffff", stroke="#000", dash=False, bold=True):
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"'
           + (' stroke-dasharray="6 4"' if dash else "") + "/>"]
    if title:
        out.append(f'<text x="{x + 8}" y="{y + 19}" font-size="15" font-weight="{"bold" if bold else "normal"}">'
                   f"{H.escape(title)}</text>")
    if sub:
        for k, line in enumerate(sub):
            out.append(f'<text x="{x + 8}" y="{y + 38 + 17 * k}" font-size="13.5">{H.escape(line)}</text>')
    return "\n".join(out)


def text(x, y, s, size=13.5, anchor="start", italic=False, weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}"'
            + (' font-style="italic"' if italic else "") + f">{H.escape(s)}</text>")


def arrow(x1, y1, x2, y2, both=False):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#000" stroke-width="1.5" marker-end="url(#ah)"'
            + (' marker-start="url(#ah)"' if both else "") + "/>")


def mark(x, y, n):
    """A failure's number: a red disc with the number in white."""
    return (f'<circle cx="{x}" cy="{y}" r="11" fill="#c00000"/>'
            f'<text x="{x}" y="{y + 4.6}" font-size="13" font-weight="bold" fill="#fff" text-anchor="middle">{n}</text>')


def figure():
    s = [f'<svg viewBox="0 0 {W} {Hh}" width="{W}" style="max-width:100%;height:auto;font-family:Times,serif" '
         'role="img" aria-label="Map of the SR2 hand bench: three computers, their processes and links, with the '
         'numbered failures of 2026-11-15 to 11-17">',
         '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#000"/></marker></defs>',
         f'<rect x="0" y="0" width="{W}" height="{Hh}" fill="#fff"/>']
    # the workstation
    s.append(box(10, 30, 190, 120, "Workstation", ["HOM server, one per plan:", "a UDP port each", "clock: its logs only"]))
    # corvus and its processes
    s.append(box(285, 10, 395, 620, "corvus, the UR5 lab computer", None, fill="#f4f4f4"))
    s.append(text(295, 50, "clock: the reference (a trial's t = 0)", italic=True))
    s.append(box(298, 62, 370, 300, "hom_trial.py: one Python process", None))
    rows = ["50 Hz control loop", "tracker: two cameras, AprilTags", "two video writers",
            "station client: frames, 20 Hz poll", "HEX-E reader", "the arm's lift (RTDE)"]
    for k, r in enumerate(rows):
        y = 92 + 42 * k
        s.append(box(312, y, 280, 32, None, None))
        s.append(text(322, y + 21, r))
    s.append(box(298, 400, 370, 80, "Fingertip estimator process (from 11-17)",
                 ["takes rests, estimates contacts: 17-19 ms", "spawned by the trial; writes the boards' records"]))
    s.append(box(298, 515, 370, 100, "Splitter: reskin5x_mux.py",
                 ["reads the I2C mux, logs every record", "writes each board to a pseudo-terminal",
                  "(one reader each)"]))
    # the CB1 and its devices
    s.append(box(770, 30, 200, 250, "CB1", ["control station: HTTP :8765", "owns both serial links", "",
                                             "clock: hours off", "(11 h 16 min behind, 11-17):",
                                             "readings have their own age"]))
    s.append(box(770, 320, 200, 120, "Nine SCS0009 servos", ["1 Mbps half-duplex bus", "5 V until 11-17, then 7.5 V",
                                                             "potentiometers over 300 deg"]))
    s.append(box(770, 480, 200, 60, "Gantry steppers", ["second serial link"]))
    # devices on the left
    s.append(box(10, 200, 190, 60, "Two RealSense D435", ["1080p at 30 Hz, USB 3"]))
    s.append(text(14, 280, "pose 65-82 ms old when used", 13))
    s.append(box(10, 295, 190, 50, "UR5 arm", ["RTDE at 125 Hz"]))
    s.append(box(10, 370, 190, 50, "HEX-E wrench sensor", ["100-200 Hz"]))
    s.append(box(10, 515, 190, 100, "Three ReSkin boards", ["fingertip magnetometers", "one I2C mux, USB serial",
                                                            "160.6 records/s a board"]))
    # links
    s.append(arrow(202, 108, 310, 108, both=True))                                  # UDP to the loop
    s.append(text(206, 98, "UDP, 50 Hz", 13))
    s.append(text(206, 125, "Tailscale", 13))
    s.append(arrow(594, 234, 768, 234, both=True))                                  # HTTP frames
    s.append(text(686, 224, "HTTP, 50 Hz", 13))
    s.append(text(686, 252, "14 ms a frame", 13))
    s.append(arrow(870, 282, 870, 318, both=True))                                  # servo bus
    s.append(text(862, 296, "9 writes + 9 reads,", 13, anchor="end"))
    s.append(text(862, 312, "about 9 ms", 13, anchor="end"))
    s.append(arrow(870, 442, 870, 478))                                             # steppers
    s.append(arrow(202, 230, 310, 152))                                             # cameras -> tracker
    s.append(arrow(202, 320, 310, 318, both=True))                                  # RTDE
    s.append(arrow(202, 395, 310, 278))                                             # HEX-E
    s.append(arrow(202, 565, 296, 565))                                             # boards -> splitter
    s.append(arrow(480, 513, 480, 482))                                             # splitter -> estimator
    s.append(text(488, 503, "pseudo-terminals", 13))
    s.append('<polyline points="630,398 630,108 596,108" fill="none" stroke="#000" stroke-width="1.5" '
             'marker-end="url(#ah)"/>')                                             # estimator -> loop
    s.append(text(312, 386, "pipe to the loop: estimates, data 41 ms old", 13))
    # the failures' numbers
    for n, (x, y) in {11: (262, 142), 7: (226, 252), 13: (251, 252), 9: (650, 80), 1: (610, 318),
                      2: (697, 196), 4: (722, 196), 14: (747, 196), 3: (893, 300), 5: (918, 300),
                      6: (790, 424), 8: (815, 424), 10: (840, 424), 12: (865, 424)}.items():
        s.append(mark(x, y, n))
    s.append("</svg>")
    return "\n".join(s)


# ------------------------------------------------------------------------------------------------ the page
def table(head, rows):
    out = ["<table>", "<tr>" + "".join(f"<th>{H.escape(h)}</th>" for h in head) + "</tr>"]
    for r in rows:
        out.append("<tr>" + "".join(f"<td>{H.escape(str(c))}</td>" for c in r) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def page():
    n_dev = len(DEVICE)
    n_plumb = len(FAILURES) - n_dev
    summary = (f"The SR2 hand bench runs on three computers joined by HTTP, UDP over Tailscale, USB, RTDE and Ethernet. "
               f"Of the {len(FAILURES)} failures recorded from 2026-11-15 to 11-17, {n_dev} sat in the devices, the "
               f"power or the sensing, and {n_plumb} in the plumbing between processes.")
    fails = [(n, d, w, where, c, now, "device" if n in DEVICE else "plumbing") for n, d, w, where, c, now in FAILURES]
    plumb = ", ".join(str(n) for n, *_ in FAILURES if n not in DEVICE)
    dev = ", ".join(str(n) for n, *_ in FAILURES if n in DEVICE)
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Data paths, rates and clocks of the SR2 hand bench</title>
<style>
body {{ max-width: 1000px; margin: 2em auto; padding: 0 16px; }}
figure {{ margin: 1.5em 0; text-align: center; }}
figcaption {{ text-align: left; font-size: 0.95em; }}
td, th {{ vertical-align: top; }}
</style>
</head><body>
<p><a href="../index.html">Documentation home</a></p>
<h1>Data paths, rates and clocks of the SR2 hand bench</h1>

<p>{H.escape(summary)} Figure&nbsp;1 maps every stream with its rate, its age when used and its clock; Table&nbsp;2
places each failure on the map. The owner asked on 2026-11-17 whether the bench should move to ROS, after repeated
trouble with servo telemetry, tag detection and fingertip streaming. The last section sorts the failures by what a
middleware changes.</p>

<h2>Terms</h2>
<ul>
<li><b>Rate</b>: messages a second on a link.</li>
<li><b>Age when used</b>: the time from a measurement to the control loop's use of it.</li>
<li><b>Clock</b>: the computer whose time stamps a stream's samples.</li>
<li><b>Read-back</b>: after the station writes a frame of nine goals, it reads the nine positions and returns them with
its answer.</li>
<li><b>Lease</b>: the time the station waits for the next frame before it ends a stream and keeps the last goals.</li>
<li><b>Plumbing</b>: the processes, threads and links between the devices, as opposed to the devices, their power and
the sensing itself.</li>
</ul>

<figure>
{figure()}
<figcaption>Figure 1. The bench on 2026-11-17. Boxes are computers, processes and devices; arrows are links, with their
rates and the age of their data when the loop uses it. Red discs number the failures of Table&nbsp;2 where they
occurred. Failure 9 sat in the trial process; its estimator moved to the process below it that evening.</figcaption>
</figure>

<h2>Streams</h2>
<p>Table 1. Each stream of a HOM trial on 2026-11-17. The ages are medians from that day's trial records and the bench
runbook.</p>
{table(["Stream", "From and to", "Link", "Rate", "Age when used", "Clock"], STREAMS)}

<p>Corvus's clock stamps every sample a trial uses. The CB1's clock runs hours off corvus's, so the station's own time
stamps are not used; each joint reading comes with its age in seconds. The workstation's clock appears only in its logs:
the HOM server's periods are counted from the states' times.</p>

<h2>Failures on the map</h2>
<p>Table 2. The failures recorded in the bench runbook (<a href="{RUNBOOK}">README.md</a>, sections "State on
2026-11-15 17:42" to "State on 2026-11-17 17:22"). The class is "device" for the devices, their power and the sensing,
and "plumbing" for the processes and links between them.</p>
{table(["No.", "Date", "What happened", "Where", "Cause", "Since then", "Class"], fails)}

<h2>Failures and a middleware</h2>
<p>Failures {plumb} sat in the plumbing. Two came from one Python process doing six jobs at once (1 and 9): a job
that blocks the interpreter stalls every other job in it. Three came from the HTTP link to the station (2, 4 and 14),
and one from UDP replies that never arrived (11). ROS&nbsp;2 runs each node as a separate process and stamps every
message with its time. Several nodes can read one topic, and one bag records all topics. Separate processes share no
interpreter (1, 9), and a topic monitor shows a stream that stops (11). Leases and timeouts (2, 4, 14) are settings of
the transport under ROS&nbsp;2 as well.</p>

<p>Failures {dev} sat in the devices, the power and the sensing. The servo bus reads nine positions in about 9 ms
whatever runs above it (3, 5). The potentiometers leave their track near their ends (8, 10). The supply sagged under
the grip's current (6, 12), and the tags were seen twice or too few times (7, 13). A middleware changes none of
these.</p>

<p>The bench is moving toward the structure ROS has without adopting ROS: since 2026-11-17 the fingertip estimator runs
in a process of its own. The next steps on that path are the tracker in its own process, one publish-and-subscribe
transport with time stamps for every stream, a recorder of all streams and a time service on all three computers.
Each has a counterpart in ROS&nbsp;2: nodes, topics, bags and a shared clock.</p>

<h2>Sources</h2>
<ul>
<li>Bench runbook: <code>docs/experiments/20261115-hom_bench_trials/README.md</code>.</li>
<li>Trial loop: <code>docs/experiments/20261115-hom_bench_trials/lab/hom_trial.py</code>; HOM server:
<code>src/sr2hand/control/hom_server.py</code>; station client limits: <code>src/sr2hand/control/servo_limits.py</code>.</li>
<li>Fingertip estimator and its process: <code>src/sr2hand/driver/reskin5x/host/tip_contacts.py</code>; splitter:
<code>src/sr2hand/driver/reskin5x/host/reskin5x_mux.py</code>.</li>
<li>This page's builder: <code>scripts/docs/bench_data_paths_page.py</code>.</li>
</ul>
</body></html>
"""
    body = docs_index.tag(body, topic="hardware", summary=summary, date="2026-11-17")
    return retro_style.apply(body)


def main():
    OUT.write_text(page(), encoding="utf-8")
    print(f"-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
