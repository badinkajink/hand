#!/usr/bin/env python3
"""Status page of the current work: docs/status.html.

The curated part is docs/status.json (the owner's requests and their state, findings with numbers, decisions waiting on
the owner), written by the sessions that do the work. A section with a "live" key gets a block read at build time:

  jobs     autopilot jobs touched in the last three days, from ~/.claude/autopilot (job records, step ledgers) and the
           usage readings each login's last check left in usage-<login>.json
  prints   the print projects at the owner's chosen seat (the sheets in
           docs/experiments/20261016-sim2real_fingertip_readiness/print/)
  dip      the DIP-link concepts' evaluation files and their exploded renders
  pages    INDEX.md rows of the last four days
  commits  this repository's commits of the last two days
  events   ~/.claude/resguard_events.jsonl entries of the last two days

The owner reads the page after compacting a chat or from another machine, so it opens from the file system: relative
links, images inlined, nothing fetched.

    .venv/bin/python scripts/status_page.py
"""
from __future__ import annotations

import base64
import html
import io
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import retro_style  # noqa: E402

KIT = Path.home() / ".claude/autopilot"
CUR = ROOT / "docs/status.json"
OUT = ROOT / "docs/status.html"
PRINT = ROOT / "docs/experiments/20261016-sim2real_fingertip_readiness/print"
SEAT_CSV = ROOT / "docs/experiments/20261018-fingertip_print_procedure/records/seat_coupon.csv"
DIP = ROOT / "docs/experiments/20261018-dip_link_board_mount"
EVENTS = Path.home() / ".claude/resguard_events.jsonl"
NOW = time.time()


def esc(s):
    return html.escape(str(s), quote=False)


def inline(s):
    """Curated text: escape, then `code` spans, [text](relative link) and bare claude.ai links."""
    s = esc(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', s)
    s = re.sub(r"(?<![\"=>])(https://claude\.ai/artifact/[A-Za-z0-9]+)", r'<a href="\1">\1</a>', s)
    return s


def table(head, rows, raw=False):
    f = (lambda c: c) if raw else inline
    th = "".join(f"<th>{esc(h)}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{f(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def hm(t):
    return time.strftime("%m-%d %H:%M", time.localtime(t)) if t else ""


# ------------------------------------------------------------------------------------------------- live blocks

def live_jobs():
    sys.path.insert(0, str(KIT))
    import aplib
    rows = []
    for p in sorted((KIT / "jobs").glob("*.json")):
        if p.name.endswith(".steps.json") or NOW - p.stat().st_mtime > 3 * 86400:
            continue
        try:
            j = json.loads(p.read_text())
        except ValueError:
            continue
        slug = j.get("slug") or p.stem
        st = aplib.steps(slug)
        steps = [s for s in st.values() if s.get("kind") != "rule"]
        done = sum(s["state"] == "done" for s in steps)
        opn = [s for s in steps if s["state"] not in ("done", "drop")]
        nxt = "; ".join(f"{s['id']}. {s['text'][:150]}" for s in opn[:3]) or "none"
        note = Path(j.get("note", ""))
        try:
            note_rel = note.relative_to(ROOT)
        except ValueError:
            note_rel = note
        rows.append([f"<code>{esc(slug)}</code>", esc(j.get("state", "")), esc(j.get("account") or "shared"),
                     f"{done} of {len(steps)}", esc(nxt), f"<code>{esc(note_rel)}</code>"])
    out = table(["Job", "State", "Login", "Steps done", "Open steps", "Note"], rows, raw=True)
    urows = []
    for acct, label in (("shared", "~/.claude (owner's Max seat)"), ("colorado", "colorado"), ("personal", "personal")):
        f = KIT / f"usage-{acct}.json"
        if not f.exists():
            continue
        c = json.loads(f.read_text())
        s, w = c.get("session") or [None, None], c.get("weekly") or [None, None]
        urows.append([esc(label), f"{s[0]:.0f} %" if s[0] is not None else "", hm(s[1]),
                      f"{w[0]:.0f} %" if w[0] is not None else "", hm(w[1]), hm(c.get("t"))])
    out += "<p>Usage of each login at its last reading (the kit reads them every 5 minutes while a job runs):</p>"
    out += table(["Login", "5-hour window used", "Resets", "Week used", "Week resets", "Read at"], urows, raw=True)
    return out


def chosen_seat():
    import csv
    if not SEAT_CSV.exists():
        return None
    ok = [float(r["seat_mm"]) for r in csv.DictReader(SEAT_CSV.open())
          if r["coupon"] == "two walls" and r["takes_to_floor"].strip().lower() == "yes"
          and r["wall_split"].strip().lower() == "no" and r["keeps_when_tapped"].strip().lower() == "yes"]
    return min(ok) if ok else None


def live_prints():
    seat = chosen_seat()
    if seat is None:
        return "<p>No seat chosen yet in the seat record.</p>"
    order = {"block": 0, "tcd_cap8.5": 1, "tcd_ell7.4x8.5x16x32": 2, "tcd_ell7.4x8.5x11x22": 3}
    names = {"block": "Block", "tcd_cap8.5": "Capsule r 8.5 mm", "tcd_ell7.4x8.5x16x32": "Half-ellipsoid 16/32 mm",
             "tcd_ell7.4x8.5x11x22": "Half-ellipsoid 11/22 mm"}
    rs = []
    for f in sorted(PRINT.glob("20261016-siraya85a_*.json")):
        r = json.loads(f.read_text())
        if "tip" not in r or abs(r.get("seat_mm", 0) - seat) > 1e-6:
            continue
        rs.append(r)
    rs.sort(key=lambda r: (r["layout"] != "four" or r.get("pocket_walls") == 1, r["infill_percent"], order[r["tip"]],
                           r.get("pocket_walls") == 1))
    rows = []
    for r in rs:
        lab = names[r["tip"]] + (", five magnets" if r["layout"].startswith("five") else "") + \
            (", one wall at the pockets" if r.get("pocket_walls") == 1 else "")
        rows.append([esc(lab), f"{r['infill_percent']} %", esc(r["slice"]["check"]["print_time"]),
                     esc(", ".join(f"{p['print_z']:g}" for p in r["pauses"])),
                     f"{r['magnet']['count']}, {r['magnet']['top_depth_mm']:g} mm", f"{r['wall_min_mm']:.2f}",
                     f"<code>{esc(Path(r['project']).name)}</code>"])
    return (f"<p>Seat {seat:g} mm from <code>{esc(SEAT_CSV.relative_to(ROOT))}</code>; projects in "
            f"<code>{esc(PRINT.relative_to(ROOT))}/</code>, the four-magnet prints at 30 % first.</p>"
            + table(["Print", "Infill", "Time", "Pauses (mm)", "Magnets, top under the apex", "Least TPU at a pocket (mm)",
                     "Project"], rows, raw=True))


def img(path, width=900):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def live_dip():
    """The concepts' exploded renders (any concept_*/media/*exploded*.png and final/**/*.png named exploded)."""
    figs = []
    for pic in sorted(DIP.glob("concept_*/media/*exploded*.png")) + sorted(DIP.glob("final/**/*exploded*.png")):
        k = pic.parts[-3] if pic.parts[-3].startswith("concept_") else "final"
        figs.append(f'<figure><img src="{img(pic)}" alt="{esc(k)} exploded"><figcaption>{esc(k.replace("_", " ").capitalize())}, '
                    f'exploded (<code>{esc(pic.relative_to(ROOT))}</code>).</figcaption></figure>')
    return "".join(figs) or "<p>No renders yet.</p>"


def live_pages():
    rows = []
    for line in (ROOT / "docs/experiments/INDEX.md").read_text().splitlines():
        m = re.match(r"\|\s*(2026-\d\d-\d\d)\s*\|", line)
        if not m or time.mktime(time.strptime(m.group(1), "%Y-%m-%d")) < NOW - 4 * 86400:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        subj = re.sub(r"\s*\(.*", "", cells[2]) if len(cells) > 2 else ""
        loc = re.search(r"\]\(([^)]+\.html)\)", line) or re.search(r"`([^`]+\.html)`", line)
        art = re.search(r"https://claude\.ai/artifact/[A-Za-z0-9]+", line)
        href = ""
        if loc:
            p = loc.group(1)
            href = p if p.startswith("experiments/") else re.sub(r"^(\.\./|docs/)*", "", p)
            href = href if href.startswith("experiments/") else "experiments/" + href.split("experiments/")[-1]
        rows.append([m.group(1), esc(subj[:160]), f'<a href="{esc(href)}">{esc(Path(href).name)}</a>' if href else "",
                     f'<a href="{art.group(0)}">{art.group(0)[26:]}</a>' if art else "pending"])
    return table(["Date", "Page", "Local file", "claude.ai"], rows, raw=True)


def live_commits():
    out = subprocess.run(["git", "-C", str(ROOT), "log", "--since=2 days ago", "--format=%h|%ad|%s",
                          "--date=format:%m-%d %H:%M"], capture_output=True, text=True).stdout
    rows = [[f"<code>{esc(h)}</code>", esc(d), esc(s[:180])] for h, d, s in
            (ln.split("|", 2) for ln in out.splitlines()[:60] if ln.count("|") >= 2)]
    return table(["Commit", "When", "Message"], rows, raw=True)


def live_events():
    rows = []
    if EVENTS.exists():
        for ln in EVENTS.read_text().splitlines():
            try:
                e = json.loads(ln)
            except ValueError:
                continue
            if e.get("t", 0) < NOW - 2 * 86400:
                continue
            cmd = e.get("cmd") or e.get("command") or ""
            why = e.get("why") or e.get("reason") or ""
            rows.append([esc(e.get("time", "")), esc(e.get("kind", "")), f"<code>{esc(cmd[:140])}</code>", esc(why[:140])])
    return table(["Time", "Event", "Command", "Reason"], rows, raw=True) if rows else "<p>None.</p>"


LIVE = dict(jobs=live_jobs, prints=live_prints, dip=live_dip, pages=live_pages, commits=live_commits, events=live_events)


# ------------------------------------------------------------------------------------------------- page

def main():
    S = json.loads(CUR.read_text())
    built = time.strftime("%Y-%m-%d %H:%M")
    toc = "".join(f'<li><a href="#{esc(s["id"])}">{esc(s["heading"])}</a></li>' for s in S["sections"])
    body = []
    for s in S["sections"]:
        part = [f'<h2 id="{esc(s["id"])}">{esc(s["heading"])}</h2>']
        part += [f"<p>{inline(p)}</p>" for p in s.get("paras", [])]
        if "table" in s:
            part.append(table(s["table"]["head"], s["table"]["rows"]))
        if "live" in s:
            try:
                part.append(LIVE[s["live"]]())
            except Exception as ex:  # a missing source must not stop the page
                part.append(f"<p>Live block {esc(s['live'])} failed at build time: {esc(type(ex).__name__)}: "
                            f"{esc(str(ex)[:200])}</p>")
        body.append("\n".join(part))
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(S['title'])}</title>
<style>
body {{ max-width: 1100px; margin: 0 auto; padding: 16px; }}
.tw {{ overflow-x: auto; }}
table {{ border-collapse: collapse; margin: 0.6em 0 1.2em; }}
th, td {{ border: 1px solid #000; padding: 4px 6px; vertical-align: top; text-align: left; }}
figure {{ margin: 1em 0; text-align: center; }} figure img {{ max-width: 100%; height: auto; }}
figcaption {{ font-size: 15px; }}
.meta {{ font-size: 15px; }}
</style></head>
<body>
<p class="meta"><a href="index.html">SR² Hand documentation</a> · <a href="pages.html">index of result pages</a></p>
<h1>{esc(S['title'])}</h1>
<p class="meta">Curated text as of {esc(S['as_of'])}; live blocks read {esc(built)} by <code>scripts/status_page.py</code>
(rebuild: <code>.venv/bin/python scripts/status_page.py</code>; curated text in <code>docs/status.json</code>).</p>
<p>{inline(S['lede'])}</p>
<ul>{toc}</ul>
{''.join(body)}
</body></html>
"""
    OUT.write_text(retro_style.apply(page))
    print(f"wrote {OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1e6:.2f} MB", flush=True)


if __name__ == "__main__":
    main()
