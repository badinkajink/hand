"""Reprint the CYLINDER tag (tag36h11 id 0) at 40.0 mm.

Why 40 mm: on 2026-08-31 the 30 mm vane tag stopped decoding partway through 5 of 12
Stage-1 runs while the 40 mm static reference tag decoded in every frame of every run,
in the same images at the same range.  Size is the only thing that differs.

The bit grid is lifted out of the EXISTING apriltags_letter.pdf rather than regenerated,
so the new print is the same pattern in the same orientation as the tag currently on the
vane.  That matters: the detector's pose frame is fixed to the pattern, so a tag printed
rotated 90 deg would rotate the measured object axis even if it is mounted "the same way".
The lift is cross-checked against OpenCV's tag36h11 dictionary and the result is
decode-verified by rasterising this PDF and running the real detector over it.
"""
import re, sys, zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
MM = 72.0 / 25.4
PW, PH = 215.9 * MM, 279.4 * MM
M = 12.0
CORE = 8                     # tag36h11 renders 8x8 cells: a 1-cell black ring + 6x6 data
QUIET_CELLS = 2.0
SIZE_MM = 40.0

# ---------------------------------------------------------------- lift id0 out of the old sheet
def old_id0_grid(pdf: Path, tag_mm: float = 30.0) -> list[list[bool]]:
    raw = pdf.read_bytes()
    body = zlib.decompress(raw.split(b"stream\n", 1)[1].rsplit(b"\nendstream", 1)[0])
    rect = re.compile(rb"^(\d) (\d) (\d) rg ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) re f$")
    cell = tag_mm / CORE
    black = []
    for ln in body.split(b"\n"):
        m = rect.match(ln.strip())
        if not m or m.group(1) != b"0":
            continue
        x, ypdf, w, h = (float(v) / MM for v in m.groups()[3:])
        if abs(h - cell) > 0.05:                       # only the 30 mm band has this cell height
            continue
        black.append((x, (PH / MM) - ypdf - h, w, h))  # x, y-from-top, w, h  (all mm)
    if not black:
        sys.exit(f"no {tag_mm} mm tag cells found in {pdf}")

    black.sort(key=lambda r: r[0])                     # split the band into its separate tags
    groups, cur = [], [black[0]]
    for r in black[1:]:
        if r[0] - max(c[0] + c[2] for c in cur) > cell:
            groups.append(cur); cur = [r]
        else:
            cur.append(r)
    groups.append(cur)
    first = groups[0]                                  # band A is ids 0,1,2 left to right

    x0 = min(r[0] for r in first); y0 = min(r[1] for r in first)
    grid = [[False] * CORE for _ in range(CORE)]
    for x, y, w, h in first:
        r = round((y - y0) / cell)
        c = round((x - x0) / cell)
        for k in range(round(w / cell)):
            grid[r][c + k] = True
    return grid

def aruco_grid(tag_id: int = 0) -> list[list[bool]]:
    import cv2, numpy as np
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    img = cv2.aruco.generateImageMarker(d, tag_id, CORE * 10)   # includes the black ring
    return [[bool(img[r * 10 + 5, c * 10 + 5] < 128) for c in range(CORE)] for r in range(CORE)]

def rot90(g): return [list(r) for r in zip(*g[::-1])]

# ---------------------------------------------------------------- PDF primitives
ops = []
def X(mm): return mm * MM
def Y(mm): return PH - mm * MM
def rect(x, y, w, h, fill=(0, 0, 0)):
    ops.append(f"{fill[0]} {fill[1]} {fill[2]} rg {X(x):.4f} {Y(y+h):.4f} {w*MM:.4f} {h*MM:.4f} re f")
def line(x1, y1, x2, y2, w=0.4, g=0.65, dash=None):
    d = f"[{dash} {dash}] 0 d " if dash else "[] 0 d "
    ops.append(f"{g} {g} {g} RG {w} w {d}{X(x1):.4f} {Y(y1):.4f} m {X(x2):.4f} {Y(y2):.4f} l S")
def esc(s): return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
def text(x, y, s, size=8, bold=False, g=0.0):
    ops.append(f"BT {g} {g} {g} rg {'/F2' if bold else '/F1'} {size} Tf "
               f"{X(x):.4f} {Y(y):.4f} Td ({esc(s)}) Tj ET")
def textc(xc, y, s, size=8, bold=False, g=0.0):
    text(xc - len(s) * size * (0.60 if bold else 0.55) / MM / 2, y, s, size, bold, g)

def draw_tag(px, py, s, bits, quiet_cells=QUIET_CELLS):
    """px,py = top-left of the quiet-zone patch; s = the black-square edge in mm."""
    n = len(bits); cell = s / CORE; q = quiet_cells * cell
    patch = n * cell + 2 * q
    rect(px, py, patch, patch, fill=(1, 1, 1))
    ox, oy = px + q, py + q
    for r, row in enumerate(bits):
        c = 0
        while c < n:
            if row[c]:
                c2 = c
                while c2 + 1 < n and row[c2 + 1]:
                    c2 += 1
                rect(ox + c * cell, oy + r * cell, (c2 - c + 1) * cell, cell)
                c = c2 + 1
            else:
                c += 1
    return patch

def cutmarks(px, py, patch, pad=2.5):
    x0, y0, x1, y1 = px - pad, py - pad, px + patch + pad, py + patch + pad
    for a, b, c, d in [(x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)]:
        line(a, b, c, d, w=0.35, g=0.72, dash=2)

# ---------------------------------------------------------------- build
bits = old_id0_grid(HERE / "apriltags_letter.pdf")
try:
    ref = aruco_grid(0)
    turns = next((k for k in range(4) if bits == ref), None) if False else None
    g = ref
    for k in range(4):
        if g == bits:
            turns = k
            break
        g = rot90(g)
    if turns is None:
        sys.exit("the grid lifted from the old sheet is not tag36h11 id 0 in any rotation")
    print(f"cross-check: matches OpenCV tag36h11 id 0 at {turns * 90} deg -- pattern and "
          f"orientation preserved from the sheet already on the vane")
except ImportError:
    print("cross-check SKIPPED (no cv2 in this interpreter)")

y = 16
text(M, y, "AprilTag reprint  -  CYLINDER tag id 0 at 40.0 mm", 13, bold=True); y += 6.0
text(M, y, 'PRINT AT 100% / "Actual size".  Turn OFF "Fit to page" and "Shrink oversized pages" - they scale silently.', 7.4, g=0.15); y += 4.2
text(M, y, "Use a LASER printer. Inkjet black dye is often transparent at 850 nm - invisible to the D435 infrared imagers.", 7.4, g=0.15); y += 4.2
text(M, y, "Check the bar below with calipers BEFORE cutting, then measure a printed tag and pass THAT number as tag_size.", 7.4, g=0.15); y += 4.2
text(M, y, "Replaces the 30 mm vane tag, which stopped decoding partway through 5 of 12 runs while the 40 mm reference tag never did.", 7.4, g=0.15); y += 7.5

bar_x, bar_w, bar_h = M, 100.0, 3.2
rect(bar_x, y, bar_w, bar_h)
for i in range(11):
    tx = bar_x + i * 10.0
    line(tx, y - 1.6, tx, y + bar_h + 1.6, w=0.5, g=0.0)
    if i % 5 == 0:
        textc(tx, y + bar_h + 5.0, f"{i*10}", 6.5, g=0.2)
text(bar_x + bar_w + 7, y + bar_h - 0.4, "100.0 mm exactly. Short = your printer scaled it.", 7.2, bold=True)
y += bar_h + 9.0
line(M, y, 215.9 - M, y, w=0.6, g=0.35); y += 8.0

text(M, y, "Four identical copies - three are spares.", 9.0, bold=True)
text(M + 62, y, "Solid black square = 40.0 mm.  White patch incl. quiet zone = 60.0 mm.", 7.2, g=0.35)
y += 5.0
cell = SIZE_MM / CORE
patch = CORE * cell + 2 * QUIET_CELLS * cell
gap = 20.0
top = y
for row in range(2):
    px = M + (191.9 - (2 * patch + gap)) / 2.0
    for col in range(2):
        draw_tag(px, top, SIZE_MM, bits)
        cutmarks(px, top, patch)
        textc(px + patch / 2, top + patch + 6.0, "tag36h11  id 0   tag_size 40.0 mm", 7.2, bold=True, g=0.1)
        px += patch + gap
    top += patch + 14.0
y = top + 2

line(M, y - 4, 215.9 - M, y - 4, w=0.6, g=0.35)
text(M, y + 1.5, "MOUNTING.  Keep the tag centre at 71.0 mm along the shaft axis (CYL_TAG_AXIAL_MM) and the vane normal along the pinch axis, so", 7.0, g=0.15)
text(M, y + 5.7, "the turn stays an in-plane rotation.  Orientation matters as much as position: the detector's pose frame is fixed to the pattern, so mount", 7.0, g=0.15)
text(M, y + 9.9, "this square the same way up as the one it replaces.  The patch is 60 mm across, 15 mm wider than the 30 mm tag's - check vane clearance.", 7.0, g=0.15)
text(M, y + 14.1, "Then set CYL_TAG_SIZE_M = 0.040 in src/morphohand/bench/tags.py.  A tag printed at 40 mm and read as 30 mm misplaces the object by 33%.", 7.0, bold=True, g=0.0)

content = "\n".join(ops).encode()
comp = zlib.compress(content)
objs = [
    b"<< /Type /Catalog /Pages 2 0 R >>",
    b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PW:.4f} {PH:.4f}] /Resources "
    f"<< /Font << /F1 5 0 R /F2 6 0 R >> >> /Contents 4 0 R >>".encode(),
    b"<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(comp) + comp + b"\nendstream",
    b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
]
out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
offs = []
for i, o in enumerate(objs, 1):
    offs.append(len(out))
    out += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
xref = len(out)
out += f"xref\n0 {len(objs)+1}\n0000000000 65535 f \n".encode()
for o in offs:
    out += f"{o:010d} 00000 n \n".encode()
out += f"trailer\n<< /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
dest = HERE / "apriltags_id0_40mm.pdf"
dest.write_bytes(out)
print(f"content ends at y = {y+18:.1f} mm  (usable bottom = {279.4-M:.1f} mm)")
print(f"wrote {dest} ({len(out)} bytes)")
