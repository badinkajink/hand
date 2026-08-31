import zlib
from PIL import Image

MM = 72.0/25.4
PW, PH = 215.9*MM, 279.4*MM          # US Letter
M    = 12.0                           # margin, mm

def Y(mm): return PH - mm*MM          # mm-from-top -> PDF y
def X(mm): return mm*MM

ops = []
def rect(x,y,w,h,fill=(0,0,0)):
    ops.append(f"{fill[0]} {fill[1]} {fill[2]} rg {X(x):.4f} {Y(y+h):.4f} {w*MM:.4f} {h*MM:.4f} re f")
def line(x1,y1,x2,y2,w=0.4,g=0.65,dash=None):
    d = f"[{dash} {dash}] 0 d " if dash else "[] 0 d "
    ops.append(f"{g} {g} {g} RG {w} w {d}{X(x1):.4f} {Y(y1):.4f} m {X(x2):.4f} {Y(y2):.4f} l S")
def esc(s): return s.replace('\\','\\\\').replace('(','\\(').replace(')','\\)')
def text(x,y,s,size=8,bold=False,g=0.0):
    f = "/F2" if bold else "/F1"
    ops.append(f"BT {g} {g} {g} rg {f} {size} Tf {X(x):.4f} {Y(y):.4f} Td ({esc(s)}) Tj ET")
def textc(xc,y,s,size=8,bold=False,g=0.0):
    w = len(s)*size*(0.60 if bold else 0.55)/MM
    text(xc-w/2, y, s, size, bold, g)

def load_bits(path, inset):
    """Return the rendered cell grid as a bool grid (True=black)."""
    im = Image.open(path).convert('L'); n = im.size[0]
    return [[im.getpixel((x,y)) < 128 for x in range(inset, n-inset)]
            for y in range(inset, n-inset)]

def draw_tag(px, py, s, bits, core, quiet_cells=2.0):
    """px,py = top-left of the QUIET-ZONE patch.  s = BLACK-SQUARE edge (mm).
    core = how many cells the black square spans; len(bits) may exceed it
    (tagStandard41h12 carries data in the ring OUTSIDE its black border)."""
    n = len(bits); cell = s/core; q = quiet_cells*cell
    patch = n*cell + 2*q
    rect(px, py, patch, patch, fill=(1,1,1))          # explicit white quiet zone
    ox, oy = px+q, py+q
    for r,row in enumerate(bits):                      # merge horizontal runs
        c = 0
        while c < n:
            if row[c]:
                c2 = c
                while c2+1 < n and row[c2+1]: c2 += 1
                rect(ox+c*cell, oy+r*cell, (c2-c+1)*cell, cell)
                c = c2+1
            else: c += 1
    return patch

def cutmarks(px,py,patch,pad=2.0):
    x0,y0,x1,y1 = px-pad, py-pad, px+patch+pad, py+patch+pad
    for (a,b,c,d) in [(x0,y0,x1,y0),(x1,y0,x1,y1),(x1,y1,x0,y1),(x0,y1,x0,y0)]:
        line(a,b,c,d,w=0.35,g=0.72,dash=2)

# ---------------------------------------------------------------- header
y = 16
text(M, y, "AprilTag sheet  -  cylinder reorientation rig", 13, bold=True); y += 6.0
text(M, y, "PRINT AT 100% / \"Actual size\".  Turn OFF \"Fit to page\" and \"Shrink oversized pages\" - they scale silently.", 7.4, g=0.15); y += 4.2
text(M, y, "Every tag is labelled with the number to pass as tag_size - the distance between the corners the detector returns.", 7.4, g=0.15); y += 4.2
text(M, y, "Verify against the bar below with calipers, then measure a printed tag and pass THAT number, not the nominal.", 7.4, g=0.15); y += 4.2
text(M, y, "Use a LASER printer. Inkjet black dye is often transparent at 850 nm - invisible to the D435 infrared imagers.", 7.4, g=0.15); y += 7.5

# ---------------------------------------------------------------- scale bar
bar_x, bar_w, bar_h = M, 100.0, 3.2
rect(bar_x, y, bar_w, bar_h)
for i in range(0, 11):
    tx = bar_x + i*10.0
    line(tx, y-1.6, tx, y+bar_h+1.6, w=0.5, g=0.0)
    if i % 5 == 0: textc(tx, y+bar_h+5.0, f"{i*10}", 6.5, g=0.2)
text(bar_x+bar_w+7, y+bar_h-0.4, "100.0 mm exactly. Short = your printer scaled it.", 7.2, bold=True)
y += bar_h + 9.0
line(M, y, 215.9-M, y, w=0.6, g=0.35); y += 7.5

FAM36 = [(i, load_bits(f"tag36_11_{i:05d}.png", 1)) for i in range(8)]
FAM41 = [(i, load_bits(f"std41_{i:05d}.png", 0)) for i in range(2)]

def band(y, title, sub, items, gap=12.0):
    text(M, y, title, 9.0, bold=True)
    tw = len(title)*9.0*0.60/MM
    text(M+tw+4, y, sub, 7.2, g=0.35)
    y += 4.0
    patches = [(len(b)/core + 2*2.0/core)*s for (fam, s, _, b, core) in items]
    total = sum(patches) + gap*(len(items)-1)
    px = M + (191.9-total)/2.0
    top = y
    hmax = 0
    for (fam, s, tid, bits, core), patch in zip(items, patches):
        p = draw_tag(px, top, s, bits, core)
        cutmarks(px, top, p)
        lbl = (f"tag36h11  id {tid}   tag_size {s:.1f} mm" if fam == 36
               else f"tagStandard41h12  id {tid}   tag_size {s*5.0/7.0:.1f} mm")
        textc(px+p/2, top+p+6.0, lbl, 7.2, bold=True, g=0.1)
        px += patch + gap
        hmax = max(hmax, patch)
    return top + hmax + 12.0

y = band(y, "A - CYLINDER, 30.0 mm", "recommended: the size you asked for",
         [(36, 30.0, FAM36[i][0], FAM36[i][1], 8) for i in (0,1,2)])
y += 2
y = band(y, "B - CYLINDER, 20.0 mm", "fallback if the mount plate ends up tight",
         [(36, 20.0, FAM36[i][0], FAM36[i][1], 8) for i in (3,4,5)], gap=26.0)
y += 2
y = band(y, "C - STATIC REFERENCE 40.0 mm  +  tagStandard41h12", "left: fix to the fixture, it never moves.  right: needs pupil-apriltags",
         [(36, 40.0, FAM36[6][0], FAM36[6][1], 8),
          (41, 30.0, FAM41[0][0], FAM41[0][1], 7),
          (41, 30.0, FAM41[1][0], FAM41[1][1], 7)], gap=8.0)

line(M, y-4, 215.9-M, y-4, w=0.6, g=0.35)
text(M, y+1.5, "tag36h11: tag_size IS the black square you can see and measure (8 cells).  tagStandard41h12: it is NOT.  That family has no quiet zone - its outer", 7.0, g=0.15)
text(M, y+5.7, "ring carries data - so there is no clean outer quad and the detector returns the INNER edge of the black ring, 5 cells across, not the 7-cell square.", 7.0, g=0.15)
text(M, y+9.9, "Measuring its black square and passing that inflates every distance by 40%.  Both families above were decode-verified at the labelled tag_size.", 7.0, g=0.15)

# ---------------------------------------------------------------- emit PDF
content = "\n".join(ops).encode()
comp = zlib.compress(content)
objs = [
 b"<< /Type /Catalog /Pages 2 0 R >>",
 b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
 f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PW:.4f} {PH:.4f}] /Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> /Contents 4 0 R >>".encode(),
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
for o in offs: out += f"{o:010d} 00000 n \n".encode()
out += f"trailer\n<< /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
open("apriltags_letter.pdf","wb").write(out)
print(f"content ends at y = {y:.1f} mm  (usable bottom = {279.4-M:.1f} mm)")
print("wrote apriltags_letter.pdf", len(out), "bytes")
