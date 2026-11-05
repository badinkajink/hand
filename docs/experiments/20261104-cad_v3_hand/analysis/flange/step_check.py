#!/usr/bin/env python3
"""Independent reading of the STEP text (no FreeCAD): the cylinder axes of the UR5 adaptor and of the robot mount's
plate-side holes, in their own files and, through the assembly's NEXT_ASSEMBLY_USAGE_OCCURRENCE transforms, in the hand
file's frame A. Compared with parts.json (FreeCAD 0.19 / OCCT). Output: step_check.json.

    .venv/bin/python logs/20261104-cad_v3/flange/step_check.py
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path("/home/humanoid/Programs/hand")
SRC = REPO / "assets/cad_v2/full_cad"
OUT = Path(__file__).resolve().parent
PARTS = json.load(open(REPO / "logs/20261104-cad_v3/extract/parts.json"))["parts"]
REF = re.compile(r"#(\d+)")


def load(path):
    txt = open(path, encoding="latin-1").read()
    data = txt[txt.index("DATA;") + 5: txt.rindex("ENDSEC;")]
    data = re.sub(r"/\*.*?\*/", "", data, flags=re.S)
    ents = {}
    for st in re.split(r";\s*(?=#\d+\s*=)", data):
        m = re.match(r"\s*#(\d+)\s*=\s*(.*)", st, re.S)
        if not m:
            continue
        body = " ".join(m.group(2).split()).rstrip(";").strip()
        ents[int(m.group(1))] = body
    unit = "mm" if re.search(r"SI_UNIT\(\s*\.MILLI\.\s*,\s*\.METRE\.\s*\)", txt) else "m"
    return ents, (1.0 if unit == "mm" else 1000.0)


def etype(body):
    return body.split("(", 1)[0].strip() if not body.startswith("(") else "COMPLEX"


def nums(body):
    return [float(x) for x in re.findall(r"[-+]?\d*\.\d*(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?", body)
            if x not in ("", ".", "-", "+")]


class Step:
    def __init__(self, path):
        self.e, self.s = load(path)
        self.by_type = defaultdict(list)
        for k, b in self.e.items():
            self.by_type[etype(b)].append(k)

    def refs(self, k):
        return [int(x) for x in REF.findall(self.e[k])]

    def point(self, k):
        b = self.e[k]
        inner = b[b.index("(", b.index("(") + 1) + 1: b.rindex(")")]
        return np.array([float(x) for x in inner.replace(")", "").split(",")])

    def placement(self, k):
        """4x4 of an AXIS2_PLACEMENT_3D (mm)."""
        b = self.e[k]
        args = b[b.index("(") + 1: b.rindex(")")].split(",")
        p = self.point(int(args[1].strip()[1:])) * self.s
        z = self.point(int(args[2].strip()[1:])) if args[2].strip() != "$" else np.array([0, 0, 1.0])
        x = self.point(int(args[3].strip()[1:])) if args[3].strip() != "$" else np.array([1.0, 0, 0])
        z = z / np.linalg.norm(z)
        x = x - z * (x @ z)
        x /= np.linalg.norm(x)
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, np.cross(z, x), z, p
        return T

    def cylinders(self, brep):
        """(radius mm, axis 4x4) of every cylindrical face reachable from a MANIFOLD_SOLID_BREP."""
        shell = self.refs(brep)[0]
        out = []
        for f in self.refs(shell):
            if etype(self.e[f]) != "ADVANCED_FACE":
                continue
            surf = self.refs(f)[-1]
            b = self.e[surf]
            if etype(b) == "CYLINDRICAL_SURFACE":
                ax = self.refs(surf)[0]
                r = float(b[b.rindex(",") + 1: b.rindex(")")]) * self.s
                out.append((r, self.placement(ax)))
        return out

    def product_of_brep(self, brep):
        """The PRODUCT_DEFINITION whose shape holds the brep."""
        absr = [k for k in self.by_type["ADVANCED_BREP_SHAPE_REPRESENTATION"] if brep in self.refs(k)]
        reps = set(absr)
        for k in self.by_type["SHAPE_REPRESENTATION_RELATIONSHIP"]:
            r = self.refs(k)
            if any(a in reps for a in r):
                reps.update(r)
        for k in self.by_type["SHAPE_DEFINITION_REPRESENTATION"]:
            pds, rep = self.refs(k)[:2]
            if rep in reps:
                return self.refs(pds)[0], reps
        raise KeyError(brep)

    def product_name(self, pd):
        pdf = self.refs(pd)[0]
        prod = self.refs(pdf)[0]
        return self.e[prod].split("'")[1]

    def instance_chain(self, pd):
        """Transforms from the part's representation to the root's, one per use of the part: list of 4x4 (two
        conventions: A(item2) A(item1)^-1 and its inverse) and the chain of product names."""
        out = []
        nauos = [k for k in self.by_type["NEXT_ASSEMBLY_USAGE_OCCURRENCE"] if self.refs(k)[1] == pd]
        if not nauos:
            return [(np.eye(4), np.eye(4), [])]
        for n in nauos:
            parent = self.refs(n)[0]
            pds = [k for k in self.by_type["PRODUCT_DEFINITION_SHAPE"] if self.refs(k)[0] == n][0]
            cdsr = [k for k in self.by_type["CONTEXT_DEPENDENT_SHAPE_REPRESENTATION"] if self.refs(k)[1] == pds][0]
            rel = self.refs(cdsr)[0]
            body = self.e[rel]
            idt = int(re.search(r"REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION\(\s*#(\d+)", body).group(1))
            a1, a2 = self.refs(idt)[:2]
            Ta = self.placement(a2) @ np.linalg.inv(self.placement(a1))
            for T1, T2, names in self.instance_chain(parent):
                out.append((T1 @ Ta, T2 @ np.linalg.inv(Ta), [self.product_name(parent)] + names))
        return out


def match(c_list, r):
    return [T for rr, T in c_list if abs(rr - r) < 1e-6]


def axis_points(Ts):
    """Distinct axis lines (as their point in the z = const plane of the first axis' start) -> (n, 3) points and the
    axis directions."""
    pts, dirs = [], []
    for T in Ts:
        p, d = T[:3, 3], T[:3, 2]
        if any(np.linalg.norm(np.cross(p - q, d)) < 1e-6 for q in pts):
            continue
        pts.append(p)
        dirs.append(d)
    return np.array(pts), np.array(dirs)


def same_lines(P1, D1, P2, D2):
    """Largest distance between each line of set 1 and the nearest line of set 2 (mm), and the largest angle (deg)."""
    dmax, amax = 0.0, 0.0
    for p, d in zip(P1, D1):
        best = min(range(len(P2)), key=lambda j: np.linalg.norm(np.cross(P2[j] - p, D2[j])))
        q, e = P2[best], D2[best]
        dmax = max(dmax, float(np.linalg.norm(np.cross(q - p, e))))
        amax = max(amax, float(np.degrees(np.arccos(min(1.0, abs(d @ e))))))
    return dmax, amax


def fc_lines(part, r):
    P, D = [], []
    for c in part["cylinders"]:
        if abs(c["r"] - r) < 1e-6:
            P.append(np.array(c["point"]))
            D.append(np.array(c["axis"]))
    return np.array(P), np.array(D)


def main():
    res = {}
    fc = {(p["file"], p["name"]): p for p in PARTS}
    # 1. standalone files
    for key, fname, radii in (("adaptor", "UR5 Adaptor.step", (3.2, 5.5, 2.2, 4.3)),
                              ("robot_mount", "robot_mount.step", (2.2, 4.25)),
                              ("base_plate", "base_plate.step", ())):
        S = Step(SRC / fname)
        breps = S.by_type["MANIFOLD_SOLID_BREP"]
        cyl = S.cylinders(breps[0])
        row = dict(n_breps=len(breps), units_scale_to_mm=S.s, n_cylinders=len(cyl),
                   radius_census=dict(sorted(Counter(round(r, 3) for r, _ in cyl).items())))
        for r in radii:
            P, D = axis_points(match(cyl, r))
            P2, D2 = fc_lines(fc[(key, "000")], r)
            dm, am = same_lines(P, D, P2, D2)
            row[f"r{r}"] = dict(n_axes=len(P), axis_points_mm=np.round(P, 4).tolist(), vs_freecad_max_mm=dm,
                                vs_freecad_max_deg=am)
        res[key] = row
        print(key, {k: v for k, v in row.items() if not k.startswith("r") or k == "radius_census"})
        for r in radii:
            print("   r", r, "axes", len(row[f"r{r}"]["axis_points_mm"]), "vs FreeCAD max line distance",
                  f'{row[f"r{r}"]["vs_freecad_max_mm"]:.2e}', "mm, angle", f'{row[f"r{r}"]["vs_freecad_max_deg"]:.2e}')
    # 2. the hand assembly
    H = Step(SRC / "assemblies/hand v3 (anthro config).step")
    found = {}
    for b in H.by_type["MANIFOLD_SOLID_BREP"]:
        cyl = H.cylinders(b)
        cen = Counter(round(r, 3) for r, _ in cyl)
        if cen.get(3.2, 0) >= 4 and cen.get(5.5, 0) >= 4 and cen.get(4.3, 0) >= 4:
            found.setdefault("adaptor", []).append((b, cyl))
        if cen.get(45.0, 0) >= 1 and cen.get(4.25, 0) >= 4:
            found.setdefault("robot_mount", []).append((b, cyl))
    hand = {}
    for key, hits in found.items():
        assert len(hits) == 1, (key, len(hits))
        b, cyl = hits[0]
        pd, _ = H.product_of_brep(b)
        chains = H.instance_chain(pd)
        fcp = fc[("hand", "004" if key == "adaptor" else "001")]
        rows = []
        for T1, T2, names in chains:
            for conv, T in (("A(item2) A(item1)^-1", T1), ("inverse", T2)):
                out = {}
                for r in ((3.2, 5.5, 2.2) if key == "adaptor" else (2.2, 4.25)):
                    P, D = axis_points(match(cyl, r))
                    PA = P @ T[:3, :3].T + T[:3, 3]
                    DA = D @ T[:3, :3].T
                    P2, D2 = fc_lines(fcp, r)
                    dm, am = same_lines(PA, DA, P2, D2)
                    out[f"r{r}"] = dict(max_line_distance_mm=dm, max_angle_deg=am,
                                        axis_points_A_mm=np.round(PA, 4).tolist())
                rows.append(dict(convention=conv, chain=names, product=H.product_name(pd),
                                 T_A_from_part=np.round(T, 6).tolist(), checks=out))
                print(key, conv, "chain", names, "product", H.product_name(pd),
                      {k: (f'{v["max_line_distance_mm"]:.2e}', f'{v["max_angle_deg"]:.2e}') for k, v in out.items()})
        hand[key] = rows
    res["hand"] = hand
    json.dump(res, open(OUT / "step_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
