"""Minimal ISO 10303-21 reader for the flange check (written for this check; no FreeCAD, no other agent's code).

Reads the DATA section into {id: entity}, where a simple entity is (TYPE, params) and a complex entity is
{"__complex__": {TYPE: params, ...}}. Gives assembly transforms (ITEM_DEFINED_TRANSFORMATION composed as
A(item_2) A(item_1)^-1), cylinders, planes and vertices of each solid, in millimetres.
"""
import re
import numpy as np

_tok = re.compile(r"""\s*(?:
    (?P<str>'(?:[^']|'')*')
  | (?P<ref>\#\d+)
  | (?P<enum>\.[A-Z_0-9]+\.)
  | (?P<num>[+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[Ee][+-]?\d+)?)
  | (?P<kw>[A-Z_][A-Z_0-9]*)
  | (?P<sym>[(),$*])
)""", re.X)


def _tokens(s):
    pos = 0
    out = []
    while pos < len(s):
        m = _tok.match(s, pos)
        if not m:
            if s[pos:].strip() == "":
                break
            raise ValueError("bad token at %r" % s[pos:pos + 40])
        pos = m.end()
        kind = m.lastgroup
        out.append((kind, m.group(kind)))
    return out


def _parse_list(toks, i):
    # toks[i] is '('
    assert toks[i] == ("sym", "("), toks[i]
    i += 1
    items = []
    while True:
        k, v = toks[i]
        if k == "sym" and v == ")":
            return items, i + 1
        if k == "sym" and v == ",":
            i += 1
            continue
        if k == "sym" and v == "(":
            sub, i = _parse_list(toks, i)
            items.append(sub)
            continue
        if k == "kw":
            # typed parameter, e.g. LENGTH_MEASURE(2.E-5)
            sub, i = _parse_list(toks, i + 1)
            items.append((v, sub))
            continue
        if k == "num":
            items.append(float(v))
        elif k == "ref":
            items.append(int(v[1:]))
        elif k == "str":
            items.append(v[1:-1].replace("''", "'"))
        elif k == "enum":
            items.append(v)
        elif k == "sym" and v in "$*":
            items.append(None)
        else:
            raise ValueError((k, v))
        i += 1


def load(path):
    txt = open(path, encoding="latin-1").read()
    data = txt.split("DATA;", 1)[1].rsplit("ENDSEC;", 1)[0]
    # strip comments
    data = re.sub(r"/\*.*?\*/", "", data, flags=re.S)
    ents = {}
    # split on ';' outside strings
    buf = []
    inq = False
    for ch in data:
        if ch == "'":
            inq = not inq
        if ch == ";" and not inq:
            stmt = "".join(buf).strip()
            buf = []
            if stmt:
                m = re.match(r"#(\d+)\s*=\s*(.*)$", stmt, re.S)
                if not m:
                    continue
                eid = int(m.group(1))
                body = m.group(2).strip()
                toks = _tokens(body)
                if toks[0] == ("sym", "("):
                    # complex entity: (A(...)B(...)...)
                    comp = {}
                    i = 1
                    while toks[i] != ("sym", ")"):
                        name = toks[i][1]
                        params, i = _parse_list(toks, i + 1)
                        comp[name] = params
                    ents[eid] = ("__complex__", comp)
                else:
                    name = toks[0][1]
                    params, _ = _parse_list(toks, 1)
                    ents[eid] = (name, params)
        else:
            buf.append(ch)
    return ents


def unit_scale_to_mm(ents):
    """Return the factor from file length units to mm (looks for SI_UNIT METRE with or without prefix)."""
    for eid, (t, p) in ents.items():
        if t == "__complex__" and "LENGTH_UNIT" in p and "SI_UNIT" in p:
            prefix, name = p["SI_UNIT"]
            if name == ".METRE.":
                return {None: 1000.0, ".MILLI.": 1.0, ".CENTI.": 10.0}[prefix]
    raise ValueError("no length unit")


class Step:
    def __init__(self, path):
        self.e = load(path)
        self.s = unit_scale_to_mm(self.e)

    def t(self, i):
        return self.e[i][0]

    def p(self, i):
        return self.e[i][1]

    def point(self, i):
        t, p = self.e[i]
        assert t == "CARTESIAN_POINT", t
        return np.array(p[1], float) * self.s

    def direction(self, i):
        t, p = self.e[i]
        assert t == "DIRECTION", t
        v = np.array(p[1], float)
        return v / np.linalg.norm(v)

    def placement(self, i):
        """AXIS2_PLACEMENT_3D -> 4x4 (mm), columns x, y, z, origin."""
        t, p = self.e[i]
        assert t == "AXIS2_PLACEMENT_3D", t
        o = self.point(p[1])
        z = self.direction(p[2]) if p[2] is not None else np.array([0, 0, 1.0])
        xr = self.direction(p[3]) if p[3] is not None else np.array([1.0, 0, 0])
        x = xr - np.dot(xr, z) * z
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, o
        return T

    # ---- assembly structure ----
    def occurrences(self):
        """List of (parent_rep, child_rep, T_parent_from_child, nauo_id) from CDSR entities."""
        out = []
        for eid, (t, p) in self.e.items():
            if t != "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION":
                continue
            rr, pds = p
            tt, comp = self.e[rr]
            assert tt == "__complex__"
            rep1, rep2 = comp["REPRESENTATION_RELATIONSHIP"][2], comp["REPRESENTATION_RELATIONSHIP"][3]
            idt = comp["REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION"][0]
            it, ip = self.e[idt]
            assert it == "ITEM_DEFINED_TRANSFORMATION"
            A1, A2 = self.placement(ip[2]), self.placement(ip[3])
            T = A2 @ np.linalg.inv(A1)
            nauo = self.p(pds)[2]
            out.append((rep2, rep1, T, nauo))
        return out

    def rep_solids(self):
        """{shape_rep_id: [brep solid ids]} via SHAPE_REPRESENTATION_RELATIONSHIP to ADVANCED_BREP_SHAPE_REPRESENTATION."""
        out = {}
        for eid, (t, p) in self.e.items():
            if t == "SHAPE_REPRESENTATION_RELATIONSHIP":
                a, b = p[2], p[3]
                for sr, br in ((a, b), (b, a)):
                    if self.t(br) == "ADVANCED_BREP_SHAPE_REPRESENTATION" and self.t(sr) == "SHAPE_REPRESENTATION":
                        items = [i for i in self.p(br)[1] if self.t(i) == "MANIFOLD_SOLID_BREP"]
                        out.setdefault(sr, []).extend(items)
        return out

    def rep_name(self, rep):
        return self.p(rep)[0]

    def world_transforms(self):
        """Every leaf solid with its transform into the root representation: list of (solid, T, chain)."""
        occ = self.occurrences()
        children = {}
        parents = set()
        childs = set()
        for par, ch, T, nauo in occ:
            children.setdefault(par, []).append((ch, T, nauo))
            parents.add(par)
            childs.add(ch)
        roots = [r for r in parents if r not in childs]
        solids = self.rep_solids()
        out = []

        def walk(rep, T, chain):
            for s in solids.get(rep, []):
                out.append((s, T, chain))
            for ch, Tc, nauo in children.get(rep, []):
                walk(ch, T @ Tc, chain + [(self.rep_name(ch), nauo)])

        if not roots:
            # single part file
            for rep, ss in solids.items():
                for s in ss:
                    out.append((s, np.eye(4), []))
        for r in roots:
            walk(r, np.eye(4), [(self.rep_name(r), None)])
        return out

    # ---- faces ----
    def faces(self, solid):
        t, p = self.e[solid]
        assert t == "MANIFOLD_SOLID_BREP"
        shell = p[1]
        return self.p(shell)[1]

    def face_vertices(self, face):
        t, p = self.e[face]
        assert t == "ADVANCED_FACE", t
        pts = []
        for b in p[1]:
            loop = self.p(b)[1]
            if self.t(loop) == "EDGE_LOOP":
                for oe in self.p(loop)[1]:
                    ec = self.p(oe)[3]
                    v1, v2 = self.p(ec)[1], self.p(ec)[2]
                    for v in (v1, v2):
                        pts.append(self.point(self.p(v)[1]))
                        if self.t(self.p(ec)[3]) == "CIRCLE":
                            pass
            elif self.t(loop) == "VERTEX_LOOP":
                pts.append(self.point(self.p(self.p(loop)[1])[1]))
        return np.array(pts)

    def face_circles(self, face):
        """Circles bounding a face: (centre, normal, radius) in mm."""
        t, p = self.e[face]
        out = []
        for b in p[1]:
            loop = self.p(b)[1]
            if self.t(loop) != "EDGE_LOOP":
                continue
            for oe in self.p(loop)[1]:
                ec = self.p(oe)[3]
                cv = self.p(ec)[3]
                if self.t(cv) == "CIRCLE":
                    T = self.placement(self.p(cv)[1])
                    out.append((T[:3, 3], T[:3, 2], self.p(cv)[2] * self.s))
        return out

    def surface(self, face):
        t, p = self.e[face]
        sid = p[2]
        st, sp = self.e[sid]
        same_sense = p[3]
        if st == "CYLINDRICAL_SURFACE":
            T = self.placement(sp[1])
            return dict(kind="cyl", origin=T[:3, 3], axis=T[:3, 2], r=sp[2] * self.s, same_sense=same_sense)
        if st == "PLANE":
            T = self.placement(sp[1])
            return dict(kind="plane", origin=T[:3, 3], normal=T[:3, 2], same_sense=same_sense)
        if st == "CONICAL_SURFACE":
            T = self.placement(sp[1])
            return dict(kind="cone", origin=T[:3, 3], axis=T[:3, 2], r=sp[2] * self.s, semi_angle=sp[3],
                        same_sense=same_sense)
        return dict(kind=st, same_sense=same_sense)
