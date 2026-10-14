#!/usr/bin/env python3
"""MuJoCo viewer for static result pages: MuJoCo 3.14.0 (WebAssembly) and three.js r147, inlined.

    from mjwasm_embed import embed
    html = embed("assets/mjcf/real_v1/real_hand.xml", element_id="hand", height_px=480, title="real_v1 hand")

embed() returns an HTML fragment (a div and its scripts) that a page builder drops into a template. The fragment
carries the MJCF and every file it references (meshes, textures, height fields, skins, included XML, sub-models),
each gzip-compressed and base64-encoded. In the browser the files are written into the Emscripten MEMFS under
/mjwasm/<element_id>/ and the model is compiled with MjModel.from_xml_path. The page opens from file:// with no
network and no server; nothing is fetched.

Shared runtime. The first fragment on a page also carries the runtime: the WebAssembly binary and its glue, three.js,
OrbitControls, the viewer code and its style (3.7 MB as inlined). Every further viewer on the same page passes
include_runtime=False, or the builder places runtime() once anywhere on the page and passes include_runtime=False to
all of them. All viewers share one WebAssembly module and each compiles its own MjModel and MjData; the order of the
runtime and the viewer fragments on the page does not matter.

Viewer. Orbit (drag), pan (right drag), zoom (wheel, once the view has been clicked, so the wheel scrolls the page
until then). One slider per hinge (degrees) and slide joint (mm), bounded by
the joint range (+-180 deg, or +-half the model extent, for an unlimited joint). Paused: a slider sets qpos and runs
mj_forward. Play: the physics steps in real time; a slider whose joint drives a position actuator sets that
actuator's ctrl (gear x target), any other slider sets qpos directly. Buttons for Play/Pause, Reset (mj_resetData),
each keyframe (mj_resetDataKeyframe) and the home view; checkboxes for geom groups 0-5 (MuJoCo's default shows 0-2),
contact points and sites. Plane, sphere, capsule, ellipsoid, cylinder, box and mesh geoms are drawn with MuJoCo's
rgba and material rules; a plane with a 2D texture (builtin checker, gradient) shows it. Height fields, SDF geoms,
flexes, skins, tendons and the skybox are not drawn.

Vendor files are pinned by version and sha256 and cached in ~/.cache/mjwasm/: the @mujoco/mujoco 3.14.0 and
three 0.147.0 npm tarballs from registry.npmjs.org (the MuJoCo files are copied from
~/.cache/contact-slides/mujoco-3.14.0/ when they are there), and MuJoCo's LICENSE from its GitHub tag.

    python scripts/mjwasm_embed.py MODEL.xml [MODEL.xml ...] -o page.html [--title T] [--height 480]
    python scripts/mjwasm_embed.py MODEL.xml --sizes          # what each inlined asset adds to a page
    python scripts/mjwasm_embed.py --vendor                   # fetch and verify the vendor cache only

Each viewer writes its state into data-mjwasm-status on its div ("loading", "ok", or "error: <MuJoCo's message>")
and a JSON summary into data-mjwasm-info; window.MJWASM.viewers holds the viewer objects (play, pause, reset, key,
setJoint, setGroup, setContacts, step) for scripted checks.
"""
from __future__ import annotations

import argparse
import base64
import functools
import gzip
import hashlib
import html
import io
import json
import os
import re
import sys
import tarfile
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

MUJOCO_VERSION = "3.14.0"
THREE_VERSION = "0.147.0"
RUNTIME_VERSION = "2026.10.13"

CACHE = Path(os.environ.get("MJWASM_CACHE", Path.home() / ".cache" / "mjwasm"))
CONTACT_SLIDES_CACHE = Path.home() / ".cache" / "contact-slides" / f"mujoco-{MUJOCO_VERSION}"

# npm tarballs; the sha512 "integrity" values the registry lists for them match these files.
TARBALLS = {
    "mujoco": (f"https://registry.npmjs.org/@mujoco/mujoco/-/mujoco-{MUJOCO_VERSION}.tgz",
               "f845a07ab81303cc6579fec07136818bed21d44a31c1911340f61e06f4d11eee"),
    "three": (f"https://registry.npmjs.org/three/-/three-{THREE_VERSION}.tgz",
              "03b50c0c820f7ae92010ce464200904c92214938ea7f751bd0e96e0499330988"),
}

# name in the cache -> (package, tarball member or URL, sha256 of the file)
VENDOR = {
    "mujoco.wasm": ("mujoco", "package/mujoco.wasm",
                    "2d380f1070a556c3003b4a7c7c10c98a5f61db4e91815de1075ec478e9487e76"),
    "mujoco.js": ("mujoco", "package/mujoco.js",
                  "bc6980e4bc5ee5f809a6cabde0743798637e113d333f9e85124714d42be1ccae"),
    "LICENSE": ("mujoco", f"https://raw.githubusercontent.com/google-deepmind/mujoco/{MUJOCO_VERSION}/LICENSE",
                "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"),
    "three.min.js": ("three", "package/build/three.min.js",
                     "f34446bf875b5fb0dcd93819ffe1d9e182d46634ee855f5d904c6c4ac7cdbc95"),
    "OrbitControls.js": ("three", "package/examples/js/controls/OrbitControls.js",
                         "b4c6e53f98538535b11fd6655627d47fb5877b3ad972568bff0a7026c4b6d5c4"),
    "three.LICENSE": ("three", "package/LICENSE",
                      "fbf3943930dacbf56aabb8dc5d816440bd16aa6f4cc78ddbdf12106ee1807832"),
}
PKG_VERSION = {"mujoco": MUJOCO_VERSION, "three": THREE_VERSION}

ELEMENT_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}")


# ------------------------------------------------------------------------------------------------ vendor cache
def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def _download(url: str, sha256: str) -> bytes:
    print(f"mjwasm_embed: downloading {url}", file=sys.stderr)
    req = urllib.request.Request(url, headers={"User-Agent": "mjwasm_embed"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    if _sha(data) != sha256:
        raise RuntimeError(f"sha256 mismatch for {url}: got {_sha(data)}, pinned {sha256}")
    return data


@functools.lru_cache(maxsize=None)
def _tarball(pkg: str) -> bytes:
    url, sha256 = TARBALLS[pkg]
    return _download(url, sha256)


def _member(pkg: str, member: str) -> bytes:
    with tarfile.open(fileobj=io.BytesIO(_tarball(pkg)), mode="r:gz") as tf:
        info = tf.getmember(member)          # read in memory: nothing from the tarball touches the disk
        if not info.isfile():
            raise RuntimeError(f"{member} in the {pkg} tarball is not a regular file")
        return tf.extractfile(info).read()


@functools.lru_cache(maxsize=None)
def vendor() -> dict[str, Path]:
    """Make sure every pinned vendor file is in ~/.cache/mjwasm/ with its pinned sha256; return name -> path."""
    out = {}
    for name, (pkg, src, sha256) in VENDOR.items():
        path = CACHE / f"{pkg}-{PKG_VERSION[pkg]}" / name
        if path.is_file() and _sha(path.read_bytes()) == sha256:
            out[name] = path
            continue
        data = None
        legacy = CONTACT_SLIDES_CACHE / name
        if pkg == "mujoco" and legacy.is_file() and _sha(legacy.read_bytes()) == sha256:
            data = legacy.read_bytes()
        elif src.startswith("https://"):
            data = _download(src, sha256)
        else:
            data = _member(pkg, src)
            if _sha(data) != sha256:
                raise RuntimeError(f"sha256 mismatch for {src} in the {pkg} tarball")
        _write_atomic(path, data)
        out[name] = path
    return out


def _gz_b64(data: bytes) -> str:
    """gzip level 9 with mtime 0 (byte-for-byte reproducible), then base64."""
    return base64.b64encode(gzip.compress(data, compresslevel=9, mtime=0)).decode("ascii")


def _gz_b64_cached(data: bytes) -> str:
    path = CACHE / "encoded" / f"{_sha(data)}.gz.b64"
    if path.is_file():
        return path.read_text("ascii")
    s = _gz_b64(data)
    _write_atomic(path, s.encode("ascii"))
    return s


def _glue(src: str) -> str:
    """Upstream mujoco.js is an ES module; the runtime runs it through new Function (no module loader, no fetch).

    Removes `export default loadMujoco;` and replaces import.meta.url (used only to locate mujoco.wasm, which the
    runtime passes in as wasmBinary) by the page URL."""
    if src.count("export default loadMujoco;") != 1:
        raise RuntimeError("mujoco.js: expected exactly one 'export default loadMujoco;'")
    body = src.replace("export default loadMujoco;", "").replace("import.meta.url", "__mjwasmBase")
    if "import.meta" in body or re.search(r"^\s*(export|import)\s", body, re.M):
        raise RuntimeError("mujoco.js still has module syntax after the rewrite; inspect it by hand")
    head = ('var __mjwasmBase = (typeof location !== "undefined" && /^(https?|file):/.test(location.href))'
            ' ? location.href : "http://localhost/";\n')
    return head + body


def _script_safe(text: str, what: str) -> str:
    """Text inside a <script> element must not contain </script or <!-- (HTML script-data rules)."""
    low = text.lower()
    if "</script" in low or "<!--" in low or "<script" in low:
        raise RuntimeError(f"{what} contains a <script or <!-- sequence and cannot be inlined as script text")
    return text


def _json_script(obj) -> str:
    s = json.dumps(obj, separators=(",", ":"), ensure_ascii=True)
    return s.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


# ------------------------------------------------------------------------------------------------ the runtime
@dataclass
class Row:
    asset: str
    raw: int          # bytes of the file itself
    inlined: int      # bytes it adds to the page as inlined (base64 of gzip, or plain text)
    how: str


@functools.lru_cache(maxsize=None)
def _runtime() -> tuple[str, tuple[Row, ...]]:
    v = vendor()
    wasm = v["mujoco.wasm"].read_bytes()
    glue = _glue(v["mujoco.js"].read_text("utf-8")).encode("utf-8")
    three = v["three.min.js"].read_bytes()
    orbit = v["OrbitControls.js"].read_bytes()
    lic_mj = _script_safe(v["LICENSE"].read_text("utf-8"), "MuJoCo LICENSE")
    lic_three = _script_safe(v["three.LICENSE"].read_text("utf-8"), "three.js LICENSE")
    js = _script_safe(_VIEWER_JS.replace("__VERSIONS__", json.dumps(
        {"runtime": RUNTIME_VERSION, "mujoco": MUJOCO_VERSION, "three": THREE_VERSION})), "viewer JS")
    css = _VIEWER_CSS

    parts, rows = [], []
    parts.append(f"<!-- mjwasm runtime {RUNTIME_VERSION} (scripts/mjwasm_embed.py): MuJoCo {MUJOCO_VERSION} "
                 f"WebAssembly (Google DeepMind, Apache-2.0, npm @mujoco/mujoco), three.js r{THREE_VERSION.split('.')[1]} "
                 f"with OrbitControls (three.js authors, MIT, npm three@{THREE_VERSION}); license texts in "
                 f"#mjwasm-license-mujoco and #mjwasm-license-three -->")
    parts.append(f'<style id="mjwasm-style">{css}</style>')
    rows.append(Row("viewer style (CSS)", len(css.encode()), len(css.encode()), "plain"))
    for pid, data, label in (("mjwasm-wasm", wasm, f"mujoco.wasm {MUJOCO_VERSION}"),
                             ("mjwasm-glue", glue, f"mujoco.js {MUJOCO_VERSION} (WebAssembly glue)"),
                             ("mjwasm-three", three, f"three.min.js {THREE_VERSION}"),
                             ("mjwasm-orbit", orbit, f"OrbitControls.js {THREE_VERSION}")):
        b64 = _gz_b64_cached(data)
        parts.append(f'<script type="application/octet-stream" id="{pid}" data-encoding="gzip+base64" '
                     f'data-bytes="{len(data)}" data-sha256="{_sha(data)}">{b64}</script>')
        rows.append(Row(label, len(data), len(b64), "gzip+base64"))
    for pid, text, label in (("mjwasm-license-mujoco", lic_mj, "MuJoCo LICENSE (Apache-2.0)"),
                             ("mjwasm-license-three", lic_three, "three.js LICENSE (MIT)")):
        parts.append(f'<script type="text/plain" id="{pid}">\n{text}</script>')
        rows.append(Row(label, len(text.encode()), len(text.encode()), "plain"))
    parts.append(f'<script id="mjwasm-runtime">{js}</script>')
    rows.append(Row("viewer runtime (JS)", len(js.encode()), len(js.encode()), "plain"))
    return "\n".join(parts) + "\n", tuple(rows)


def runtime() -> str:
    """The shared runtime block. Place it once per page (anywhere) when the viewers use include_runtime=False."""
    return _runtime()[0]


def runtime_inventory() -> list[Row]:
    return list(_runtime()[1])


# ------------------------------------------------------------------------------------------------ model files
# Elements whose attributes name files, and which compiler directory MuJoCo resolves them against.
_FILE_ATTRS = {
    "mesh": ("mesh", ("file",)),
    "skin": ("mesh", ("file",)),
    "hfield": ("mesh", ("file",)),
    "flexcomp": ("mesh", ("file",)),
    "texture": ("texture", ("file", "fileright", "fileleft", "fileup", "filedown", "filefront", "fileback")),
    "model": ("model", ("file",)),
}


@dataclass
class Bundle:
    """A model and every file it references, laid out for MEMFS under /mjwasm/<element_id>/."""
    root: str
    main: str = ""
    files: dict[str, bytes] = field(default_factory=dict)       # MEMFS path -> bytes
    sources: dict[str, str] = field(default_factory=dict)       # MEMFS path -> local file it came from
    by_disk: dict[Path, str] = field(default_factory=dict)
    dirs: dict[Path, int] = field(default_factory=dict)

    def add_disk(self, disk: Path, data: bytes | None = None) -> str:
        disk = disk.resolve()
        if disk in self.by_disk:
            return self.by_disk[disk]
        if data is None:
            if not disk.is_file():
                raise FileNotFoundError(f"MJCF references {disk}, which does not exist")
            data = disk.read_bytes()
        k = self.dirs.setdefault(disk.parent, len(self.dirs))
        path = f"{self.root}/assets/{k}/{disk.name}"      # keeps the file name: a nameless <mesh file> takes it
        self.by_disk[disk] = path
        self.files[path] = data
        self.sources[path] = str(disk)
        return path

    def payload(self) -> dict:
        """The JSON the viewer reads: byte-identical files (a servo mesh used under three names) are stored once."""
        by_sha: dict[str, dict] = {}
        for path, data in self.files.items():
            h = _sha(data)
            if h in by_sha:
                by_sha[h]["paths"].append(path)
            else:
                by_sha[h] = {"paths": [path], "bytes": len(data), "gz": _gz_b64(data)}
        return {"main": self.main, "files": list(by_sha.values())}

    def inventory(self) -> list[Row]:
        rows, seen = [], set()
        for path, data in self.files.items():
            h = _sha(data)
            inl = 0 if h in seen else len(_gz_b64(data))
            seen.add(h)
            rows.append(Row(Path(self.sources[path]).name, len(data), inl, "gzip+base64" if inl else "same bytes as above"))
        return rows


def _resolve(base: Path, subdir: str | None, name: str, strip: bool) -> Path:
    if strip:
        name = name.replace("\\", "/").rsplit("/", 1)[-1]
    p = Path(name)
    if p.is_absolute():
        return p
    if subdir:
        d = Path(subdir)
        return (d if d.is_absolute() else base / d) / p
    return base / p


def _expand_includes(elem: ET.Element, main_dir: Path, cur: Path, stack: tuple[Path, ...]) -> None:
    """Replace each <include file=...> by the children of the included file's top element, as MuJoCo does. The file
    is resolved against the main model's directory (MuJoCo's rule), then against the including file's directory."""
    i = 0
    while i < len(elem):
        child = elem[i]
        if child.tag != "include":
            _expand_includes(child, main_dir, cur, stack)
            i += 1
            continue
        name = child.get("file") or ""
        cands = [Path(name)] if Path(name).is_absolute() else [main_dir / name, cur.parent / name]
        inc = next((c.resolve() for c in cands if c.is_file()), None)
        if inc is None:
            raise FileNotFoundError(f"<include file={name!r}> in {cur}: not found at " + " or ".join(map(str, cands)))
        if inc in stack:
            raise ValueError(f"<include> cycle through {inc}")
        sub = ET.parse(inc).getroot()
        _expand_includes(sub, main_dir, inc, stack + (inc,))
        elem.remove(child)
        for k, c in enumerate(list(sub)):
            elem.insert(i + k, c)
        i += len(sub)


def _rewrite(xml_path: Path, bundle: Bundle, stack: tuple[Path, ...] = ()) -> str:
    """Parse one MJCF file, inline its includes, point every file reference at its MEMFS copy; return the XML text.

    The compiler's meshdir/texturedir/assetdir/strippath are removed because every reference becomes an absolute
    MEMFS path; the XML comments are dropped."""
    xml_path = xml_path.resolve()
    root = ET.parse(xml_path).getroot()
    if root.tag != "mujoco":
        raise ValueError(f"{xml_path}: top element is <{root.tag}>, not <mujoco>")
    base = xml_path.parent
    _expand_includes(root, base, xml_path, stack + (xml_path,))
    meshdir = texturedir = assetdir = None
    strip = False
    for comp in root.iter("compiler"):
        meshdir = comp.get("meshdir", meshdir)
        texturedir = comp.get("texturedir", texturedir)
        assetdir = comp.get("assetdir", assetdir)
        if "strippath" in comp.attrib:
            strip = comp.get("strippath", "").strip().lower() == "true"
        for k in ("meshdir", "texturedir", "assetdir", "strippath"):
            comp.attrib.pop(k, None)
    dirs = {"mesh": meshdir if meshdir is not None else assetdir,
            "texture": texturedir if texturedir is not None else assetdir,
            "model": None}
    for el in root.iter():
        spec = _FILE_ATTRS.get(el.tag)
        if spec is None:
            continue
        kind, attrs = spec
        for attr in attrs:
            val = el.get(attr)
            if not val:
                continue
            disk = _resolve(base, dirs[kind], val, strip and kind != "model").resolve()
            if kind == "model":                       # <asset><model file=...>: a sub-model, rewritten in turn
                if disk in stack:
                    raise ValueError(f"<model file> cycle through {disk}")
                text = _rewrite(disk, bundle, stack + (xml_path,))
                el.set(attr, bundle.add_disk(disk, text.encode("utf-8")))
            else:
                el.set(attr, bundle.add_disk(disk))
    return ET.tostring(root, encoding="unicode")


def collect(xml_path, *, element_id: str, extra_files=None) -> Bundle:
    """Read an MJCF file and every file it references into a Bundle rooted at /mjwasm/<element_id>/.

    extra_files: files the XML scan cannot see (a plugin's data, say), as {path under /mjwasm/<element_id>/: local
    file} or as a list of local files (each placed at /mjwasm/<element_id>/<file name>)."""
    if not ELEMENT_ID.fullmatch(element_id):
        raise ValueError(f"element_id {element_id!r}: use a letter, then letters, digits, '_' or '-' (at most 64)")
    xml_path = Path(xml_path)
    b = Bundle(root=f"/mjwasm/{element_id}")
    text = _rewrite(xml_path, b)
    b.main = f"{b.root}/{xml_path.name}"
    b.files = {b.main: text.encode("utf-8"), **b.files}
    b.sources = {b.main: str(xml_path.resolve()), **b.sources}
    if extra_files:
        items = extra_files.items() if isinstance(extra_files, dict) else ((Path(p).name, p) for p in extra_files)
        for rel, local in items:
            rel = str(rel).replace("\\", "/")
            if rel.startswith("/") or ".." in rel.split("/"):
                raise ValueError(f"extra_files key {rel!r} must be a relative path inside the model directory")
            path = f"{b.root}/{rel}"
            b.files[path] = Path(local).read_bytes()
            b.sources[path] = str(Path(local).resolve())
    return b


# ------------------------------------------------------------------------------------------------ the fragment
def embed(xml_path, *, element_id: str, height_px: int = 480, title: str | None = None, extra_files=None,
          include_runtime: bool = True, groups=(0, 1, 2), keyframe=None, camera: dict | None = None,
          contacts: bool = False, sites: bool = True, shadows: bool = True, background: str = "#ffffff",
          autoplay: bool = False) -> str:
    """HTML fragment (div + scripts) showing the MJCF model at xml_path in a live MuJoCo viewer.

    element_id     id of the viewer's div; also names its MEMFS directory, so it must be unique on the page.
    height_px      height of the 3D view; the width follows the container.
    title          text above the view (plain text, escaped).
    extra_files    files the XML scan cannot find; see collect().
    include_runtime  False for every viewer after the first on a page (or when runtime() is placed separately).
    groups         geom groups shown at load (MuJoCo's default 0, 1, 2); the checkboxes change them.
    keyframe       name or index of the keyframe to load at start (default: qpos0).
    camera         {"azimuth", "elevation" (deg), "distance" (m), "lookat" [x, y, z]}; default from the model's
                   <visual><global> and statistic, as MuJoCo's free camera.
    contacts, sites, shadows, autoplay   initial state of those toggles.
    """
    b = collect(xml_path, element_id=element_id, extra_files=extra_files)
    if keyframe is not None:                               # fail at build time, not in the reader's browser
        keys = [k.get("name") for k in ET.fromstring(b.files[b.main]).iter("key")]
        if isinstance(keyframe, int) and not 0 <= keyframe < len(keys):
            raise ValueError(f"keyframe {keyframe}: {xml_path} has {len(keys)} keyframes")
        if isinstance(keyframe, str) and keyframe not in keys:
            raise ValueError(f"keyframe {keyframe!r}: {xml_path} has " + (", ".join(map(repr, keys)) or "no keyframes"))
    model = b.payload()
    cfg = {"id": element_id, "model": f"{element_id}-mjwasm-model", "height": int(height_px), "title": title,
           "groups": [int(g) for g in groups], "keyframe": keyframe, "camera": camera, "contacts": bool(contacts),
           "sites": bool(sites), "shadows": bool(shadows), "background": background, "autoplay": bool(autoplay),
           "source": Path(xml_path).name}
    t = html.escape(title) if title else ""
    out = [runtime()] if include_runtime else []
    out.append(
        f'<div id="{element_id}" class="mjw" data-mjwasm-status="loading">'
        + (f'<div class="mjw-title">{t}</div>' if t else "")
        + f'<div class="mjw-stage" style="height:{int(height_px)}px"><div class="mjw-msg">Loading MuJoCo '
        f'{MUJOCO_VERSION} &hellip;</div></div>'
        '<noscript>This MuJoCo viewer needs JavaScript and WebGL.</noscript></div>')
    out.append(f'<script type="application/json" id="{element_id}-mjwasm-model">{_json_script(model)}</script>')
    out.append(f"<script>(window.MJWASM_QUEUE=window.MJWASM_QUEUE||[]).push({_json_script(cfg)});</script>")
    return "\n".join(out) + "\n"


def inventory(xml_path=None, *, element_id: str = "viewer", extra_files=None, include_runtime: bool = True) -> list[Row]:
    """Rows of (asset, raw bytes, bytes as inlined) for the runtime and, when given, one model."""
    rows = runtime_inventory() if include_runtime else []
    if xml_path is not None:
        rows += collect(xml_path, element_id=element_id, extra_files=extra_files).inventory()
    return rows


def page(body: str, *, title: str, lede: str = "") -> str:
    """A minimal standalone page around one or more fragments (Times on white, as the result pages)."""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
html,body{{background:#fff;color:#000}}
body{{font:18px/1.4 "Times New Roman",Times,serif;margin:0}}
main{{max-width:860px;margin:0 auto;padding:24px 16px 48px}}
h1{{font-size:27px;line-height:1.2;margin:12px 0 8px}}
</style></head>
<body><main>
<h1>{html.escape(title)}</h1>
{lede}
{body}
</main></body></html>
"""


# ------------------------------------------------------------------------------------------------ CSS and JS
_VIEWER_CSS = """
.mjw{margin:14px 0;font:16px/1.35 "Times New Roman",Times,serif;color:#000;max-width:100%;text-align:left}
.mjw-title{font-weight:bold;margin:0 0 4px}
.mjw-stage{position:relative;border:1px solid #000;background:#fff;overflow:hidden;max-width:100%}
.mjw-canvas{display:block;width:100%;height:100%;outline:none;cursor:grab;touch-action:none}
.mjw-canvas:active{cursor:grabbing}
.mjw-msg{position:absolute;left:12px;right:12px;top:50%;transform:translateY(-50%);text-align:center;color:#444;
  white-space:pre-wrap;pointer-events:none}
.mjw-msg.mjw-err{color:#c00;text-align:left;font:14px/1.35 "Courier New",Courier,monospace}
.mjw-hint{position:absolute;right:0;bottom:0;font-size:13px;color:#444;background:rgba(255,255,255,.75);
  padding:0 4px;pointer-events:none}
.mjw-bar{display:flex;flex-wrap:wrap;align-items:center;gap:4px 14px;margin:6px 0 2px}
.mjw-bar button{font:15px "Times New Roman",Times,serif;padding:1px 8px;margin:0}
.mjw-bar label{white-space:nowrap;cursor:pointer}
.mjw-set{display:inline-flex;flex-wrap:wrap;align-items:center;gap:2px 8px}
.mjw-joints{column-width:16em;column-gap:20px;margin:6px 0 2px}
.mjw-joint{display:grid;grid-template-columns:6.6em minmax(5em,1fr) 3.9em;align-items:center;
  column-gap:5px;break-inside:avoid;font-size:15px;line-height:1.55}
.mjw-jn{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font:13px "Courier New",Courier,monospace}
.mjw-jv{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}
.mjw-joint input{width:100%;min-width:0;margin:0}
.mjw-note,.mjw-status{font-size:14px;color:#444;margin:2px 0}
.mjw-status{font-variant-numeric:tabular-nums}
"""

_VIEWER_JS = r"""/* mjwasm viewer runtime (scripts/mjwasm_embed.py in the MorphoHand repository). */
(function () {
  "use strict";
  if (window.MJWASM && window.MJWASM.mount) return;     // a second copy of the runtime on the page does nothing
  const VERSIONS = __VERSIONS__;
  const MJW = (window.MJWASM = Object.assign(window.MJWASM || {}, { versions: VERSIONS, viewers: [] }));
  const PLANE = 0, SPHERE = 2, CAPSULE = 3, ELLIPSOID = 4, CYLINDER = 5, BOX = 6, MESH = 7;
  const GEOM_NAMES = ["plane", "hfield", "sphere", "capsule", "ellipsoid", "cylinder", "box", "mesh", "sdf"];
  const SLIDE = 2, HINGE = 3, NGROUP = 6, DEG = 180 / Math.PI;

  // ---------------------------------------------------------------- payloads (gzip + base64 in script blocks)
  function b64bytes(s) {
    if (typeof Uint8Array.fromBase64 === "function") return Uint8Array.fromBase64(s);
    const bin = atob(s), out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  async function gunzip(u8) {
    if (typeof DecompressionStream === "undefined")
      throw new Error("this browser lacks DecompressionStream (needs Chrome 80, Firefox 113 or Safari 16.4)");
    const s = new Blob([u8]).stream().pipeThrough(new DecompressionStream("gzip"));
    return new Uint8Array(await new Response(s).arrayBuffer());
  }
  function block(id) {
    const el = document.getElementById(id);
    if (!el) throw new Error("payload #" + id + " is missing: the page needs the mjwasm runtime once");
    return el.textContent.trim();
  }
  const pending = {};
  const payload = (id) => pending[id] || (pending[id] = gunzip(b64bytes(block(id))));
  const utf8 = (u8) => new TextDecoder("utf-8").decode(u8);

  let threeP = null, modP = null;
  function loadThree() {
    if (!threeP) threeP = (async () => {
      const [core, orbit] = await Promise.all([payload("mjwasm-three"), payload("mjwasm-orbit")]);
      const exports = {}, module = { exports };
      (new Function("exports", "module", "define", utf8(core) + "\n//# sourceURL=mjwasm/three.min.js"))(exports, module, undefined);
      const THREE = module.exports;
      (new Function("THREE", utf8(orbit) + "\n//# sourceURL=mjwasm/OrbitControls.js"))(THREE);
      delete pending["mjwasm-three"]; delete pending["mjwasm-orbit"];
      return THREE;
    })();
    return threeP;
  }
  // One WebAssembly module for every viewer on the page.
  MJW.module = function () {
    if (!modP) modP = (async () => {
      const t0 = performance.now();
      const [glue, wasm] = await Promise.all([payload("mjwasm-glue"), payload("mjwasm-wasm")]);
      const loadMujoco = (new Function(utf8(glue) + "\nreturn loadMujoco;\n//# sourceURL=mjwasm/mujoco.js"))();
      delete pending["mjwasm-glue"]; delete pending["mjwasm-wasm"];
      const mod = await loadMujoco({
        wasmBinary: wasm.buffer,
        print: (s) => console.log("[mujoco] " + s),
        printErr: (s) => console.warn("[mujoco] " + s),
      });
      MJW.loadMs = Math.round(performance.now() - t0);
      return mod;
    })();
    return modP;
  };

  // ---------------------------------------------------------------- small helpers
  const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
  function h(tag, attrs, parent, text) {
    const e = document.createElement(tag);
    if (attrs) for (const k in attrs) { if (k === "class") e.className = attrs[k]; else e.setAttribute(k, attrs[k]); }
    if (text !== undefined && text !== null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function button(parent, label, fn, title) {
    const b = h("button", { type: "button" }, parent, label);
    if (title) b.title = title;
    b.addEventListener("click", (ev) => { ev.preventDefault(); fn(); });
    return b;
  }
  function checkbox(parent, label, on, fn, title) {
    const l = h("label", null, parent), c = h("input", { type: "checkbox" }, l);
    c.checked = !!on;
    l.appendChild(document.createTextNode("\u00a0" + label));
    if (title) l.title = title;
    c.addEventListener("change", () => fn(c.checked));
    return c;
  }
  function errText(mod, e) {
    if (mod && typeof mod.getExceptionMessage === "function" &&
        (typeof e === "number" || (typeof WebAssembly.Exception === "function" && e instanceof WebAssembly.Exception))) {
      try { return [].concat(mod.getExceptionMessage(e)).filter(Boolean).join(": "); } catch (_) { /* below */ }
    }
    return String((e && e.message) || e).replace(/\s+$/, "");
  }
  function setMat4(M, P, R, i) {
    const p = 3 * i, r = 9 * i;
    M.set(R[r], R[r + 1], R[r + 2], P[p], R[r + 3], R[r + 4], R[r + 5], P[p + 1], R[r + 6], R[r + 7], R[r + 8], P[p + 2], 0, 0, 0, 1);
  }
  function setMat(obj, P, R, i) { setMat4(obj.matrix, P, R, i); obj.matrixWorldNeedsUpdate = true; }
  // three.js r147's Blinn-Phong adds a Schlick Fresnel term and a (s/2 + 1)/4 normalisation (about 8x at MuJoCo's
  // shininess 0.5); MuJoCo's renderer is fixed-function: specular x light x (N.H)^s. The chunk text is pinned with
  // three.js; when it does not match, the stock Phong shader is kept.
  let PHONG_CHUNK;
  function fixedPhong(T) {
    const src = T.ShaderChunk && T.ShaderChunk.lights_phong_pars_fragment;
    const old = "reflectedLight.directSpecular += irradiance * BRDF_BlinnPhong( directLight.direction, geometry.viewDir, geometry.normal, material.specularColor, material.specularShininess ) * material.specularStrength;";
    if (!src || src.indexOf(old) < 0) return null;
    return src.replace(old, "if ( dotNL > 0.0 ) reflectedLight.directSpecular += directLight.color * RECIPROCAL_PI * material.specularColor"
      + " * pow( saturate( dot( geometry.normal, normalize( directLight.direction + geometry.viewDir ) ) ), material.specularShininess ) * material.specularStrength;");
  }
  // Segments for round primitives, by size relative to the model: 8 for a 1 mm pad on a hand, up to 40.
  const segments = (r, ext) => clamp(Math.round(6 + 48 * Math.sqrt(Math.max(r, 0) / ext)), 8, 40);
  const niceSpan = (x) => { const p = Math.pow(10, Math.floor(Math.log10(x))); return [1, 2, 5, 10].map((k) => k * p).find((v) => v >= x); };
  // The 3.14.0 bindings cannot return mjtByte (bool) arrays ("memory_view<bool>" has no JS type). MuJoCo lays the
  // model arrays out in field order, each aligned, so a byte array sits in the gap after its predecessor `prev`;
  // the successor `next` must then start exactly where the alignment puts it, which verifies the location. Returns
  // a Uint8Array view, or null when the layout does not match (the caller falls back to MuJoCo's default).
  function byteField(m, prev, next, n) {
    try {
      const a = m[prev], b = m[next];
      if (!n || !ArrayBuffer.isView(a) || !ArrayBuffer.isView(b) || a.buffer !== b.buffer) return null;
      for (const al of [64, 8]) {
        const up = (x) => Math.ceil(x / al) * al, at = up(a.byteOffset + a.byteLength);
        if (up(at + n) === b.byteOffset) return new Uint8Array(a.buffer, at, n);
      }
    } catch (_) { /* fall through */ }
    return null;
  }

  // ---------------------------------------------------------------- viewer
  class Viewer {
    constructor(cfg) {
      this.cfg = cfg;
      this.id = cfg.id;
      this.root = document.getElementById(cfg.id);
      if (!this.root) throw new Error("mjwasm: no element #" + cfg.id);
      this.playing = false; this.visible = true; this.raf = 0; this.last = null; this.acc = 0;
      this.speed = 1; this.steps = 0; this.frames = 0; this.error = null; this.info = null; this.rtf = 1;
      const g0 = cfg.groups || [0, 1, 2];
      this.groups = Array.from({ length: NGROUP }, (_, g) => g0.indexOf(g) >= 0);
      this.showContacts = !!cfg.contacts;
      this.showSites = cfg.sites !== false;
      this.frameFn = (t) => this.frame(t);
      this.shell();
      this.ready = this.start().then(() => this, (e) => { this.fail(e); return this; });
    }

    shell() {
      const r = this.root, c = this.cfg;
      r.classList.add("mjw");
      r.textContent = "";
      if (c.title) h("div", { class: "mjw-title" }, r, c.title);
      this.stage = h("div", { class: "mjw-stage" }, r);
      this.stage.style.height = (c.height || 480) + "px";
      this.canvas = h("canvas", { class: "mjw-canvas", "aria-label": "MuJoCo view of " + (c.source || "the model") }, this.stage);
      this.msg = h("div", { class: "mjw-msg" }, this.stage, "Loading MuJoCo " + VERSIONS.mujoco + " \u2026");
      this.bar = h("div", { class: "mjw-bar" }, r);
      this.jointsEl = h("div", { class: "mjw-joints" }, r);
      this.noteEl = h("div", { class: "mjw-note" }, r);
      this.statusEl = h("div", { class: "mjw-status" }, r);
      this.setStatus("loading");
    }

    setStatus(s) { this.root.setAttribute("data-mjwasm-status", s); }

    fail(e) {
      this.error = e;
      const msg = errText(this.mod, e);
      console.error("mjwasm #" + this.id + ": " + msg);
      this.msg.style.display = "";
      this.msg.className = "mjw-msg mjw-err";
      this.msg.textContent = "MuJoCo could not show " + (this.cfg.source || "this model") + ".\n" + msg;
      this.setStatus("error: " + msg);
    }

    async start() {
      const t0 = performance.now();
      const data = JSON.parse(block(this.cfg.model));
      const [mod, THREE] = await Promise.all([MJW.module(), loadThree()]);
      this.mod = mod; this.T = THREE;
      const files = await Promise.all(data.files.map(async (f) => [f.paths, await gunzip(b64bytes(f.gz))]));
      const FS = mod.FS;
      for (const [paths, bytes] of files) for (const path of paths) {
        const dir = path.slice(0, path.lastIndexOf("/"));
        if (dir) FS.mkdirTree(dir);
        FS.writeFile(path, bytes);
      }
      let m = null, d = null;
      try {
        m = mod.MjModel.from_xml_path(data.main);
        d = new mod.MjData(m);
      } catch (e) {
        if (d) d.delete();
        if (m) m.delete();
        throw new Error(errText(mod, e));
      }
      this.m = m; this.d = d;
      this.readModel();
      const k = this.keyIndex(this.cfg.keyframe);
      if (k >= 0) mod.mj_resetDataKeyframe(m, d, k);
      mod.mj_forward(m, d);
      this.scene3();
      this.controlsUI();
      this.observe();
      this.msg.style.display = "none";
      this.live = true;                                  // from here on requestRender() schedules frames
      this.render();
      this.info = {
        mujoco: typeof mod.mj_versionString === "function" ? mod.mj_versionString() : VERSIONS.mujoco,
        nq: m.nq, nv: m.nv, nu: m.nu, nbody: m.nbody, njnt: m.njnt, ngeom: m.ngeom, nsite: m.nsite,
        nmesh: m.nmesh, nkey: m.nkey, nlight: m.nlight, geomsDrawn: this.geomObjs.length,
        sitesDrawn: this.siteObjs.length, sliders: this.sliders.length, servos: this.sliders.filter((s) => s.acts.length).length,
        notDrawn: this.notDrawn, webgl2: !!this.renderer.capabilities.isWebGL2,
        byteFieldsRead: Object.keys(this.flags).filter((k) => this.flags[k]),
        triangles: this.lastDraw.triangles, drawCalls: this.lastDraw.calls,
        wasmLoadMs: MJW.loadMs, readyMs: Math.round(performance.now() - t0),
        bytes: data.files.reduce((a, f) => a + f.bytes * f.paths.length, 0),
      };
      this.root.setAttribute("data-mjwasm-info", JSON.stringify(this.info));
      this.status();
      this.setStatus("ok");
      if (this.cfg.autoplay) this.play();
    }

    readModel() {
      const mod = this.mod, m = this.m;
      const en = (e, k, dflt) => { try { const v = mod[e][k].value; return typeof v === "number" ? v : dflt; } catch (_) { return dflt; } };
      this.OBJ = { body: en("mjtObj", "mjOBJ_BODY", 1), joint: en("mjtObj", "mjOBJ_JOINT", 3), geom: en("mjtObj", "mjOBJ_GEOM", 5),
                   site: en("mjtObj", "mjOBJ_SITE", 6), actuator: en("mjtObj", "mjOBJ_ACTUATOR", 19), key: en("mjtObj", "mjOBJ_KEY", 24) };
      const stat = m.stat, vis = m.vis;
      this.extent = stat.extent > 0 ? stat.extent : 1;
      this.center = Array.from(stat.center);
      this.meansize = stat.meansize > 0 ? stat.meansize : 0.05 * this.extent;
      const g = vis.global, mp = vis.map, q = vis.quality, hl = vis.headlight, sc = vis.scale, rg = vis.rgba;
      this.V = {
        azimuth: g.azimuth, elevation: g.elevation, fovy: g.fovy, znear: mp.znear, zfar: mp.zfar, shadowsize: q.shadowsize,
        head: !!hl.active, ambient: Array.from(hl.ambient), diffuse: Array.from(hl.diffuse), specular: Array.from(hl.specular),
        contactwidth: sc.contactwidth, contactheight: sc.contactheight, contactpoint: Array.from(rg.contactpoint),
      };
      this.timestep = m.opt.timestep;
      this.flags = {                                    // copies, so heap growth cannot detach them
        light_active: byteField(m, "light_range", "light_pos", m.nlight),
        light_castshadow: byteField(m, "light_texid", "light_bulbradius", m.nlight),
        mat_texuniform: byteField(m, "mat_texid", "mat_texrepeat", m.nmat),
      };
      for (const k in this.flags) if (this.flags[k]) this.flags[k] = this.flags[k].slice();
    }

    flag(name, i, dflt) { const f = this.flags[name]; return f ? !!f[i] : dflt; }

    nameOf(kind, i) { return this.mod.mj_id2name(this.m, this.OBJ[kind], i) || ""; }

    keyIndex(k) {
      if (k === undefined || k === null) return -1;
      const n = this.m.nkey;
      if (typeof k === "number") {
        if (k < 0 || k >= n) throw new Error("keyframe index " + k + " is out of range (the model has " + n + ")");
        return k;
      }
      const i = this.mod.mj_name2id(this.m, this.OBJ.key, String(k));
      if (i < 0) throw new Error("the model has no keyframe named '" + k + "'");
      return i;
    }

    // ---------------------------------------------------------------- three.js scene
    scene3() {
      const T = this.T, c = this.cfg, V = this.V, ext = this.extent;
      const renderer = (this.renderer = new T.WebGLRenderer({ canvas: this.canvas, antialias: true, preserveDrawingBuffer: true }));
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.setClearColor(new T.Color(c.background || "#ffffff"), 1);
      if (c.shadows !== false) { renderer.shadowMap.enabled = true; renderer.shadowMap.type = T.PCFSoftShadowMap; }
      this.scene = new T.Scene();
      const cam = (this.camera = new T.PerspectiveCamera(V.fovy || 45, 1, Math.max(1e-6, (V.znear || 0.01) * ext), Math.max(V.zfar || 50, 20) * ext));
      cam.up.set(0, 0, 1);                                  // before OrbitControls, which reads it once
      this.controls = new T.OrbitControls(cam, this.canvas);
      this.controls.addEventListener("change", () => this.requestRender());
      // The wheel scrolls the page until the view is clicked, then zooms until the pointer leaves the view.
      this.controls.enableZoom = false;
      this.canvas.addEventListener("pointerdown", () => { this.controls.enableZoom = true; });
      this.stage.addEventListener("pointerleave", () => { this.controls.enableZoom = false; });
      this.home();
      this.meshCache = {};
      this.notDrawn = {};
      this.M4 = new T.Matrix4();
      this.lights();
      this.geomObjs = this.buildGeoms();
      this.siteObjs = this.buildSites();
      const cw = V.contactwidth * this.meansize, ch = V.contactheight * this.meansize;
      this.contactGeo = new T.CylinderGeometry(cw, cw, 2 * ch, 20).rotateX(Math.PI / 2);
      this.contactMat = this.material([V.contactpoint[0], V.contactpoint[1], V.contactpoint[2], 1], -1, null);
      this.cinst = null;
      this.resize();
      if ("ResizeObserver" in window) new ResizeObserver(() => this.resize()).observe(this.stage);
      else window.addEventListener("resize", () => this.resize());
      h("div", { class: "mjw-hint" }, this.stage, "Drag: rotate \u00b7 right-drag: pan \u00b7 click, then wheel: zoom");
    }

    home() {
      const c = this.cfg.camera || {}, V = this.V;
      const az = (c.azimuth !== undefined ? c.azimuth : V.azimuth) / DEG;
      const el = (c.elevation !== undefined ? c.elevation : V.elevation) / DEG;
      const dist = c.distance !== undefined ? c.distance : 1.5 * this.extent;
      const look = c.lookat || this.center;
      const f = [Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el)];   // MuJoCo's free camera
      this.camera.position.set(look[0] - dist * f[0], look[1] - dist * f[1], look[2] - dist * f[2]);
      this.controls.target.set(look[0], look[1], look[2]);
      this.controls.update();
      this.requestRender();
    }

    lights() {
      const T = this.T, m = this.m, V = this.V, ext = this.extent, scene = this.scene;
      const col = (a) => new T.Color(a[0], a[1], a[2]);
      const amb = V.head ? V.ambient.slice() : [0, 0, 0];
      if (V.head) { this.head = new T.DirectionalLight(col(V.diffuse), 1); scene.add(this.head, this.head.target); }
      this.mlights = [];
      const shadows = this.cfg.shadows !== false;
      const size = Math.min(V.shadowsize || 2048, 2048);
      for (let i = 0; i < m.nlight; i++) {
        if (!this.flag("light_active", i, true)) continue;
        for (let k = 0; k < 3; k++) amb[k] += m.light_ambient[3 * i + k];
        const dif = [m.light_diffuse[3 * i], m.light_diffuse[3 * i + 1], m.light_diffuse[3 * i + 2]];
        const type = m.light_type ? m.light_type[i] : (m.light_directional && m.light_directional[i] ? 1 : 0);
        let L;
        if (type === 1) L = new T.DirectionalLight(col(dif), 1);
        else if (type === 0) L = new T.SpotLight(col(dif), 1, 0, clamp((m.light_cutoff ? m.light_cutoff[i] : 45) / DEG, 0.02, 1.55), 0.3, 0);
        else L = new T.PointLight(col(dif), 1, 0, 0);
        if (shadows && this.flag("light_castshadow", i, true) && type !== 2) {
          L.castShadow = true;
          L.shadow.mapSize.set(size, size);
          L.shadow.bias = -0.0002;
          L.shadow.normalBias = 0.02 * this.meansize;
          const sc = L.shadow.camera;
          if (type === 1) { sc.left = -ext; sc.right = ext; sc.top = ext; sc.bottom = -ext; }
          sc.near = 0.02 * ext; sc.far = 5 * ext;
          sc.updateProjectionMatrix();
        }
        scene.add(L);
        if (L.target) scene.add(L.target);
        this.mlights.push({ i, L, type });
      }
      scene.add(new T.AmbientLight(col(amb.map((x) => Math.min(1, x))), 1));
      let sd = V.head ? V.diffuse[0] : 0, ss = V.head ? V.specular[0] : 0;
      for (const o of this.mlights) { sd += m.light_diffuse[3 * o.i]; ss += m.light_specular ? m.light_specular[3 * o.i] : 0.3; }
      this.specRatio = sd > 0 ? ss / sd : 0.3;
    }

    updateLights() {
      if (this.head) {
        this.head.position.copy(this.camera.position);
        this.head.target.position.copy(this.controls.target);
        this.head.target.updateMatrixWorld();
      }
      if (!this.mlights.length) return;
      const d = this.d, P = d.light_xpos, D = d.light_xdir, c = this.center, r = 2 * this.extent;
      for (const o of this.mlights) {
        const i = 3 * o.i, p = [P[i], P[i + 1], P[i + 2]], dir = [D[i], D[i + 1], D[i + 2]];
        if (o.type === 1) { o.L.position.set(c[0] - r * dir[0], c[1] - r * dir[1], c[2] - r * dir[2]); o.L.target.position.set(c[0], c[1], c[2]); }
        else { o.L.position.set(p[0], p[1], p[2]); if (o.L.target) o.L.target.position.set(p[0] + dir[0], p[1] + dir[1], p[2] + dir[2]); }
        if (o.L.target) o.L.target.updateMatrixWorld();
      }
    }

    // MuJoCo's rule: a geom's own rgba wins when it is not the default (0.5 0.5 0.5 1) or there is no material.
    rgba(A, M, i) {
      const r = [A[4 * i], A[4 * i + 1], A[4 * i + 2], A[4 * i + 3]], mat = M[i];
      const dflt = Math.abs(r[0] - 0.5) < 1e-6 && Math.abs(r[1] - 0.5) < 1e-6 && Math.abs(r[2] - 0.5) < 1e-6 && r[3] === 1;
      if (mat >= 0 && dflt) { const R = this.m.mat_rgba; return [R[4 * mat], R[4 * mat + 1], R[4 * mat + 2], R[4 * mat + 3]]; }
      return r;
    }

    material(rgba, matid, planeSize) {
      const T = this.T, m = this.m;
      let spec = 0.5, shin = 0.5, emis = 0;
      if (matid >= 0) { spec = m.mat_specular[matid]; shin = m.mat_shininess[matid]; emis = m.mat_emission[matid]; }
      // A three.js light has one colour for diffuse and specular, MuJoCo's lights have two: the material's specular
      // carries the ratio of summed light specular to summed light diffuse.
      const s = clamp(spec * this.specRatio, 0, 1);
      const p = {
        color: new T.Color(rgba[0], rgba[1], rgba[2]), specular: new T.Color(s, s, s), shininess: clamp(128 * shin, 1, 128),
        emissive: new T.Color(rgba[0] * emis, rgba[1] * emis, rgba[2] * emis),
      };
      if (rgba[3] < 1) { p.transparent = true; p.opacity = rgba[3]; p.depthWrite = false; }
      if (planeSize && matid >= 0) { const tex = this.texture(matid, planeSize); if (tex) p.map = tex; }
      const mat = new T.MeshPhongMaterial(p);
      if (PHONG_CHUNK === undefined) PHONG_CHUNK = fixedPhong(T);
      if (PHONG_CHUNK) {
        mat.onBeforeCompile = (sh) => { sh.fragmentShader = sh.fragmentShader.replace("#include <lights_phong_pars_fragment>", PHONG_CHUNK); };
        mat.customProgramCacheKey = () => "mjwasm-fixed-phong";
      }
      return mat;
    }

    texture(matid, size) {
      const m = this.m, T = this.T;
      if (!m.ntex || !m.mat_texid) return null;
      const nrole = m.nmat > 0 ? Math.round(m.mat_texid.length / m.nmat) : 1;
      const texid = nrole > 1 ? m.mat_texid[matid * nrole + 1] : m.mat_texid[matid];   // mjTEXROLE_RGB
      if (texid < 0 || m.tex_type[texid] !== 0) return null;                              // 2D textures only
      const w = m.tex_width[texid], ht = m.tex_height[texid], nc = m.tex_nchannel[texid], adr = m.tex_adr[texid];
      const src = m.tex_data, n = w * ht, out = new Uint8Array(4 * n);
      for (let k = 0; k < n; k++) {
        const a = adr + nc * k;
        out[4 * k] = src[a];
        out[4 * k + 1] = nc > 1 ? src[a + 1] : src[a];
        out[4 * k + 2] = nc > 2 ? src[a + 2] : src[a];
        out[4 * k + 3] = nc > 3 ? src[a + 3] : 255;
      }
      const tex = new T.DataTexture(out, w, ht, T.RGBAFormat);
      tex.wrapS = tex.wrapT = T.RepeatWrapping;
      tex.magFilter = T.LinearFilter; tex.minFilter = T.LinearMipmapLinearFilter; tex.generateMipmaps = true;
      tex.anisotropy = Math.min(8, this.renderer.capabilities.getMaxAnisotropy());
      const rx = m.mat_texrepeat[2 * matid], ry = m.mat_texrepeat[2 * matid + 1];
      // Copies over the plane, as MuJoCo's renderer draws them (measured on 3.6): texrepeat, or with texuniform
      // texrepeat x the half-size (texrepeat 20 on a plane of size 0.1 gives 2 copies, 4 checker squares).
      if (this.flag("mat_texuniform", matid, false)) tex.repeat.set(rx * size[0], ry * size[1]); else tex.repeat.set(rx, ry);
      tex.needsUpdate = true;
      return tex;
    }

    geometry(type, s, meshid) {
      const T = this.T, ext = this.extent;
      const n = segments(type === ELLIPSOID ? Math.max(s[0], s[1], s[2]) : s[0], ext), h = Math.max(4, n >> 1);
      switch (type) {
        case PLANE: return new T.PlaneGeometry(2 * s[0], 2 * s[1]);
        case SPHERE: return new T.SphereGeometry(s[0], n, h);
        case CAPSULE: return new T.CapsuleGeometry(s[0], 2 * s[1], Math.max(3, n >> 2), n).rotateX(Math.PI / 2);
        case ELLIPSOID: return new T.SphereGeometry(1, n, h).scale(s[0], s[1], s[2]);
        case CYLINDER: return new T.CylinderGeometry(s[0], s[0], 2 * s[1], n).rotateX(Math.PI / 2);
        case BOX: return new T.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
        case MESH: return meshid >= 0 ? this.meshGeometry(meshid) : null;
        default: return null;
      }
    }

    meshGeometry(id) {
      if (this.meshCache[id]) return this.meshCache[id];
      const m = this.m, T = this.T;
      const va = m.mesh_vertadr[id], fa = m.mesh_faceadr[id], fn = m.mesh_facenum[id];
      const Vx = m.mesh_vert, F = m.mesh_face, pos = new Float32Array(9 * fn);
      for (let f = 0; f < fn; f++) for (let k = 0; k < 3; k++) {
        const v = 3 * (va + F[3 * (fa + f) + k]), o = 9 * f + 3 * k;
        pos[o] = Vx[v]; pos[o + 1] = Vx[v + 1]; pos[o + 2] = Vx[v + 2];
      }
      const geo = new T.BufferGeometry();
      geo.setAttribute("position", new T.BufferAttribute(pos, 3));
      const nn = m.mesh_normalnum ? m.mesh_normalnum[id] : 0;
      if (nn > 0 && m.mesh_facenormal && m.mesh_normal) {
        // MuJoCo keeps one normal per deduplicated vertex, so at a sharp CAD edge a corner's normal can point up to
        // 180 deg away from its face. As MuJoCo's renderer does, use the face normal wherever the stored normal is
        // more than acos(0.8) = 37 deg from it: flat faces stay flat, curved surfaces stay smooth.
        const N = m.mesh_normal, FN = m.mesh_facenormal, na = m.mesh_normaladr[id], nrm = new Float32Array(9 * fn);
        for (let f = 0; f < fn; f++) {
          const o = 9 * f;
          const ux = pos[o + 3] - pos[o], uy = pos[o + 4] - pos[o + 1], uz = pos[o + 5] - pos[o + 2];
          const vx = pos[o + 6] - pos[o], vy = pos[o + 7] - pos[o + 1], vz = pos[o + 8] - pos[o + 2];
          let cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
          const cl = Math.hypot(cx, cy, cz);
          if (cl > 0) { cx /= cl; cy /= cl; cz /= cl; }
          for (let k = 0; k < 3; k++) {
            const q = 3 * (na + FN[3 * (fa + f) + k]), w = o + 3 * k;
            if (cl > 0 && N[q] * cx + N[q + 1] * cy + N[q + 2] * cz < 0.8) { nrm[w] = cx; nrm[w + 1] = cy; nrm[w + 2] = cz; }
            else { nrm[w] = N[q]; nrm[w + 1] = N[q + 1]; nrm[w + 2] = N[q + 2]; }
          }
        }
        geo.setAttribute("normal", new T.BufferAttribute(nrm, 3));
      } else geo.computeVertexNormals();
      geo.computeBoundingSphere();
      return (this.meshCache[id] = geo);
    }

    // Geoms that share type, size, mesh, material, colour and group are drawn as one InstancedMesh (a hand with
    // sphere pads has thousands of identical spheres); the others are single meshes sharing geometry and material.
    buildGeoms() {
      const m = this.m, T = this.T, out = [], L = 10 * this.extent, lists = new Map();
      this.instanced = [];
      for (let i = 0; i < m.ngeom; i++) {
        const type = m.geom_type[i];
        const rgba = this.rgba(m.geom_rgba, m.geom_matid, i);
        if (rgba[3] <= 0) continue;
        let s = [m.geom_size[3 * i], m.geom_size[3 * i + 1], m.geom_size[3 * i + 2]];
        if (type === PLANE) s = [s[0] > 0 ? s[0] : L, s[1] > 0 ? s[1] : L, s[2]];     // size 0: infinite
        const group = clamp(m.geom_group[i], 0, NGROUP - 1), matid = m.geom_matid[i], dataid = type === MESH ? m.geom_dataid[i] : -1;
        const key = [type, s[0], s[1], s[2], dataid, matid, rgba.join(","), group].join("|");
        const it = { i, type, s, rgba, matid, dataid, group };
        if (lists.has(key)) lists.get(key).push(it); else lists.set(key, [it]);
      }
      for (const list of lists.values()) {
        const g0 = list[0];
        const geo = this.geometry(g0.type, g0.s, g0.dataid);
        if (!geo) { const k = GEOM_NAMES[g0.type] || String(g0.type); this.notDrawn[k] = (this.notDrawn[k] || 0) + list.length; continue; }
        const mat = this.material(g0.rgba, g0.matid, g0.type === PLANE ? g0.s : null);
        const shadow = g0.type !== PLANE && g0.rgba[3] >= 1;
        if (list.length >= 8 && g0.type !== PLANE) {
          const obj = new T.InstancedMesh(geo, mat, list.length);
          obj.instanceMatrix.setUsage(T.DynamicDrawUsage);
          obj.frustumCulled = false;                       // r147 culls by the geometry's own bounds
          obj.castShadow = shadow; obj.receiveShadow = true; obj.visible = this.groups[g0.group];
          this.scene.add(obj);
          this.instanced.push(obj);
          list.forEach((x, k) => out.push({ i: x.i, obj, group: x.group, k }));
        } else {
          for (const x of list) {
            const obj = new T.Mesh(geo, mat);
            obj.matrixAutoUpdate = false;
            obj.castShadow = shadow; obj.receiveShadow = true; obj.visible = this.groups[x.group];
            this.scene.add(obj);
            out.push({ i: x.i, obj, group: x.group, k: -1 });
          }
        }
      }
      return out;
    }

    buildSites() {
      const m = this.m, T = this.T, out = [];
      for (let i = 0; i < m.nsite; i++) {
        if (m.site_group[i] > 2) continue;                                   // MuJoCo shows site groups 0-2
        const rgba = this.rgba(m.site_rgba, m.site_matid, i);
        if (rgba[3] <= 0) continue;
        const s = [m.site_size[3 * i], m.site_size[3 * i + 1], m.site_size[3 * i + 2]];
        const geo = this.geometry(m.site_type[i], s, -1);
        if (!geo) continue;
        const obj = new T.Mesh(geo, this.material(rgba, m.site_matid[i], null));
        obj.matrixAutoUpdate = false;
        obj.visible = this.showSites;
        this.scene.add(obj);
        out.push({ i, obj });
      }
      return out;
    }

    poses() {
      const d = this.d;
      const P = d.geom_xpos, R = d.geom_xmat, M = this.M4;          // fresh views: heap growth detaches old ones
      for (const g of this.geomObjs) {
        if (g.k < 0) setMat(g.obj, P, R, g.i);
        else { setMat4(M, P, R, g.i); g.obj.setMatrixAt(g.k, M); }
      }
      for (const o of this.instanced) o.instanceMatrix.needsUpdate = true;
      if (this.siteObjs.length && this.showSites) {
        const SP = d.site_xpos, SR = d.site_xmat;
        for (const s of this.siteObjs) setMat(s.obj, SP, SR, s.i);
      }
    }

    contacts() {
      const d = this.d, T = this.T, M = this.M4;
      let used = 0;
      if (this.showContacts && d.ncon > 0) {
        const vec = d.contact;                                     // a copy; it and its elements must be deleted
        try {
          const n = Math.min(d.ncon, vec.size());
          if (!this.cinst || this.cinst.instanceMatrix.count < n) {         // grow the instanced marker pool
            if (this.cinst) { this.scene.remove(this.cinst); this.cinst.dispose(); }
            this.cinst = new T.InstancedMesh(this.contactGeo, this.contactMat, Math.max(64, 2 * n));
            this.cinst.instanceMatrix.setUsage(T.DynamicDrawUsage);
            this.cinst.frustumCulled = false;
            this.scene.add(this.cinst);
          }
          for (let i = 0; i < n; i++) {
            const c = vec.get(i);
            if (!c) continue;
            try {
              const p = c.pos, f = c.frame;                          // frame rows: normal, tangent 1, tangent 2
              M.set(f[3], f[6], f[0], p[0], f[4], f[7], f[1], p[1], f[5], f[8], f[2], p[2], 0, 0, 0, 1);
              this.cinst.setMatrixAt(used++, M);
            } finally { c.delete(); }
          }
        } finally { vec.delete(); }
      }
      if (this.cinst) {
        this.cinst.count = used;
        this.cinst.visible = used > 0;
        this.cinst.instanceMatrix.needsUpdate = true;
      }
      this.contactsShown = used;
    }

    // ---------------------------------------------------------------- controls
    controlsUI() {
      const m = this.m, bar = this.bar;
      this.playBtn = button(bar, "Play", () => this.toggle(), "Step the physics in real time");
      button(bar, "Reset", () => this.reset(), "mj_resetData: qpos0, zero velocity and ctrl");
      if (m.nkey > 0) {
        const keys = h("span", { class: "mjw-set" }, bar, "Keyframes:");
        for (let k = 0; k < m.nkey; k++) button(keys, this.nameOf("key", k) || "key " + k, () => this.key(k), "mj_resetDataKeyframe " + k);
      }
      button(bar, "Home view", () => this.home(), "Camera of the model's <visual><global>");
      const count = new Array(NGROUP).fill(0);
      for (let i = 0; i < m.ngeom; i++) count[clamp(m.geom_group[i], 0, NGROUP - 1)]++;
      const grp = h("span", { class: "mjw-set" }, bar, "Geom groups:");
      this.groupBoxes = [];
      for (let g = 0; g < NGROUP; g++)
        this.groupBoxes.push(checkbox(grp, String(g), this.groups[g], (on) => this.setGroup(g, on), count[g] + " geoms in group " + g));
      this.contactBox = checkbox(bar, "Contacts", this.showContacts, (on) => this.setContacts(on), "Contact points (mjData.contact)");
      if (this.siteObjs.length) this.siteBox = checkbox(bar, "Sites", this.showSites, (on) => this.setSites(on), "Sites in groups 0-2");
      this.buildJoints();
      const servos = this.sliders.filter((s) => s.acts.length).length;
      this.noteEl.textContent = this.sliders.length === 0 ? "" :
        "While paused, a slider sets its joint position. While playing, a slider marked \u2022 sets the target of its"
        + " position actuator" + (servos < this.sliders.length ? " and the other sliders set their joints directly." : ".");
      window.addEventListener("pointerup", () => { for (const s of this.sliders) s.dragging = false; });
    }

    buildJoints() {
      const m = this.m, wrap = this.jointsEl;
      this.sliders = [];
      const servo = {};
      const ng = m.nu ? m.actuator_gainprm.length / m.nu : 10, nb = m.nu ? m.actuator_biasprm.length / m.nu : 10;
      for (let a = 0; a < m.nu; a++) {
        const trn = m.actuator_trntype[a];
        if (trn !== 0 && trn !== 1) continue;                       // mjTRN_JOINT, mjTRN_JOINTINPARENT
        const kp = m.actuator_gainprm[ng * a], b0 = m.actuator_biasprm[nb * a], b1 = m.actuator_biasprm[nb * a + 1];
        if (m.actuator_gaintype[a] !== 0 || m.actuator_biastype[a] !== 1 || !(kp > 0) || b0 !== 0 || Math.abs(b1 + kp) > 1e-9 * kp) continue;
        const j = m.actuator_trnid[2 * a];
        const lo = m.actuator_ctrlrange[2 * a], hi = m.actuator_ctrlrange[2 * a + 1];
        (servo[j] = servo[j] || []).push({                          // ctrlrange given <=> ctrllimited (autolimits)
          a, gear: m.actuator_gear[6 * a] || 1, limited: hi > lo, lo, hi, name: this.nameOf("actuator", a),
        });
      }
      const span = niceSpan(0.5 * this.extent);
      for (let j = 0; j < m.njnt; j++) {
        const type = m.jnt_type[j];
        if (type !== HINGE && type !== SLIDE) continue;
        const hinge = type === HINGE, k = hinge ? DEG : 1000;
        let lo = m.jnt_range[2 * j], hi = m.jnt_range[2 * j + 1];
        if (!(hi > lo)) { lo = hinge ? -Math.PI : -span; hi = hinge ? Math.PI : span; }   // range "0 0": unlimited
        const name = this.nameOf("joint", j) || "joint " + j, acts = servo[j] || [];
        const row = h("div", { class: "mjw-joint" }, wrap);
        const lab = h("span", { class: "mjw-jn" }, row, (acts.length ? "\u2022 " : "") + name);
        lab.title = name + (hinge ? " (hinge, deg)" : " (slide, mm)") + (acts.length ? "; position actuator " + acts.map((x) => x.name || x.a).join(", ") : "");
        const input = h("input", { type: "range", "aria-label": name }, row);
        input.min = String(lo * k); input.max = String(hi * k); input.step = String((hi - lo) * k / 1000);
        const val = h("span", { class: "mjw-jv" }, row);
        const s = { j, name, hinge, k, unit: hinge ? "\u00b0" : " mm", qadr: m.jnt_qposadr[j], dadr: m.jnt_dofadr[j], lo, hi, acts, input, val, dragging: false };
        input.addEventListener("input", () => this.onSlider(s));
        input.addEventListener("pointerdown", () => { s.dragging = true; });
        this.sliders.push(s);
      }
      this.syncSliders();
    }

    onSlider(s) {
      const q = parseFloat(s.input.value) / s.k, d = this.d, ctrl = d.ctrl;
      for (const a of s.acts) ctrl[a.a] = a.limited ? clamp(a.gear * q, a.lo, a.hi) : a.gear * q;
      if (!this.playing || !s.acts.length) {
        d.qpos[s.qadr] = q;
        if (this.playing) d.qvel[s.dadr] = 0;
        else this.mod.mj_forward(this.m, d);
      }
      this.readouts();
      this.requestRender();
    }

    syncSliders() {
      const d = this.d, qpos = d.qpos, ctrl = d.ctrl;
      for (const s of this.sliders) {
        let q = qpos[s.qadr];
        if (this.playing && s.acts.length) q = ctrl[s.acts[0].a] / s.acts[0].gear;
        s.input.value = String(q * s.k);
      }
      this.readouts();
    }

    readouts() {
      const qpos = this.d.qpos;
      for (const s of this.sliders) {
        const q = qpos[s.qadr];
        s.val.textContent = (q * s.k).toFixed(1) + s.unit;
        if (this.playing && !s.acts.length && !s.dragging) s.input.value = String(q * s.k);
      }
    }

    status() {
      const m = this.m, d = this.d;
      let s = "MuJoCo " + VERSIONS.mujoco + " \u00b7 t = " + d.time.toFixed(3) + " s \u00b7 ";
      s += this.playing ? (this.rtf < 0.97 ? (this.rtf.toFixed(2) + "\u00d7 real time") : "real time") : "paused";
      s += " \u00b7 " + m.ngeom + " geoms, " + m.njnt + " joints, " + m.nu + " actuators";
      const nd = Object.keys(this.notDrawn || {});
      if (nd.length) s += " \u00b7 not drawn: " + nd.map((k) => this.notDrawn[k] + " " + k).join(", ");
      if (this.showContacts) s += " \u00b7 " + (this.contactsShown || 0) + " contacts";
      this.statusEl.textContent = s;
    }

    // ---------------------------------------------------------------- public API (buttons, and scripted checks)
    toggle() { if (this.playing) this.pause(); else this.play(); }
    play() {
      if (!this.d || this.playing) return;
      this.playing = true; this.last = null; this.acc = 0;
      this.playBtn.textContent = "Pause";
      this.syncSliders(); this.requestRender();
    }
    pause() {
      if (!this.playing) return;
      this.playing = false;
      this.playBtn.textContent = "Play";
      this.syncSliders(); this.status(); this.requestRender();
    }
    reset() { this.mod.mj_resetData(this.m, this.d); this.after(); }
    key(k) { this.mod.mj_resetDataKeyframe(this.m, this.d, this.keyIndex(k)); this.after(); }
    after() { this.acc = 0; this.mod.mj_forward(this.m, this.d); this.syncSliders(); this.status(); this.requestRender(); }
    step(n) { for (let i = 0; i < (n || 1); i++) this.mod.mj_step(this.m, this.d); this.steps += n || 1; this.syncSliders(); this.status(); this.requestRender(); }
    slider(name) { const s = this.sliders.find((x) => x.name === name); if (!s) throw new Error("no slider for joint '" + name + "'"); return s; }
    setJoint(name, q) { const s = this.slider(name); s.input.value = String(q * s.k); this.onSlider(s); }
    joint(name) { return this.d.qpos[this.slider(name).qadr]; }
    setGroup(g, on) {
      this.groups[g] = !!on;
      if (this.groupBoxes && this.groupBoxes[g]) this.groupBoxes[g].checked = !!on;
      for (const o of this.geomObjs) o.obj.visible = this.groups[o.group];
      this.requestRender();
    }
    setContacts(on) { this.showContacts = !!on; if (this.contactBox) this.contactBox.checked = !!on; this.status(); this.requestRender(); }
    setSites(on) {
      this.showSites = !!on; if (this.siteBox) this.siteBox.checked = !!on;
      for (const o of this.siteObjs) o.obj.visible = this.showSites;
      this.requestRender();
    }
    visibleGeoms() { return this.geomObjs.filter((o) => o.obj.visible).length; }

    // ---------------------------------------------------------------- loop
    observe() {
      if (!("IntersectionObserver" in window)) return;
      new IntersectionObserver((es) => {
        for (const e of es) this.visible = e.isIntersecting;
        if (this.visible) this.requestRender();
      }, { threshold: 0 }).observe(this.stage);
    }
    resize() {
      const w = Math.max(1, this.stage.clientWidth), hh = Math.max(1, this.stage.clientHeight);
      if (w === this.w && hh === this.h) return;
      this.w = w; this.h = hh;
      this.renderer.setSize(w, hh, false);
      this.camera.aspect = w / hh;
      this.camera.updateProjectionMatrix();
      this.requestRender();
    }
    requestRender() { if (!this.raf && this.live) this.raf = requestAnimationFrame(this.frameFn); }
    frame(now) {
      this.raf = 0;
      if (this.playing && this.visible) {
        const dt = this.last === null ? 0 : Math.min((now - this.last) / 1000, 0.2);   // >= 5 frames/s keeps real time
        this.last = now;
        if (dt > 0) this.advance(dt);
        this.raf = requestAnimationFrame(this.frameFn);
      } else this.last = null;
      try { this.render(); } catch (e) {
        if (this.raf) cancelAnimationFrame(this.raf);
        this.raf = 0; this.live = false; this.playing = false;
        this.fail(e);
        return;
      }
      if (!this.playing) this.status();                                      // paused frames come from user input
      else if (now - (this.uiAt || 0) > 100) { this.uiAt = now; this.readouts(); this.status(); }
    }
    advance(dt) {
      const hstep = this.timestep, mod = this.mod, m = this.m, d = this.d, t0 = performance.now(), sim0 = d.time;
      this.acc += dt * this.speed;
      let n = 0;
      while (this.acc >= hstep) {
        mod.mj_step(m, d);
        this.acc -= hstep;
        n++;
        if ((n & 7) === 0 && performance.now() - t0 > 25) { this.acc = 0; break; }   // slower than real time
      }
      this.steps += n;
      this.wallPlayed = (this.wallPlayed || 0) + dt;
      const r = (d.time - sim0) / dt;
      if (isFinite(r) && r >= 0) this.rtf = 0.9 * this.rtf + 0.1 * r;
    }
    render() {
      this.poses();
      this.contacts();
      this.updateLights();
      this.renderer.render(this.scene, this.camera);
      this.frames++;
      const ri = this.renderer.info.render;
      this.lastDraw = { calls: ri.calls, triangles: ri.triangles };
    }
  }

  MJW.Viewer = Viewer;
  MJW.mount = function (cfg) {
    const v = new Viewer(cfg);
    MJW.viewers.push(v);
    return v;
  };
  MJW.all = () => Promise.all(MJW.viewers.map((v) => v.ready));
  const queued = Array.isArray(window.MJWASM_QUEUE) ? window.MJWASM_QUEUE : [];
  window.MJWASM_QUEUE = { push: (cfg) => { MJW.mount(cfg); return 0; } };
  for (const cfg of queued) MJW.mount(cfg);
})();
"""


# ------------------------------------------------------------------------------------------------ CLI
def _fmt_rows(rows: list[Row], total_page: int | None = None) -> str:
    w = max(len(r.asset) for r in rows) if rows else 10
    out = [f"{'asset':<{w}}  {'raw bytes':>12}  {'inlined bytes':>13}  encoding"]
    for r in rows:
        out.append(f"{r.asset:<{w}}  {r.raw:>12,}  {r.inlined:>13,}  {r.how}")
    out.append(f"{'sum':<{w}}  {sum(r.raw for r in rows):>12,}  {sum(r.inlined for r in rows):>13,}")
    if total_page is not None:
        out.append(f"{'whole fragment(s) as written':<{w}}  {'':>12}  {total_page:>13,}")
    return "\n".join(out)


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", type=Path, help="MJCF files")
    ap.add_argument("-o", "--out", type=Path, help="write a standalone page with one viewer per model")
    ap.add_argument("--title", default=None)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--keyframe", default=None)
    ap.add_argument("--sizes", action="store_true", help="print what each inlined asset adds to the page")
    ap.add_argument("--vendor", action="store_true", help="fetch and verify the vendor cache, then exit")
    a = ap.parse_args(argv)
    paths = vendor()
    if a.vendor or not a.models:
        for k, p in paths.items():
            print(f"{k:18s} {p}  sha256 {_sha(p.read_bytes())}")
        return 0
    frags, rows = [], []
    for n, mpath in enumerate(a.models):
        eid = "mjw-" + re.sub(r"[^A-Za-z0-9_-]", "-", mpath.stem)[:40] + (f"-{n}" if len(a.models) > 1 else "")
        frags.append(embed(mpath, element_id=eid, height_px=a.height, title=mpath.name, include_runtime=(n == 0),
                           keyframe=a.keyframe))
        rows += inventory(mpath, element_id=eid, include_runtime=(n == 0))
    body = "\n".join(frags)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(page(body, title=a.title or ", ".join(p.name for p in a.models)), encoding="utf-8")
        print(f"wrote {a.out} ({a.out.stat().st_size:,} bytes)")
    if a.sizes or not a.out:
        print(_fmt_rows(rows, len(body.encode("utf-8"))))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
