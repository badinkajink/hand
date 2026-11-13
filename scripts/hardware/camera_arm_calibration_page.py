#!/usr/bin/env python3
"""Result page of 2026-11-13: lens and arm calibration of the two D435 tracking cameras on the UR5, with the owner
jogging the arm (owner: "sure, build it here, something that we can ideally store, replay, (render a nice mujoco video
of the whole scene would be cool) don't go crazy with it").

    data   (corvus only: reads logs/20261113-camera_arm_calibration/) writes docs/experiments/20261113-camera_arm_calibration/data/:
           poses.jsonl without per-frame metadata, the hand-held views, the fits, and the checks of the hex block and the
           depth (data/fits/20261113-block_and_depth.json, 20261113-block_and_table.json). The hex block's map
           (data/fits/20261113-block_map_fit.json, 20261113-hex_block_map.json) comes from lab/block_map_fit.py, and
           836612071918's pose after its move (data/fits/20261113-handeye_tcp_fitted_lens_moved836.json,
           data/camera_move/) from lab/stereo_pose_fit.py.
    figs   the matplotlib figures into media/ from data/
    page   docs/experiments/20261113-camera_arm_calibration/20261113-camera_arm_calibration.html from data/ and media/

    DOCS_REFRESH=0 /usr/bin/python3 scripts/hardware/camera_arm_calibration_page.py data|figs|page
The film and the overlays come from scripts/hardware/camera_arm_calibration_film.py.
"""
from __future__ import annotations

import base64
import html
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path

ROOT = Path(__file__).resolve().parents[2]

D = ROOT / "docs/experiments/20261113-camera_arm_calibration"
DATA, FITS, MEDIA, LAB = D / "data", D / "data/fits", D / "media", D / "lab"
LOGS = ROOT / "logs/20261113-camera_arm_calibration"
PAGE = D / "20261113-camera_arm_calibration.html"
TEMPLATE = ROOT / "scripts/ur5/ur5_tip_calibration_page.template.html"
SNS = ["832412070344", "836612071918"]
TAGS = [12.9, 13.0, 13.1, 13.2, 13.33]


def j(p):
    return json.loads(Path(p).read_text())


def rows(p):
    return [json.loads(l) for l in open(p)]


def inv(M):
    N = np.eye(4)
    N[:3, :3], N[:3, 3] = M[:3, :3].T, -M[:3, :3].T @ M[:3, 3]
    return N


# ------------------------------------------------------------------ data (corvus)
def data():
    import cv2
    from sr2hand.tracking import pose as tp
    from sr2hand.tracking.tags import HexBlock
    DATA.mkdir(parents=True, exist_ok=True)
    FITS.mkdir(parents=True, exist_ok=True)
    with open(DATA / "poses.jsonl", "w") as f:
        for r in rows(LOGS / "poses.jsonl"):
            for c in r["cams"].values():
                c.pop("depth_frames", None)
                for s in ("color", "ir1_off"):
                    c[s].pop("frames", None)
            f.write(json.dumps(r) + "\n")
    for d in ("lens_paper", "lens_paper_b"):
        (DATA / d).mkdir(exist_ok=True)
        for n in ("views.jsonl", "session.json", "lens_views.out"):
            shutil.copy(LOGS / d / n, DATA / d / n)
    for p in (LOGS / "fits").glob("*.json"):
        shutil.copy(p, FITS / p.name)
    # hand-eye with a board scale factor (the arm's metric against the print)
    he = LAB / "handeye_fit.py"
    lens = FITS / "20261113-lens_board_k3_arm.json"
    for v, name in (("--tcp", "tcp"), ("", "fk")):
        cmd = ["/usr/bin/python3", str(he), str(DATA), str(lens), "--fit-scale", "--json",
               str(FITS / f"20261113-handeye_{name}_fitted_lens_scale.json")] + ([v] if v else [])
        subprocess.run(cmd, check=True, env=dict(os.environ, OPENBLAS_NUM_THREADS="1"), capture_output=True)
    L = j(lens)
    he_tcp, he_fk = j(FITS / "20261113-handeye_tcp_fitted_lens.json"), j(FITS / "20261113-handeye_fk_fitted_lens.json")
    recs = rows(DATA / "poses.jsonl")
    det = tp.detector()

    def cam(sn, factory=False):
        if factory:
            I = recs[0]["cams"][sn]["streams"]["color"]
            return tp.Camera(np.array([[I["fx"], 0, I["cx"]], [0, I["fy"], I["cy"]], [0, 0, 1.0]]), np.zeros(5), 1920, 1080)
        l = L["cams"][sn]["lens"]
        return tp.Camera(np.array([[l["fx"], 0, l["cx"]], [0, l["fy"], l["cy"]], [0, 0, 1.0]]),
                         np.array([l["k1"], l["k2"], l["p1"], l["p2"], l["k3"]]), 1920, 1080)

    obs = {(r["pose"], sn): tp.detect(det, cv2.imread(str(LOGS / f"p{r['pose']:02d}" / f"{sn}_color.png"), 0),
                                      ids=[10, 11, 12, 13, 14, 15]) for r in recs for sn in SNS}
    raw = {(r["pose"], sn): np.median(np.load(LOGS / f"p{r['pose']:02d}" / f"{sn}_depth_raw.npz")["z16"].astype(float), 0)
           for r in recs for sn in SNS}

    def depth_at(Z, uv, rr=2):
        u, v = int(round(uv[0])), int(round(uv[1]))
        w = Z[max(v - rr, 0):v + rr + 1, max(u - rr, 0):u + rr + 1]
        w = w[w > 0]
        return float(np.median(w)) if w.size else np.nan

    def block(ts, he, factory=False):
        TM = HexBlock(across_flats=29.985, tag_size=ts).tag_map()
        cen, Ts, dz = {sn: [] for sn in SNS}, {sn: [] for sn in SNS}, []
        for r in recs:
            for sn in SNS:
                T, info = tp.solve_body(obs[(r["pose"], sn)], TM, cam(sn, factory))
                if T is None:
                    continue
                TB = inv(np.array(he["cams"][sn]["T_CB"])) @ T
                cen[sn].append(TB[:3, 3].tolist())
                Ts[sn].append(TB.tolist())
                st = r["cams"][sn]["streams"]
                Rdc, tdc = np.array(st["depth_to_color"]["R"]), 1000 * np.array(st["depth_to_color"]["t"])
                di = st["depth"]
                Kd = np.array([[di["fx"], 0, di["cx"]], [0, di["fy"], di["cy"]], [0, 0, 1.0]])
                for i in info["tags"]:
                    pc = T[:3, :3] @ TM.frames[i][0] + T[:3, 3]
                    pd = Rdc.T @ (pc - tdc)
                    uv = (Kd @ pd)[:2] / pd[2]
                    z = depth_at(raw[(r["pose"], sn)], uv) * st["depth_scale"] * 1000
                    if np.isfinite(z):
                        dz.append((sn, r["pose"], i, z - pd[2], pd[2]))
        a, b = np.mean(cen[SNS[0]], 0), np.mean(cen[SNS[1]], 0)
        return dict(tag_size=ts, centre={sn: np.mean(cen[sn], 0).tolist() for sn in SNS},
                    sd={sn: np.std(cen[sn], 0).tolist() for sn in SNS}, n={sn: len(cen[sn]) for sn in SNS},
                    diff_mm=float(np.linalg.norm(a - b)), depth_minus_solve=dz, T=Ts)

    out = dict(scan=[], factory_1333=None)
    for ts in TAGS:
        b = block(ts, he_tcp)
        b.pop("T")
        out["scan"].append(b)
    b = block(13.33, j(FITS / "20261113-handeye_tcp_factory_lens.json"), factory=True)
    b.pop("T")
    out["factory_1333"] = b
    # tag 13 triangulated from both cameras (fitted lens, TCP route)
    tri = []
    for r in recs:
        o0, o1 = obs[(r["pose"], SNS[0])], obs[(r["pose"], SNS[1])]
        if 13 in o0 and 13 in o1:
            P = []
            for sn, o in ((SNS[0], o0), (SNS[1], o1)):
                c = cam(sn)
                P.append((np.array(he_tcp["cams"][sn]["T_CB"])[:3, :], cv2.undistortPoints(o[13].reshape(-1, 1, 2), c.K, c.dist).reshape(-1, 2)))
            X = cv2.triangulatePoints(P[0][0], P[1][0], P[0][1].T, P[1][1].T)
            tri.append((X[:3] / X[3]).T.tolist())
    tri = np.array(tri)
    m = tri.mean(0)
    out["tag13"] = dict(n=len(tri), edges_mm=np.linalg.norm(np.roll(m, -1, 0) - m, axis=1).tolist(),
                        corner_sd_mm=float(np.linalg.norm(tri.std(0), axis=1).mean()))
    # depth at the board in IR1 (projector off), board shape fitted, no colour lens
    A = cv2.aruco
    BOARD = A.CharucoBoard_create(7, 7, 15.0, 10.5, A.getPredefinedDictionary(A.DICT_5X5_50))
    Pb = np.array(BOARD.chessboardCorners, float) + np.array(L["board_offsets_mm"])
    ir = []
    for r in recs:
        for sn in SNS:
            c = r["cams"][sn]["ir1_off"]
            if len(c["corner_ids"]) < 12:
                continue
            ids, uv = np.array(c["corner_ids"]), np.array(c["corners_mean"], float)
            ii = r["cams"][sn]["streams"]["ir1"]
            K = np.array([[ii["fx"], 0, ii["cx"]], [0, ii["fy"], ii["cy"]], [0, 0, 1.0]])
            ok, rv, tv = cv2.solvePnP(Pb[ids], uv, K, None, flags=cv2.SOLVEPNP_SQPNP)
            ok, rv, tv = cv2.solvePnP(Pb[ids], uv, K, None, rv, tv, True, cv2.SOLVEPNP_ITERATIVE)
            X = (cv2.Rodrigues(rv)[0] @ Pb[ids].T).T + tv.ravel()
            sc = r["cams"][sn]["streams"]["depth_scale"] * 1000
            z = np.array([depth_at(raw[(r["pose"], sn)], q, 1) * sc for q in uv])
            k = np.isfinite(z)
            if k.sum() >= 6:
                ir.append(dict(cam=sn, pose=r["pose"], z_mm=float(np.median(X[k, 2])),
                               depth_minus_solve_mm=float(np.median(z[k] - X[k, 2])), n=int(k.sum())))
    out["ir1"] = ir
    (FITS / "20261113-block_and_depth.json").write_text(json.dumps(out, indent=1))
    # block and table for the film (the model's kinematics: the fk route; tags 13.0)
    b = block(13.0, he_fk)
    Ts = [np.array(T) for sn in SNS for T in b["T"][sn]]
    c = np.mean([T[:3, 3] for T in Ts], 0)
    R = np.array(b["T"][SNS[1]][0])[:3, :3]
    st = recs[0]["cams"][SNS[1]]["streams"]
    di = st["depth"]
    Z = raw[(0, SNS[1])] * st["depth_scale"] * 1000
    v, u = np.mgrid[0:720:3, 0:1280:3]
    z = Z[v, u]
    k = z > 0
    pd = np.stack([(u[k] - di["cx"]) / di["fx"] * z[k], (v[k] - di["cy"]) / di["fy"] * z[k], z[k]], 1)
    Rdc, tdc = np.array(st["depth_to_color"]["R"]), 1000 * np.array(st["depth_to_color"]["t"])
    pc = (Rdc @ pd.T).T + tdc
    T_BC = inv(np.array(he_fk["cams"][SNS[1]]["T_CB"]))
    pb = (T_BC[:3, :3] @ pc.T).T + T_BC[:3, 3]
    rho = np.linalg.norm(pb[:, :2] - c[:2], axis=1)                  # the table around the block's stand
    pb = pb[(rho > 25) & (rho < 120) & (pb[:, 2] < c[2] - 8.5) & (pb[:, 2] > c[2] - 120)]
    T_B = np.eye(4)
    T_B[:3, :3], T_B[:3, 3] = R, c
    (FITS / "20261113-block_and_table.json").write_text(json.dumps(dict(
        T_B_block=T_B.tolist(), across_flats=29.985, height=17.0, tag_size=13.0,
        table_z_mm=float(np.median(pb[:, 2])), table_z_iqr_mm=np.percentile(pb[:, 2], [25, 75]).tolist(),
        table_points=int(len(pb))), indent=1))
    print("data written to", DATA)


# ------------------------------------------------------------------ figures
def figs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13, "font.family": "serif"})
    MEDIA.mkdir(parents=True, exist_ok=True)
    recs = rows(DATA / "poses.jsonl")
    views = [v for d in ("lens_paper", "lens_paper_b") for v in rows(DATA / d / "views.jsonl")]
    bd = j(FITS / "20261113-block_and_depth.json")
    # coverage
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.9))
    for a, sn in zip(ax, SNS):
        hv = np.concatenate([np.array(v["corners"]) for v in views if v["cam"] == sn])
        av = np.concatenate([np.array(r["cams"][sn]["color"]["corners_mean"]).reshape(-1, 2) for r in recs])
        a.scatter(hv[:, 0], hv[:, 1], s=2, c="0.6", label="hand-held views")
        a.scatter(av[:, 0], av[:, 1], s=3, c="C3", label="arm poses")
        tg = np.concatenate([np.array(t["corners"]) for t in recs[0]["cams"][sn]["color"]["tags"] if 10 <= t["id"] <= 15])
        a.plot(*tg.mean(0), "kx", ms=12, mew=3, label="hex block")
        a.set_xlim(0, 1920)
        a.set_ylim(1080, 0)
        a.set_aspect("equal")
        a.set_title(f"Camera {sn}")
        a.set_xlabel("u (px)")
    ax[0].set_ylabel("v (px)")
    ax[0].legend(loc="lower left", fontsize=11, markerscale=4)
    fig.tight_layout()
    fig.savefig(MEDIA / "20261113-corner_coverage.png", dpi=110)
    plt.close(fig)
    # board shape and per-pose hand-eye error
    L = j(FITS / "20261113-lens_board_k3_arm.json")
    import cv2
    A = cv2.aruco
    P0 = np.array(A.CharucoBoard_create(7, 7, 15.0, 10.5, A.getPredefinedDictionary(A.DICT_5X5_50)).chessboardCorners)
    dB = np.array(L["board_offsets_mm"])
    z = (P0 + dB)[:, 2]
    M = np.c_[P0[:, :2], np.ones(36)]
    zr = z - M @ np.linalg.lstsq(M, z, rcond=None)[0]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw=dict(width_ratios=[1, 1.6]))
    im = ax[0].imshow(zr.reshape(6, 6), cmap="RdBu_r", vmin=-0.55, vmax=0.55, extent=[7.5, 97.5, 97.5, 7.5])
    for (r_, c_), v in np.ndenumerate(zr.reshape(6, 6)):
        ax[0].text(15 + 15 * c_, 15 + 15 * r_, f"{v:+.2f}", ha="center", va="center", fontsize=10)
    ax[0].set_xlabel("x on the board (mm)")
    ax[0].set_ylabel("y on the board (mm)")
    ax[0].set_title("Corner height off the best plane (mm)")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    w = 0.2
    for k, (name, f) in enumerate((("TCP pose, fitted lens", "tcp_fitted_lens"), ("fk(q), fitted lens", "fk_fitted_lens"),
                                   ("TCP pose, factory lens", "tcp_factory_lens"))):
        he = j(FITS / f"20261113-handeye_{f}.json")
        for s, sn in enumerate(SNS):
            pp = np.array(he["cams"][sn]["per_pose"])
            ax[1].bar(pp[:, 0] + (k - 1) * w + (s - 0.5) * w / 2, pp[:, 1], w / 2, color=f"C{k}", alpha=1.0 if s == 0 else 0.55,
                      label=name if s == 0 else None)
    ax[1].set_xlabel("arm pose")
    ax[1].set_ylabel("reprojection RMS (px)")
    ax[1].set_title("Hand-eye fit at each arm pose")
    ax[1].set_xticks(range(17))
    ax[1].legend(fontsize=11, title="dark 832412070344, light 836612071918", title_fontsize=11)
    fig.tight_layout()
    fig.savefig(MEDIA / "20261113-board_shape_and_handeye.png", dpi=110)
    plt.close(fig)
    # tag-size scan and depth vs IR1 solve
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    ts = [s["tag_size"] for s in bd["scan"]]
    ax[0].plot(ts, [s["diff_mm"] for s in bd["scan"]], "o-", color="C0")
    e = bd["tag13"]["edges_mm"]
    ax[0].axvline(np.mean(e), color="C3", ls="--", label=f"tag 13 triangulated, {np.mean(e):.2f} mm")
    ax[0].set_xlabel("tag size in the block's map (mm)")
    ax[0].set_ylabel("distance between the two\ncameras' block centres (mm)")
    ax[0].set_title("Hex block seen by both cameras")
    ax[0].legend(fontsize=11)
    for sn, c in zip(SNS, ("C1", "C0")):
        pts = np.array([(r["z_mm"], r["depth_minus_solve_mm"]) for r in bd["ir1"] if r["cam"] == sn])
        ax[1].plot(pts[:, 0], pts[:, 1], "o", color=c, label=f"camera {sn}")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xlabel("board distance in the depth frame (mm)")
    ax[1].set_ylabel("depth minus IR1 solve (mm)")
    ax[1].set_title("D435 depth at the board's corners")
    ax[1].legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(MEDIA / "20261113-tag_size_and_depth.png", dpi=110)
    plt.close(fig)
    print("figures written to", MEDIA)


# ------------------------------------------------------------------ page
def esc(s):
    return html.escape(str(s), quote=True)


def img_uri(p, width=1100, quality=86):
    from PIL import Image
    im = Image.open(p).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def fig(n, p, caption, width=1100):
    return (f'<figure id="fig{n}"><img src="{img_uri(p, width)}" alt="{esc(caption.split(".")[0])}">'
            f"<figcaption>Figure {n}. {caption}</figcaption></figure>")


def film_html(n, mp4, poster, caption):
    v = "data:video/mp4;base64," + base64.b64encode(Path(mp4).read_bytes()).decode()
    return (f'<figure id="fig{n}"><video controls muted loop playsinline autoplay preload="metadata" '
            f'poster="{img_uri(poster, 1400)}" src="{v}" style="width:100%"></video>'
            f"<figcaption>Figure {n}. {caption}</figcaption></figure>")


def table(head, body, num=(), caption=None, n=None):
    cap = f"<caption>Table {n}. {caption}</caption>" if caption else ""
    th = "".join(f'<th{" class=n" if k in num else ""}>{h}</th>' for k, h in enumerate(head))
    tr = "".join("<tr>" + "".join(f'<td{" class=n" if k in num else ""}>{c}</td>' for k, c in enumerate(r)) + "</tr>"
                 for r in body)
    return f'<div class="tw"><table>{cap}<thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def code(s):
    return f"<code>{esc(s)}</code>"


def page():
    sys.path.insert(0, str(ROOT / "scripts/docs"))
    import docs_index
    import retro_style
    recs = rows(DATA / "poses.jsonl")
    L = j(FITS / "20261113-lens_board_k3_arm.json")
    Lh = j(FITS / "20261113-lens_board_k3_handheld.json")
    Ln = j(FITS / "20261113-lens_board_nok3_handheld.json")
    H = {k: j(FITS / f"20261113-handeye_{k}.json") for k in ("tcp_fitted_lens", "fk_fitted_lens", "tcp_factory_lens")}
    Hs = {k: j(FITS / f"20261113-handeye_{k}_fitted_lens_scale.json") for k in ("tcp", "fk")}
    bd = j(FITS / "20261113-block_and_depth.json")
    ovl = j(FITS / "20261113-render_overlap.json")
    ovm = {sn: np.median([x["rms_px"] for x in ovl["rows"] if x["cam"] == sn and x["rms_px"] is not None]) for sn in SNS}
    ovx = {sn: max(x["rms_px"] for x in ovl["rows"] if x["cam"] == sn and x["rms_px"] is not None) for sn in SNS}
    views = {d: rows(DATA / d / "views.jsonl") for d in ("lens_paper", "lens_paper_b")}
    fac = {sn: recs[0]["cams"][sn]["streams"]["color"] for sn in SNS}
    he = H["tcp_fitted_lens"]
    T_BC = {sn: inv(np.array(he["cams"][sn]["T_CB"])) for sn in SNS}
    sep = np.linalg.norm(T_BC[SNS[0]][:3, 3] - T_BC[SNS[1]][:3, 3])
    ang = np.degrees(np.arccos(T_BC[SNS[0]][:3, 2] @ T_BC[SNS[1]][:3, 2]))
    dpm = {sn: np.median(he["cams"][sn]["board_arm_vs_pnp_mm"]) for sn in SNS}
    dmx = max(max(he["cams"][sn]["board_arm_vs_pnp_mm"]) for sn in SNS)
    scan = {s["tag_size"]: s for s in bd["scan"]}
    e13 = bd["tag13"]["edges_mm"]
    ir = {sn: np.median([r["depth_minus_solve_mm"] for r in bd["ir1"] if r["cam"] == sn]) for sn in SNS}
    irz = [r["z_mm"] for r in bd["ir1"]]
    dz13 = np.median([d[3] for d in scan[13.0]["depth_minus_solve"]])
    dz133 = np.median([d[3] for d in scan[13.33]["depth_minus_solve"]])
    fx = {sn: L["cams"][sn]["lens"]["fx"] for sn in SNS}
    sc = Hs["tcp"]["board_scale"]
    bm = j(FITS / "20261113-block_map_fit.json")
    bA, bB, bc = bm["fits"]["A"], bm["fits"]["B_across_flats"], bm["checks"]
    bviews = rows(DATA / "20261113-block_views.jsonl")
    lying = [np.degrees(np.arccos(abs(np.array(r["T_PB_nominal"])[2, 2]))) > 45 for r in bviews]
    pv = bA["rms_px_per_view"]
    up_med = np.median([x for x, l in zip(pv, lying) if not l and x is not None])
    ly_med = np.median([x for x, l in zip(pv, lying) if l and x is not None])
    bsz = list(bA["sizes_mm"].values())
    bin_ = [-t["along_normal_mm"] for t in bA["tags"].values()]
    bturn = [t["turn_deg"] for t in bA["tags"].values()]
    bside = max(t["in_face_mm"] for t in bA["tags"].values())
    bopp, bBopp = list(bA["opposite_mm"].values()), list(bB["opposite_mm"].values())
    arm_fit, arm_13 = bc["fitted A"]["arm_poses"]["cameras_apart_mm"], bc["nominal 13.0"]["arm_poses"]["cameras_apart_mm"]
    mv_he = j(FITS / "20261113-handeye_tcp_fitted_lens_moved836.json")
    mv, mvc = mv_he["moved"], mv_he["moved"]["checks"]
    mvt = j(DATA / "camera_move/20261113-stereo_check_unmoved.json")["moved"]
    MV, FX = mv["camera"], mv["fixed"]
    T_BC2 = {sn: inv(np.array(mv_he["cams"][sn]["T_CB"])) for sn in SNS}
    sep2 = np.linalg.norm(T_BC2[SNS[0]][:3, 3] - T_BC2[SNS[1]][:3, 3])

    lede = (f"Both D435 colour cameras now have fitted lenses and fitted poses in the UR5's base. The lenses reproject the "
            f"board's corners at {L['cams'][SNS[0]]['rms_px']:.2f} and {L['cams'][SNS[1]]['rms_px']:.2f} px. Their focal "
            f"lengths are {fx[SNS[0]]:.0f} and {fx[SNS[1]]:.0f} px, 1.5 % under the factory values, with radial "
            f"distortion that the factory model leaves out. The board's position from the arm, through the controller's TCP "
            f"pose, differs from each camera's own solve by {dpm[SNS[0]]:.2f} and {dpm[SNS[1]]:.2f} mm (median) and "
            f"{dmx:.2f} mm at worst. The depth is good to 1 to 2 mm. The hex block's tags measure 13.0 mm, not the "
            f"13.33 mm of its 11-09 map. A map of the block fitted from {bm['views']} views in both cameras brings the two "
            f"cameras' placements of it from {arm_13:.1f} mm apart to {arm_fit:.2f} mm. Afterwards {MV} was moved for a "
            f"better view of the hand; its new pose, solved from {mv['views']} views of the board in both cameras, "
            f"varies by {mv['bootstrap_sd_mm']:.2f} mm.")
    summary = (f"D435 lens and arm calibration on the UR5 (17 owner-jogged poses, 120 hand-held views): focal lengths "
               f"{fx[SNS[0]]:.0f}/{fx[SNS[1]]:.0f} px with distortion, board from the arm vs each camera "
               f"{dpm[SNS[0]]:.2f}/{dpm[SNS[1]]:.2f} mm median through the TCP pose; depth within 1-2 mm; "
               f"hex block map fitted from {bm['views']} two-camera views (tags 13.0 mm, not 13.33, recessed 0.3 mm): "
               f"cameras {arm_fit:.2f} mm apart on the block, from {arm_13:.2f}; {MV} later moved and re-solved from "
               f"{mv['views']} board views in both cameras (sd {mv['bootstrap_sd_mm']:.2f} mm).")

    ov = "".join(f'<img src="{img_uri(MEDIA / f"20261113-overlay_p{p:02d}.jpg", 1400)}" alt="overlay pose {p}" style="width:100%">'
                 for p in (0, 3, 12))
    ov_fig = (f'<figure id="fig3">{ov}<figcaption>Figure 3. Each camera\'s frame blended half and half with the render from '
              f'the fitted camera, at arm poses 0, 3 and 12 (rows). The frames are undistorted with the fitted lenses; '
              f'{SNS[0]} is on the left, {SNS[1]} on the right. The left part of {SNS[1]}\'s frames, where the owner sat, '
              f'is blurred.</figcaption></figure>')
    S = []
    S.append(("terms", "Terms and metrics", f"""
<dl class="terms">
<dt>Board</dt><dd>The paper ChArUco board printed by the tag-block session and glued to a stiff back: 7 &times; 7
squares of 15.0 mm as printed, with markers of DICT_5X5_100. Its 36 inner corners are detected in each image.</dd>
<dt>Arm pose</dt><dd>One still position of the UR5 with the board on the SR2 hand, jogged by the owner on the pendant.
Each camera records 15 colour frames, 15 depth frames and 15 left infrared frames there; the corners are averaged over
the 15 frames.</dd>
<dt>Hand-held view</dt><dd>One colour frame of the board in the owner's hand. A frame was saved when the board was
still and in a new place.</dd>
<dt>Reprojection RMS</dt><dd>The root mean square distance, in pixels, between the detected corners and the corners
projected through the fitted model.</dd>
<dt>Lens</dt><dd>The focal lengths fx and fy, the principal point cx and cy (px), and OpenCV's distortion terms k1, k2
and k3 (radial) and p1 and p2 (tangential).</dd>
<dt>Hand-eye fit</dt><dd>The fit of each camera's pose in the UR5's base and the board's pose on the flange (tool0)
from all arm poses at once.</dd>
<dt>TCP pose, fk(q)</dt><dd>The two ways to get the flange's pose from the arm. The TCP pose is the controller's
reading, computed with the arm's own calibrated kinematics. fk(q) is the nominal UR5 table applied to the joint
angles (<code>sr2hand.robot.ur_kinematics.fk</code>).</dd>
<dt>Arm vs camera</dt><dd>At each arm pose the board's centre is placed in two ways: through the arm and the hand-eye
fit, and by that camera's solve alone. Given are the distance between the two (mm) and the angle between the two
orientations.</dd>
</dl>"""))

    S.append(("setup", "Setup and procedure", f"""
<p>The two D435s stand on the table, both upright at 1920 &times; 1080, serials {SNS[0]} and {SNS[1]}. The board was
fixed to the SR2 hand on the UR5. The owner jogged the arm with the pendant through 17 poses. Nine were moves of the
tool, &plusmn;30 to &plusmn;120 mm sideways, 30 mm up, and 40 mm toward and away from the cameras. Six were turns of
0.3 rad (17&deg;) each way about the tool's three axes. Two combined a side move with a turn. Table 5 lists them.
Colour exposure, gain and white balance were fixed for the session; the white paper reads about 200 of 255.</p>
<p>The owner then took the board off the hand and held it in front of both cameras, for 60 views per camera. The
views were tilted up to 70&deg;, and their corners reached x 43 to 1833 px and y 59 to 1006 px (Figure 2). Views tilted more than 70&deg; are left out of
the fits. During both parts the hex block stood still on the table.</p>
<p>RealSense Viewer and the recording scripts cannot open the cameras at the same time on corvus, even with the
viewer's streams off. The viewer is a source build that claims the USB interfaces itself. A small window of both
cameras with the detected corners (<code>lab/preview.py</code>) replaced it while the owner jogged; it closes the
cameras for the 20 s of each recording.</p>"""))

    lens_rows = []
    for sn in SNS:
        f = fac[sn]
        lens_rows.append([f"{sn} factory", f"{f['fx']:.1f}", f"{f['fy']:.1f}", f"{f['cx']:.1f}", f"{f['cy']:.1f}", "0", "0", "0", "&ndash;"])
        for name, LL in (("hand-held, k3 = 0", Ln), ("hand-held", Lh), ("hand-held and arm", L)):
            l, s = LL["cams"][sn]["lens"], LL["cams"][sn]["sd"]
            lens_rows.append([f"{sn} {name}", f"{l['fx']:.1f} &plusmn; {s['fx']:.1f}", f"{l['fy']:.1f}", f"{l['cx']:.1f}",
                              f"{l['cy']:.1f}", f"{l['k1']:+.3f}", f"{l['k2']:+.3f}", f"{l['k3']:+.3f}",
                              f"{LL['cams'][sn]['rms_px']:.3f}"])
    S.append(("lens", "Lenses and the board's shape", f"""
<p>The board is not flat. Fitted as flat, the lenses reproject at 1.04 and 0.87 px, though the corners of one burst
scatter by only 0.07 px. Each of the board's 36 corners was therefore given a free 3D position, shared by every view
of both cameras. The board's mean spacing was kept at the printed 15.0 mm. The corners lie 0.25 mm RMS off the best
plane, with a 1.0 mm dip toward the middle (Figure 4, left). The shape found from the hand-held views alone also fits
the arm poses, which were taken with the board on the hand. Their RMS falls from 0.61 to 0.19 px and from 0.77 to
0.29 px. The back kept its shape when the owner took the board off.</p>
<p>Table 1 gives the lenses. The final fit uses the hand-held views and the arm poses together, with five distortion
terms. Its focal lengths are 1.5 % under the factory values, and its radial terms are large. The radial terms trade
off against each other: k3 changes the focal length by 11 px (Table 1, rows with k3 = 0).</p>
{table(["camera and fit", "fx (px)", "fy", "cx", "cy", "k1", "k2", "k3", "RMS (px)"], lens_rows, num=range(1, 9),
       caption="Each camera's lens: the factory model and three fits with the board's shape free. &plusmn; is one standard deviation of fx.", n=1)}
{fig(2, MEDIA / "20261113-corner_coverage.png", "The board's corners in each image: the hand-held views (grey) and the arm poses (red). The cross marks the hex block.")}"""))

    he_rows = []
    for name, k in (("TCP pose, fitted lens", "tcp_fitted_lens"), ("fk(q), fitted lens", "fk_fitted_lens"),
                    ("TCP pose, factory lens", "tcp_factory_lens")):
        h = H[k]
        for sn in SNS:
            c = h["cams"][sn]
            he_rows.append([name, sn, f"{c['rms_px']:.2f}", f"{np.median(c['board_arm_vs_pnp_mm']):.2f}",
                            f"{max(c['board_arm_vs_pnp_mm']):.2f}", f"{np.median(c['board_arm_vs_pnp_deg']):.2f}",
                            f"{max(c['board_arm_vs_pnp_deg']):.2f}"])
    S.append(("handeye", "Camera poses in the arm's base", f"""
<p>In the hand-eye fit, each camera's lens and the board's shape are fixed. It solves six numbers for each camera and six
for the board on tool0 from all 34 camera views of the 17 arm poses. Table 2 compares three versions. Through the
controller's TCP pose with the fitted lenses, the reprojection RMS is {he['cams'][SNS[0]]['rms_px']:.2f} and
{he['cams'][SNS[1]]['rms_px']:.2f} px. The board's position from the arm agrees with each camera's own solve to
{dpm[SNS[0]]:.2f} and {dpm[SNS[1]]:.2f} mm (median), {dmx:.2f} mm at worst. The plan's acceptance for step 5 is 1 mm
and 0.5&deg; (<code>docs/notes/20261105-object_seating_and_tracking/</code>); one pose of {SNS[0]} exceeds it.
Through fk(q), the nominal table, the RMS is 1.7 times as large. With the factory lenses the disagreement reaches
3.1 mm and 1.6&deg;.</p>
<p>The cameras are {sep:.0f} mm apart and their viewing axes {ang:.1f}&deg; apart. The film (Figure 1) replays the 17
poses in the MuJoCo model of the hand on the UR5, with both cameras placed by this fit. In the model the gantries stand
at the station's home layout, and the fingers hang at the angles read from the limp servos on 11-13. Figure 3 lays the
render from each fitted camera over that camera's undistorted frame. The board's corners, detected in both, lie
{ovm[SNS[0]]:.2f} and {ovm[SNS[1]]:.2f} px apart (median over the 17 poses; {ovx[SNS[0]]:.1f} and
{ovx[SNS[1]]:.1f} px at worst). The render places the arm through fk(q), the model's own kinematics.</p>
{ov_fig}
<p>In a second fit, one more number was free: a scale factor of the board. The arm's own distances then set the
board's size. It comes out at {sc:.4f} (&plusmn; {Hs['tcp']['board_scale_sd']:.4f}) through the TCP pose and
{Hs['fk']['board_scale']:.4f} through fk(q): the squares measure {15 * sc:.3f} mm against the printed 15.0 mm.</p>
{table(["arm route and lens", "camera", "RMS (px)", "arm vs camera, median (mm)", "max (mm)", "median (deg)", "max (deg)"],
       he_rows, num=range(2, 7), caption="Hand-eye fits over the 17 arm poses.", n=2)}
{fig(4, MEDIA / "20261113-board_shape_and_handeye.png", "Left: the fitted board's corners off their best plane. Right: the hand-eye fit's reprojection RMS at each arm pose for the three versions of Table 2.")}"""))

    sc_rows = [[f"{s['tag_size']:.2f}", f"{s['diff_mm']:.2f}"] for s in bd["scan"]]
    sc_rows.append(["13.33, factory lens", f"{bd['factory_1333']['diff_mm']:.2f}"])
    S.append(("block", "Depth and the hex block", f"""
<p>The depth was checked without the colour lens. In the left infrared image, with the projector off, the board's
corners give its pose in the depth camera's own frame. The median depth at those corners lies {ir[SNS[0]]:+.2f} mm
beyond that pose for {SNS[0]} and {ir[SNS[1]]:+.2f} mm for {SNS[1]}. The board stood {min(irz):.0f} to
{max(irz):.0f} mm from the cameras (Figure 5, right). A board 0.2 % larger than printed, as the arm's scale says,
accounts for about 0.6 mm of that.</p>
<p>The hex block stood still throughout. Solved with its 11-09 map (across flats 29.985 mm, tags 13.33 mm) and the
fitted lenses, each camera places it within 0.2 mm across the 17 poses. The two cameras place it
{scan[13.33]['diff_mm']:.1f} mm apart in the base. Tag 13, seen by both, triangulates at
{', '.join(f'{x:.2f}' for x in e13)} mm on its four edges (mean {np.mean(e13):.2f} mm, {bd['tag13']['n']} poses).
With 13.0 mm tags the two cameras differ by {scan[13.0]['diff_mm']:.1f} mm (Table 3). The depth at {SNS[1]}'s
tag faces then lies {dz13:+.1f} mm from the solve, against {dz133:+.1f} mm with 13.33 mm tags. {SNS[0]} stands 240 to 246 mm
from the block, closer than the D435's depth range, and gives no depth there. Section 6 fits the block's own map.</p>
{table(["tag size in the map (mm)", "distance between the two cameras' block centres (mm)"], sc_rows, num=(1,),
       caption="The hex block placed through each camera's fitted pose, against the tag size of its map (fitted lenses unless noted).", n=3)}
{fig(5, MEDIA / "20261113-tag_size_and_depth.png", "Left: the distance between the block centres placed through the two cameras, against the tag size of the block's map. Right: depth minus the infrared solve of the board at each arm pose.")}"""))

    ck = [("CAD map, 13.33 mm tags", bc["nominal 13.33"]), ("CAD map, 13.0 mm tags", bc["nominal 13.0"]),
          ("fitted map", bc["fitted A"])]
    ck_rows = [[n, f"{c['arm_poses']['cameras_apart_mm']:.2f}",
                f"{c['held_out']['cameras_apart_mm']['median']:.2f} / {c['held_out']['cameras_apart_mm']['max']:.2f}",
                f"{c['held_out']['joint_rms_px']['median']:.2f}"] for n, c in ck]
    S.append(("blockmap", "A map of the hex block from both cameras", f"""
<p>After the arm session the owner turned the hex block by hand in front of both cameras. This is the test block
printed on 11-07. It stood upright at several places on the table in {len(lying) - sum(lying)} views and lay on a side
face in {sum(lying)}. <code>lab/block_views.py</code> saved a view when the block was still in both cameras and had
turned 15&deg; or moved 25 mm from every saved view. Each tag's corners are the mean of six frames.</p>
<p>The fit (<code>sr2hand.tracking.pose.bundle_adjust_multi</code>) keeps both cameras at their fitted poses and lenses.
Their {sep:.0f} mm baseline then sets the scale, so no caliper length enters. Each tag's pose and size on the block are
free, and so is the block's pose in each view. Tag 10 fixes the block's frame. The result is fitted rigidly onto the CAD
map (across flats 29.985 mm, tags 13.0 mm). Tags seen more than 65&deg; from their normal are left out, as in the
tracker.</p>
<p>The corners reproject at {bA['rms_px']:.2f} px: a median {up_med:.2f} px in the upright views and {ly_med:.2f} px in
the lying ones. The tags measure {min(bsz):.2f} to {max(bsz):.2f} mm. Each sits {min(bin_):.2f} to {max(bin_):.2f} mm
inside its face and is turned {min(bturn):.1f} to {max(bturn):.1f}&deg;, with at most {bside:.2f} mm of sideways shift.
Opposite tags lie {min(bopp):.2f} to {max(bopp):.2f} mm apart; the calipers measure 29.985 mm across the faces. The
tags are printed slightly recessed (owner). A second fit that ties the opposite tags to the calipers' length, within 0.03 mm, still puts them
{min(bBopp):.2f} to {max(bBopp):.2f} mm apart. Fitted from {len(bm['train'])} of the {bm['views']} views, the map moves
by {bA['map_A_train_vs_A_mm'][0]:.2f} mm RMS.</p>
<p>With the fitted map the two cameras place the block {arm_fit:.2f} mm apart over the 17 arm poses. With 13.0 mm tags
they were {arm_13:.2f} mm apart (Table 4). On the held-out views the median distance is
{bc['fitted A']['held_out']['cameras_apart_mm']['median']:.1f} mm. The worst held-out views show one tag to one camera,
whose own solve is then weak in depth. Solving from both cameras at once avoids that: the held-out views then reproject
at {bc['fitted A']['held_out']['joint_rms_px']['median']:.2f} px, against {bc['nominal 13.0']['held_out']['joint_rms_px']['median']:.2f}
px with the CAD map.</p>
<p>The recess and the spacing come from the same fit. They reconcile the arm's scale with the calipers, but they do not
check one against the other. A depth gauge between a tag's surface and its face's rim would: a reading near 0.3 mm
means the two scales agree to about 0.05 mm over 30 mm.</p>
{table(["map of the block", "cameras apart, 17 arm poses (mm)", f"cameras apart, {len(bm['test'])} held-out views, median / max (mm)",
        "both cameras at once, held-out RMS (px)"], ck_rows, num=(1, 2, 3),
       caption="The hex block placed by each camera's own solve through its fitted pose: the distance between the two cameras' block origins. The held-out views are every fifth view, left out of the fit that checks them. The last column solves the block from both cameras at once.", n=4)}"""))

    S.append(("move", f"Camera {MV} moved after the session", f"""
<p>From where it stood, {MV} saw only half of the hand. At 16:00 the owner moved it, and its pose was solved again
without the arm. The lenses travel with the cameras, so only its pose changed. {FX} kept its pose from the arm.</p>
<p>The owner held the board in front of both cameras in {mv['views']} places (<code>lab/board_pair_views.py</code>).
Each view averages six frames per camera. The fit (<code>lab/stereo_pose_fit.py</code>) keeps both lenses, the
board's fitted shape and the arm's board scale fixed. It solves the moved camera's pose relative to {FX}, and the board's
pose in each view. The corners reproject at {mv['rms_px'][FX]:.2f} px in {FX} and {mv['rms_px'][MV]:.2f} px in {MV}.
Over 40 bootstrap draws of the views, the camera's centre varies by {mv['bootstrap_sd_mm']:.2f} mm and its axes by
{mv['bootstrap_sd_deg']:.2f}&deg;. The camera moved {np.linalg.norm(mv['centre_moved_mm']):.1f} mm and turned
{mv['turned_deg']:.1f}&deg;. The two cameras now stand {sep2:.0f} mm apart.</p>
<p>The method was first tried on the hand-held lens views that both cameras happened to save within 0.3 s
({mvt['views']} pairs, the camera still in its old place). It put {MV} within {np.linalg.norm(mvt['centre_moved_mm']):.2f} mm
and {mvt['turned_deg']:.2f}&deg; of its pose from the arm.</p>
<p>Between frames taken before and after the move, {FX}'s image shifted by
{np.hypot(*mvc['fixed_image_shift_px']):.2f} px, so it stayed put. Solved from both cameras at once, the hex block reprojects at
{mvc['block_joint_rms_px_new_pose']:.2f} px after the move with the new pose, and at
{mvc['block_joint_rms_px_before_move_pose']:.1f} px with the old one. Before the move, the same solve reprojected at
{mvc['block_joint_rms_px_before_move']:.2f} px. The film and Figures 1 to 3 show {MV} where it stood during the
session.</p>"""))

    S.append(("use", "Use in the tracker", f"""
<ul>
<li>Lenses and the board's shape: {code('data/fits/20261113-lens_board_k3_arm.json')} (<code>cams.SERIAL.lens</code>,
<code>board_offsets_mm</code>).</li>
<li>Camera poses in the base, T_CB in mm: {code('data/fits/20261113-handeye_tcp_fitted_lens_moved836.json')}, the
poses since {MV} moved (Section 7). {code('data/fits/20261113-handeye_tcp_fitted_lens.json')} has both poses of the
session, through the TCP pose. Use the controller's TCP pose for the flange, as in that fit.</li>
<li>Hex block, the 11-07 test block only: {code('data/fits/20261113-hex_block_map.json')}, read with
<code>TagMap.from_corners</code> from its <code>corners_mm</code>. The new cylinder and sphere blocks use the same
tags 10 to 15. Each needs its own run of <code>lab/block_views.py</code> and <code>lab/block_map_fit.py</code>, and only
one block may be in view at a time.</li>
</ul>"""))

    S.append(("limits", "Limitations and next steps", f"""
<ul>
<li>The arm poses cover a band around the image centre (Figure 2); the image corners come from the hand-held views
alone.</li>
<li>The block's tags sit 0.3 mm inside its faces by the map's fit alone. A depth-gauge reading would check the arm's
scale against the calipers (Section 6).</li>
<li>Views of the block lying on a side fit at {ly_med:.2f} px against {up_med:.2f} px upright, mostly in {SNS[0]}, which saw
those tags at 35 to 55&deg;. The recess's edges may shift the corners at such angles.</li>
<li>One arm pose of camera 832412070344 misses the 1 mm acceptance (1.18 mm). The session was not repeated.</li>
<li>The servos have been limp since 11-06, so the fingers hung under gravity during the session. Their angles in the
render were read on 11-13 with the arm within 6&deg; of the untilted poses; in the tilted poses the real fingers swing
by up to about 19&deg;.</li>
<li>The cad_v3 model draws hand tag 6 turned 180&deg; in its plane against the printed tag (Figure 3, left). The tag's
pose in the model needs the same turn before the tracker uses the hand tags.</li>
</ul>"""))

    pose_rows = []
    for r in recs:
        q = np.degrees(r["arm_before"]["q"])
        pose_rows.append([str(r["pose"]), esc(r["label"]), ", ".join(f"{x:.1f}" for x in q)]
                         + [f"{len(r['cams'][sn]['color']['corner_ids'])} / {len(r['cams'][sn]['ir1_off']['corner_ids'])}" for sn in SNS])
    S.append(("files", "Data, rerun and files", f"""
<p>The compact data are in this folder's <code>data/</code>. <code>poses.jsonl</code> has the 17 arm poses: the
joints, the TCP pose, each camera's averaged corners and tags, and the streams' factory models. The hand-held views and
every fit are beside it. The
raw frames and depth (1.4 GB) stay on corvus in <code>logs/20261113-camera_arm_calibration/</code>. The fits rerun from
<code>data/</code> alone:</p>
<pre class="cmd">L=docs/experiments/20261113-camera_arm_calibration/lab; O=docs/experiments/20261113-camera_arm_calibration/data
python3 $L/lens_board_fit.py $O --arm --json lens.json
python3 $L/handeye_fit.py $O lens.json --tcp [--fit-scale]
python3 $L/lens_board_fit.py $O --flat --no-k3      # the flat-board fit
python3 $L/block_map_fit.py $O/20261113-block_views.jsonl $O/fits/20261113-lens_board_k3_arm.json \\
    $O/fits/20261113-handeye_tcp_fitted_lens.json - map_fit.json map.json   # the block's map; - skips the arm-pose check
python3 $L/stereo_pose_fit.py $O/camera_move/20261113-board_pairs.jsonl $O/fits/20261113-lens_board_k3_arm.json \\
    $O/fits/20261113-handeye_tcp_fitted_lens.json $O/fits/20261113-handeye_tcp_fitted_lens_scale.json \\
    --moved 836612071918 --out moved.json \\
    --before $O/camera_move/20261113-before_tags.json --after $O/camera_move/20261113-after_tags.json   # the moved camera</pre>
<p>Recording: <code>lab/preview.py</code> (the live window), <code>lab/record.sh</code> and
<code>lab/record_pose.py</code> (one arm pose), <code>lab/lens_views.py</code> (hand-held views),
<code>lab/block_views.py</code> (the block's views, kept in <code>data/20261113-block_views.jsonl</code>),
<code>lab/snap.py</code> and <code>lab/board_pair_views.py</code> (the camera's move, kept in
<code>data/camera_move/</code>). The checks of the
block and the depth need the raw frames: <code>lab/block_depth_check.py</code> and this page's builder,
<code>scripts/hardware/camera_arm_calibration_page.py data</code>. The film: <code>scripts/hardware/camera_arm_calibration_film.py</code>.</p>
{table(["pose", "label", "joints (deg)", f"{SNS[0]} corners, colour / IR", f"{SNS[1]}"], pose_rows, num=(0, 3, 4),
       caption="The arm poses with the corners each camera found (of 36). Pose 15's turn was recorded as rx +0.3 rad; its joints match the rx &minus;0.3 rad pose. Pose 2 was recorded as up from x +30 mm; its joints put it at x 0.", n=5)}"""))

    filmfig = film_html(1, MEDIA / "20261113-calibration_replay.mp4", MEDIA / "20261113-calibration_replay_poster.jpg",
                        f"The 17 arm poses replayed in the MuJoCo model of the SR2 hand on the UR5 (cad_v3). The gantries stand at the "
                        f"station's home layout; the fingers hang at the angles read from the servos on 11-13. "
                        f"Left: the cell, with each camera's viewing pyramid to 0.36 m ({SNS[0]} orange, {SNS[1]} blue). The board "
                        f"on tool0, the cameras, the hex block and the table are placed by the fits through fk(q). Right: the renders from the "
                        f"two fitted cameras.")
    toc = "\n".join(f'<li><a href="#{k}">{i + 1}. {t}</a></li>' for i, (k, t, _) in enumerate(S))
    secs = "\n".join(f'<section id="{k}"><h2>{i + 1}. {t}</h2>{b}</section>' for i, (k, t, b) in enumerate(S))
    title = "Lens and arm calibration of the two D435 tracking cameras on the UR5"
    h = TEMPLATE.read_text()
    for k, v in dict(TITLE=title, TEXCSS="", EYEBROW="SR2 hand on the UR5 &middot; object tracking, step 5 of the plan",
                     H1=title, LEDE=lede, BYLINE="2026-11-13, corvus. The owner jogged the arm, held the board, turned the block and moved a camera.",
                     FILM=filmfig, TOC=toc, SECTIONS=secs,
                     FOOTER=f"Built by <code>scripts/hardware/camera_arm_calibration_page.py</code> from <code>data/</code>.").items():
        h = h.replace("{{" + k + "}}", v)
    h = docs_index.tag(h, topic="hardware", summary=summary, date="2026-11-13")
    h = retro_style.apply(h)
    PAGE.write_text(h)
    print(PAGE, f"{PAGE.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    {"data": data, "figs": figs, "page": page}[sys.argv[1]]()
