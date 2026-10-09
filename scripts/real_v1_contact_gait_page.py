#!/usr/bin/env python3
"""Build the dated contact-gait result page from raw experiment JSON."""
from __future__ import annotations

import html
import json
import math
import re
from datetime import datetime
from pathlib import Path
import retro_style

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "docs/experiments/20261007-contact-gait"
OUT = D / "20261007-contact_release_real_v1.html"
TEMPLATE = ROOT / "scripts/real_v1_contact_gait_page.template.html"
COLORS = {"direct_gradient":"#a65a3a", "cem":"#3c79a4", "hybrid":"#247e72"}
LABELS = {"direct_gradient":"Local derivative", "cem":"CEM", "hybrid":"CEM + derivative"}


def read(name):
    return json.loads((D/name).read_text())


def num(value, places=3):
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{value:.{places}f}"


def table(head, rows):
    body = ["<table><thead><tr>" + "".join(f"<th>{html.escape(str(v))}</th>" for v in head) + "</tr></thead><tbody>"]
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>")
    return "".join(body) + "</tbody></table>"


def fig_svg(parts, width, height, label):
    return (f'<div class="viz"><svg viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="{html.escape(label)}">' + "".join(parts) + "</svg></div>")


def search_figure(data):
    groups = [("1 mm packed",data["pads"]),("Convex mesh",data["mesh"]),
              ("Single sphere",data["sphere"])]
    w,h,left,right,top,bottom = 920,360,74,28,54,83
    ymax=.19
    plot_h=h-top-bottom
    y=lambda v: top+plot_h*(1-v/ymax)
    parts=[f'<text x="{left}" y="27" class="label">Orientation gain after 400 ms</text>',
           f'<text x="{left}" y="45" class="small">vertical cosine; seed 1; 256 evaluations per search</text>']
    for tick in (0,.05,.10,.15):
        yy=y(tick)
        parts += [f'<line x1="{left}" y1="{yy:.1f}" x2="{w-right}" y2="{yy:.1f}" stroke="var(--rule2)"/>',
                  f'<text x="{left-9}" y="{yy+4:.1f}" text-anchor="end" class="small">{tick:.2f}</text>']
    gw=(w-left-right)/3
    for i,(group,d) in enumerate(groups):
        gx=left+(i+.5)*gw
        for j,m in enumerate(("direct_gradient","cem","hybrid")):
            r=d["candidates"][m]
            value=r["cos_gain"]
            x=gx+(j-1)*48-17
            yy=y(max(value,0))
            outline=' stroke="#b64334" stroke-width="3"' if not r["held_final"] else ''
            parts.append(f'<rect x="{x:.1f}" y="{yy:.1f}" width="34" height="{max(0,y(0)-yy):.1f}" rx="3" fill="{COLORS[m]}"{outline}><title>{group}, {LABELS[m]}: {value:+.3f}; held {r["held_final"]}</title></rect>')
            parts.append(f'<text x="{x+17:.1f}" y="{yy-7:.1f}" text-anchor="middle" class="small">{value:+.3f}</text>')
        parts.append(f'<text x="{gx:.1f}" y="{h-bottom+25}" text-anchor="middle" class="label">{group}</text>')
    for j,m in enumerate(("direct_gradient","cem","hybrid")):
        x=left+j*215
        parts.append(f'<rect x="{x}" y="{h-27}" width="12" height="12" fill="{COLORS[m]}"/>')
        parts.append(f'<text x="{x+18}" y="{h-17}" class="small">{LABELS[m]}</text>')
    return fig_svg(parts,w,h,"Orientation gain by search method and fingertip model")


def trace_figure(pads, mesh):
    curves=[("Packed CEM",pads["candidates"]["cem"]["contact_trace"],"#3c79a4"),
            ("Packed hybrid",pads["candidates"]["hybrid"]["contact_trace"],"#247e72"),
            ("Mesh CEM",mesh["candidates"]["cem"]["contact_trace"],"#b75f48")]
    w,h,left,right=920,560,84,35
    x=lambda t:left+t/400*(w-left-right)
    panels=[("Index normal force", "N",0,3.2,lambda r:r["finger_normal_force_N"]["index"]),
            ("Tool height", "m",.045,.102,lambda r:r["tool_z_m"]),
            ("Tool vertical cosine", "",-.1,.8,lambda r:r["vertical_cos"])]
    parts=[f'<text x="{left}" y="27" class="label">Opening, recontact and tool motion</text>',
           f'<text x="{left}" y="45" class="small">0–400 ms; shaded command interval 0–180 ms</text>']
    for k,(title,unit,lo,hi,getter) in enumerate(panels):
        top=66+k*155
        ph=105
        y=lambda v:top+ph*(1-(v-lo)/(hi-lo))
        parts.append(f'<rect x="{left}" y="{top}" width="{x(180)-left:.1f}" height="{ph}" fill="#d9e9e5" opacity=".45"/>')
        for tick in (lo,(lo+hi)/2,hi):
            yy=y(tick)
            parts += [f'<line x1="{left}" y1="{yy:.1f}" x2="{w-right}" y2="{yy:.1f}" stroke="var(--rule2)"/>',
                      f'<text x="{left-9}" y="{yy+4:.1f}" text-anchor="end" class="small">{tick:.3g}</text>']
        parts.append(f'<text x="{left}" y="{top-8}" class="label">{title} {unit}</text>')
        if k==0:
            parts.append(f'<line x1="{left}" y1="{y(.05):.1f}" x2="{w-right}" y2="{y(.05):.1f}" stroke="#af4b42" stroke-dasharray="5 4" opacity=".7"/>')
        if k==1:
            parts.append(f'<line x1="{left}" y1="{y(pads["initial_tool_z_m"]):.1f}" x2="{w-right}" y2="{y(pads["initial_tool_z_m"]):.1f}" stroke="#657783" stroke-dasharray="5 4" opacity=".8"/>')
        for label,trace,color in curves:
            coords=" ".join(f'{x(r["t_s"]*1000):.1f},{y(getter(r)):.1f}' for r in trace)
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"><title>{label}</title></polyline>')
        for tick in (0,100,200,300,400):
            parts.append(f'<text x="{x(tick):.1f}" y="{top+ph+18}" text-anchor="middle" class="small">{tick}</text>')
    parts.append(f'<text x="{(left+w-right)/2:.1f}" y="{h-31}" text-anchor="middle" class="small">time after supported grasp (ms)</text>')
    for i,(label,_,color) in enumerate(curves):
        xx=left+i*230
        parts += [f'<line x1="{xx}" y1="{h-12}" x2="{xx+18}" y2="{h-12}" stroke="{color}" stroke-width="3"/>',
                  f'<text x="{xx+25}" y="{h-8}" class="small">{label}</text>']
    return fig_svg(parts,w,h,"Index force, tool height and orientation across forced-release plans")


def transfer_stats(rows, destination, plan):
    subset=[r for r in rows if r["destination"]==destination and r["plan"]==plan]
    held=[r for r in subset if r["held_final"]]
    return len(held), sum(r["index_recontact"] for r in subset), (sum(r["cos_gain"] for r in held)/len(held) if held else None)


def warp_stats(rows, plan):
    subset=[r for r in rows if r["plan"]==plan]
    held=[r for r in subset if r["held_final"]]
    return len(held), sum(r["index_recontact"] for r in subset), (sum(r["cos_gain"] for r in held)/len(held) if held else None)


def transfer_figure(cpu,warp_free,warp_release,warp_free_seed,warp_release_seed):
    plans=[("Free local","tpu6_padsT_direct_gradient","direct_gradient",False),
           ("Free CEM","tpu6_padsT_cem","cem",False),
           ("Free CEM128","tpu6_padsT_hybrid_seed","hybrid_seed",False),
           ("Free hybrid","tpu6_padsT_hybrid","hybrid",False),
           ("Release CEM","tpu6_padsT_release_cem","cem",True),
           ("Release CEM128","tpu6_padsT_release_hybrid_seed","hybrid_seed",True),
           ("Release hybrid","tpu6_padsT_release_hybrid","hybrid",True)]
    colors=["#247e72","#3c79a4","#b75f48"]
    w,h,left,top=920,424,165,65
    width=w-left-35
    x=lambda held:left+held/8*width
    parts=[f'<text x="{left}" y="28" class="label">Held continuations across eight grip starts</text>',
           f'<text x="{left}" y="47" class="small">packed CPU / packed Warp / convex-mesh CPU</text>']
    for tick in (0,2,4,6,8):
        xx=x(tick)
        parts += [f'<line x1="{xx:.1f}" y1="{top-12}" x2="{xx:.1f}" y2="{h-46}" stroke="var(--rule2)"/>',
                  f'<text x="{xx:.1f}" y="{h-25}" text-anchor="middle" class="small">{tick}/8</text>']
    for i,(label,plan,method,release) in enumerate(plans):
        yy=top+i*44
        parts.append(f'<text x="{left-12}" y="{yy+19}" text-anchor="end" class="small">{label}</text>')
        warp=(warp_release_seed if release else warp_free_seed) if method=="hybrid_seed" else (warp_release if release else warp_free)
        sets=[transfer_stats(cpu["rows"],"tpu6_padsT",plan),
              warp_stats(warp["rows"],method),
              transfer_stats(cpu["rows"],"tpu6_pt",plan)]
        for j,(held,_,_) in enumerate(sets):
            y=yy+j*9
            parts.append(f'<rect x="{left}" y="{y}" width="{max(1,x(held)-left):.1f}" height="7" fill="{colors[j]}"><title>{label}: {held}/8 held</title></rect>')
            parts.append(f'<text x="{x(held)+5:.1f}" y="{y+7}" class="small">{held}</text>')
    return fig_svg(parts,w,h,"Held rollouts in CPU MuJoCo, MuJoCo-Warp, and mesh transfer")


def main():
    source={"pads":read("20261007-padsT-seed1-256.json"),
            "mesh":read("20261007-mesh-seed1-256.json"),
            "sphere":read("20261007-sphere-seed1-256.json")}
    release={"pads":read("20261007-padsT-release-seed1-256.json"),
             "mesh":read("20261007-mesh-release-seed1-256.json"),
             "sphere":read("20261007-sphere-release-seed1-256.json")}
    cpu=read("20261007-transfer-mj360.json")
    cpu331=read("20261007-transfer-mj331.json")
    wf=read("20261007-warp-padsT-free-8.json")
    wr=read("20261007-warp-padsT-release-8.json")
    wfs=read("20261007-warp-padsT-free-seed-8.json")
    wrs=read("20261007-warp-padsT-release-seed-8.json")
    jvp=read("20261007-fork-padsT2-recontact-jvp.json")
    cfd=read("20261007-fork-padsT2-recontact-cfd-jvp.json")
    assert jvp["start_ms"]==cfd["start_ms"] and jvp["steps"]==cfd["steps"]
    assert jvp["cpu_fd"]==cfd["cpu_fd"]
    coarse_free=read("20261007-padsT2-seed1-256.json")
    coarse_release=read("20261007-padsT2-release-seed1-256.json")
    coarse=read("20261007-transfer-padsT2-to-1mm-mj360.json")
    attempt=read("20261007-mjx-exact1mm-attempt.json")
    gpu={"sphere":read("20261007-mjx-gpu-sphere-20-warm.json"),
         "mesh":read("20261007-mjx-gpu-mesh-capsule-20-warm.json"),
         "pads2":read("20261007-mjx-gpu-padsT2-20-warm.json")}
    native={"sphere":read("20261007-native-mjx-gpu-sphere-20-warm.json"),
            "pads2_x64":read("20261007-native-mjx-gpu-padsT2-20-warm.json"),
            "pads2_f32":read("20261007-native-mjx-gpu-padsT2-f32-20-warm.json")}
    style=re.search(r"<style>(.*?)</style>",(ROOT/"scripts/hom_chain_page.template.html").read_text(),re.S).group(1)
    style+='''\n.viz{background:var(--card);border:1px solid var(--rule);border-radius:10px;padding:12px;box-shadow:var(--shadow)}
.viz svg{display:block;width:100%;height:auto}.viz .label{fill:var(--ink);font:600 14px var(--f-display)}
.viz .small{fill:var(--ink2);font:400 11px var(--f-mono)}
@media print{.col>.tw{width:100%}table{font-size:11px;table-layout:fixed}th{font-size:9px;white-space:normal;padding:7px 6px}td{padding:7px 6px;overflow-wrap:anywhere}.tw{overflow:visible}}
'''
    search_rows=[]
    for model,title in (("pads","1 mm packed"),("mesh","Convex mesh"),("sphere","Single sphere")):
        d=source[model]
        for method in ("direct_gradient","cem","hybrid"):
            r=d["candidates"][method]
            search_rows.append([title,LABELS[method],f'{r["cos_gain"]:+.3f}',
                                "yes" if r["held_final"] else "no",str(r["index_release_ms"]),
                                num(d["methods"]["hybrid_gradient" if method=="hybrid" else method]["wall_s"]
                                    +(d["methods"]["hybrid_cem"]["wall_s"] if method=="hybrid" else 0),1)])
    release_rows=[]
    for model,title,methods in (("pads","1 mm packed",("hold","direct_gradient","cem","hybrid")),
                                 ("mesh","Convex mesh",("cem",)),("sphere","Single sphere",("cem",))):
        for method in methods:
            r=release[model]["candidates"][method]
            release_rows.append([title,LABELS.get(method,"Starting plan"),f'{r["cos_gain"]:+.3f}',
                                 num(r["min_tool_z_m"],3),"yes" if r["held_final"] else "no",
                                 str(r["index_release_ms"]),"yes" if r["index_recontact"] else "no"])
    jvp_rows=[]
    for i,(label,unit) in enumerate((("Vertical cosine","per rad"),
                                     ("Tool height","m/rad"),
                                     ("Tool angular speed x","rad/s per rad"),
                                     ("Contact torque x","N m/rad"))):
        cpu_der=jvp["cpu_fd"]["0.01"]["derivative"][i]
        fork_fd=jvp["fork_fd_0.01"][i]
        ad=jvp["fork_ad"][i]
        cfd_ad=cfd["fork_ad"][i]
        jvp_rows.append([label+' <small>('+unit+')</small>',f'{cpu_der:.4g}',
                         f'{fork_fd:.4g}',f'{ad:.4g}',f'{cfd_ad:.4g}'])
    selected=[("Free local","tpu6_padsT_direct_gradient","direct_gradient",False),
              ("Free CEM","tpu6_padsT_cem","cem",False),
              ("Free CEM128","tpu6_padsT_hybrid_seed","hybrid_seed",False),
              ("Free hybrid","tpu6_padsT_hybrid","hybrid",False),
              ("Release CEM","tpu6_padsT_release_cem","cem",True),
              ("Release CEM128","tpu6_padsT_release_hybrid_seed","hybrid_seed",True),
              ("Release hybrid","tpu6_padsT_release_hybrid","hybrid",True)]
    transfer_rows=[]
    for label,plan,method,is_release in selected:
        warp=(wrs if is_release else wfs) if method=="hybrid_seed" else (wr if is_release else wf)
        for backend,dest,stat in (("CPU packed","tpu6_padsT",None),
                                  ("Warp packed",None,warp_stats(warp["rows"],method)),
                                  ("CPU mesh","tpu6_pt",None)):
            held,recontact,gain=stat if stat else transfer_stats(cpu["rows"],dest,plan)
            transfer_rows.append([label,backend,f"{held}/8",f"{recontact}/8",
                                  f"{gain:+.3f}" if gain is not None else "—"])
    free_seed=transfer_stats(cpu["rows"],"tpu6_padsT","tpu6_padsT_hybrid_seed")
    free_hybrid=transfer_stats(cpu["rows"],"tpu6_padsT","tpu6_padsT_hybrid")
    free_seed_mesh=transfer_stats(cpu["rows"],"tpu6_pt","tpu6_padsT_hybrid_seed")
    free_hybrid_mesh=transfer_stats(cpu["rows"],"tpu6_pt","tpu6_padsT_hybrid")
    release_seed=transfer_stats(cpu["rows"],"tpu6_padsT","tpu6_padsT_release_hybrid_seed")
    release_hybrid=transfer_stats(cpu["rows"],"tpu6_padsT","tpu6_padsT_release_hybrid")
    release_mesh=transfer_stats(cpu["rows"],"tpu6_pt","tpu6_padsT_release_hybrid")
    direct=transfer_stats(cpu["rows"],"tpu6_padsT","tpu6_padsT_direct_gradient")
    for _,plan,method,is_release in selected:
        cpu_count=transfer_stats(cpu["rows"],"tpu6_padsT",plan)[0]
        old_count=transfer_stats(cpu331["rows"],"tpu6_padsT",plan)[0]
        warp=(wrs if is_release else wfs) if method=="hybrid_seed" else (wr if is_release else wf)
        warp_count=warp_stats(warp["rows"],method)[0]
        assert cpu_count==old_count==warp_count,(plan,cpu_count,old_count,warp_count)
    coarse_plans=[("Free CEM128","tpu6_padsT_2mm_hybrid_seed"),
                  ("Free hybrid","tpu6_padsT_2mm_hybrid"),
                  ("Release CEM256","tpu6_padsT_2mm_release_cem"),
                  ("Release CEM128","tpu6_padsT_2mm_release_hybrid_seed"),
                  ("Release hybrid","tpu6_padsT_2mm_release_hybrid")]
    coarse_rows=[]
    for label,plan in coarse_plans:
        src=transfer_stats(coarse["rows"],"tpu6_padsT_2mm",plan)
        dst=transfer_stats(coarse["rows"],"tpu6_padsT",plan)
        mesh=transfer_stats(coarse["rows"],"tpu6_pt",plan)
        coarse_rows.append([label,f'{src[0]}/8',f'{dst[0]}/8',f'{mesh[0]}/8',
                            f'{dst[2]:+.3f}' if dst[2] is not None else '—'])
    coarse_cem=transfer_stats(coarse["rows"],"tpu6_padsT_2mm","tpu6_padsT_2mm_release_cem")
    fine_cem=transfer_stats(coarse["rows"],"tpu6_padsT","tpu6_padsT_2mm_release_cem")
    compute=[]
    for model,title in (("sphere","CPU single sphere"),("mesh","CPU convex mesh"),("pads","CPU 1 mm packed")):
        seconds=source[model]["methods"]["cem"]["wall_s"]/256
        compute.append([title,"1",f"{seconds*1e3:.2f} ms / 400 ms plan","—","—"])
    compute.append(["CPU 2 mm packed","1",
                    f'{coarse_free["methods"]["cem"]["wall_s"]/256*1e3:.2f} ms / 400 ms plan',"—","—"])
    compute.append(["Warp 1 mm packed","32",f'{wf["mean_batch_step_s"]*1e3:.3f} ms / batch step',"warm","—"])
    for model,title in (("sphere","Fork MJX sphere / cylinder"),("mesh","Fork MJX mesh / capsule"),
                        ("pads2","Fork MJX 2 mm packed / cylinder")):
        d=gpu[model]
        compute.append([title,"1",f'{d["rollout_warm_median_s"]*1e3:.2f} ms / 20 steps',
                        f'{d["initial_compile_s"]:.1f} + {d["rollout_compile_s"]:.1f} s',
                        f'{d["cpu_rollout_s"]*1e3:.2f} ms CPU / 20 steps'])
    for model,title in (("sphere","Native MJX sphere / cylinder"),
                        ("pads2_x64","Native MJX 2 mm packed (64-bit)"),
                        ("pads2_f32","Native MJX 2 mm packed (32-bit)")):
        d=native[model]
        compute.append([title,"1",f'{d["mjx_warm_s"]*1e3:.2f} ms / 20 steps',
                        f'{d["forward_compile_s"]:.1f} + {d["mjx_first_s"]:.1f} s',
                        f'{d["cpu_rollout_s"]*1e3:.2f} ms CPU / 20 steps'])
    compute.append(["Fork MJX 1 mm packed / cylinder","1","unavailable",
                    f'{attempt["initial_forward_compile_s"]:.1f} s + unfinished rollout',"—"])
    cpu_pad_step=source["pads"]["methods"]["cem"]["wall_s"]/256/400
    warp_world_step=wf["mean_batch_step_s"]/wf["worlds"]
    native32_ratio=native["pads2_f32"]["mjx_warm_s"]/native["pads2_f32"]["cpu_rollout_s"]
    fidelity=(f'The 2 mm packed MJX scene had a maximum initial fingertip-gap difference of '
              f'{max(abs(gpu["pads2"]["fork_initial_finger_dist_m"][f]-gpu["pads2"]["cpu_initial_contact"]["finger_min_dist_m"][f]) for f in ("thumb","index","middle"))*1e6:.2f} µm '
              f'from CPU MuJoCo. The corresponding single-sphere error was '
              f'{max(abs(gpu["sphere"]["fork_initial_finger_dist_m"][f]-gpu["sphere"]["cpu_initial_contact"]["finger_min_dist_m"][f]) for f in ("thumb","index","middle"))*1e3:.2f} mm. '
              'The mesh/capsule comparator disagreed by 13–22 mm at thumb and index; its rollout has a different contact geometry from the mesh-pad CPU scene. '
              'The prior exact 1 mm single-step study measured near-micrometre packed contact parity.')
    slots=(f'The fork instantiated {len(gpu["sphere"]["fork_initial_contact_geom"])} candidate contact slots for '
           f'the one-sphere tips and {len(gpu["pads2"]["fork_initial_contact_geom"])} for the 2 mm pads; '
           f'the exact 1 mm model instantiated {attempt["candidate_contact_slots"]}. '
           f'The held 2 mm CPU state had {gpu["pads2"]["cpu_initial_contact"]["total_contacts"]} active contacts.')
    decision=(f'Numerical local derivatives improved the same sampled packed-pad plan from '
              f'{free_seed[2]:.3f} to {free_hybrid[2]:.3f} mean orientation gain across '
              f'{free_hybrid[0]} held grip jitters. Its mesh replay retained {free_seed_mesh[0]} grasps '
              f'before refinement and {free_hybrid_mesh[0]} after. Direct local search reached '
              f'{direct[2]:.3f} gain on its {direct[0]} held jitters. From a fully opened index, '
              'local search failed to discover a retained grasp; sampling found recontact. Refinement of that '
              f'sampled release plan kept {release_hybrid[0]} of eight grasps and changed held gain '
              f'from {release_seed[2]:.3f} to {release_hybrid[2]:.3f}. The packed release hybrid '
              f'held {release_mesh[0]} of eight mesh replays. The 400 ms result supports local refinement '
              'within a favorable contact regime and shows no stronger behavior from deliberate release in this task slice. '
              'Plain AD tracked the physical recontact derivative; the contacts-from-distance backward rule changed '
              'that derivative by severalfold at the measured onset.')
    next_rows=[
        ["Longer reorientation", "Run 1–3 s receding-horizon plans with the same D1–D8 held starts; report held turn, force and wall time at a fixed planning budget.", "Extend real_v1_contact_gait_search.py"],
        ["AD at contact transition", "Compile a 2 mm packed 400 ms control derivative; compare its chosen plan with CPU finite differences and replay on 1 mm pads.", "Extend real_v1_diffmjx_official_gate.py"],
        ["Policy learning", "Train mjlab and Playground baselines on the same scene and score after a long-horizon planner survives transfer; compare three seeds.", "Port the D1 scene and live-grasp reset"]]
    lede=(f'In a 400 ms continuation of the D1 screwdriver grasp, numerical control derivatives improved the same '
          f'sampled 1 mm packed-pad plan from {free_seed[2]:.3f} to {free_hybrid[2]:.3f} mean orientation gain with '
          f'{free_hybrid[0]}/8 perturbed grasps held. A sampled index release recontacted in '
          f'{release_hybrid[1]}/8 starts but gained {release_hybrid[2]:.3f} when held and failed on the convex mesh. '
          f'Plain DiffMJX produced a local derivative across recontact on 2 mm pads; exact 1 mm MJX '
          f'rollout compilation remained unfinished when the run stopped at {attempt["elapsed_before_stop_s"]//60} min '
          f'{attempt["elapsed_before_stop_s"]%60} s.')
    replacements={
        "STYLE":"<style>"+style+"</style>","BUILT":datetime.now().strftime("%Y-%m-%d %H:%M"),
        "LEDE":lede,"SEARCH_FIG":search_figure(source),
        "SEARCH_TABLE":table(["Tip","Search","Gain","Held","Index release (ms)","Search (s)"],search_rows),
        "TRACE_FIG":trace_figure(release["pads"],release["mesh"]),
        "RELEASE_TEXT":(f'The packed-pad CEM plan released the index for '
                        f'{release["pads"]["candidates"]["cem"]["index_release_ms"]} ms, '
                        f'recontacted it, and gained '
                        f'{release["pads"]["candidates"]["cem"]["cos_gain"]:.3f} vertical cosine in seed 1. '
                        'The corresponding mesh and single-sphere searches did not restore a held grasp.'),
        "RELEASE_TABLE":table(["Tip","Search","Gain","Minimum height (m)","Held","Release (ms)","Recontact"],release_rows),
        "JVP_TEXT":(f'The sampled 2 mm pad plan reached a positive index gap of '
                    f'{jvp["initial_fork_index_min_gap_m"]*1e6:.1f} µm at 280 ms. CPU index normal force '
                    f'rose from {jvp["initial_cpu_index_force_N"]:.3f} to '
                    f'{jvp["final_cpu_index_force_N"]:.3f} N during the next 10 ms. '
                    'The released fork used its ordinary contact derivative with contacts from distance disabled. '
                    f'Its AD derivative compiled in {jvp["derivative_compile_s"]:.1f} s and took '
                    f'{jvp["derivative_warm_s"]:.2f} s warm. The CPU finite differences were stable '
                    'at 0.01, 0.003 and 0.001 rad perturbations.'),
        "JVP_TABLE":table(["Final quantity","CPU FD","Fork FD","Plain AD","CFD AD"],jvp_rows),
        "CFD_TEXT":(f'Plain AD differed from CPU finite differences by '
                    f'{abs(jvp["fork_ad"][0]-jvp["cpu_fd"]["0.01"]["derivative"][0])/abs(jvp["cpu_fd"]["0.01"]["derivative"][0])*100:.1f}% '
                    f'for orientation and '
                    f'{abs(jvp["fork_ad"][2]-jvp["cpu_fd"]["0.01"]["derivative"][2])/abs(jvp["cpu_fd"]["0.01"]["derivative"][2])*100:.1f}% '
                    f'for angular speed. Contacts from distance changed the orientation derivative to '
                    f'{abs(cfd["fork_ad"][0]/jvp["cpu_fd"]["0.01"]["derivative"][0]):.1f} times '
                    f'the CPU value and the height derivative to '
                    f'{abs(cfd["fork_ad"][1]/jvp["cpu_fd"]["0.01"]["derivative"][1]):.1f} times. '
                    f'The plain and CFD forward vertical cosines differed by '
                    f'{abs(cfd["fork_zero"][0]-jvp["fork_zero"][0]):.2g}. '
                    f'CFD derivative compilation took {cfd["derivative_compile_s"]:.1f} s and its warm '
                    f'evaluation took {cfd["derivative_warm_s"]:.2f} s.'),
        "TRANSFER_FIG":transfer_figure(cpu,wf,wr,wfs,wrs),
        "TRANSFER_TEXT":(f'The free hybrid gained {free_hybrid[2]-free_seed[2]:.3f} vertical cosine over '
                         f'its own 128-sample seed while retaining {free_hybrid[0]} of eight packed-pad grasps; '
                         f'convex-mesh retention fell from {free_seed_mesh[0]} to {free_hybrid_mesh[0]}. '
                         f'The release hybrid kept the same {release_hybrid[0]} of eight packed-pad grasps as '
                         f'its sampled seed and added {release_hybrid[2]-release_seed[2]:.3f} held gain. '
                         'MuJoCo 3.3.1 and Warp reproduced these packed-pad held counts; failed rollouts sometimes '
                         'followed different orientation paths.'),
        "TRANSFER_TABLE":table(["Plan","Replay","Held","Recontact","Gain if held"],transfer_rows),
        "PAD2_COUNT":str(sum(gpu["pads2"]["n_pads"].values())),
        "EXACT_STOP_MIN":num(attempt["elapsed_before_stop_s"]/60,2),
        "EXACT_RSS_GIB":num(attempt["peak_resident_kib"]/1048576,2),
        "COMPUTE_TABLE":table(["Backend / tip","Worlds","Warm execution","Compile","CPU reference"],compute),
        "THROUGHPUT_TEXT":(f'At 32 worlds, Warp took {wf["mean_batch_step_s"]*1e3:.3f} ms per batch step, '
                           f'or {warp_world_step*1e6:.1f} µs per world step. The 1 mm CPU CEM batch took '
                           f'{cpu_pad_step*1e6:.1f} µs per step of one world, giving '
                           f'{cpu_pad_step/warp_world_step:.1f} times more aggregate Warp throughput. '
                           f'Native 32-bit MJX took {native32_ratio:,.0f} times the CPU reference time '
                           'for one 20 ms 2 mm packed rollout.'),
        "SLOT_TEXT":slots,"FIDELITY_TEXT":fidelity,"DECISION_TEXT":decision,
        "COARSE_TEXT":(f'At 256 evaluations, the 2 mm packed CEM search took '
                       f'{coarse_free["methods"]["cem"]["wall_s"]:.2f} s versus '
                       f'{source["pads"]["methods"]["cem"]["wall_s"]:.2f} s at 1 mm. '
                       f'The 2 mm forced-release CEM plan held {coarse_cem[0]}/8 grasps on its '
                       f'own geometry and {fine_cem[0]}/8 on the exact 1 mm surface. '
                       'The coarser free-turn plan transferred more reliably than its release plans.'),
        "COARSE_TABLE":table(["2 mm source plan","2 mm held","1 mm held","Mesh held","1 mm gain if held"],
                             coarse_rows),
        "NEXT_TABLE":table(["Next measurement","Outcome to record","Starting point"],next_rows),
    }
    page=TEMPLATE.read_text()
    for key,value in replacements.items():
        page=page.replace("{{"+key+"}}",str(value))
    missing=re.findall(r"{{[A-Z_]+}}",page)
    if missing:
        raise RuntimeError(f"Unfilled placeholders: {missing}")
    OUT.write_text(retro_style.apply(page))  # plain page style (owner, 2026-10-09)
    print(OUT)


if __name__=="__main__":
    main()
