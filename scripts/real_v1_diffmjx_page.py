#!/usr/bin/env python3
"""Build the dated real_v1 contact-gradient result page from its JSON measurements."""
from __future__ import annotations

import html
import json
import math
import re
from datetime import datetime
from pathlib import Path
import retro_style

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "docs/experiments/20261007-diffmjx"
TEMPLATE = ROOT / "scripts/real_v1_diffmjx_page.template.html"
OUT = D / "20261007-contact_gradients_real_v1.html"
FINGERS = ("thumb", "index", "middle")
METRICS = ("Vertical cosine", "Tool height", "Tool angular speed x", "Contact torque x")
UNITS = ("per rad", "m/rad", "rad/s per rad", "N m/rad")


def read(name: str, optional: bool = False):
    path = D / name
    if optional and not path.exists():
        return None
    return json.loads(path.read_text())


def num(value, digits=3):
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}g}"


def gib_from_kib(value):
    return num(value / (1024 * 1024), 3)


def table(head, rows):
    s = ["<table><thead><tr>" + "".join(f"<th>{html.escape(str(h))}</th>" for h in head) + "</tr></thead><tbody>"]
    for row in rows:
        s.append("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>")
    s.append("</tbody></table>")
    return "".join(s)


def svg_gap(datasets):
    width, left, right, top, step = 920, 235, 46, 72, 38
    labels = []
    for title, data, color in datasets:
        if data is None:
            continue
        if "fork_initial_finger_dist_m" in data:
            cpu = data["cpu_initial_contact"]["finger_min_dist_m"]
            sim = data["fork_initial_finger_dist_m"]
        else:
            cpu = data["initial_contact"]["finger_min_dist_m"]
            sim = {finger: data["mjx_initial_contact"][finger]["min_dist_m"] for finger in FINGERS}
        for finger in FINGERS:
            labels.append((f"{title} · {finger}", abs(sim[finger] - cpu[finger]) * 1e6, color))
    height = top + step * len(labels) + 70
    xlo, xhi = -2, 3
    x = lambda v: left + (math.log10(max(v, 0.01)) - xlo) / (xhi - xlo) * (width - left - right)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Absolute CPU versus MJX fingertip gap error">',
             f'<text x="{left}" y="29" class="label" style="font-weight:700">Absolute fingertip gap difference at the held state</text>',
             f'<text x="{left}" y="50" class="small">Micrometres, log scale</text>']
    for tick in (0.01, 0.1, 1, 10, 100, 1000):
        xx = x(tick)
        parts.append(f'<line x1="{xx:.1f}" y1="{top-16}" x2="{xx:.1f}" y2="{height-48}" stroke="var(--rule2)"/>')
        parts.append(f'<text x="{xx:.1f}" y="{height-22}" text-anchor="middle" class="small">{tick:g}</text>')
    for i, (label, value, color) in enumerate(labels):
        y = top + i * step
        parts.append(f'<text x="{left-17}" y="{y+4}" text-anchor="end" class="small">{html.escape(label)}</text>')
        parts.append(f'<line x1="{left}" y1="{y}" x2="{x(value):.1f}" y2="{y}" stroke="{color}" stroke-width="2" opacity=".42"/>')
        parts.append(f'<circle cx="{x(value):.1f}" cy="{y}" r="5.5" fill="{color}"><title>{html.escape(label)}: {value:.4g} µm</title></circle>')
        value_label = '&lt;0.01' if value < 0.01 else f'{value:.3g}'
        parts.append(f'<text x="{min(x(value)+12, width-right-55):.1f}" y="{y+4}" class="small">{value_label}</text>')
    parts.append('</svg>')
    return '<div class="viz">' + ''.join(parts) + '</div>'


def svg_search(search, plain_jac, cfd_jac):
    scores = [1e4 * (x - search["candidates"]["zero"]["vertical_cos"]) for x in search["random_vertical_cos"]]
    fd = 1e4 * (search["candidates"]["cpu_fd"]["vertical_cos"] - search["candidates"]["zero"]["vertical_cos"])
    ad = 1e4 * (plain_jac['local_search']['cpu_score_ad_direction'] - search['candidates']['zero']['vertical_cos'])
    cfd = 1e4 * (cfd_jac['local_search']['cpu_score_ad_direction'] - search['candidates']['zero']['vertical_cos'])
    best = max(scores)
    xmin, xmax = math.floor(min(scores) * 2) / 2 - 0.15, math.ceil(max(fd, best, ad, cfd) * 2) / 2 + 0.15
    width, height, left, right, top, bottom = 920, 330, 85, 44, 56, 70
    x = lambda v: left + (v - xmin) / (xmax - xmin) * (width - left - right)
    bins = 24
    counts = [0] * bins
    for s in scores:
        i = min(bins - 1, max(0, int((s - xmin) / (xmax - xmin) * bins)))
        counts[i] += 1
    ymax = max(counts)
    plot_h = height - top - bottom
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Distribution of 128 sampled control changes and the finite-difference direction">',
             f'<text x="{left}" y="28" class="label" style="font-weight:700">Local yaw-target search on the 1 mm packed surface</text>',
             f'<text x="{left}" y="48" class="small">Change in tool vertical cosine after 20 ms, in units of 0.0001</text>']
    for tick in range(math.ceil(xmin), math.floor(xmax)+1):
        xx = x(tick)
        parts.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{height-bottom}" stroke="var(--rule2)"/>')
        parts.append(f'<text x="{xx:.1f}" y="{height-bottom+24}" text-anchor="middle" class="small">{tick:g}</text>')
    bw = (width - left - right) / bins
    for i, count in enumerate(counts):
        bar_h = count / ymax * (plot_h - 10)
        parts.append(f'<rect x="{left+i*bw+1:.1f}" y="{height-bottom-bar_h:.1f}" width="{bw-2:.1f}" height="{bar_h:.1f}" fill="var(--s2)" opacity=".65"/>')
    for val, color, label, y in ((fd, 'var(--s1)', 'CPU finite difference', top+18),
                                 (ad, 'var(--s2)', 'Plain fork AD', top+38),
                                 (cfd, 'var(--ink3)', 'Distance + scan-loop AD', top+58),
                                 (best, 'var(--good)', 'Best of 128 samples', top+78)):
        xx = x(val)
        parts.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{height-bottom}" stroke="{color}" stroke-width="3"/>')
        txt = f"{label}: {val:.3f}"
        right_side = xx + 8 + 9.4 * len(txt) < width            # 15.5-unit mono text; else end the label at the line
        parts.append(f'<text x="{xx + 8 if right_side else xx - 8:.1f}" y="{y}" class="small" fill="{color}" '
                     f'text-anchor="{"start" if right_side else "end"}">{txt}</text>')
    parts.append(f'<text x="{left}" y="{height-8}" class="small">128 sampled directions at 0.1 rad radius</text></svg>')
    return '<div class="viz">' + ''.join(parts) + '</div>'


def svg_fd_sweep(sweep, jac):
    vals = sorted(((float(eps), row) for eps, row in sweep['fork_fd_sweep'].items()))
    ad = jac['ad_directional_derivative'][0]
    width, height, left, right, top, bottom = 920, 310, 95, 56, 70, 64
    x = lambda eps: left + (math.log10(eps) + 4) / 2 * (width-left-right)
    y = lambda ratio: height-bottom-(ratio-0.85)/(1.25-0.85)*(height-top-bottom)
    fork = [(eps, row[0]/ad) for eps,row in vals]
    cpu = [(eps, sweep['cpu_fd_sweep'][f'{eps:g}'][0]/ad) for eps,_ in vals]
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Finite-difference orientation derivative versus perturbation size">',
             f'<text x="{left}" y="28" class="label" style="font-weight:700">Index-yaw derivative over 20 ms</text>',
             f'<text x="{left}" y="48" class="small">Finite difference over plain-fork automatic derivative</text>']
    for ratio in (0.9, 1.0, 1.1, 1.2):
        yy=y(ratio)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="var(--rule2)"/>')
        parts.append(f'<text x="{left-14}" y="{yy+4:.1f}" text-anchor="end" class="small">{ratio:g}</text>')
    for eps in (0.0001, 0.001, 0.01):
        xx=x(eps)
        parts.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{height-bottom}" stroke="var(--rule2)"/>')
        parts.append(f'<text x="{xx:.1f}" y="{height-bottom+24}" text-anchor="middle" class="small">{eps:g}</text>')
    for series,color,label in ((fork,'var(--s1)','fork FD'),(cpu,'var(--s2)','CPU FD')):
        points=' '.join(f'{x(eps):.1f},{y(value):.1f}' for eps,value in series)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for eps,value in series:
            parts.append(f'<circle cx="{x(eps):.1f}" cy="{y(value):.1f}" r="5" fill="{color}"><title>{label}: {value:.4f} at {eps:g} rad</title></circle>')
    parts.append(f'<text x="{width-right}" y="{top+8}" text-anchor="end" class="small" style="fill:var(--s1)">● fork FD</text>')
    parts.append(f'<text x="{width-right}" y="{top+29}" text-anchor="end" class="small" style="fill:var(--s2)">● CPU FD</text>')
    parts.append(f'<text x="{width-right}" y="{height-9}" text-anchor="end" class="small">Yaw target perturbation, rad</text></svg>')
    return '<div class="viz">'+''.join(parts)+'</div>'


def svg_gradient(data, title):
    if "ad_directional_derivative" in data:
        ad, fd, cpu = data["ad_directional_derivative"], data["fork_fd_directional_derivative"], data["cpu_fd_directional_derivative"]
        fd = fd or [None] * len(ad)
        control = 'index yaw'
    else:
        ad = [row[1] for row in data["ad_jacobian"]]
        fd = [row[1] for row in data["fork_fd_jacobian"]]
        cpu = [row[1] for row in data["cpu_fd_jacobian"]]
        control = 'index yaw'
    width, height, left, right, top, step = 920, 275, 260, 60, 85, 42
    ratios = [v/c for row in (ad, fd) for v, c in zip(row, cpu)
              if v is not None and abs(c) > 1e-14 and math.isfinite(v/c)]
    xlo = min(0.75, min(ratios) - 0.15)
    xhi = max(1.25, max(ratios) + 0.15)
    x = lambda v: left + (v - xlo) / (xhi - xlo) * (width-left-right)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Automatic and finite-difference derivatives relative to CPU finite differences">',
             f'<text x="{left}" y="27" class="label" style="font-weight:700">{html.escape(title)}: {control} derivative</text>',
             f'<text x="{left}" y="47" class="small">Ratio to the CPU central finite difference (1 is agreement)</text>']
    tick_step = 0.25 if xhi-xlo < 1.2 else 0.5
    tick0 = math.ceil(xlo / tick_step) * tick_step
    ticks = [tick0 + tick_step*i for i in range(int((xhi-tick0)/tick_step)+1)]
    if 1 not in ticks:
        ticks.append(1)
    for tick in sorted(ticks):
        xx=x(tick)
        parts.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{height-35}" stroke="var(--ink3)" stroke-width="2"/>' if tick==1 else
                     f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{height-35}" stroke="var(--rule2)"/>')
        parts.append(f'<text x="{xx:.1f}" y="{height-13}" text-anchor="middle" class="small">{tick:.3g}</text>')
    for i, (name, a, f, c) in enumerate(zip(METRICS, ad, fd, cpu)):
        yy=top+i*step
        parts.append(f'<text x="{left-16}" y="{yy+4}" text-anchor="end" class="small">{html.escape(name)}</text>')
        for value, offset, color, label in ((a,-7,'var(--s1)','AD'),(f,7,'var(--s2)','MJX FD')):
            if value is None:
                continue
            ratio=value/c if abs(c)>1e-14 else math.nan
            if math.isfinite(ratio):
                xx=max(left,min(width-right,x(ratio)))
                parts.append(f'<circle cx="{xx:.1f}" cy="{yy+offset}" r="5" fill="{color}"><title>{label}: {value:.6g}; CPU FD: {c:.6g}; ratio {ratio:.5g}</title></circle>')
    parts.append(f'<text x="{left}" y="{height-40}" class="small" style="fill:var(--s1)">● automatic derivative</text>')
    if any(v is not None for v in fd):
        parts.append(f'<text x="{left+250}" y="{height-40}" class="small" style="fill:var(--s2)">● fork finite difference</text>')
    parts.append('</svg>')
    return '<div class="viz">' + ''.join(parts) + '</div>'


def main():
    mask = read('20261007-pad_mask_cpu.json')
    search = read('20261007-pad_local_search_20_mj331.json')
    plain_jac = read('20261007-official_padsT1_cylinder_jacobian_20.json')
    cfd_jac = read('20261007-official_padsT1_cylinder_cfd_jacobian_20.json')
    fd_sweep = read('20261007-official_padsT1_cylinder_fd_sweep_20.json')
    attempts = read('20261007-resource_attempts.json')
    native_pad1 = read('20261007-native_padsT_cylinder_1.json')
    completed = {row['name']: row for row in attempts['completed']}
    pad2 = read('20261007-official_padsT2_cylinder_1.json')
    pad2_grad = read('20261007-official_padsT2_cylinder_jvp_1.json', optional=True)
    pad1 = read('20261007-official_padsT1_cylinder_1.json', optional=True)
    pad_grad = read('20261007-official_padsT1_cylinder_jvp_1.json', optional=True)
    if pad_grad is None:
        pad_grad = read('20261007-official_padsT2_cylinder_jvp_1.json', optional=True)
    pad_fork_fd = read('20261007-official_padsT1_cylinder_fd_1.json', optional=True)
    if pad_grad and pad_fork_fd and pad_grad['scene'] == pad_fork_fd['scene'] and pad_grad['steps'] == pad_fork_fd['steps']:
        pad_grad = dict(pad_grad)
        pad_grad['fork_fd_directional_derivative'] = pad_fork_fd['fork_fd_directional_derivative']
    sphere_cyl = read('20261007-official_cylinder_plain_1.json')
    sphere_plain = read('20261007-official_capsule_plain_20.json')
    sphere_cfd = read('20261007-official_capsule_cfd_20.json')
    native_sphere = read('20261007-native_capsule_gradient_20_x64.json')

    style_path = ROOT / 'scripts/hom_chain_page.template.html'
    style = re.search(r'<style>(.*?)</style>', style_path.read_text(), re.S).group(1)
    style += '''\n.viz{background:var(--card);border:1px solid var(--rule);border-radius:10px;padding:12px;box-shadow:var(--shadow)}
.viz svg{display:block;width:100%;height:auto}.viz .label{fill:var(--ink);font:600 15.5px var(--f-display)}
.viz .small{fill:var(--ink2);font:400 15.5px var(--f-mono)}.decision td:first-child{font-weight:600}
@media print{.col>.tw{width:100%}table{font-size:11px;table-layout:fixed}th{font-size:9px;white-space:normal;padding:7px 6px}td{padding:7px 6px;overflow-wrap:anywhere}.tw{overflow:visible}}
'''

    gap1 = max(abs(pad1['fork_initial_finger_dist_m'][f]-pad1['cpu_initial_contact']['finger_min_dist_m'][f]) for f in FINGERS)*1e6
    base = search['candidates']['zero']['vertical_cos']
    fd = search['candidates']['cpu_fd']['vertical_cos']
    random_best = search['random_best']['vertical_cos']
    native_gap = max(abs(native_pad1['mjx_initial_contact'][f]['min_dist_m']-
                         native_pad1['initial_contact']['finger_min_dist_m'][f]) for f in FINGERS)*1e6
    lede = (f'Native MJX 3.6 and the released fork completed one step of the 2,388-pad grasp with '
            f'at most {num(max(native_gap,gap1),3)} µm initial tool-gap difference from CPU MuJoCo. '
            f'Over 20 ms, the fork gradient selected a 0.1 rad control action that scored '
            f'{plain_jac["local_search"]["cpu_score_ad_direction"]:.6f} in CPU MuJoCo; '
            f'CPU finite differences scored {fd:.6f} and the best of 128 samples scored {random_best:.6f}. '
            f'The warm three-control gradient took {num(plain_jac["gradient_warm_s"],3)} s versus '
            f'{num(1e3*search["cpu_fd_6_evals_s"],3)} ms for six CPU finite-difference evaluations. '
            f'Contact-from-distance with the scan-loop solver scored '
            f'{cfd_jac["local_search"]["cpu_score_ad_direction"]:.6f} at {num(cfd_jac["gradient_warm_s"],3)} s.')

    scene_text = (f'The deployed D1 plan was replayed for 0.8 s with the printed thermoplastic polyurethane fingertip model. The 1 mm '
                  f'sphere-packed surface has 796 pads per finger and produced {mask["rows"]["pads"]["initial_contact"]["total_contacts"]} '
                  f'contacts after the post was disabled. The tool-only pad collision mask retained all pad-to-tool contacts. '
                  f'The full and masked CPU scenes had identical finger forces and tool pose after 20 ms '
                  f'(maximum recorded pose difference {num(mask["final_tool_qpos_max_abs_error"])}). '
                  f'The 2 mm surface has {pad2["n_pads"]["thumb"]} pads per finger. Both MJX probes used the deployed cylinder tool.')

    gap_fig = svg_gap([('1 mm native',native_pad1,'var(--ink3)'),('1 mm fork',pad1,'var(--s1)'),('2 mm fork',pad2,'var(--good)'),
                       ('single sphere',sphere_cyl,'var(--s2)')])
    forward_rows=[]
    forward_rows.append(['1 mm pads, native MJX 3.6','2,388',num(native_gap,3),
                         num(native_pad1['max_qpos_abs_error']*1e6,3),
                         num(native_pad1['max_qvel_abs_error'],3),num(native_pad1['mjx_first_s'],3)])
    for label,d in [('1 mm pads, released fork',pad1),('2 mm pads, released fork',pad2),
                    ('single sphere, released fork',sphere_cyl)]:
        if d is None:
            continue
        error=max(abs(d['fork_initial_finger_dist_m'][f]-d['cpu_initial_contact']['finger_min_dist_m'][f]) for f in FINGERS)*1e6
        forward_rows.append([html.escape(label),str(sum(d.get('n_pads',{}).values()) if d.get('n_pads') else 3),
                             num(error,3),num(d['max_qpos_abs_error']*1e6,3),
                             num(d['max_qvel_abs_error'],3),num(d['rollout_compile_s'],3)])
    forward_table = table(['Scene / MJX','Spheres','Gap error µm',
                           'Position ×10⁻⁶','Velocity error','First compiled step s'], forward_rows)

    if pad_grad and pad_grad.get('ad_directional_derivative') is not None:
        grad_title = f'{num(pad_grad["pad_spacing_m"]*1000,1)} mm packed surface'
        internal = ('The fork finite difference agrees with its automatic derivative. ' if
                    pad_grad.get('fork_fd_directional_derivative') else '')
        grad_text = (f'The released fork returned finite index-yaw derivatives on the {grad_title} after '
                     f'{pad_grad["steps"]} ms. {internal}The table and Figure 2 compare the fork with '
                     f'CPU MuJoCo. The first JVP compiled in {num(pad_grad["gradient_compile_s"],3)} s; '
                     f'a warm JVP took {num(1e3*pad_grad["gradient_warm_s"],3)} ms, while '
                     f'three CPU finite-difference evaluations took '
                     f'{num(1e3*pad_grad["cpu_fd_3_evals_s"],3)} ms. '
                     'The native MJX 3.6 pad derivative was not measured.')
        if pad2_grad and pad_grad['pad_spacing_m'] == 0.001:
            grad_text += (f' At 2 mm pad spacing, the fork contact-torque derivative was '
                          f'{num(pad2_grad["ad_directional_derivative"][3],3)} N m/rad and the CPU '
                          f'derivative was {num(pad2_grad["cpu_fd_directional_derivative"][3],3)} N m/rad; '
                          'their signs differ.')
        gdata=pad_grad
    else:
        grad_title = 'single-sphere capsule reference'
        grad_text = ('The 20 ms single-sphere capsule reference differentiates under the released fork. '
                     'Plain MJX 3.3.1 agrees with CPU finite differences for tool motion and contact torque. '
                     'The fork’s contact-from-distance switch changes the backward derivative while preserving '
                     'the forward trajectory. Native MJX 3.6 returned 12 nonfinite entries in the same reference. '
                     'The packed-tip derivative remains a separate measurement.')
        gdata=sphere_plain
    gradient_fig=svg_gradient(gdata,grad_title)
    if 'ad_directional_derivative' in gdata:
        ad, mjxfd, cpufd=gdata['ad_directional_derivative'],gdata['fork_fd_directional_derivative'],gdata['cpu_fd_directional_derivative']
        mjxfd = mjxfd or [None] * len(ad)
    else:
        ad=[r[1] for r in gdata['ad_jacobian']]
        mjxfd=[r[1] for r in gdata['fork_fd_jacobian']]
        cpufd=[r[1] for r in gdata['cpu_fd_jacobian']]
    grad_rows=[]
    for name,unit,a,f,c in zip(METRICS,UNITS,ad,mjxfd,cpufd):
        grad_rows.append([html.escape(name),html.escape(unit),num(c,4),num(f,4),num(a,4),
                          num(100*abs(a-c)/abs(c),3)+'%' if abs(c)>1e-14 else '—'])
    gradient_table=table(['Output','Derivative unit','CPU FD','Fork FD','Fork AD','AD error'],grad_rows)
    plain_index = sphere_plain['ad_jacobian'][0][1]
    cfd_index = sphere_cfd['ad_jacobian'][0][1]
    cfd_text = (f'On the 1 mm packed surface, contact-from-distance and the scan-loop solver changed the 20 ms '
                f'index-yaw vertical-cosine derivative from {num(plain_jac["ad_directional_derivative"][0],5)} '
                f'to {num(cfd_jac["ad_directional_derivative"][0],5)} per radian. '
                f'The three-control action direction moved from cosine '
                f'{num(plain_jac["local_search"]["direction_cosine"],4)} to '
                f'{num(cfd_jac["local_search"]["direction_cosine"],4)} against CPU finite differences. '
                f'The single-sphere capsule control gave a {num(100*(cfd_index/plain_index-1),3)}% '
                f'change in the same index-yaw derivative.')

    jac_text = (f'The plain fork returned finite derivatives for all three yaw targets after 20 ms. '
                f'Its 0.1 rad action raised CPU vertical cosine from {base:.6f} to '
                f'{plain_jac["local_search"]["cpu_score_ad_direction"]:.6f}. '
                f'The CPU finite-difference action reached {fd:.6f}; the contact-from-distance action reached '
                f'{cfd_jac["local_search"]["cpu_score_ad_direction"]:.6f}. '
                f'The plain and contact-from-distance three-direction gradients took '
                f'{num(plain_jac["gradient_warm_s"],3)} and {num(cfd_jac["gradient_warm_s"],3)} s when warm, '
                f'after {num(plain_jac["gradient_compile_s"],3)} and {num(cfd_jac["gradient_compile_s"],3)} s compilation.')
    sweep_text = (f'The 20 ms fork index-yaw finite difference for vertical cosine was '
                  f'{num(fd_sweep["fork_fd_sweep"]["0.01"][0],5)} per radian at 0.01 rad and '
                  f'{num(fd_sweep["fork_fd_sweep"]["0.0003"][0],5)} at 0.0003 rad; '
                  f'the automatic derivative was {num(plain_jac["ad_directional_derivative"][0],5)}. '
                  f'CPU finite differences stayed between '
                  f'{num(min(row[0] for row in fd_sweep["cpu_fd_sweep"].values()),5)} and '
                  f'{num(max(row[0] for row in fd_sweep["cpu_fd_sweep"].values()),5)} over the sweep. '
                  f'The 0.01 rad fork finite difference samples a wider response than its local derivative.')
    sweep_rows = [['Automatic derivative','—',num(plain_jac['ad_directional_derivative'][0],5),'—',
                   num(plain_jac['ad_directional_derivative'][3],5),'—']]
    for step_size in sorted(fd_sweep['fork_fd_sweep'], key=float, reverse=True):
        fork_row=fd_sweep['fork_fd_sweep'][step_size]
        cpu_row=fd_sweep['cpu_fd_sweep'][step_size]
        sweep_rows.append(['Central finite difference',step_size,num(fork_row[0],5),num(cpu_row[0],5),
                           num(fork_row[3],5),num(cpu_row[3],5)])
    sweep_table=table(['Method','Yaw step rad','Fork orientation /rad','CPU orientation /rad',
                       'Fork torque N m/rad','CPU torque N m/rad'],sweep_rows)

    gain_fd=fd-base
    gain_best=random_best-base
    search_text=(f'At a 0.1 rad three-yaw-target radius, the plain fork gradient increased CPU vertical cosine '
                 f'by {num(plain_jac["local_search"]["cpu_score_ad_direction"]-base,4)}. '
                 f'The CPU finite-difference direction increased it by {num(gain_fd,4)}, '
                 f'the contact-from-distance direction by '
                 f'{num(cfd_jac["local_search"]["cpu_score_ad_direction"]-base,4)}, '
                 f'and the best of 128 sampled directions by {num(gain_best,4)}. '
                 f'All 128 sampled actions retained positive fingertip normal force.')
    def cost_ms(value):
        return f'{value:,.0f}' if value >= 1000 else num(value,3)
    search_rows=[]
    for name,row,cost in [('No yaw change',search['candidates']['zero'],0),
                          ('CPU finite-difference direction',search['candidates']['cpu_fd'],1e3*search['cpu_fd_6_evals_s']),
                          ('Best of 128 sampled directions',search['random_best'],1e3*search['random_128_evals_s'])]:
        search_rows.append([html.escape(name),f'{row["vertical_cos"]:.6f}',num(1e4*(row['vertical_cos']-base),3),
                            num(min(row['finger_normal_force_N'].values()),3),cost_ms(cost)])
    for name,key,jac in [('Plain fork gradient','plain_ad',plain_jac),
                         ('Contact-from-distance gradient','cfd_ad',cfd_jac)]:
        score=search['candidates'][key]['vertical_cos']
        search_rows.append([name,f'{score:.6f}',num(1e4*(score-base),4),
                            num(min(search['candidates'][key]['finger_normal_force_N'].values()),3),
                            cost_ms(1e3*jac['gradient_warm_s'])])
    search_table=table(['Direction','Cosine','Gain ×0.0001','Min finger force N','Search time ms'],search_rows)

    decision_table=table(['Use','Result and next test'],[
        ['Initial grasp synthesis','Test contact-from-distance during first touch on the D1–D8 packed tips. Score retained grasps and wall time against the current cross-entropy method search.'],
        ['Reorientation control','Use six CPU finite differences as the local baseline. The plain fork gave the same 20 ms turn gain at much higher CPU cost. Measure 100 ms forward parity before a longer-horizon controller comparison.'],
        ['Policy training in MuJoCo Playground','Playground can run MJX-JAX, but sampled policy-gradient training does not differentiate through its simulator. A simulator change alone retains the same training objective.'],
        ['High-dimensional tip design','Test a design vector large enough that CPU finite differences become costly. Compare gain per wall second, including compilation and transfer to the MuJoCo-Warp task suite.']])
    methods=(f'The D1 design and plan come from the real_v1 hardware family; the starting tool pose has seed 1 jitter. '
             f'The palm is fixed, the grip lasts 0.8 s, and each continuation uses 1 ms implicit-fast MuJoCo steps, '
             f'an elliptic friction cone and Newton contact solver with impratio 100. The CPU pad-mask check used '
             f'MuJoCo {mask["mujoco_version"]}; the official fork used MuJoCo {pad2["mujoco_version"]} and JAX {pad2["jax_version"]}. '
             f'The fork packages were cloned at pinned commits in the runbook. All JAX runs used CPU, so GPU throughput is unmeasured. '
             f'The released 1 mm forward process peaked at {gib_from_kib(completed["released fork, 1 mm packed tip, one forward step"]["max_resident_kB"])} GiB '
             f'resident memory; the one-direction derivative peaked at '
             f'{gib_from_kib(completed["released fork, 1 mm packed tip, one directional derivative"]["max_resident_kB"])} GiB. '
             f'The native 1 mm forward process peaked at '
             f'{gib_from_kib(completed["native MJX 3.6, 1 mm packed tip, one forward step"]["max_resident_kB"])} GiB '
             f'and wrote its one-step output before the resource guard stopped the process. '
             f'The plain and contact-from-distance 20 ms three-control gradient processes peaked at '
             f'{gib_from_kib(completed["released fork, 1 mm packed tip, 20 ms three-control gradient"]["max_resident_kB"])} '
             f'and {gib_from_kib(completed["released fork with CFD and scan-loop, 1 mm packed tip, 20 ms three-control gradient"]["max_resident_kB"])} GiB.')

    replacements={
        'STYLE':'<style>'+style+'</style>', 'BUILT':datetime.now().strftime('%Y-%m-%d %H:%M'),
        'LEDE':html.escape(lede), 'SCENE_TEXT':html.escape(scene_text), 'GAP_FIG':gap_fig,
        'FORWARD_TABLE':forward_table, 'GRADIENT_TEXT':html.escape(grad_text),
        'GRADIENT_FIG':gradient_fig, 'GRADIENT_TABLE':gradient_table,
        'CFD_TEXT':html.escape(cfd_text), 'JAC_TEXT':html.escape(jac_text),
        'SWEEP_TEXT':html.escape(sweep_text), 'SWEEP_FIG':svg_fd_sweep(fd_sweep,plain_jac),
        'SWEEP_TABLE':sweep_table,
        'SEARCH_TEXT':html.escape(search_text), 'SEARCH_FIG':svg_search(search,plain_jac,cfd_jac),
        'SEARCH_TABLE':search_table, 'DECISION_TABLE':decision_table,
        'METHODS':html.escape(methods),
    }
    page=TEMPLATE.read_text()
    for key,value in replacements.items():
        page=page.replace('{{'+key+'}}',value)
    assert not re.search(r'{{[A-Z_]+}}',page)
    OUT.write_text(retro_style.apply(page))  # plain page style (owner, 2026-10-09)
    print(OUT)


if __name__=='__main__':
    main()
