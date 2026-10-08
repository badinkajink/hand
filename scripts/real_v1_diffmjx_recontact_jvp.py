#!/usr/bin/env python3
"""Measure a local DiffMJX control derivative immediately before index recontact."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from real_v1_contact_gait_search import (CLOSE_SCALE_RAD, OPEN_UNTIL_MS,
                                         PHASE_EDGES_MS, Scene, YAW_SCALE_RAD)
import real_v1_diffmjx_gate as gate


def prepare(scene, x, start_ms):
    x = np.asarray(x, float)
    d = gate.clone_data(scene.model, scene.held)
    phase = -1
    for k in range(start_ms):
        next_phase = int(k >= PHASE_EDGES_MS[0]) + int(k >= PHASE_EDGES_MS[1])
        if next_phase != phase or k == OPEN_UNTIL_MS:
            phase = next_phase
            d.ctrl[:] = scene.ctrl0
            ids = scene.yaw_ids
            d.ctrl[ids] = np.clip(scene.ctrl0[ids] + YAW_SCALE_RAD*x[3*phase:3*phase+3],
                                  scene.low[ids], scene.high[ids])
            for i, finger in enumerate(("index", "middle")):
                ids = scene.flex_ids[finger]
                if k < OPEN_UNTIL_MS:
                    amplitude = 0.5*(x[9+i]+1)
                    d.ctrl[ids] = scene.ctrl0[ids] + amplitude*(scene.low[ids]-scene.ctrl0[ids])
                else:
                    d.ctrl[ids] = np.clip(scene.ctrl0[ids]+CLOSE_SCALE_RAD*x[11+i],
                                          scene.low[ids],scene.high[ids])
        scene.mujoco.mj_step(scene.model,d)
    return d


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans",type=Path,required=True)
    parser.add_argument("--method",default="cem")
    parser.add_argument("--start-ms",type=int,default=280)
    parser.add_argument("--steps",type=int,default=10)
    parser.add_argument("--cfd",action="store_true")
    parser.add_argument("--scan-loop",action="store_true")
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()

    import jax
    jax.config.update("jax_enable_x64",True)
    import jax.numpy as jnp
    from mujoco import mjx

    x=np.asarray(json.loads(args.plans.read_text())["candidates"][args.method]["x"])
    scene=Scene("tpu6","padsT",1,pad_spacing_mm=2)
    model=scene.model
    state=prepare(scene,x,args.start_ms)
    index_id=scene.model.actuator("a_index_mcp").id
    obj=scene.model.body(gate.rb.OBJ).id
    vadr=int(scene.model.jnt_dofadr[scene.model.body_jntadr[obj]])
    qa=scene.qadr
    initial_cpu=gate.contact_summary(model,state)

    def cpu_metrics(delta):
        d=gate.clone_data(model,state)
        d.ctrl[index_id]+=delta
        for _ in range(args.steps):
            scene.mujoco.mj_step(model,d)
        quat=d.qpos[qa+3:qa+7]
        metric=np.array([1-2*(quat[1]**2+quat[2]**2),d.qpos[qa+2],
                         d.qvel[vadr+3],d.qfrc_constraint[vadr+3]])
        return metric,gate.contact_summary(model,d)

    cpu_zero,cpu_contact=cpu_metrics(0.0)
    cpu_fd={}
    for eps in (0.01,0.003,0.001):
        plus,cp=cpu_metrics(eps)
        minus,cm=cpu_metrics(-eps)
        cpu_fd[str(eps)]={"derivative":((plus-minus)/(2*eps)).tolist(),
                          "plus_index_force_N":cp["finger_normal_force_N"]["index"],
                          "minus_index_force_N":cm["finger_normal_force_N"]["index"]}
    print(f"CPU index force {initial_cpu['finger_normal_force_N']['index']:.3f} -> "
          f"{cpu_contact['finger_normal_force_N']['index']:.3f} N at "
          f"{args.start_ms + args.steps} ms", flush=True)

    t0=time.perf_counter()
    mx=mjx.put_model(model)
    put_model_s=time.perf_counter()-t0
    mx=mx.replace(opt=mx.opt.replace(cfd_enable=args.cfd,
                                    cfd_solimp=jnp.array([0.0,0.01,0.05,1.0,2.0]),
                                    scan_loop=args.scan_loop))
    md=mjx.put_data(model,state)
    md=md.replace(contact=md.contact.replace(geom1=md.contact.geom1.astype(jnp.int64),
                                              geom2=md.contact.geom2.astype(jnp.int64),
                                              geom=md.contact.geom.astype(jnp.int64)))

    @jax.jit
    def forward(d):
        return mjx.forward(mx,d)

    t0=time.perf_counter()
    jinitial=forward(md)
    jinitial.qpos.block_until_ready()
    forward_compile_s=time.perf_counter()-t0
    tool_geoms={i for i in range(model.ngeom) if model.geom_bodyid[i]==obj}
    index_geoms={i for i in range(model.ngeom) if model.body(model.geom_bodyid[i]).name.startswith("index_")}
    index_dist=[float(dist) for (g0,g1),dist in zip(np.asarray(jinitial.contact.geom),
                                                    np.asarray(jinitial.contact.dist))
                if (int(g0) in tool_geoms and int(g1) in index_geoms)
                or (int(g1) in tool_geoms and int(g0) in index_geoms)]

    @jax.jit
    def metrics(delta):
        s=md.replace(ctrl=md.ctrl.at[index_id].add(delta))
        def step(carry,_):
            return mjx.step(mx,carry),None
        final=jax.lax.scan(step,s,None,length=args.steps)[0]
        quat=final.qpos[qa+3:qa+7]
        return jnp.array([1-2*(quat[1]**2+quat[2]**2),final.qpos[qa+2],
                          final.qvel[vadr+3],final.qfrc_constraint[vadr+3]])

    @jax.jit
    def derivative(delta):
        return jax.jvp(metrics,(delta,),(jnp.array(1.0),))[1]

    zero=jnp.array(0.0)
    t0=time.perf_counter()
    base=metrics(zero)
    base.block_until_ready()
    rollout_compile_s=time.perf_counter()-t0
    t0=time.perf_counter()
    ad=derivative(zero)
    ad.block_until_ready()
    derivative_compile_s=time.perf_counter()-t0
    t0=time.perf_counter()
    derivative(zero).block_until_ready()
    derivative_warm_s=time.perf_counter()-t0
    eps=.01
    fork_fd=(np.asarray(metrics(jnp.array(eps)))-np.asarray(metrics(jnp.array(-eps))))/(2*eps)
    out={"backend":"released DiffMJX fork of MJX 3.3.1","jax_backend":jax.default_backend(),
         "precision":"float64","scene":str(scene.scene),"plans":str(args.plans),
         "method":args.method,"pad_spacing_mm":2,"start_ms":args.start_ms,
         "steps":args.steps,"control":"index MCP target bias (rad)","cfd":args.cfd,
         "scan_loop":args.scan_loop,"metric_names":["vertical_cos","tool_z_m",
                         "tool_omega_x_rad_s","tool_contact_torque_x_Nm"],
         "initial_cpu_index_force_N":initial_cpu["finger_normal_force_N"]["index"],
         "initial_fork_index_min_gap_m":min(index_dist) if index_dist else None,
         "final_cpu_index_force_N":cpu_contact["finger_normal_force_N"]["index"],
         "cpu_zero":cpu_zero.tolist(),"fork_zero":np.asarray(base).tolist(),
         "cpu_fd":cpu_fd,"fork_fd_0.01":fork_fd.tolist(),
         "fork_ad":np.asarray(ad).tolist(),"put_model_s":put_model_s,
         "forward_compile_s":forward_compile_s,"rollout_compile_s":rollout_compile_s,
         "derivative_compile_s":derivative_compile_s,
         "derivative_warm_s":derivative_warm_s}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
    print(json.dumps(out,indent=2,allow_nan=False),flush=True)


if __name__=="__main__":
    main()
