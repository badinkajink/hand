#!/usr/bin/env python3
"""Clearances of the index finger at the poses of its stall on the night of 2026-11-04, in the full model on the UR5 at k = 1,
from the home layout (docs/experiments/20261104-sr2_hand_crosstalk/README.md, step 38 of run1b_home_rest). Prints the nearest
palm and gantry parts for the moving links; run from the repository root under resguard. Result at mcp 45 + pip 80.3 deg: 30.4 mm."""
import numpy as np, mujoco, json, sys
m=mujoco.MjModel.from_xml_path('assets/mjcf/experimental/20261104-cad_v3/ur5_full_k1.xml')
d=mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m,d,m.key('sr2_night').id)
d.qpos[:3]=(0,0,-5)
def setq(j,v): d.qpos[m.jnt_qposadr[m.joint(j).id]]=v
home={'thumb':(-.03,-.055),'index':(.03,.03),'middle':(.03,-.03)}
for f,(x,y) in home.items():
    setq(f'{f}_x',x); setq(f'{f}_y',y); setq(f'{f}_bridge',y)
    for j in ('yaw','mcp','pip'): setq(f'{f}_{j}',0.0)
gn=lambda g: mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or f'#{g}'
bn=lambda b: mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,b) or f'#{b}'
def subtree(root):
    ids=[]
    for b in range(m.nbody):
        x=b
        while x>0:
            if x==root: ids.append(b); break
            x=m.body_parentid[x]
    return ids
fing=[g for g in range(m.ngeom) if m.geom_bodyid[g] in subtree(m.body('index_mcp_frame').id) and m.geom_type[g]==mujoco.mjtGeom.mjGEOM_SPHERE]
base=[g for g in range(m.ngeom) if m.geom_bodyid[g] in (m.body('index_mount').id,m.body('index_yaw_frame').id) and m.geom_type[g]==mujoco.mjtGeom.mjGEOM_SPHERE]
palm_id=m.body('palm_pose').id
others=[g for g in range(m.ngeom) if m.geom_bodyid[g] in (palm_id,m.body('index_bridge').id,m.body('thumb_bridge').id,m.body('middle_bridge').id) and m.geom_type[g] in (mujoco.mjtGeom.mjGEOM_MESH,mujoco.mjtGeom.mjGEOM_BOX,mujoco.mjtGeom.mjGEOM_CYLINDER)]
print(len(fing),'finger spheres',len(others),'others',[gn(g) for g in others][:60])
for label,(mcp,pip) in {'zero':(0,0),'37.5+90':(37.5,90),'50+67.5':(50,67.5),'45+80.3 stalled':(45,80.3),'50+90 commanded':(50,90)}.items():
    setq('index_mcp',np.radians(mcp)); setq('index_pip',np.radians(pip))
    mujoco.mj_kinematics(m,d); mujoco.mj_collision(m,d) if False else None
    mujoco.mj_forward(m,d)
    best=[]
    ft=np.zeros(6)
    # bound by distance from sphere centres to other geoms' bounding boxes first: use mj_geomDistance with a cap
    for go in others:
        dmin=1e9; pair=None
        for g in fing:
            dist=mujoco.mj_geomDistance(m,d,g,go,0.05,ft)
            if dist<dmin: dmin=dist; pair=g
        best.append((round(dmin*1000,1),gn(pair),gn(go)))
    best.sort()
    Tw=np.eye(4); 
    palm=d.body('palm_pose'); R=palm.xmat.reshape(3,3)
    zs=[(R.T@(d.geom_xpos[g]-palm.xpos))[2]*1000+m.geom_size[g,0]*1000 for g in fing]
    bb=(1e9,None,None)
    C=d.geom_xpos[fing]; rf=m.geom_size[fing,0]; Cb=d.geom_xpos[base]; rb=m.geom_size[base,0]
    D=np.linalg.norm(C[:,None]-Cb[None],axis=2)-rf[:,None]-rb[None]
    i,j=np.unravel_index(np.argmin(D),D.shape)
    print(label,'moving links top z_P (mm)',round(max(zs),1),'nearest palm parts:',best[:5],'nearest own base:',round(float(D[i,j])*1000,1),gn(fing[i]),gn(base[j]))
