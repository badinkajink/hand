"""Extra checks: where c_A, the adaptor faces, the M6 counterbores, the 4.4 mm holes, the flats and the
flange's dowel position land in E for each claimed T; and how many vertices the all-hand check used."""
import sys, json, numpy as np
sys.path.insert(0, '.')
np.set_printoptions(suppress=True, precision=5, linewidth=160)
v = json.load(open('verify.json'))
Tm = [np.array(T) for T in v['T_mine']]
c6 = np.array(v['m6_centre_A'])
P44 = np.array(v['adaptor_44_holes_A'])
for k, T in enumerate(Tm):
    R, t = T[:3, :3], T[:3, 3]
    m = lambda p: R @ np.asarray(p, float) + t
    cE = m([*c6, 28.0])
    top = m([*c6, 28.0])[2]; bot = m([*c6, 6.0])[2]; seat = m([*c6, 13.0])[2]
    # the flange's 6 mm pin hole at -90 deg (convention under test), 25 mm out: where is it in A?
    pin_E = np.array([0, -25.0, 0])
    pin_A = R.T @ (pin_E - t)
    # nearest adaptor hole to that point (6.4 mm M6, 4.4 mm) in A
    m6A = np.array([[c6[0] + 25 * np.cos(np.radians(a)), c6[1] + 25 * np.sin(np.radians(a))] for a in v['m6_angles_A_deg']])
    d_m6 = np.linalg.norm(m6A - pin_A[:2], axis=1).min(); d_44 = np.linalg.norm(P44 - pin_A[:2], axis=1).min()
    flats_E = sorted(((np.degrees(np.arctan2(*(R[:2, :2] @ [np.cos(np.radians(a)), np.sin(np.radians(a))])[::-1]))) % 360) for a in v['adaptor_flat_normals_A_deg'])
    r44 = np.linalg.norm(P44 - c6, axis=1)
    print('k=%d flange centre -> %s; adaptor faces at z_E %.6f (flange side) and %.6f, head seats at z_E %.6f; flats face %s deg in E;' % (k, np.round(cE, 9), top, bot, seat, np.round(flats_E, 6)))
    print('     the flange pin hole (E: 25 mm at -90 deg) is at A %s, %.2f mm from the nearest 6.4 mm hole and %.2f mm from the nearest 4.4 mm hole (4.4 mm holes at radii %s)' % (np.round(pin_A, 3), d_m6, d_44, np.round(r44, 3)))
