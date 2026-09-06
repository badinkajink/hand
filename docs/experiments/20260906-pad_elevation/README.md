# Pad contact elevation, 2026-09-06 — how to regenerate

The result page is `20260906-pad_elevation.html`, built by
`scripts/real_v1_pad_elevation_page.py`. Its inputs, and the command behind each:

| file | command |
|---|---|
| `20260906-griploop/grasp_closure.json` | `real_v1_grasp_closure.py --sets A --squeeze-mm 2.0` |
| `20260906-griploop/grasp_closure_elev.json` | same, `--elevations 0,-6,-12,-18,-24` |
| `20260906-griploop/grasp_closure_d1d8.json` | same, `--hands sv1_w6689_b060,sv1_w0099_b100,rv05_manual_b85 --depths 44,48,52,56,60,64,68 --straddles 32,40` |
| `20260906-d1_graspgrid/chain_hands.json` | `real_v1_chain_hands.py --sets A --hands sv1_w6689_b060,sv1_w2360_b075 --straddles 32,36,40 --depths 50.5,56.5,62.5 --stand table --reps 6 --loads 0` |
| `20260906-griploop/chain_hands.json` | `real_v1_chain_hands.py --sets A --stand table --reps 6 --loads 0,150,250` |
| `20260906-screened_turn/chain_hands.json` | `real_v1_chain_hands.py --sets A --stand air --reps 6 --loads 0,250` |
| `20260906-elevation/chain_hands.json` | `real_v1_chain_hands.py --sets A --elevations 0,-6,-12,-18,-24 --stand table --reps 6 --loads 0` |
| `20260906-turn_band/chain_hands.json` | `real_v1_chain_hands.py --sets A --stand air --reps 4 --loads 0 --budgets 0.3,0.45,0.6 --turn-steps 250,550` |
| `20260906-turn_elev/chain_hands.json` | `real_v1_chain_hands.py --sets A --stand air --reps 4 --loads 0 --elevations 0,-12 --budgets 0.6,0.8 --turn-steps 250,550` |

All under `uv run --extra rl --extra arm python scripts/<name>`.
