# Perp compact-design sweep (scripted open-loop swing)

50 designs from `perp_compact_design`, gated on self-collision, grasp
retargeted to the reference fingertip world targets. `held` is asked of the physics
(tip force, height, floor contact) after a 3.2 s hold, never inferred from cos.

| rank | design | thumb x | pair x | pair \|y\| | peak cos | final cos | grip N | obj z | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `t0.00_x0.25_y0.00` | -65.0 | +29.4 | 48.0 | +0.995 | +0.984 | 3.2 | 0.078 | **HELD** |
| 2 | `t0.00_x0.00_y0.00` | -65.0 | +35.0 | 48.0 | +0.991 | +0.987 | 1.2 | 0.077 | **HELD** |
| 3 | `t0.25_x0.25_y0.00` | -51.9 | +29.4 | 48.0 | +0.987 | +0.983 | 2.1 | 0.080 | **HELD** |
| 4 | `t0.50_x0.50_y0.00` | -38.8 | +23.8 | 48.0 | +0.879 | +0.879 | 13.0 | 0.084 | **HELD** |
| 5 | `t0.00_x0.50_y0.00` | -65.0 | +23.8 | 48.0 | +0.746 | +0.746 | 11.7 | 0.088 | **HELD** |
| 6 | `t0.25_x0.50_y0.00` | -51.9 | +23.8 | 48.0 | +0.711 | +0.711 | 11.6 | 0.092 | **HELD** |
| 7 | `t0.00_x1.00_y0.00` | -65.0 | +12.5 | 48.0 | +0.672 | +0.672 | 3.3 | 0.096 | **HELD** |
| 8 | `t0.50_x1.00_y0.00` | -38.8 | +12.5 | 48.0 | +0.660 | +0.660 | 3.2 | 0.097 | **HELD** |
| 9 | `t0.25_x1.00_y0.00` | -51.9 | +12.5 | 48.0 | +0.647 | +0.647 | 3.2 | 0.098 | **HELD** |
| 10 | `t0.00_x0.75_y0.00` | -65.0 | +18.1 | 48.0 | +0.582 | +0.574 | 6.5 | 0.096 | **HELD** |
| 11 | `t0.25_x0.75_y0.00` | -51.9 | +18.1 | 48.0 | +0.579 | +0.553 | 6.4 | 0.098 | **HELD** |
| 12 | `t0.50_x0.75_y0.00` | -38.8 | +18.1 | 48.0 | +0.575 | +0.568 | 6.4 | 0.097 | **HELD** |
| 13 | `t0.75_x0.00_y0.00` | -25.6 | +35.0 | 48.0 | +1.000 | +0.999 | 0.0 | 0.050 | on floor |
| 14 | `t0.75_x0.25_y0.00` | -25.6 | +29.4 | 48.0 | +1.000 | +0.999 | 0.0 | 0.050 | on floor |
| 15 | `t0.50_x0.00_y0.00` | -38.8 | +35.0 | 48.0 | +1.000 | +0.997 | 0.0 | 0.051 | on floor |
| 16 | `t0.50_x0.25_y0.00` | -38.8 | +29.4 | 48.0 | +0.999 | +1.000 | 0.0 | 0.050 | on floor |
| 17 | `t0.25_x0.00_y0.00` | -51.9 | +35.0 | 48.0 | +0.990 | +0.976 | 0.2 | 0.078 | released |

## Gate-rejected

| design | reason |
|---|---|
| `t0.00_x0.00_y0.25` | self-collide index_len_frame<->middle_len_frame 26N |
| `t0.00_x0.25_y0.25` | self-collide index_len_frame<->middle_len_frame 26N |
| `t0.00_x0.50_y0.25` | self-collide index_len_frame<->middle_len_frame 26N |
| `t0.00_x0.75_y0.25` | self-collide index_len_frame<->middle_len_frame 26N |
| `t0.00_x1.00_y0.25` | self-collide index_len_frame<->middle_len_frame 26N |
| `t0.25_x0.00_y0.25` | self-collide index_len_frame<->middle_len_frame 28N |
| `t0.25_x0.25_y0.25` | self-collide index_len_frame<->middle_len_frame 28N |
| `t0.25_x0.50_y0.25` | self-collide index_len_frame<->middle_len_frame 28N |
| `t0.25_x0.75_y0.25` | self-collide index_len_frame<->middle_len_frame 28N |
| `t0.25_x1.00_y0.25` | self-collide index_len_frame<->middle_len_frame 28N |
| `t0.50_x0.00_y0.25` | self-collide index_len_frame<->middle_len_frame 29N |
| `t0.50_x0.25_y0.25` | self-collide index_len_frame<->middle_len_frame 29N |
| `t0.50_x0.50_y0.25` | self-collide index_len_frame<->middle_len_frame 29N |
| `t0.50_x0.75_y0.25` | self-collide index_len_frame<->middle_len_frame 29N |
| `t0.50_x1.00_y0.25` | self-collide index_len_frame<->middle_len_frame 29N |
| `t0.75_x0.00_y0.25` | self-collide index_len_frame<->middle_len_frame 29N |
| `t0.75_x0.25_y0.25` | self-collide index_len_frame<->middle_len_frame 29N |
| `t0.75_x0.50_y0.00` | self-collide thumb_tip<->middle_len_frame 1N |
| `t0.75_x0.50_y0.25` | self-collide index_len_frame<->middle_len_frame 29N |
| `t0.75_x0.75_y0.00` | self-collide thumb_tip<->index_len_frame 10N |
| `t0.75_x0.75_y0.25` | self-collide index_len_frame<->middle_len_frame 29N |
| `t0.75_x1.00_y0.00` | self-collide thumb_tip<->index_len_frame 22N |
| `t0.75_x1.00_y0.25` | self-collide index_len_frame<->middle_len_frame 29N |
| `t1.00_x0.00_y0.00` | self-collide thumb_tip<->middle_len_frame 4N |
| `t1.00_x0.00_y0.25` | self-collide index_len_frame<->middle_len_frame 29N |
| `t1.00_x0.25_y0.00` | self-collide thumb_tip<->middle_len_frame 13N |
| `t1.00_x0.25_y0.25` | self-collide index_len_frame<->middle_len_frame 29N |
| `t1.00_x0.50_y0.00` | self-collide thumb_tip<->middle_len_frame 28N |
| `t1.00_x0.50_y0.25` | self-collide index_len_frame<->middle_len_frame 29N |
| `t1.00_x0.75_y0.00` | self-collide thumb_pip_frame<->index_len_frame 16N |
| `t1.00_x0.75_y0.25` | self-collide index_len_frame<->middle_len_frame 29N |
| `t1.00_x1.00_y0.00` | self-collide thumb_pip_frame<->index_len_frame 30N |
| `t1.00_x1.00_y0.25` | self-collide index_len_frame<->middle_len_frame 29N |
