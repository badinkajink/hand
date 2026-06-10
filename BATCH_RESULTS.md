
## Batch run 2026-06-09 18:13 — co-adaptation wave 1
| run | warmstart | eval pairing | min-z (bar .05) | z@handoff | status |
|---|---|---|---|---|---|
| coadapt_B_toAtol20 | Badapt | model_270×NEWMARK | 0.0076 | 0.10652279108762741 | COLLAPSED |
| B_complete_fromBadapt | Badapt | model_500×NEWMARK | 0.0022 | 0.11348770558834076 | ok |
| branchB_w6_tol20 | frozenA | NEWMARK×model_270 | 0.0042 | 0.012292750179767609 | COLLAPSED |
| branchB_w4_tol15 | frozenA | NEWMARK×model_270 | 0.0073 | 0.10471154004335403 | ok |
| inject_velOnly | frozenA | model_500×NEWMARK | 0.0023 | 0.11437230557203293 | ok |
| inject_lastactOnly | frozenA | model_500×NEWMARK | 0.0073 | 0.1113230362534523 | ok |

## Batch run 2026-06-09 21:16 — wave 2 (proper co-adapt, STATIC inject)
| run | warmstart | eval pairing | min-z (bar .05) | z@handoff | status |
|---|---|---|---|---|---|

_(the wave-2 header above this line was an aborted launch — run 1 relaunched with lenient watchdog)_

## Batch run 2026-06-09 21:21 — wave 2 (proper co-adapt, STATIC inject)
| run | warmstart | eval pairing | min-z (bar .05) | z@handoff | status |
|---|---|---|---|---|---|
| coadapt_B10_Atol20_static | B10 | Atol20×NEW | 0.0113 | 0.10365995019674301 | ok |
| coadapt_Badapt_Atol20_static | Badapt | Atol20×NEW | 0.0118 | 0.10567375272512436 | ok |
