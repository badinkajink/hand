# ATI Mini40 / Mini45 calibration handoff — October 2026

**Purpose:** Reference for an engineering agent (Claude) already operating the laboratory robot and developing a *safe, low-cost, empirical calibration* for two used ATI six-axis F/T transducers. This is an instrument-building/calibration task, not a research-novelty exercise.

**Status:** The laboratory transducers' true factory calibration matrices **have not been obtained**. Three **other** ATI transducers' factory calibration files have been recovered as structural priors. Their matrix values are not safely interchangeable without verifying electronics, channel order, signs, scaling and actual loads. **Never enable force-feedback control, contact motion, or safety thresholds based on an unvalidated borrowed matrix.**

## 1. Actual hardware inventory (from supplied RMA form)

| Item | Serial | Part / hardware | Stated current configuration | Desired calibration | Status |
|---|---|---|---|---|---|
| **Lab sensor A** | **FT11452** | Mini45, `9105-TW-Mini45` | `US-120-160` (per RMA section 2) | **`SI-145-5`** (per RMA section 3) | No serial-specific calibration recovered |
| **Lab sensor B** | **FT16552** | Mini40, `9105-TW-Mini40` | `SI-20-1` | **`SI-20-1`** | No serial-specific calibration recovered |
| **Lab interface** | **FT35796** | `9105-NETBA`, Net F/T Box | Ethernet box with analog front end | Not a mechanical transducer calibration | Actual active calibration/configuration not yet read |

**Source:** User-uploaded `RMA Request form 2024 (2) (1).xlsx`, sheet `Return Materials Req Form`, cells C20:H22 and E32. **Important discrepancy:** `US-120-160` is explicitly what the form lists for the Mini45 as received, whereas `SI-145-5` is the requested replacement. Treat the requested calibration as a goal, **not** as proof that such a calibration is currently installed. The form indicates an RMA *request*; it is not evidence that ATI subsequently performed the service. The precise sensor condition, wiring, service history and presence of any preexisting calibrations in NETBA are presently unknown.

**Sensor families / units:**

| Calibration type | ±Fx, ±Fy | ±Fz | ±Tx, ±Ty, ±Tz | Provenance |
|---|---:|---:|---:|---|
| Mini45 `US-120-160` | 120 lbf (~534 N) | 240 lbf (~1068 N) | 160 lbf-in (~18.08 N·m) | [ATI Mini45 model page](https://www.ati-ia.com/products/ft/ft_models.aspx?id=Mini45) |
| Mini45 `SI-145-5` | 145 N | 290 N | 5 N·m | ATI Mini45 model page; RMA desired range |
| Mini45 `US-30-40` | 30 lbf (~133.45 N) | 60 lbf (~266.89 N) | 40 lbf-in (~4.519 N·m) | ATI Mini45 model page; FT31565 certificate |
| Mini45 `SI-290-10` | 290 N | 580 N | 10 N·m | ATI Mini45 model page; FT32816 certificate |
| Mini40 `SI-20-1` | 20 N | 60 N | 1 N·m | [ATI Mini40 model page](https://www.ati-ia.com/products/ft/ft_models.aspx?id=Mini40); RMA and FT21031 |

These numbers are the **calibrated measurement ranges**, not permissible combined-contact load limits or a substitute for mechanical overload/combined-loading specifications. For used sensors with unknown condition, begin well below rated limits and use independent safety stops.

### What is TWE / TW and what is NETBA?

- **TWE = Transducer Without Electronics.** ATI supplies the passive strain-gage transducer with calibration information; it requires external excitation, amplification/conditioning and digitization. Source: [ATI F/T system interfaces](https://www.ati-ia.com/products/ft/ft_SystemInterfaces.aspx) and [TWE manual](https://www.ati-ia.com/app_content/documents/9610-05-1018%20TWE.pdf).
- **NETBA** is the Net F/T interface **with analog front-end board**, intended for Mini/Nano transducers or other sensors without internal electronics. It connects to the transducer and provides digitized data / processed F/T over Ethernet. Source: [ATI manuals and hardware list](https://www.ati-ia.com/products/ft/ft_literature.aspx).
- Anticipated acquisition path: `FT11452 OR FT16552 -> appropriate cable -> FT35796 NETBA -> Ethernet -> robot/logger`. **Verify connector/cable pinout, compatible excitation and one-sensor-at-a-time switching procedure with ATI docs before use.** The RMA alone does not establish a working connection, correct channel mapping or actual NETBA firmware/configuration.

## 2. Recovered factory calibration files (for OTHER sensors)

All three archives were found using [ATI's serial-number calibration file request page](https://ati.novanta.com/products/force-torque-sensors/calibration-files/) (files are available there for instruments shipped/recalibrated after 2017-11-28). The ZIPs were uploaded in this conversation. **The downloadable originals are user-provided local files; their ATI form results are not public permanent URLs.** Numerical tables below were extracted from the original native `.xlsx`, `.dat` and `.xml` members, *not* transcribed from screenshots.

| Reference serial | Model / manufacturer calibration | Electronics family in certificate | Certificate calibration date | Original archive | Native matrix output units | Main relevance |
|---|---|---|---|---|---|---|
| **FT31565** | Mini45 `US-30-40` | **TWE** | 2020-07-27 | `FT31565 (1).zip` | lbf, lbf-in from unamplified normalized bridge signals | Best same-family / same-TWE prior for **FT11452** |
| **FT21031** | Mini40 `SI-20-1` | **MUX** | 2020-01-31 | `FT21031.zip` | N, N·m from this MUX calibration's signal representation | Best same-model / same-range **structural** prior for **FT16552** |
| **FT32816** | Mini45 `SI-290-10` | **Net F/T** | 2020-12-11 | `FT32816.zip` | N, N·m in Net calibration-file's signal domain | Independent Mini45 structural comparison; NET file schema example |

The three certificates report ATI factory acceptance (`Passed`), with hardware temperature compensation, at ~22.2 ± 1.1 °C. Their stated maximum 95% confidence measurement uncertainty (% full-scale) is:

| Serial | Fx | Fy | Fz | Tx | Ty | Tz |
|---|---:|---:|---:|---:|---:|---:|
| FT31565 | 1.25% | 1.00% | 0.75% | 1.25% | 1.50% | 1.25% |
| FT21031 | 1.25% | 1.00% | 0.75% | 1.25% | 1.25% | 2.00% |
| FT32816 | 1.00% | 0.75% | 0.75% | 1.00% | 1.25% | 1.25% |

**Do not transfer these factory uncertainty values to our sensors.** These apply only to the serials and complete calibration setups on the certificates.

### 2.1 Mini45 FT31565 — 2020 TWE raw bridge-output calibration

Original archive: `FT31565 (1).zip` -> `FT31565 TWE.xlsx`, sheet **`Calibration Info (Raw)`**, cells B10:G15. The certificate is `FT31565 Calibration Certificate and Accuracy Report.pdf` (certificate number `FT31565-20200727`).

- Sensor: Mini45 `US-30-40`; interface family: **TWE**, no user-defined amplifier specification built into the valid raw sheet.
- Force units: **lbf**; torque units: **lbf-in**.
- Matrix maps **six low-level bridge ratios (V / V_excitation)** to wrench. *They are not NETBA streaming counts or amplified sensor volts.* Apply it only to exactly the required signal domain.
- Bias must be treated separately: loaded-minus-unloaded input. The worksheet also lists factory hardware biases in V/V_excitation; do **not** reuse FT31565's bias vector for FT11452.

**Original 6×6 matrix** `C_31565_raw_US`: output rows `Fx,Fy,Fz,Tx,Ty,Tz`; input columns `G0..G5`, wrench = matrix @ (normalized_bridge_loading - normalized_bridge_unloaded):

```text
          G0             G1             G2             G3             G4             G5
Fx     -196.121139       2.075379     421.630226   -5241.153554    -245.072084    5282.728968
Fy     -691.510384    5732.553881     109.427023   -3028.105257     193.141180   -3054.432428
Fz     7361.997734     367.541865    7310.572229     346.626291    7475.106333     295.590677
Tx     -230.120609    1606.876411   -4632.577406   -1073.676070    4689.654846    -671.367569
Ty     5419.830460     268.246506   -2945.507406    1337.543751   -2562.554419   -1586.116360
Tz      411.213577   -2848.744084      48.887808   -3035.820711     225.091740   -3084.323116
```

**Equivalent SI output conversion**, `C_31565_raw_SI`, computed only by multiplying force rows by `4.4482216152605 N/lbf` and torque rows by `0.1129848290276167 N·m/(lbf-in)`; input is **still V/V_excitation**, not NETBA ADC counts:

```text
          G0             G1             G2             G3             G4             G5
Fx     -872.390290       9.231746    1875.504685  -23313.812527   -1090.134940   23498.749183
Fy    -3075.991435   25499.670084     486.755650  -13469.683256     859.134770  -13586.792348
Fz    32747.797452    1634.907667   32519.045410    1541.870560   33250.929566    1314.852839
Tx      -26.000138     181.552657    -523.410966    -121.309107     529.859851     -75.854350
Ty      612.358618      30.307786    -332.797651     151.122152    -289.529773    -179.207086
Tz       46.460896    -321.864863       5.523581    -343.001684      25.431952    -348.481720
```

Selected additional calibration-sheet metadata:

- `Output Range (V/Vin)` gage G0..G5: `0.0163223769, 0.0167771566, 0.0160112723, 0.0159430768, 0.0157950307, 0.0156104292`
- `Hardware Bias (V/Vin)` G0..G5: `3.7137627560e-07, -3.5334121891e-08, -7.9521115602e-07, 1.5026859977e-06, -1.8608572403e-06, -8.0710693614e-07` **(FT31565 only)**.
- `Hardware Temperature Compensation = Yes`, `Software Temperature Compensation = No`, `Gain Multiplier = 1`.
- **Critical:** `Signal Conditioning` tab has unset excitation / ADC / amplifier settings; `Calibration Info (Amplified)` and `Sample Calculations` display `#DIV/0!` formulas. **Only the original raw matrix above is valid as provided.** Do not use those empty/failed amplified formulas.

### 2.2 Mini40 FT21031 — 2020 MUX calibration

Original archive: `FT21031.zip` -> `FT21031.dat`, paired with `FT21031 Calibration Certificate and Accuracy Report.pdf`, certificate `FT21031-20200131`.

- Sensor: **Mini40 `SI-20-1`**, same physical model and requested range as lab sensor FT16552, but different electronics: **MUX**.
- Native DAT uses first six nonzero calibration columns; two additional zero columns are padded and omitted below.
- Matrix rows are in **N and N·m**. The DAT is a MUX device/driver matrix; its input units/ADC scaling are **not equivalent** to a NETBA's raw gage counts merely because both expose six channels.
- Other DAT metadata: `Counts per N = 50`, `Counts per N-m = 2000`, `Gain Multiplier = 1.00`, `HW Temp Comp = Yes`, `Default Resolution = 16`, ranges `20,20,60,1,1,1`.
- DAT includes unit-specific `UnbiasedVector`, bias metadata: **do not copy** to FT16552.

**Native first six columns from `.dat`** `C_21031_MUX_SI`:

```text
          G0             G1             G2             G3             G4             G5
Fx        0.012856       0.003043       0.000085       0.800831       0.005727      -0.798956
Fy       -0.015610      -0.917848       0.030969       0.463292      -0.019066       0.459829
Fz        1.348824      -0.055740       1.368480       0.055565       1.361107      -0.047595
Tx       -0.009511      -0.200230       0.785713       0.132712      -0.784223       0.129090
Ty       -0.898313       0.037993       0.447534      -0.155043       0.455086       0.156617
Tz        0.000989      -0.442191      -0.003124      -0.443989       0.002116      -0.442626
```

### 2.3 Mini45 FT32816 — 2020 Net F/T calibration

Original archive: `FT32816.zip` -> `FT32816 Net.xml`, paired with `FT32816 Calibration Certificate and Accuracy Report.pdf`, certificate `FT32816-20201211`.

- Sensor Mini45 `SI-290-10` (higher range than FT31565 / desired FT11452).
- Firmware/interface family is **Net F/T**. XML has a 6×6 set of `MatrixFX`, `MatrixFy`, `MatrixFz`, `MatrixTx`, `MatrixTy`, `MatrixTz` fields and electronic conditioning fields.
- Outputs specified as **N and N·m**, but **do not interpret the large coefficients as directly mapping arbitrary NETBA raw counts to SI units** without reconstructing the original scaling and processing pipeline.

**Original native XML coefficients** `C_32816_NET_SI` (six numbers per XML Matrix* field):

```text
          G0             G1             G2             G3             G4             G5
Fx      244.899880     122.665891     504.513087  -14623.326502    -454.065068   14271.427992
Fy     -428.357757   16563.783909     762.394477   -8472.677462     148.145679   -8204.225926
Fz    20644.543735     435.232780   20899.360903     575.880343   19996.727260     611.411431
Tx       -5.875063     115.768759    -324.671182     -69.795067     329.989907     -46.923261
Ty      379.943333       5.531701    -201.100684      99.854951    -186.969075    -106.578726
Tz        2.191109    -209.681036      11.557499    -211.700516       8.172437    -209.489931
```

Other XML fields (**unit-specific, not transferable**):

- `GaugeGains = 299 307 295 301 293 301`
- `GaugeOffsets = 32211 34574 31368 29962 32168 32526`
- `CountsPerForce = 1000000`, `CountsPerTorque = 1000000` (Net device representation; not analog gain).
- `16BitScaleFactors = 17701 17701 17701 306 306 306`; `Resolutions = 24 24 24 24 24 24`.
- `MaxRatings = 290 290 580 10 10 10`, `CalibrationIndex = 0`.

### 2.4 Extra independent public matrix for FT21031 (2017)

A published patent reproduces a **previous ATI factory Mini40 FT21031 certificate**, `FT21031-20171219`, in Table 4. This is the **same reference transducer as above**, calibrated earlier, but in a different analog-voltage and imperial-unit representation.

- **Primary public provenance:** [U.S. Patent 12,318,117, Table 4](https://patents.justia.com/patent/12318117); [Google Patents, WO2020206297A1, section [0161] and Table 4](https://patents.google.com/patent/WO2020206297A1/en).
- The 2017 matrix is in **lbf / lbf-in from analog voltages**; it is not numerically interchangeable with the 2020 `.dat` SI MUX matrix or with NETBA raw counts.

**Publicly reproduced 2017 matrix** `C_21031_2017_analog_US`:

```text
          G0             G1             G2             G3             G4             G5
Fx        0.039690       0.008630      -0.001570       2.818510       0.034570      -2.816630
Fy       -0.051640      -3.261390       0.100990       1.627740      -0.006100       1.627470
Fz        4.797150      -0.192680       4.788570       0.202820       4.769460      -0.176570
Tx       -0.035230      -0.679520       2.701420       0.445840      -2.727340       0.438220
Ty       -3.107600       0.138890       1.580200      -0.531370       1.575220       0.527920
Tz       -0.002530      -1.519860      -0.029330      -1.534820       0.011230      -1.540860
```

2017 vs 2020 same-device *row-direction* cosine similarity (unit/scale invariance) = `0.999992, 0.999870, 0.999972, 0.999972, 0.999961, 0.999958` for Fx,Fy,Fz,Tx,Ty,Tz. This is unusually stable **within that same transducer**, but does not prove absolute gain portability across units or readout electronics.

## 3. Quantitative comparison and interpretation

Normalize each row separately: `R_i = C_i / ||C_i||_2`. This deliberately **discards all row gain information** and is meaningful only if the gage channel ordering and mapping are consistent. For cross-electronics files this comparison is **structural evidence, not calibration accuracy**.

| Output | Cosine: Mini45 FT31565 TWE vs Mini45 FT32816 Net | Cosine: Mini40 FT21031 MUX vs Mini45 FT31565 TWE |
|---|---:|---:|
| Fx | 0.998556 | -0.997881 |
| Fy | 0.996363 | -0.992663 |
| Fz | 0.999401 | 0.997616 |
| Tx | 0.999688 | -0.996419 |
| Ty | 0.999342 | -0.990864 |
| Tz | 0.996316 | 0.995513 |

- FT31565 and FT32816 have same-sign normalized rows and **0.9963–0.9997** row-direction cosines despite differing ranges and interfaces. FT32816/FT31565 *SI-matrix* row norm ratios = `0.616, 0.635, 0.624, 0.621, 0.621, 0.620` (Fx through Tz). **This ratio must not be used as an estimated physical sensor gain ratio**: differing electronics / conversion domains confound it.
- FT21031 vs FT31565 row cosines have sign flips in Fx,Fy,Tx,Ty and matching signs in Fz,Tz. This could resemble an axis-convention rotation, but **do not assume this means the actual devices were physically rotated by 180 degrees**; channel/strain-gage ordering and sign choices must be checked experimentally.
- **Very important:** row cosines near 1 do *not* imply <1% wrench accuracy, good cross-axis decoupling, or safe loading. A small discrepancy in a cancellation-heavy calibration row can still produce substantial cross-talk. Per-gage analog gain differences (and sensor damage) can also change apparent row geometry.
- Normalizing a matrix gives a useful *shape prior* that can reduce the number of fitted parameters, not a valid calibration for a different transducer.

**Best Mini45 shape prior** for lab FT11452: FT31565 normalized row matrix `R_31565` (dimensionless):

```text
          G0             G1             G2             G3             G4             G5
Fx       -0.026289       0.000278       0.056518      -0.702557      -0.032851       0.708130
Fy       -0.095998       0.795815       0.015191      -0.420373       0.026813      -0.424028
Fz        0.575115       0.028712       0.571098       0.027078       0.583951       0.023091
Tx       -0.033322       0.232680      -0.670809      -0.155471       0.679074      -0.097216
Ty        0.774309       0.038323      -0.420813       0.191089      -0.366102      -0.226602
Tz        0.079040      -0.547565       0.009397      -0.583523       0.043265      -0.592846
```

**Best Mini40 shape prior** for lab FT16552: FT21031 normalized row matrix `R_21031` (dimensionless):

```text
          G0             G1             G2             G3             G4             G5
Fx        0.011364       0.002690       0.000075       0.707878       0.005062      -0.706220
Fy       -0.013851      -0.814428       0.027480       0.411090      -0.016918       0.408017
Fz        0.572382      -0.023654       0.580723       0.023579       0.577595      -0.020197
Tx       -0.008320      -0.175155       0.687318       0.116092      -0.686015       0.112924
Ty       -0.798898       0.033788       0.398006      -0.137885       0.404722       0.139284
Tz        0.001289      -0.576372      -0.004072      -0.578715       0.002758      -0.576939
```

## 4. How to acquire meaningful data from NETBA FT35796

**Do not assume NETBA's nominal F/T packet already reports calibrated forces** when the attached transducer's exact calibration file is missing. Before overwriting anything, save screenshots/exports of *all* current calibration slots, identity labels, gage gains/offsets, device firmware, configured unit scales, sensor status and raw sample values. A previously installed slot might coincidentally be useful—but is trustworthy only if serial/model/range match the connected transducer.

ATI's **Net F/T Quick Start manual**, section 3.4.2, gives a factory raw-gage mode:

1. Via the Net F/T web configuration, select **calibration `#16-FT00000`**, **only if it is the unmodified factory-default entry**.
2. Select **force units `lbf` and torque units `lbf-in`**.
3. Apply; the documentation says `Counts per Force = 1` and `Counts per Torque = 1` in this mode.
4. The six output channels then represent **raw strain-gage values**, *not* force/torque, even if plotting software labels them Fx/Fy/Fz/Tx/Ty/Tz.
5. Inspect device health/saturation; do not continue applying load after any saturation or overload indication.
6. Log all six gage values **and status**, timestamps, unit/config selection, sensor serial, wiring arrangement, ambient temperature (if available), bias/zero procedure and physical loading fixture; put the NetBox back into the appropriate verified configuration when done.

Primary source: [ATI Quick Start, document 9610-05-1022, §3.4.2](https://www.ati-ia.com/app_content/documents/9610-05-1022%20Quick%20Start.pdf); detailed [ATI Net F/T manual](https://ati.novanta.com/wp-content/uploads/sites/2/2025/05/9620-05-Net-FT.pdf).

**Crucial signal-domain limitation:** In `FT00000` mode, the six network fields are uncalibrated *digital gage values*, **not guaranteed direct measurements of FT31565's raw V/V_excitation bridge ratios**. Actual per-gage gain/offset, ADC scale, calibration slot and temperature handling matter. Do *not* simply paste the FT31565 raw 6×6 numerical matrix into NETBA or multiply raw counts by it. Channel indexing/sign/order should be validated against hardware behavior.

For Net F/T's normal calibrated stream, force = digital force counts / configured `CountsPerForce`, and torque = torque counts / `CountsPerTorque`; these are **different from raw uncalibrated strain-gage counts** in the FT00000 mode. Source: [netft.dev sensor measurements](https://netft.dev/docs/tutorials/fundamentals/sensor-measurements/) (third-party tooling, not ATI-authoritative).

## 5. Calibration model options and practical design

### 5.1 No borrowed matrix required: full reference regression

For calibration experiment sample `k`, log:

- `g_k` = six **raw NETBA gage readings** (counts), and unloaded/bias sample `g0_k` gathered under controlled mechanical conditions;
- `w_ref,k` = independently known 6D wrench (SI units) **expressed at the ATI transducer origin in its coordinate frame**;
- fixture transform / location / reference frame, temperatures, contact approach direction, robot pose and acquisition metadata.

Model:

```text
w_hat_k = C_net @ (g_k - g0_k),
C_net shape (6,6), outputs [N,N,N,N*m,N*m,N*m], inputs NETBA gage counts.
```

Fit `C_net` by regularized linear least squares when data span six linearly independent wrench cases. Validate *each wrench component* on held-out independent loading configurations and after remounting; evaluate cross-axis error, repeatability, load/unload hysteresis, thermal drift, bias stability, saturation and residual vs load magnitude. Check excitation matrix rank/singular values, especially after physically realistic force/torque scaling; otherwise an apparently accurate least-squares matrix can be badly non-identifiable.

### 5.2 Low-parameter bootstrap using recovered row shapes

If the raw NETBA gage ordering and electronics behavior agree with the source matrix's expected pattern, define dimensionless `R` as `R_31565` for Mini45 or `R_21031` for Mini40. Try **only as a hypothesis**:

```text
q = R @ (g - g0)          # arbitrary units; 6 components
w_hat = diag(d) @ q       # 6 unknown output scales d_i
```

Fit each `d_i` from known reference loads **without ignoring torque produced by an off-axis force**. For each wrench output row `i`, simple weighted scalar LS:

```text
d_i = sum_k(weight_k * q_i[k] * w_ref_i[k]) / sum_k(weight_k * q_i[k]^2)
```

This *six-scalar fit* is a possible **first bootstrap**, not a guarantee: mismatch in per-gage conditioning changes effective matrix *columns*, so six output scalars may not be adequate. Next test `w_hat = D_out @ R @ D_gage @ (g-g0)` (shared six gage factors + six output scales; gauge ambiguity must be fixed), or directly fit a regularized correction/full 6×6 matrix if data support it. Use held-out error to determine which complexity is warranted—**do not force the low-parameter model if its crosstalk is bad**.

`D_gage` and `D_out` have multiplicative gauge ambiguity; constrain one factor (e.g. geometric mean of `D_gage` = 1) and use sign/near-unity priors to avoid ill-conditioning. If possible, check sensor frame against CAD/ATI axis drawing and collect paired loading and unloading data.

### 5.3 Wrench reference sources

The lab has (per project conversation): **OnRobot HEX-E external F/T sensor** and **UR16e internal wrist F/T sensor**. The user reports the UR16e is empirically **better** than its published accuracy number, while HEX-E is empirically **worse** than its published number. Those lab observations are qualitative as supplied; **no noise/bias/error time series have been provided** here.

- Use known masses + gravity vectors and measured lever arms as an *independent absolute anchor* whenever geometry permits.
- If comparing series-connected F/T references, transform both to the same wrench origin/frame; measure their relative pose and structural deflection rather than subtracting unregistered readings.
- Reference loads: a point force `F` applied at vector `r` from the sensor origin generates moment `tau = r × F`; **six mechanically driven DOFs or six pure-moment load cases are not inherently required** to identify a full six-row map. But confirm the resulting wrench design matrix's rank and condition number.
- Use both HEX-E and UR16e as cross-checks and characterize independent bias/drift/systematic errors on known applied loads. **Fusing two biased references does not magically create absolute ground truth**. Covariance-weighted fusion helps if residual errors are independently characterized; shared biases and hysteresis will not average away.
- Before using any robot-mounted reference: account for end-effector tool weight/COM, robot-side gravity compensation convention, tare state, sign, wrench frame and imposed moment arms.

## 6. Engineering workflow for Claude (proposal to evaluate, not blindly execute)

1. **Inventory without contact:** Confirm physical serials on sensors, cables, NETBA firmware and existing calibration slots; back up current NETBA settings/any resident calibration files. Verify strain channels are healthy/unclipped at rest. Confirm actual coordinate frame, fasteners and maximum *combined* load limitations from ATI manual. Do not overwrite slots prematurely.
2. **Get raw data safely:** Use documented `#16-FT00000` mode if available and unmodified; log continuous six-channel gage counts and status. Baseline sensor unloaded, then install rigidly and quantify fixture/gravity bias. Recover signs and channel order from small manually applied known loads only after mechanical and electrical checks.
3. **Test six-output structural prior:** For each sensor compare observed responses to the proposed normalized `R`; document whether every expected axis changes with reasonable sign/contrast. If it does, fit six output scalars from an independently labeled calibration dataset. If not, inspect channel permutation, gains, wiring/compensation, physical damage or wrong reference matrix.
4. **Build a known-wrench fixture:** For zero and loaded tests, use accurately weighed masses and known, repeatable contact lever arms at multiple orientations. Robot must not push or drag based on guessed force estimates. For torque `tau = r × F`, correct for applied force line-of-action and other fixture masses; **avoid unknown contact friction** when using simple weights.
5. **Fit and compare:** Full 6×6 (regularized) baseline vs row-shaped six-gain prior vs 12-ish parameter variant. Report held-out error in SI per axis, normalized full-scale error, maximum errors, crosstalk, repeatability, conditioning, load/unload effects and noise vs the actual expected fingertip-force regime.
6. **Validate with external sensors:** Stack tested sensor in series with HEX-E / UR16e only with a rigid, characterized transform, and use known-mass comparisons to detect systematic disagreement. Do not force agreement between two mutually inconsistent references.
7. **Only then instrument Ender3**: Mount trusted ATI on moving head with stiff short probe, measure probe-tip-to-ATI transform, monitor saturation and **use independent travel/force limits**, perform quasi-static indentation and shear with logged robot pose/tactile observation/wrench. Assess whether an extra rotary DOF is justified by unreachable surface regions or needed contact torsion—not by abstract 6D dimensionality alone.

**Safety and validity constraints:** An ATI strain gage can saturate or be mechanically overloaded before the guessed reconstructed force shows it. Unknown offsets, combined loads and moment arms are especially dangerous on Mini40's ±1 N·m calibrated torque range. Keep robot collision/stop constraints independent of the speculative calibration; avoid using absolute force control or feedback with an unverified calibration.

## 7. What Claude should explicitly determine before implementing

- Which NETBA firmware and physical wiring/channel mapping are present? Is `#16-FT00000` intact? What range/scaling do its output integers use?
- Does NETBA have **any existing** serial-specific calibration (perhaps one of the original eBay combinations)? Can it be exported without modifying it?
- Are the actual Mini45/Mini40 strain bridges undamaged and healthy (bias/saturation/drift under no load)? Are the mounting adapters mechanically sound?
- Which target calibration is desired in practice: recapture original Mini45 `US-120-160` sensing regime, use requested finer `SI-145-5`, or characterize a limited force regime for tactile fingers? **Do not confuse changing units with changing calibrated range.**
- Can the computer and robot logger collect **raw unmodified gage samples, status and reference sensor telemetry synchronized** at sufficiently low-noise stable conditions?
- Do the source row priors fit real per-axis known-wrench data? Is any channel permutation/sign correction necessary? Are residuals systematic (nonlinearity, cross-talk, hysteresis, temperature)?
- What actual calibration error is acceptable for the planned tactile-finger studies (e.g. low-load sensitivity, tangential force accuracy, contact torque)? Pick validation cases accordingly.

## 8. Reference and provenance links

**Original user-provided calibration archives:** All exact numeric matrix data (except 2017 published table) come from these local attachments. They are *not* manufacturer's calibration files for lab-owned FT11452/FT16552:

| Local archive | SHA-256 checksum |
|---|---|
| `FT21031.zip` | `de24803e9ff8562b9f6c545d47f8a6fa34400dbdb0ef5dc143d1b54994892805` |
| `FT31565 (1).zip` | `77008e539619e692ed19f357bdef61c80d7235c933a31df05778611b548b6edf` |
| `FT32816.zip` | `ea4ae12a4589772b2421bfd076856f7976c3a4ea0ed70bcdd4d7114f3b9c27fd` |
| `RMA Request form 2024 (2) (1).xlsx` | `1c257979c13a600212e7eaf01237e9f02062bdb343689c58310a6aeffdc1f06b` |

**Official ATI / public source URLs:**

1. ATI calibration retrieval by serial number: https://ati.novanta.com/products/force-torque-sensors/calibration-files/
2. ATI Mini40 model, ranges, limits and drawings: https://www.ati-ia.com/products/ft/ft_models.aspx?id=Mini40
3. ATI Mini45 model, ranges, limits and drawings: https://www.ati-ia.com/products/ft/ft_models.aspx?id=Mini45
4. ATI F/T system interfaces, TWE explanation: https://www.ati-ia.com/products/ft/ft_SystemInterfaces.aspx
5. ATI Transducer Without Electronics / TWE manual: https://www.ati-ia.com/app_content/documents/9610-05-1018%20TWE.pdf
6. ATI Net F/T Quick Start (factory raw-gage slot and caution): https://www.ati-ia.com/app_content/documents/9610-05-1022%20Quick%20Start.pdf
7. ATI Net F/T manual: https://ati.novanta.com/wp-content/uploads/sites/2/2025/05/9620-05-Net-FT.pdf
8. ATI NETBA description and hardware manuals: https://www.ati-ia.com/products/ft/ft_literature.aspx
9. Public 2017 FT21031 matrix in patent Table 4: https://patents.justia.com/patent/12318117
10. Same matrix in international patent: https://patents.google.com/patent/WO2020206297A1/en
11. Original user-shared NET interface driver repository: https://github.com/badinkajink/ati_ft_sensor
12. Third-party Net F/T count/unit conventions: https://netft.dev/docs/tutorials/fundamentals/sensor-measurements/

### Scope / evidence levels

- **Confirmed from user attachments:** Actual lab sensor/interface serials and RMA requested ranges; all three downloaded factory matrices and certificate dates/uncertainty for their respective serials; TWE spreadsheet missing amplified-stage inputs.
- **Confirmed from ATI primary documentation:** TWE meaning, NETBA analog-front-end role, nominal model/range specs, existence of raw NET F/T mode (`#16-FT00000`) subject to intact default configuration.
- **Computed directly from originals:** SI conversion of FT31565 matrix; row-normalized patterns; row cosines and ratio comparisons; 2017-vs-2020 same-serial comparison.
- **Hypotheses / proposed tests:** Borrowed row shapes will work with our NETBA channels; fitting only six output scales will be accurate enough; current lab devices may have useful remaining factory calibrations; UR and HEX can be fused to improve absolute ground truth. These are **not yet established**.
