# manta-hand wire protocol

ASCII, one command per line (`\n`-terminated), sent over the USB-CDC virtual
serial port the firmware presents (`/dev/ttyACM0` on the CB1). Chosen over a
binary framing so bring-up can be done with a plain serial terminal
(`screen /dev/ttyACM0 115200` -- baud is irrelevant over USB-CDC but pick
something conventional) before any Python is involved.

Every command gets exactly one reply line: `OK[ <data>]` or `ERR <reason>`.

Joint indices are `0`-`7`, corresponding to M1-M8 (see `docs/pinout.md`).

| Command | Args | Effect |
|---|---|---|
| `EN J<i>` | - | Enable driver on joint i |
| `DIS J<i>` | - | Disable driver on joint i |
| `USTEP J<i> <n>` | 1/2/4/8/16/32/64/128/256 | Set microstep resolution |
| `MOVE J<i> <abs_steps> <max_vel_sps> <accel_sps2>` | signed step position, steps/s, steps/s^2 | Trapezoidal move to an absolute step position |
| `SETSCALE J<i> <steps_per_mm>` | float | RAM-only mm calibration for `MOVEMM` (lost on reset, re-run after every power cycle) |
| `MOVEMM J<i> <mm> <max_vel_sps> <accel_sps2>` | signed mm, steps/s, steps/s^2 | Same as `MOVE` but in mm via `SETSCALE`'s calibration; 0mm is wherever this joint's step-position 0 currently is, so home first. `ERR UNCALIBRATED` if `SETSCALE` hasn't been set yet |
| `SETRUN5160 J<i> <ihold_irun-hex>` | TMC5160 IHOLD_IRUN register value, hex | Sets this joint's steady-state TMC5160 current; a MOVE/JOG starting from a standstill automatically kicks current briefly first, then restores this value -- see `docs/bringup.md` |
| `WREG5160 J<i> <reg-hex> <val-hex>` | register address, value, both hex | Raw TMC5160 register write (e.g. `WREG5160 J0 6D <SGT<<16>` sets COOLCONF/SGT for StallGuard homing). Prefer `SETRUN5160` over a raw write to IHOLD_IRUN (reg 0x10) specifically -- see that command's row |
| `RREG5160 J<i> <reg-hex>` | register address, hex | Raw TMC5160 register read: `OK <8-hex-digit value>`. `RREG5160 J<i> 6F` (DRV_STATUS) is how live StallGuard tuning reads `sg_result`/the stall flag -- see `docs/bringup.md` |
| `JOG J<i> <vel_sps> <accel_sps2>` | signed steps/s, steps/s^2 | Continuous velocity move (ramps to `vel_sps` and holds; `JOG J<i> 0 <accel>` ramps to a stop) |
| `HOME J<i> <direction: 1 or -1> <vel_sps> <accel_sps2> <timeout_ms>` | as shown | Non-blocking StallGuard2 sensorless home -- poll `STAT`'s `homing_result` field for the outcome. Requires COOLCONF/SGT already tuned for this joint (`WREG5160`) and a populated TMC5160 (`ERR NODIAG` otherwise, or if the joint isn't enabled). See `docs/bringup.md`'s tuning method |
| `STOP J<i>` | - | Decelerate joint i to a stop at its current accel |
| `STOPALL` | - | STOP every joint |
| `ZERO J<i>` | - | Zero joint i's step counter at its current physical position. Also used to commit a successful `HOME`'s stall position as the new mm-reference (0mm) |
| `STAT J<i>` | - | `OK <position> <target> <moving 0/1> <enabled 0/1> <homing_result>` -- `homing_result`: 0=idle, 1=homing in progress, 2=stalled (position is the home reference), 3=timed out with no stall seen (don't trust position as a home reference from this alone -- see `docs/bringup.md`'s guaranteed-timeout note) |
| `STATALL` | - | `OK` followed by 8 lines, one per joint in order 0-7, each `<position> <target> <moving 0/1> <enabled 0/1>` (no `homing_result` field -- use per-joint `STAT` for that) |

Legacy/diagnostic only, not used by any populated axis on this build (J0-J5
are TMC5160, see `docs/pinout.md`): `CUR J<i> <run_mA> <hold_mA>` (TMC2209
IRUN/IHOLD), `WREG J<i> <reg-hex> <val-hex>` / `RREGA J<i> <addr> <reg-hex>`
(raw TMC2209 UART read/write). These route through `tmc2209_uart.c` and will
simply fail/time out against a TMC5160-populated socket -- don't use them on
this hardware.

Errors: `ERR BADCMD` (unrecognized command), `ERR BADARG` (wrong arg count /
unparsable), `ERR RANGE J<i>` (joint index out of 0-7), `ERR TMC J<i>` (TMC
read/write failed for that joint -- on this hardware this means a
`WREG`/`RREGA` legacy command against a socket with no TMC2209, not a real
fault), `ERR UNCALIBRATED` (`MOVEMM` before `SETSCALE`), `ERR NODIAG`
(`HOME` on an unpopulated/disabled joint).
