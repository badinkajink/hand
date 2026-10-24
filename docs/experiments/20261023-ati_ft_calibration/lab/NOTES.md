# Notes from the lab computer for the ATI calibration (only the lab writes this file; requests come in `../QUEUE.md`)

## 2026-10-23 evening: the box on corvus, and gauge 5

### Connection (A1's "the owner's setup of the box")

- The Net box (NETBA FT35796, MAC 00:16:BD:00:2D:E4, firmware F/T BNI v2.2.59) sits at a static **192.168.123.50/24**,
  not ATI's default 192.168.1.1. It is on PoE from the Netgear GS305PP switch, whose uplink goes to the lab router's
  port 4.
- corvus reaches it through a live second address on its LAN port, which keeps its DHCP address and the internet:
  `nmcli device modify enp3s0 +ipv4.addresses 192.168.123.100/24`. Never set the connection's IPv4 method to Manual.
  The runbook is `../../20261022-ur5_tip_calibration/lab/README.md`, section *Network on corvus*.
- To find a box at an unknown address, run `/usr/bin/python3 scripts/ati_netft_discover.py --iface-ip 192.168.0.5`, ATI's
  discovery broadcast. Subnet sweeps missed this box; the workstation found it this way first.
- Browsers: use `http://192.168.123.50` in full. The box has no HTTPS, and a browser that upgrades to https fails. Only
  corvus can reach it.
- Both tools take the address: `scripts/ati_netft_logger.py --ip 192.168.123.50`, and
  `scripts/ati_calibration_session.py --ati 192.168.123.50`.

### As found (saved 2026-10-23 18:5x in `20261023-ati_box_as_found/`)

- Saved: every web page, `netftapi2.xml`, and `netftcalapi.xml?index=0..15` (all 16 calibration slots). The owner's
  photograph of the DIP switches is still to come; the box reports `dipswitch` 0xB6.
- Configuration #16 was active (`setcfgsel` 15), using calibration #1 (`cfgcalsel` 0). The status was 0x80020000.
- The calibration slots:
  - **#1**, FT35796 SI-125-3: the Axia80's, per the owner. Gains `calggn` 0x5B, 0x5B, 0x55, 0x5D, 0x57, 0x59 and
    hardware offsets `calgof` 0x7D46, 0x8088, 0x7B69, 0x7CBC, 0x7E11, 0x81E8.
  - **#16**, FT0000 "Identity" (the raw-gauge mode A1 asks for): identity matrix, counts per unit 1, and the **same
    gains and offsets as #1**.
  - **#2 to #15**, FT0000 "empty": zero matrix, counts per unit 1, and factory-default gains 0x0200 and offsets 0x8000
    on every gauge.

### Gauge 5 is saturated

- Status 0x80020000: the error bit plus bit 17, "Transducer Saturation or A/D operation error" (manual 9620-05-NET FT-23,
  Table 17.1). Gauge 5 (G4 in the box's numbering) reads +32767 in every sample. The other five read about 354,
  −5035, 1823, 1917 and 701.
- The owner saw the same on the workstation before the remount. None of these moved gauge 5 off the rail:
  - loosening the mount's screws, which moved gauges 2, 4 and 6 by 270 to 400 counts;
  - a power cycle with the cables reseated;
  - hand presses from 18:37 to 18:40 (519 reads), which moved gauge 1 over 1935 counts and gauge 3 over 1235.
- Under #1 the forces mean nothing for the Mini45 (FT11452). Gauge 5's stuck value supplies 257 of the 274 N Fz.
  Presses move Fz only through gauges 1 and 3: about 2/3 of a Z load, plus crosstalk.
- No page or CGI writes `calggn` or `calgof`; the interfaces are setting, config, comm, moncon and rundata `.cgi`.
  `setbias<n>` is a software bias applied after the A/D, so it cannot recover a clipped gauge.
- A1 cannot pass on #16 as it stands, because gauge 5 would stay at the rail with #1's tuning.

### Open, for the ATI conversation and the owner

1. **The owner's checks:** wiggle the transducer cable while corvus watches gauge 5 (an intermittent lead can be
   repaired). Measure gauge 5's bridge resistance against the others at the connector. Look in July's logs: an RDT
   status word of 0x80020000 there would mean gauge 5 was already stuck when the golem data were taken.
2. **A test, not yet run (the owner's go).** Select an "empty" calibration (#2) for the active configuration on the
   Configurations page, read `runsg` in `netftapi2.xml`, then switch back to #1. With the factory-default offsets and
   gains, gauge 5 may land in range: that would show its channel alive and the Axia80's tuning the cause. The
   catch: under an empty calibration RDT outputs zeros (zero matrix). Raw gauges would come only from polling
   `netftapi2.xml`, which is slow but may do for the static mass stations.
3. **A fallback if gauge 5 stays out** (the workstation, 18:4x). Fit gauges 1–4 and 6, each row on its own (six
   coefficients and an offset), against the weighed-mass poses; the plan already spans all six wrench directions. Use
   `F = (A' B(r))^+ (g' − g0')` with `B(r) = [I; [r]x]` and r from the registration: 5 equations for 3 unknowns. Check
   `cond(A' B(r))` for the pad's contact points right after the calibration run. This covers presses, not twists.
