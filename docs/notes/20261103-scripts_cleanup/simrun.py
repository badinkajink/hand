"""The lab program's --sim session with the simulator's clock started at a fixed time, so that two runs of the same code
give the same streams: simrun.py TREE T0 -- <the program's arguments>.

The simulator starts its clock at time.time(); at 1.8e9 s a float64 resolves 2.4e-7 s, so the start's last bits decide
where the samples fall and two runs of one program differ. Here World.__init__ gets t0 = T0. The program is imported as
the module ur5_tip_calibration, which the simulator imports too, and run through its main()."""
import sys

tree, t0 = sys.argv[1], float(sys.argv[2])
argv = sys.argv[sys.argv.index("--") + 1:]
host = f"{tree}/src/sr2hand/driver/reskin5x/host"
sys.path.insert(0, host)
import ur5_tip_calibration as C  # noqa: E402
import ur5_tip_calibration_sim as SIM  # noqa: E402

assert C.__file__.startswith(tree) and SIM.__file__.startswith(tree), (C.__file__, SIM.__file__)
_init = SIM.World.__init__


def _fixed(self, plan, truth, t0_=None):
    _init(self, plan, truth, t0 if t0_ is None else t0_)


SIM.World.__init__ = _fixed
mods = sorted(m.__file__ for m in list(sys.modules.values()) if getattr(m, "__file__", None) and "sr2hand" in m.__file__)
print(f"[simrun] {C.__file__}; sr2hand modules from {sorted(set(p.split('/src/')[0] for p in mods))}", flush=True)
C.main(argv)
