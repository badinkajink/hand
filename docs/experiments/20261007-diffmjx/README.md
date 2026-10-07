# DiffMJX contact-gradient runbook

The result page is `20261007-contact_gradients_real_v1.html`. Its builder is
`scripts/real_v1_diffmjx_page.py`. The JSON files in this directory are the
measurements used by the builder. All commands below run on the workstation;
none connects to the hand controller.

## Source revisions

| repository | commit |
|---|---|
| `martius-lab/diffmjx` | `b5453eac0b19274754ef2ae86840ce45ad62480a` |
| `martius-lab/mujoco`, branch `diffmjx` | `ced2cffdc4a8c62056dce596b6df53e5fdc59ac7` |
| `martius-lab/mjx_diffrax` | `a7756386337cdebaba409936a37b34662303b66c` |
| `a-paulus/softjax` | `82fa3c908bc5f7378ee044f0171cbe8560d52f61` |

The isolated interpreter is Python 3.13.9 with MuJoCo 3.3.1, `mujoco-mjx`
3.3.1 from the DiffMJX fork, JAX 0.11.2, Diffrax 0.7.2 and SoftJAX 0.1.1.
The project's existing environment has MuJoCo 3.6 and native MJX. Both tests
used the CPU JAX backend and 64-bit arithmetic for gradient checks.

## Isolated environment

Clone the four repositories at the commits above into
`logs/20261007-diffmjx/{upstream,mujoco,mjx_diffrax,softjax}`. The `mujoco`
clone needs its `mjx/` subtree. Then install:

```bash
uv venv --python 3.13 logs/20261007-diffmjx/venv
uv pip install --python logs/20261007-diffmjx/venv/bin/python \
  'jax[cpu]==0.11.2' 'mujoco==3.3.1' \
  -e logs/20261007-diffmjx/mujoco/mjx \
  -e logs/20261007-diffmjx/mjx_diffrax \
  -e logs/20261007-diffmjx/softjax
```

The release's umbrella `pyproject.toml` requests `jax[cuda12]>=0.8`. This
study installed CPU JAX in the isolated environment to keep GPU memory free
for the user's work. The test calls the fork's actual `mujoco.mjx` code.

## Measurements

Check `~/.claude/bin/resguard.sh status` before each long launch. The commands
below show their inner Python invocation; wrap long runs with
`~/.claude/bin/resguard.sh run --mem 8G --cpu 400 --` and use the machine's
current resource reading to choose a smaller cap if needed. The packed 20 ms
contact-from-distance gradient peaked at 10.3 GiB resident memory and used an
11 GB guard. Run MJX jobs sequentially.

```bash
uv run --no-sync python scripts/real_v1_diffmjx_pad_cpu.py
uv run --no-sync python scripts/real_v1_diffmjx_local_search.py

JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=1 uv run --no-sync python \
  scripts/real_v1_diffmjx_gate.py --steps 1 --no-post --tip tpu6 \
  --contact-model padsT \
  --out docs/experiments/20261007-diffmjx/20261007-native_padsT_cylinder_1.json

JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=1 uv run --no-sync python \
  scripts/real_v1_diffmjx_gate.py --steps 20 --no-post --tool-capsule \
  --gradient --out docs/experiments/20261007-diffmjx/20261007-native_capsule_gradient_20_x64.json

JAX_PLATFORM_NAME=cpu logs/20261007-diffmjx/venv/bin/python \
  scripts/real_v1_diffmjx_official_gate.py --steps 20 --tool-capsule \
  --out docs/experiments/20261007-diffmjx/20261007-official_capsule_plain_20.json

JAX_PLATFORM_NAME=cpu logs/20261007-diffmjx/venv/bin/python \
  scripts/real_v1_diffmjx_official_gate.py --steps 20 --tool-capsule --cfd --scan-loop \
  --out docs/experiments/20261007-diffmjx/20261007-official_capsule_cfd_20.json

JAX_PLATFORM_NAME=cpu logs/20261007-diffmjx/venv/bin/python \
  scripts/real_v1_diffmjx_official_gate.py --steps 1 --tip tpu6 \
  --contact-model padsT --skip-gradient \
  --out docs/experiments/20261007-diffmjx/20261007-official_padsT1_cylinder_1.json

JAX_PLATFORM_NAME=cpu logs/20261007-diffmjx/venv/bin/python \
  scripts/real_v1_diffmjx_pad_jvp.py --steps 1 \
  --out docs/experiments/20261007-diffmjx/20261007-official_padsT1_cylinder_jvp_1.json

JAX_PLATFORM_NAME=cpu logs/20261007-diffmjx/venv/bin/python \
  scripts/real_v1_diffmjx_pad_jvp.py --steps 1 --fork-fd-only \
  --out docs/experiments/20261007-diffmjx/20261007-official_padsT1_cylinder_fd_1.json

JAX_PLATFORM_NAME=cpu logs/20261007-diffmjx/venv/bin/python \
  scripts/real_v1_diffmjx_pad_jvp.py --steps 20 --all-directions \
  --out docs/experiments/20261007-diffmjx/20261007-official_padsT1_cylinder_jacobian_20.json

JAX_PLATFORM_NAME=cpu logs/20261007-diffmjx/venv/bin/python \
  scripts/real_v1_diffmjx_pad_jvp.py --steps 20 --fork-fd-only \
  --fd-eps-list 0.01,0.003,0.001,0.0003,0.0001 \
  --out docs/experiments/20261007-diffmjx/20261007-official_padsT1_cylinder_fd_sweep_20.json

JAX_PLATFORM_NAME=cpu logs/20261007-diffmjx/venv/bin/python \
  scripts/real_v1_diffmjx_pad_jvp.py --steps 20 --cfd --scan-loop --all-directions \
  --out docs/experiments/20261007-diffmjx/20261007-official_padsT1_cylinder_cfd_jacobian_20.json

logs/20261007-diffmjx/venv/bin/python scripts/real_v1_diffmjx_local_search.py \
  --gradient-actions \
  --out docs/experiments/20261007-diffmjx/20261007-pad_local_search_20_mj331.json

python scripts/real_v1_diffmjx_page.py
```

`--tip tpu6 --contact-model pads` builds the full 1 mm sphere-packed surface.
`padsT` retains those spheres for tool contact and masks their pad-to-pad
collision candidates. `20261007-pad_mask_cpu.json` checks this local mask
against the full scene at the D1 held state and through 20 ms of CPU MuJoCo.
`--pad-spacing-mm 2` builds the coarser 2 mm packed surface without changing
the generator's default.

The D1 base is `assets/mjcf/real_v1/real_hand.xml` via the deployed plan
`docs/experiments/20260902-residual-bench/deploy/sv1_w6689_b060_plan.json`.
The probe simulates the plan's 0.8 s grip, then removes the post for a local
three-finger continuation. The palm display plate is non-colliding; it has
zero contacts at the measured state. The exploratory capsule gradient tests
keep the shaft radius and change tool mass and end shape. The packed-surface
tests use the actual cylinder tool.

The 20 ms action search uses MuJoCo 3.3.1 for the CPU score, finite differences
and random samples, alongside the released fork. The separate
`20261007-pad_local_search_20.json` records the same search in the project's
MuJoCo 3.6 environment.
