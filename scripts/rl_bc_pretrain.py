from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

import sys
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from morphohand.rl.ppo_config import PPOConfig
from morphohand.rl.reference_trajectory import ReferenceTrajectory
from morphohand.sampling.scene import freeze_scene_for_eval


class Actor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: tuple[int, ...],
                 activation: str, init_std: float) -> None:
        super().__init__()
        from rsl_rl.modules.mlp import MLP
        from rsl_rl.modules.distribution import GaussianDistribution

        self.mlp = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
        )
        self.distribution = GaussianDistribution(
            output_dim=output_dim,
            init_std=init_std,
            std_type="scalar",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def _load_summary(run_dir: Path) -> dict:
    with (run_dir / "summary.json").open() as f:
        return json.load(f)


def _resolve_frozen_scene(run_dir: Path, summary: dict) -> Path:
    frozen = run_dir / "frozen_scene.xml"
    if frozen.exists():
        return frozen
    scene_xml = Path(summary["scene_xml"]).resolve()
    keyframe = summary.get("keyframe", "open")
    frozen = run_dir / f"frozen_{scene_xml.stem}.xml"
    freeze_scene_for_eval(scene_xml, keyframe, frozen)
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser(description="Behavior cloning pretrain for the actor.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    summary = _load_summary(run_dir)
    frozen = _resolve_frozen_scene(run_dir, summary)

    npz = np.load(run_dir / "best_rollout.npz")
    best_finger_ctrl = np.asarray(npz["best_finger_ctrl"], dtype=np.float64).reshape(-1)

    import mujoco
    model = mujoco.MjModel.from_xml_path(str(frozen))
    ref = ReferenceTrajectory.from_run_dir(run_dir, model)

    qpos = ref.qpos[:: max(1, int(args.stride))]
    finger_idx = list(ref.layout.fingers)
    finger_qpos = qpos[:, finger_idx]

    # The env consumes 9 finger residuals (palm is scripted, action_dim=0)
    # and uses best_finger_ctrl as the finger default_offset. So the
    # optimal residual is zero (CEM holds a constant setpoint = offset).
    # BC here just biases the policy toward zero residuals so PPO starts
    # near the CEM behaviour instead of from random noise. NOTE: with this
    # setup the same effect is obtained by lowering --init-noise-std in
    # rl_train_cube.py; BC mostly matters if you want a non-trivial input
    # → output mapping (e.g. switching to phase- or qpos-conditional refs).
    _ = best_finger_ctrl  # input parity with previous BC target
    target = np.zeros_like(finger_qpos, dtype=np.float32)

    x = torch.tensor(finger_qpos.astype(np.float32))
    y = torch.tensor(target)

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    ppo = PPOConfig()
    model = Actor(
        input_dim=x.shape[1],
        output_dim=y.shape[1],
        hidden_dims=ppo.actor_hidden_dims,
        activation=ppo.activation,
        init_std=ppo.init_noise_std,
    ).to(args.device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(args.epochs):
        total = 0.0
        count = 0
        for xb, yb in loader:
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total += float(loss.item())
            count += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"epoch {epoch+1:04d}  loss {total / max(1, count):.6f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"actor_state_dict": model.state_dict()}, args.output)
    print(f"wrote actor checkpoint: {args.output}")


if __name__ == "__main__":
    main()
