from __future__ import annotations

from pathlib import Path

import numpy as np

from morphohand.backends.base import RolloutResult


class MJXNativeBackend:
    """Baseline differentiable backend.

    This skeleton intentionally keeps a lightweight NumPy fallback so the project can run
    before full MJX model wiring is completed.
    """

    name = "mjx-native"
    supports_autodiff = True

    def __init__(self) -> None:
        self._xml_path: Path | None = None
        self._state = np.zeros(1, dtype=np.float32)
        self._steps = 0

    def load_model(self, xml_path: Path) -> None:
        self._xml_path = xml_path

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        self._state = np.zeros_like(self._state)
        self._steps = 0
        return self._state.copy()

    def step(self, control: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        self._state = self._state + 0.001 * np.sum(control)
        self._steps += 1
        reward = float(-np.linalg.norm(control))
        done = self._steps >= 200
        return self._state.copy(), reward, done, {"backend": self.name}

    def rollout(self, controls: np.ndarray) -> RolloutResult:
        obs = []
        rewards = []
        done = []
        info = {"backend": self.name}
        for ctrl in controls:
            o, r, d, _ = self.step(ctrl)
            obs.append(o)
            rewards.append(r)
            done.append(d)
            if d:
                break
        return RolloutResult(
            observations=np.asarray(obs),
            rewards=np.asarray(rewards),
            done=np.asarray(done),
            info=info,
        )

    def metrics(self) -> dict:
        return {"backend": self.name, "loaded_xml": str(self._xml_path), "steps": self._steps}
