"""Fake base policy for smoke: returns (T, n) action chunks (physical or normalized)."""
from __future__ import annotations

import numpy as np


class FakeBase:
    def __init__(
        self,
        n: int = 8,
        chunk_size: int = 50,
        seed: int = 0,
        home: np.ndarray | None = None,
        offset: np.ndarray | None = None,
    ):
        self.n = int(n)
        self.chunk_size = chunk_size
        self.rng = np.random.default_rng(seed)
        if home is not None:
            self._home = np.asarray(home, dtype=np.float32).reshape(self.n).copy()
        elif self.n == 8:
            # OpenArm v1 home-ish: j4 ~ pi/2, fingers open ~0.044
            self._home = np.array([0, 0, 0, 1.570796, 0, 0, 0, 0.044], dtype=np.float32)
        elif self.n == 7:
            # Default 7-DoF: ReBot B601-DM arm zeros, finger_left open ~0.04 m.
            # LIBERO relative EE should pass home=LIBERO_NOOP explicitly (zeros + gripper -1).
            self._home = np.array([0, 0, 0, 0, 0, 0, 0.04], dtype=np.float32)
        else:
            self._home = np.zeros(self.n, dtype=np.float32)

        if offset is not None:
            self._offset = np.asarray(offset, dtype=np.float32).reshape(self.n).copy()
        elif self.n == 8:
            self._offset = np.array(
                [0.02, -0.01, 0.01, 0.0, 0.01, 0.0, -0.01, 0.0], dtype=np.float32
            )
        elif self.n == 7:
            self._offset = np.array(
                [0.02, -0.01, 0.01, 0.0, 0.01, 0.0, 0.0], dtype=np.float32
            )
        else:
            self._offset = np.zeros(self.n, dtype=np.float32)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def infer_chunk(self, obs: dict | None = None) -> np.ndarray:
        # Slow drift toward a fixed offset — deterministic given home/offset
        t = np.linspace(0, 1, self.chunk_size, dtype=np.float32)[:, None]
        chunk = self._home + t * self._offset
        return chunk.astype(np.float32)


# LIBERO relative EE no-op (LeRobot get_libero_dummy_action); gripper open ≈ -1
LIBERO_NOOP_ACTION = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
LIBERO_FAKE_OFFSET = np.array([0.01, -0.005, 0.005, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
