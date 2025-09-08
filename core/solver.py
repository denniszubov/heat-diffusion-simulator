from abc import ABC
from abc import abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from .boundary_condition import BoundaryHandler
from .config import GridSpec
from .config import PhysicalSpec
from .config import TimeSpec
from .types import Array1D
from .types import Snapshot


class Stepper(ABC):
    def __init__(self, grid: GridSpec, time: TimeSpec, phys: PhysicalSpec, bc: BoundaryHandler) -> None:
        self.grid = grid
        self.time = time
        self.phys = phys
        self.bc = bc
        self.x = np.linspace(0.0, grid.L, grid.Nx, dtype=np.float64)
        self.dx = float(self.x[1] - self.x[0]) if grid.Nx > 1 else 1.0

    @abstractmethod
    def step(self, u: Array1D, t: float) -> Array1D:
        raise NotImplementedError


@dataclass
class Runner:
    stepper: Stepper
    u0: Array1D

    def run(self) -> Iterator[Snapshot]:
        T, dt = self.stepper.time.T, self.stepper.time.dt
        n_steps = int(np.floor(T / dt))
        u = self.u0.astype(np.float64).copy()
        t = 0.0

        # Apply boundary conditions to initial state
        self.stepper.bc.apply(u, t, self.stepper.dx)
        yield Snapshot(step=0, t=t, u=u.copy())

        for k in range(1, n_steps + 1):
            # Step from current time t to next time t + dt
            # (stepper handles boundary conditions internally)
            u = self.stepper.step(u, t)
            t = t + dt
            yield Snapshot(step=k, t=t, u=u.copy())
