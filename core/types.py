from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Array1D = NDArray[np.float64]


@dataclass(frozen=True)
class Snapshot:
    step: int
    t: float
    u: Array1D  # the temperature profile


class InitialCondition(Protocol):
    def __call__(self, x: Array1D) -> Array1D: ...
