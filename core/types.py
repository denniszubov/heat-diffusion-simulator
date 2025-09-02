from collections.abc import Callable
from typing import Protocol
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


Array1D = NDArray[np.float64]
ScalarFunc = Callable[[float], float]  # time-dependent boundary value f(t)


@dataclass(frozen=True)
class Snapshot:
    step: int
    t: float
    u: Array1D  # the temperature profile
    

class InitialCondition(Protocol):
    def __call__(self, x: Array1D) -> Array1D: ...
