from dataclasses import dataclass
from enum import Enum


class Method(Enum):
    FTCS_EXPLICIT = "FTCS (Explicit)"


class BoundaryType(Enum):
    DIRICHLET = "Dirichlet"
    NEUMANN = "Neumann"


@dataclass(frozen=True)
class GridSpec:
    L: float
    Nx: int  # >= 2


@dataclass(frozen=True)
class TimeSpec:
    T: float
    dt: float


@dataclass(frozen=True)
class PhysicalSpec:
    alpha: float  # thermal diffusivity


@dataclass(frozen=True)
class BoundaryConfig:
    boundary_type: BoundaryType
    left_value: float = 0.0
    right_value: float = 0.0


@dataclass(frozen=True)
class SolverSpec:
    method: Method
    boundary_type: BoundaryType
