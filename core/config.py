from dataclasses import dataclass
from enum import Enum


class Method(Enum):
    FTCS_EXPLICIT = "FTCS (Explicit)"


class BoundaryType(Enum):
    DIRICHLET = "Dirichlet"


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
    # Future fields for other BC types:
    # left_flux: float = 0.0
    # right_flux: float = 0.0
    # left_alpha: float = 1.0
    # etc.


@dataclass(frozen=True)
class SolverSpec:
    method: Method
    boundary_type: BoundaryType
