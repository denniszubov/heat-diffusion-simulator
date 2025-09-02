from dataclasses import dataclass
from enum import Enum


class Method(Enum):
    FTCS_EXPLICIT = "FTCS (Explicit)"


class BoundaryKind(Enum):
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
class SolverSpec:
    method: Method
    boundary_kind: BoundaryKind
