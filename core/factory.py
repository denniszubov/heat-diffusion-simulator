from .boundary_condition import BoundaryHandler, DirichletBC
from .config import BoundaryType, GridSpec, Method, PhysicalSpec, TimeSpec
from .methods.explicit_ftcs import FTCS
from .solver import Stepper


def build_stepper(method: Method, grid: GridSpec, time: TimeSpec, phys: PhysicalSpec, bc: BoundaryHandler) -> Stepper:
    if method is Method.FTCS_EXPLICIT:
        return FTCS(grid, time, phys, bc)
    raise ValueError(f"Unsupported method: {method}")


def build_boundary(boundary_type: BoundaryType) -> BoundaryHandler:
    if boundary_type is BoundaryType.DIRICHLET:
        return DirichletBC()
    raise ValueError(f"Unknown boundary type: {boundary_type}")
