from .boundary_condition import BoundaryHandler, DirichletBC, NeumannBC
from .config import BoundaryConfig, BoundaryType, GridSpec, Method, PhysicalSpec, TimeSpec
from .methods.explicit_ftcs import FTCS
from .solver import Stepper


def build_stepper(method: Method, grid: GridSpec, time: TimeSpec, phys: PhysicalSpec, bc: BoundaryHandler) -> Stepper:
    if method is Method.FTCS_EXPLICIT:
        return FTCS(grid, time, phys, bc)
    raise ValueError(f"Unsupported method: {method}")


def build_boundary(config: BoundaryConfig) -> BoundaryHandler:
    if config.boundary_type is BoundaryType.DIRICHLET:
        return DirichletBC(left=config.left_value, right=config.right_value)
    elif config.boundary_type is BoundaryType.NEUMANN:
        return NeumannBC()
    raise ValueError(f"Unknown boundary type: {config.boundary_type}")
