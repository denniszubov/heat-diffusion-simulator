from core.boundary_condition import BoundaryHandler
from core.config import BoundaryConfig, BoundaryType
from core.factory import build_boundary
from core.types import Array1D


def setup_boundary_conditions(
    boundary_type_choice: str, u0: Array1D
) -> tuple[BoundaryHandler, BoundaryType, tuple[float, float]]:
    """
    Set up boundary conditions based on user choice and initial condition.

    Args:
        boundary_type_choice: String describing the boundary condition type
        u0: Initial temperature distribution

    Returns:
        Tuple containing:
        - BoundaryHandler: The configured boundary condition handler
        - BoundaryType: The boundary type enum for solver configuration
        - Tuple[float, float]: (y_min, y_max) for plot limits
    """

    if boundary_type_choice == "Dirichlet (Fixed)":
        # Set boundary conditions to match initial condition endpoints
        left_bc_val = float(u0[0])
        right_bc_val = float(u0[-1])

        boundary_config = BoundaryConfig(
            boundary_type=BoundaryType.DIRICHLET, left_value=left_bc_val, right_value=right_bc_val
        )
        bc_type = BoundaryType.DIRICHLET

        # For Dirichlet, use the boundary values for plot limits
        y_min = float(min(u0.min(), left_bc_val, right_bc_val)) - 0.1
        y_max = float(max(u0.max(), left_bc_val, right_bc_val)) + 0.1

    elif boundary_type_choice == "Neumann (Isolated)":
        boundary_config = BoundaryConfig(boundary_type=BoundaryType.NEUMANN)
        bc_type = BoundaryType.NEUMANN

        # For isolated boundaries, allow more dynamic range
        y_buffer = 0.2 * (u0.max() - u0.min())
        y_min = float(u0.min()) - y_buffer
        y_max = float(u0.max()) + y_buffer

    else:
        raise ValueError(f"Unknown boundary type choice: {boundary_type_choice}")

    bc = build_boundary(boundary_config)

    return bc, bc_type, (y_min, y_max)


def get_boundary_type_options():
    return ["Neumann (Isolated)", "Dirichlet (Fixed)"]


def get_boundary_type_help():
    return "Neumann: Isolated boundaries (zero flux)\n" "Dirichlet: Fixed temperatures at boundaries"
