from abc import ABC, abstractmethod
from dataclasses import dataclass

from .types import Array1D


class BoundaryHandler(ABC):
    @abstractmethod
    def apply(self, u: Array1D, t: float, dx: float) -> None:
        """Apply boundary conditions to the array u."""
        pass


@dataclass
class DirichletBC(BoundaryHandler):
    left: float
    right: float
    
    def apply(self, u: Array1D, t: float, dx: float) -> None:
        u[0] = self.left
        u[-1] = self.right


@dataclass
class NeumannBC(BoundaryHandler):
    """Zero flux (isolated) boundary conditions.
    
    Implements ∂u/∂x = 0 at both boundaries using ghost points.
    This allows the boundaries to evolve naturally during the simulation.
    """
    
    def apply(self, u: Array1D, t: float, dx: float) -> None:
        # Zero flux at left boundary: ∂u/∂x|_{x=0} = 0
        # Using central difference: (u[1] - u[-1]) / (2*dx) = 0
        # This gives us: u[0] = u[1] (reflecting the gradient)
        u[0] = u[1]
        
        # Zero flux at right boundary: ∂u/∂x|_{x=L} = 0  
        # Using central difference: (u[ghost] - u[-2]) / (2*dx) = 0
        # This gives us: u[-1] = u[-2] (reflecting the gradient)
        u[-1] = u[-2]
