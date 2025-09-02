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
