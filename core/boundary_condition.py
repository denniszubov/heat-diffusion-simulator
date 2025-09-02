from abc import ABC, abstractmethod
from dataclasses import dataclass

from .types import Array1D


class BoundaryHandler(ABC):
    @abstractmethod
    def apply(self, u: Array1D, t: float, dx: float) -> None:
        raise NotImplementedError()


@dataclass
class DirichletBC(BoundaryHandler):
    def apply(self, u: Array1D, t: float, dx: float) -> None:
        raise NotImplementedError("To be implemented")
