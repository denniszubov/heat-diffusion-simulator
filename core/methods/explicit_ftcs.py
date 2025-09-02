from ..solver import Stepper
from ..types import Array1D


class FTCS(Stepper):
    """
    Forward-Time Central-Space (FTCS) explicit method for 1D heat diffusion.
    """
    def step(self, u: Array1D, t: float) -> Array1D:
        raise NotImplementedError("FTCS.step: implement the explicit update here.")

    def stability_note(self) -> str | None:
        r = self.phys.alpha * self.time.dt / (self.dx * self.dx)
        return f"FTCS recommended stability: r = alpha*dt/dx^2 = {r:.4f} (≤ 0.5 for classical 1D)."
