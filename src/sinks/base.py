"""Base ``Sink`` contract for everything that consumes composited frames.

A sink is the output end of the pipeline: hand it a finished BGR frame via
``send()`` each tick. Keeping this one-method contract means the loop can drive
a preview window today and additionally a virtual camera later (Phase 6) from
the same composited frame, with no branching in the loop.
"""

from abc import ABC, abstractmethod

import numpy as np


class Sink(ABC):
    """A consumer of composited BGR frames. Subclasses implement ``send()``."""

    @abstractmethod
    def send(self, frame: np.ndarray) -> None:
        """Present/emit one composited BGR frame."""

    def close(self) -> None:
        """Release any underlying resources. Default is a no-op."""
