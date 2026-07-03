"""Frame sinks: the output end of the pipeline.

A ``Sink`` consumes a composited BGR frame each tick. ``PreviewSink`` shows it
in an OpenCV window; later phases add a virtual-camera sink (Phase 6) that
satisfies the same contract, so the loop can feed several sinks from one frame.
"""

from src.sinks.base import Sink
from src.sinks.preview import PreviewSink

__all__ = ["Sink", "PreviewSink"]
