"""Frame sinks: the output end of the pipeline.

A ``Sink`` consumes a composited BGR frame each tick. ``PreviewSink`` shows it
in an OpenCV window; ``VirtualCamSink`` (Phase 6) broadcasts it to a virtual
webcam device. Both satisfy the same contract, so the loop can feed several
sinks from one frame.
"""

from src.sinks.base import Sink, SinkError
from src.sinks.preview import PreviewSink
from src.sinks.virtualcam import VirtualCamSink

__all__ = ["Sink", "SinkError", "PreviewSink", "VirtualCamSink"]
