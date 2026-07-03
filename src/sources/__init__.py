"""Frame sources: objects that produce the latest frame as a numpy array.

A ``Source`` is the input end of the pipeline (``sources -> scene -> compositor
-> sinks``). ``WebcamSource`` is the live background feed; ``ImageSource`` is a
static overlay image. Later phases add live window/monitor capture sources that
satisfy the same contract, so the rest of the pipeline never changes.
"""

from src.sources.base import Size, Source, SourceError
from src.sources.image import ImageSource
from src.sources.webcam import WebcamSource

__all__ = ["Size", "Source", "SourceError", "ImageSource", "WebcamSource"]
