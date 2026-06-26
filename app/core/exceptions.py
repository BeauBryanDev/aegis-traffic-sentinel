"""
Custom application exceptions.
"""

class AuthenticationError(Exception):
    """Raised when user authentication fails."""

    pass
class VideoValidationError(Exception):
    """Raised when an uploaded video fails validation."""

    pass


class InferenceError(Exception):
    """Raised when ONNX inference fails."""

    pass


class ModelNotLoadedError(Exception):
    """Raised when the ONNX model has not been initialized."""

    pass


class StreamDisconnectedError(Exception):
    """Raised when a WebSocket client disconnects unexpectedly."""

    pass


class FrameExtractionError(Exception):
    """Raised when video frames cannot be extracted."""

    pass


class InvalidInputShapeError(Exception):
    """Raised when an input tensor has an unexpected shape."""

    pass