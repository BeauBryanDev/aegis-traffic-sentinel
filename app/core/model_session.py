
from pathlib import Path

import onnxruntime as ort

from app.core.config import BASE_DIR, settings


_model_session: ort.InferenceSession | None = None


def initialize_model() -> None:
    """
    Load the ONNX model into memory.

    Raises:
        FileNotFoundError:
            If the model file does not exist.
    """

    global _model_session

    if _model_session is not None:
        return

    model_path = BASE_DIR / settings.MODEL_PATH

    if not model_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {model_path}"
        )

    _model_session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )


def get_model_session() -> ort.InferenceSession:
    """
    Return the shared ONNX Runtime session.

    Raises:
        RuntimeError:
            If the model has not been initialized.
    """

    if _model_session is None:
        raise RuntimeError(
            "Model session has not been initialized."
        )

    return _model_session


def close_model_session() -> None:
    """
    Release the shared ONNX Runtime session.
    """

    global _model_session

    _model_session = None