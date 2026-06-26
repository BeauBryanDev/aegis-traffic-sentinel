from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import settings
from app.core.model_session import (
    close_model_session,
    initialize_model,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize application resources during startup and
    release them during shutdown.
    """

    settings.VIDEO_UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    settings.RECORDING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    settings.TEMP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    initialize_model()

    yield

    close_model_session()