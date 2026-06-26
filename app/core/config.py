
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings loaded from environment (or `.env` at project root).

    Security-sensitive values use `SecretStr` to avoid accidental logging
    and introspection. Defaults are aligned with the project's `.env`.
    """

    # Application
    APP_NAME: str = "Aegis Traffic Sentinel"
    DEBUG: bool = True

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    API_PREFIX: str = "/api/v1"

    # Database
    POSTGRES_HOST: str = "db"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "aegis_traffic_db"
    POSTGRES_USER: str = "aegis_manager"
    POSTGRES_PASSWORD: SecretStr = Field(..., repr=False)

    # Uploads
    MAX_UPLOAD_MB: int = 250
    VIDEO_UPLOAD_DIR: Path = BASE_DIR / "uploads" / "videos"
    RECORDING_DIR: Path = BASE_DIR / "uploads" / "recordings"
    TEMP_DIR: Path = BASE_DIR / "uploads" / "temp"

    # ML Model
    MODEL_PATH: str = "ml/models/crash_detector.onnx"
    N_FRAMES: int = 16
    IMAGE_SIZE: int = 224
    CRASH_THRESHOLD: float = 0.50

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_DIR: Path = BASE_DIR / "logs"

    # Security
    SECRET_KEY: SecretStr = Field(..., repr=False)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Pydantic settings
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        case_sensitive=True,
        extra="ignore",
    )

    @property
    def DATABASE_URL(self) -> str:
        """Build a SQLAlchemy-compatible database URL using secret values."""

        password = self.POSTGRES_PASSWORD.get_secret_value()
        return (
            f"postgresql+psycopg2://"
            f"{self.POSTGRES_USER}:"
            f"{password}@"
            f"{self.POSTGRES_HOST}:"
            f"{self.POSTGRES_PORT}/"
            f"{self.POSTGRES_DB}"
        )


settings = Settings()


# Ensure application runtime directories exist (safe to call on import)
def _ensure_directories():
    for p in (settings.VIDEO_UPLOAD_DIR, settings.RECORDING_DIR, settings.TEMP_DIR, settings.LOG_DIR):
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Avoid raising on startup if filesystem is read-only — the app
            # should still be able to start and surface errors where appropriate.
            pass


_ensure_directories()