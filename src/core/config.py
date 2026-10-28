"""Environment-backed settings, loaded from ``.env`` and the process environment.

The process environment wins over ``.env``. Defaults live here so model ids,
keys, endpoints and runtime paths are not scattered across modules.
"""

from __future__ import annotations

import os
from pathlib import Path

from core import BASE_DIR

ENV_FILE = BASE_DIR / ".env"


def load_dotenv(path: str | Path = ENV_FILE) -> dict[str, str]:
    """Load ``KEY=VALUE`` lines from ``path`` into ``os.environ``.

    Existing environment variables win, so a real export always overrides the
    file. Returns the pairs found in the file.
    """
    file = Path(path)
    if not file.is_file():
        return {}
    found: dict[str, str] = {}
    for raw in file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        found[key] = value
        os.environ.setdefault(key, value)
    return found


def env(name: str, default: str = "") -> str:
    """A non-empty environment value, or ``default``."""
    return os.environ.get(name, "").strip() or default


def carcass_env() -> str:
    """Read at call time so tests and CLI runs can override it after import."""
    return env("CARCASS_ENV", "dev")


def env_path(name: str, default: Path) -> Path:
    """A path from the environment; relative paths are anchored at the repo root."""
    path = Path(env(name, str(default))).expanduser()
    return path if path.is_absolute() else BASE_DIR / path


def _env_int(name: str, default: int) -> int:
    return int(env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(env(name, str(default)))


load_dotenv()

SERVICE_NAME = env("OTEL_SERVICE_NAME", "carcass")
LOG_LEVEL = env("LOG_LEVEL", "INFO")
LOG_RETENTION = env("LOG_RETENTION", "30 days")

OPENROUTER_API_KEY_ENV = "CARCASS_OPENROUTER_API_KEY"
MISTRAL_API_KEY_ENV = "MISTRAL_API_KEY"

OPENROUTER_URL = env("OPENROUTER_URL", "https://openrouter.ai/api/v1")
CHAT_MAX_RETRIES = _env_int("CHAT_MAX_RETRIES", 2)
CHAT_TIMEOUT_S = _env_float("CHAT_TIMEOUT_S", 720.0)
OPENROUTER_EXTRA_BODY = {"usage": {"include": True}}

DEFAULT_CHAT_MODEL = env("DEFAULT_CHAT_MODEL", "anthropic/claude-haiku-5.5")
DEFAULT_CAPTION_MODEL = env("DEFAULT_CAPTION_MODEL", DEFAULT_CHAT_MODEL)
DEFAULT_OCR_MODEL = env("DEFAULT_OCR_MODEL", "mistral-ocr-4-1")

DEFAULT_OCR_CHUNK_SIZE = _env_int("DEFAULT_OCR_CHUNK_SIZE", 10)
DEFAULT_OCR_TIMEOUT_S = _env_float("DEFAULT_OCR_TIMEOUT_S", 600.0)
DEFAULT_OCR_INLINE_MAX_BYTES = _env_int(
    "DEFAULT_OCR_INLINE_MAX_BYTES", 45 * 1024 * 1024
)
DEFAULT_TABLE_MAX_PIXELS = _env_int("DEFAULT_TABLE_MAX_PIXELS", 2_000_000)

DEFAULT_PARCEL_ADDRESS = env(
    "DEFAULT_PARCEL_ADDRESS", "4 rue Joliot Curie 91190 Gif-sur-Yvette"
)
MOCK_CHAT_MODEL = env("MOCK_CHAT_MODEL", "deepseek/deepseek-v4.1-flash-1017")

LOGS_DIR = env_path("CARCASS_LOGS_DIR", BASE_DIR / "data" / "logs")
TRACES_DIR = env_path("CARCASS_TRACES_DIR", BASE_DIR / "data" / "traces")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
TRACES_DIR.mkdir(parents=True, exist_ok=True)


def openrouter_api_key() -> str | None:
    return env(OPENROUTER_API_KEY_ENV) or None


def mistral_api_key() -> str | None:
    return env(MISTRAL_API_KEY_ENV) or None


def require_openrouter_api_key() -> str:
    key = openrouter_api_key()
    if not key:
        raise RuntimeError(
            f"{OPENROUTER_API_KEY_ENV} is not set; copy .env.example to .env"
        )
    return key
