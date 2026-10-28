import sys

from loguru import logger

from core.config import LOG_LEVEL, LOG_RETENTION, LOGS_DIR

logger.remove()

logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=LOG_LEVEL,
)

logger.add(
    f"{LOGS_DIR!s}/{{time:YYYYMMDD}}.log",
    rotation="00:00",
    retention=LOG_RETENTION,
    level=LOG_LEVEL,
    encoding="utf-8",
)
