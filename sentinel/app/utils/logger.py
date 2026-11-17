"""
Central Logger Module.
Configures loguru logging with rotation, formatting, and file sinks.
"""

import sys
from pathlib import Path
from loguru import logger

from config import settings

# Configure loguru format
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

logger.remove()
logger.add(sys.stderr, format=LOG_FORMAT, level=settings.log_level)

# Create logs directory
log_dir = settings.base_dir / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
logger.add(
    str(log_dir / "sentinel_{time:YYYY-MM-DD}.log"),
    format=LOG_FORMAT,
    level="DEBUG",
    rotation="100 MB",
    retention="14 days",
    compression="zip",
)

__all__ = ["logger"]

