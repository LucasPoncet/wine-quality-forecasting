"""
Centralized logger configuration for the project.

Usage:
    from utils.config_logger import setup_logging
    setup_logging()

    import logging
    logger = logging.getLogger(__name__)
    logger.info("Training started")
"""

import logging
import sys
from pathlib import Path


def setup_logging(
    level: int = logging.INFO,
    log_dir: str | None = None,
    filename: str = "training.log",
) -> None:
    """Configure logging globally across the project.

    Parameters
    ----------
    level : int, optional
        Global log level (default=logging.INFO).
    log_dir : str | None, optional
        Directory to store log file. If None, logs only to stdout.
    filename : str, optional
        Name of log file if `log_dir` is provided.
    """
    # Prevent duplicate configuration if setup_logging is called multiple times
    if logging.getLogger().handlers:
        return

    fmt = "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(log_dir) / filename)
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
    )

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
