import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

console = Console()


def setup_logging(level: int = logging.INFO, log_file: str | None = None) -> None:
    handlers: list[logging.Handler] = [
        RichHandler(console=console, rich_tracebacks=True, show_time=False)
    ]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.FileHandler(log_file, encoding="utf-8")
        )

    logging.basicConfig(
        level=level,
        format="%(name)s: %(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
