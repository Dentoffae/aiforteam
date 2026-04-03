"""Настройка логирования: консоль + опционально ротируемый файл."""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

from team_telegram_bot.config import Settings


def setup_logging(settings: Settings) -> None:
    level_name = settings.log_level.upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setLevel(level)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    if settings.log_file:
        path = Path(settings.log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            path,
            maxBytes=settings.log_max_bytes,
            backupCount=settings.log_backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

    # Шум сторонних библиотек при INFO
    for name in ("telebot", "urllib3", "httpx", "httpcore", "openai", "pinecone"):
        logging.getLogger(name).setLevel(logging.WARNING)

    haystack_level = logging.DEBUG if level <= logging.DEBUG else logging.WARNING
    logging.getLogger("haystack").setLevel(haystack_level)
