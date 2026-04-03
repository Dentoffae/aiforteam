#!/usr/bin/env python3
"""Запуск телеграм-бота: python run_bot.py"""

from team_telegram_bot.bot_app import run
from team_telegram_bot.config import Settings

if __name__ == "__main__":
    run(Settings.from_env())
