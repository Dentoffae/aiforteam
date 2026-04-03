"""
Telegram bot: индексация всех сообщений в Pinecone, сессии listen start/stop, ответы по @упоминанию.
"""

from __future__ import annotations

import logging
import re
import threading
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import telebot
from telebot.apihelper import ApiTelegramException
from telebot.types import BotCommand, Message

from team_telegram_bot.config import Settings
from team_telegram_bot.logging_setup import setup_logging
from team_telegram_bot.pipelines import (
    TeamHaystackRuntime,
    new_session_id,
    split_transcript,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

LISTEN_COMMANDS_START = {"listen_start", "начать_прослушивание", "startlisten"}
LISTEN_COMMANDS_STOP = {"listen_stop", "стоп_прослушивание", "stoplisten"}

# Кириллические алиасы обрабатываем только в on_group_text (нет в setMyCommands).
CYRILLIC_LISTEN_START = frozenset({"начать_прослушивание"})
CYRILLIC_LISTEN_STOP = frozenset({"стоп_прослушивание"})

SKIP_INDEX_COMMANDS = LISTEN_COMMANDS_START | LISTEN_COMMANDS_STOP | {"help", "start"}


class ListenSession:
    __slots__ = ("session_id", "lines")

    def __init__(self) -> None:
        self.session_id = new_session_id()
        self.lines: list[str] = []


class ChatState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.listen: ListenSession | None = None


def _ts(message: Message) -> str:
    if message.date:
        return datetime.fromtimestamp(message.date, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return ""


def _full_name(u) -> str:
    parts = [u.first_name or "", u.last_name or ""]
    return " ".join(p for p in parts if p).strip() or "Без имени"


def _parse_command(text: str | None, bot_username: str | None) -> tuple[str | None, str]:
    """
    Returns (command_without_slash_lower, rest_text).
    Handles /cmd@BotName in groups.
    """
    if not text or not text.startswith("/"):
        return None, text or ""
    line = text.split("\n", 1)[0].strip()
    token = line.split()[0]
    cmd_part = token[1:]
    if "@" in cmd_part:
        name, at, uname = cmd_part.partition("@")
        if bot_username and uname.lower() != bot_username.lower():
            return None, text
        cmd_part = name
    cmd = cmd_part.lower()
    rest = text[len(token) :].strip()
    return cmd, rest


def _is_group(message: Message) -> bool:
    return message.chat.type in ("group", "supergroup")


def _strip_bot_mention(text: str, bot_username: str) -> str:
    pattern = re.compile(rf"@?{re.escape(bot_username)}\s*", re.IGNORECASE)
    return pattern.sub("", text).strip()


def _is_addressed_to_bot(message: Message, bot_id: int, bot_username: str) -> bool:
    if message.reply_to_message and message.reply_to_message.from_user:
        if message.reply_to_message.from_user.id == bot_id:
            return True
    text = message.text or ""
    if f"@{bot_username}".lower() in text.lower():
        return True
    if message.entities:
        for e in message.entities:
            if e.type == "mention":
                frag = text[e.offset : e.offset + e.length]
                if frag.lower() == f"@{bot_username.lower()}":
                    return True
    return False


def create_bot(settings: Settings) -> telebot.TeleBot:
    return telebot.TeleBot(settings.telegram_token, parse_mode=None)


def run(settings: Settings) -> None:
    setup_logging(settings)
    bot = create_bot(settings)
    me = bot.get_me()
    bot_username = me.username or ""
    bot_id = me.id

    runtime = TeamHaystackRuntime(settings)
    runtime.warm_up()

    states: dict[int, ChatState] = {}
    states_lock = threading.Lock()

    def state_for(chat_id: int) -> ChatState:
        with states_lock:
            if chat_id not in states:
                states[chat_id] = ChatState()
            return states[chat_id]

    def set_my_commands() -> None:
        # В setMyCommands допустимы только a–z, 0–9 и _ (кириллица в имени команды → BOT_COMMAND_INVALID).
        try:
            bot.set_my_commands(
                [
                    BotCommand(
                        "listen_start",
                        "Начать сессию для резюме (ещё: /startlisten)",
                    ),
                    BotCommand(
                        "listen_stop",
                        "Стоп сессии и резюме (ещё: /stoplisten)",
                    ),
                    BotCommand("startlisten", "Алиас listen_start"),
                    BotCommand("stoplisten", "Алиас listen_stop"),
                    BotCommand("help", "Справка"),
                ]
            )
        except ApiTelegramException as e:
            logger.warning("set_my_commands failed (бот всё равно работает): %s", e)

    set_my_commands()

    if settings.telegram_log_incoming:

        @bot.middleware_handler(update_types=["message"])
        def _log_incoming(_bot_instance, message: Message) -> None:
            uid = message.from_user.id if message.from_user else None
            preview = (message.text or "")[:160].replace("\n", " ")
            logger.info(
                "tg_in type=%s chat_id=%s msg_id=%s user_id=%s text=%r",
                message.chat.type,
                message.chat.id,
                message.message_id,
                uid,
                preview,
            )

    @bot.message_handler(commands=["help"])
    def cmd_help(message: Message) -> None:
        bot.reply_to(
            message,
            "Команды (в меню Telegram только латиница; кириллические имена в API запрещены):\n"
            "/listen_start или /startlisten — начать сессию: сообщения копятся для резюме.\n"
            "/listen_stop или /stoplisten — стоп: диалог в модель и в векторную базу, резюме в чат.\n\n"
            "В группе на обычные сообщения бот не отвечает (только индексирует). Ответ — если написать @"
            f"{bot_username} с вопросом или ответить реплаем на сообщение бота.\n\n"
            "Все обычные сообщения в группе индексируются в Pinecone (нужен режим без Privacy в @BotFather).\n\n"
            "Работает в группах и супергруппах; в личке — только эта справка и команды.",
        )

    @bot.message_handler(commands=["start"])
    def cmd_start(message: Message) -> None:
        cmd_help(message)

    def handle_listen_start(message: Message) -> None:
        st = state_for(message.chat.id)
        with st.lock:
            st.listen = ListenSession()
            sid = st.listen.session_id
        logger.info(
            "listen_start chat_id=%s session_id=%s user_id=%s",
            message.chat.id,
            sid,
            message.from_user.id if message.from_user else None,
        )
        bot.reply_to(
            message,
            "Сессия прослушивания начата. Пишите сообщения — они пойдут в итоговое резюме. "
            "/listen_stop — завершить.",
        )

    def handle_listen_stop(message: Message) -> None:
        st = state_for(message.chat.id)
        with st.lock:
            sess = st.listen
            st.listen = None
        if not sess or not sess.lines:
            logger.info("listen_stop ignored: no session chat_id=%s", message.chat.id)
            bot.reply_to(message, "Нет активной сессии или буфер пуст. Сначала /listen_start.")
            return
        transcript = "\n".join(sess.lines)
        session_id = sess.session_id
        chat_id = message.chat.id
        logger.info(
            "listen_stop chat_id=%s session_id=%s lines=%s transcript_chars=%s",
            chat_id,
            session_id,
            len(sess.lines),
            len(transcript),
        )
        try:
            chunks = split_transcript(transcript)
            runtime.index_session_transcript(chat_id=chat_id, session_id=session_id, chunks=chunks)
        except Exception:
            logger.exception("index_session_transcript")
            bot.reply_to(message, "Ошибка при сохранении сессии в векторную базу. Резюме всё равно сгенерирую.")
        try:
            summary = runtime.summarize_session(transcript)
            logger.info("summarize_session ok chat_id=%s session_id=%s out_chars=%s", chat_id, session_id, len(summary))
        except Exception:
            logger.exception("summarize_session")
            bot.reply_to(message, "Не удалось получить резюме от модели.")
            return
        if len(summary) > 4000:
            for i in range(0, len(summary), 4000):
                bot.send_message(chat_id, summary[i : i + 4000])
        else:
            bot.reply_to(message, summary)

    @bot.message_handler(commands=["listen_start", "startlisten"])
    def cmd_listen_start_registered(message: Message) -> None:
        if _is_group(message):
            handle_listen_start(message)
        else:
            bot.reply_to(
                message,
                "Команда только для группы: добавьте бота в чат, в @BotFather отключите Group Privacy, "
                "затем /listen_start в группе.",
            )

    @bot.message_handler(commands=["listen_stop", "stoplisten"])
    def cmd_listen_stop_registered(message: Message) -> None:
        if _is_group(message):
            handle_listen_stop(message)
        else:
            bot.reply_to(
                message,
                "Команда только для группы (см. /help).",
            )

    def index_if_applicable(message: Message, text: str) -> None:
        if not _is_group(message):
            return
        if not text or not text.strip():
            return
        if message.from_user and message.from_user.is_bot:
            return
        cmd, _ = _parse_command(text, bot_username)
        if cmd and cmd.lower() in SKIP_INDEX_COMMANDS:
            return
        uid = message.from_user.id if message.from_user else 0
        uname = message.from_user.username if message.from_user else None
        fname = _full_name(message.from_user) if message.from_user else "Unknown"
        try:
            runtime.index_chat_message(
                chat_id=message.chat.id,
                message_id=message.message_id,
                user_id=uid,
                username=uname,
                full_name=fname,
                text=text,
                ts_iso=_ts(message),
            )
            logger.debug(
                "indexed chat_id=%s msg_id=%s user_id=%s chars=%s",
                message.chat.id,
                message.message_id,
                uid,
                len(text),
            )
        except Exception:
            logger.exception("index_chat_message")

    def append_listen_buffer(message: Message, text: str) -> None:
        if not _is_group(message):
            return
        if message.from_user and message.from_user.is_bot:
            return
        cmd, _ = _parse_command(text, bot_username)
        if cmd:
            return
        st = state_for(message.chat.id)
        with st.lock:
            if st.listen is None:
                return
        uid = message.from_user.id if message.from_user else 0
        uname = message.from_user.username if message.from_user else None
        fname = _full_name(message.from_user) if message.from_user else "Unknown"
        author = fname + (f" (@{uname})" if uname else "")
        line = f"[{_ts(message)}] {author} [id:{uid}]: {text}"
        with st.lock:
            if st.listen is not None:
                st.listen.lines.append(line)
                logger.debug(
                    "listen_buffer chat_id=%s lines=%s",
                    message.chat.id,
                    len(st.listen.lines),
                )

    def _group_text_ok(m: Message) -> bool:
        if not _is_group(m) or not m.text:
            return False
        if not m.text.startswith("/"):
            return True
        cmd, _ = _parse_command(m.text, bot_username)
        # /help и /start обрабатываются отдельными хендлерами — не дублировать
        return cmd not in ("help", "start")

    @bot.message_handler(content_types=["text"], func=_group_text_ok)
    def on_group_text(message: Message) -> None:
        text = message.text or ""
        cmd, _ = _parse_command(text, bot_username)

        if cmd in CYRILLIC_LISTEN_START:
            handle_listen_start(message)
            return
        if cmd in CYRILLIC_LISTEN_STOP:
            handle_listen_stop(message)
            return

        index_if_applicable(message, text)
        append_listen_buffer(message, text)

        if cmd:
            return

        if not _is_addressed_to_bot(message, bot_id, bot_username):
            return

        question = _strip_bot_mention(text, bot_username).strip()
        if message.reply_to_message and message.reply_to_message.text:
            question = (question + "\n\n(контекст: ответ на сообщение: " + message.reply_to_message.text[:500] + ")").strip()
        if not question:
            bot.reply_to(message, "Напишите вопрос в том же сообщении, где есть упоминание бота, или ответьте на сообщение бота текстом.")
            return

        q_preview = question.replace("\n", " ")[:200]
        logger.info("rag_query chat_id=%s msg_id=%s q=%r", message.chat.id, message.message_id, q_preview)
        try:
            bot.send_chat_action(message.chat.id, "typing")
            answer = runtime.mention_answer_with_summary(chat_id=message.chat.id, question=question)
            logger.info("mention_answer chat_id=%s out_chars=%s", message.chat.id, len(answer))
        except Exception:
            logger.exception("mention_answer_with_summary")
            bot.reply_to(message, "Ошибка при поиске по базе или вызове модели.")
            return

        if len(answer) > 4000:
            answer = answer[:3997] + "..."
        bot.reply_to(message, answer)

    @bot.edited_message_handler(content_types=["text"], func=_is_group)
    def on_edited(message: Message) -> None:
        text = message.text or ""
        logger.debug(
            "message_edited chat_id=%s msg_id=%s chars=%s",
            message.chat.id,
            message.message_id,
            len(text),
        )
        index_if_applicable(message, text)

    def _private_plain_ok(m: Message) -> bool:
        return (
            m.chat.type == "private"
            and bool(m.text)
            and not m.text.startswith("/")
        )

    @bot.message_handler(content_types=["text"], func=_private_plain_ok)
    def on_private_plain(message: Message) -> None:
        bot.reply_to(
            message,
            f"Я настроен на рабочие группы: там индексируются сообщения и работают /listen_*.\n\n"
            f"Добавьте @{bot_username} в группу, в @BotFather → Bot Settings → Group Privacy → Turn off.\n\n"
            f"В группе ответ будет только если упомянуть @{bot_username} или ответить на моё сообщение.\n\n"
            f"Команды в личке: /help",
        )

    logger.info(
        "Bot @%s started | OpenAI base=%s | Pinecone index=%s namespace=%s | log_file=%s | tg_log_incoming=%s",
        bot_username,
        settings.openai_base_url,
        settings.pinecone_index,
        settings.pinecone_namespace,
        settings.log_file or "(console only)",
        settings.telegram_log_incoming,
    )
    bot.infinity_polling(skip_pending=True, interval=0, timeout=30)
