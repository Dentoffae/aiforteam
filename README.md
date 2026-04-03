# AIFORTEAM — ассистент для рабочего чата в Telegram

Проект объединяет **Telegram-бота** на [pyTelegramBotAPI](https://github.com/eternnoir/pyTelegramBotAPI), **Haystack 2.x** (пайплайны индексации и поиска), **Pinecone** и **OpenAI-совместимый API** (в том числе [ProxyAPI](https://proxyapi.ru/) через `OPENAI_BASE_URL`). Бот рассчитан на **группы и супергруппы**: индексирует переписку, по **упоминанию** строит ответ в два этапа (саммари контекста + финальный ответ) и поддерживает **резюме сессий** `/listen_*` (спор vs решение).

---

## Возможности

| Функция | Описание |
|--------|----------|
| **Постоянная индексация** | Обычные текстовые сообщения в группе (кроме служебных команд) → документы с метаданными автора → Pinecone: `OpenAIDocumentEmbedder → DocumentWriter`. |
| **Сессия «прослушивания»** | `/listen_start` / `/listen_stop` (и алиасы): буфер сообщений → LLM-анализ + индексация транскрипта чанками (`session_transcript`). |
| **Обращение по @боту или reply** | Векторный поиск по `chat_id`, затем **`mention_answer_with_summary`**: (1) саммари найденных фрагментов под вопрос, (2) финальный промпт (саммари + усечённые фрагменты + вопрос) → ответ. **Два вызова чат-модели** на одно обращение. |
| **Классический RAG одним проходом** | Метод `rag_answer` в `pipelines.py` сохранён для экспериментов (без промежуточного саммари); в `bot_app.py` для упоминаний используется именно mention-пайплайн. |
| **Правки сообщений** | Переиндексация по тому же `message_id` (`DuplicatePolicy.OVERWRITE`). |
| **Логирование** | Консоль, опционально файл с ротацией; опционально лог **каждого** входящего Telegram-сообщения (`TELEGRAM_LOG_INCOMING`). |

---

## Архитектура (кратко)

```
                    ┌─────────────────────────────────────────┐
                    │  Индексация (каждое сообщение в группе) │
                    │  OpenAIDocumentEmbedder → DocumentWriter│
                    │              → Pinecone                 │
                    └─────────────────────────────────────────┘

  @бот или reply    ┌─────────────────────────────────────────┐
        ──────────▶│  1) OpenAITextEmbedder + Retriever      │
                   │     (фильтр meta.chat_id)                 │
                   │  2) LLM: саммари контекста под вопрос     │
                   │  3) LLM: ответ по саммари + фрагментам    │
                   └─────────────────────────────────────────┘
```

- Эмбеддинги и чат идут через **`OPENAI_BASE_URL`** (например `https://api.proxyapi.ru/openai/v1`).
- Размерность индекса Pinecone должна соответствовать **`EMBEDDING_MODEL`** (для `text-embedding-3-small` обычно **1536**).

Учебный пример RAG на Haystack — в **`pipeline example.ipynb`**; в боте вместо `InMemoryDocumentStore` используется **Pinecone**.

---

## Структура репозитория

```
AIFORTEAM/
├── README.md
├── requirements.txt
├── .env.example
├── run_bot.py
├── pipeline example.ipynb
└── team_telegram_bot/
    ├── config.py           # Settings + .env
    ├── logging_setup.py    # консоль + RotatingFileHandler
    ├── pipelines.py        # Pinecone, индексация, rag_answer, mention_answer_with_summary, резюме сессии
    └── bot_app.py          # хендлеры Telegram, listen-сессии, вызов mention при @боте
```

---

## Требования

- **Python 3.10+** (рекомендуется 3.11).
- Ключи: Telegram (`TELEGRAM_BOT_TOKEN`), Pinecone, OpenAI-совместимый API.

---

## Установка

```bash
cd AIFORTEAM
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

Скопируйте `.env.example` в `.env` и заполните значения.

---

## Переменные окружения

| Переменная | Обязательно | Описание |
|------------|-------------|----------|
| `OPENAI_API_KEY` | да | Ключ для Chat Completions и Embeddings. |
| `OPENAI_BASE_URL` | нет | Базовый URL без завершающего `/`. По умолчанию: `https://api.proxyapi.ru/openai/v1`. |
| `OPENAI_MODEL` | нет | Модель чата, по умолчанию `gpt-4o-mini`. |
| `EMBEDDING_MODEL` | нет | Модель эмбеддингов, по умолчанию `text-embedding-3-small`. |
| `PINECONE_API_KEY` | да | Ключ Pinecone. |
| `PINECONE_INDEX_NAME` | нет | Имя индекса, по умолчанию `aiforteam`. |
| `PINECONE_NAMESPACE` | нет | Namespace, по умолчанию `default`. |
| `PINECONE_DIMENSION` | нет | Размерность при **создании** индекса, по умолчанию `1536`. |
| `PINECONE_REGION` | нет | Регион serverless, по умолчанию `us-east-1`. |
| `PINECONE_CLOUD` | нет | Облако, по умолчанию `aws`. |
| `TELEGRAM_BOT_TOKEN` | да | Токен бота. |
| `TELEGRAM_LOG_INCOMING` | нет | `1` / `true` / `yes` — в лог пишется каждое входящее сообщение (`tg_in`), удобно проверять доставку апдейтов. |
| `LOG_LEVEL` | нет | `DEBUG`, `INFO`, `WARNING` … По умолчанию `INFO`. |
| `LOG_FILE` | нет | Путь к файлу лога; пусто — только консоль. Пример: `logs/bot.log`. |
| `LOG_MAX_BYTES` | нет | Размер до ротации (байты), по умолчанию 10 MB. |
| `LOG_BACKUP_COUNT` | нет | Число архивных файлов, по умолчанию `5`. |

---

## Настройка Telegram

### 1. Создание бота

В [@BotFather](https://t.me/BotFather): `/newbot` → токен в `TELEGRAM_BOT_TOKEN`.

### 2. Группа и приватность

Чтобы бот **видел все сообщения** в группе:

1. @BotFather → **Bot Settings** → **Group Privacy** → **Turn off**.
2. Добавьте бота в группу.

Иначе в вектор попадут только команды и упоминания — контекст для mention будет бедным.

### 3. Команды в меню

В `setMyCommands` допустимы только **латиница**, цифры и `_`. Кириллица в имени команды даёт `BOT_COMMAND_INVALID`, поэтому в меню:

- `/listen_start`, `/startlisten` — начать сессию для резюме.
- `/listen_stop`, `/stoplisten` — завершить сессию + резюме и индексация.
- `/help`, `/start` — справка.

Регистрация команд обёрнута в `try/except`: при ошибке API бот всё равно запускается.

В группе по-прежнему можно обработать редкие варианты вроде `/начать_прослушивание` (если клиент передаёт их как текст команды).

### 4. Личные сообщения

Основная логика (индексация, listen, mention) — **в группах**. В ЛС на обычный текст бот отвечает краткой подсказкой; `/listen_*` в ЛС дают пояснение, что команды нужно вызывать в группе.

### 5. Ответы в группе

На обычные сообщения без `@бота` бот не отвечает (только индексирует). Ответ — если в тексте есть `@username_бота` или вы отвечаете реплаем на сообщение бота.

### 6. Процесс должен работать постоянно

`run_bot.py` крутит long polling. Если процесс завершился (в логе `Infinity polling: polling exited`), новые сообщения **не обрабатываются** — перезапустите скрипт.

---

## Запуск

```bash
python run_bot.py
```

В логе: старт бота, `OPENAI_BASE_URL`, индекс Pinecone, namespace, файл лога, флаг `tg_log_incoming`.

---

## Поведение подробно

### Индексация

Метаданные включают `chat_id`, `message_id`, `user_id`, `username`, `full_name`, `author_label`, `ts_iso`, `source`. Тело документа: строка с временем и автором + текст. ID: `msg-{chat_id}-{message_id}`.

### Listen-сессия

Буфер по `chat_id`; по `/listen_stop` — вызов `summarize_session`, чанки в Pinecone с `source=session_transcript`.

### Mention (`mention_answer_with_summary` в `pipelines.py`)

1. `_retrieve_for_chat` — те же компоненты, что в RAG-пайплайне (`text_embedder` + `retriever`, фильтр `chat_id`, `top_k` по умолчанию 12).
2. Промпт с вопросом и полными фрагментами → **первый** вызов LLM: краткое **саммари** (без прямого ответа на вопрос).
3. Промпт с саммари, усечёнными фрагментами (`excerpts_max_chars` по умолчанию 8000) и вопросом → **второй** вызов LLM: итоговый ответ.

Параметры `top_k` и `excerpts_max_chars` при необходимости меняются в вызове из `bot_app.py` или в сигнатуре метода.

---

## Логирование

- **INFO**: старт, listen, mention (число документов, длины), ошибки пайплайнов.
- **DEBUG**: индексация по одному сообщению, буфер listen, размеры промптов, `rag_answer` / retrieval.
- Сторонние пакеты (`telebot`, `httpx`, `pinecone`, …) по умолчанию не громче **WARNING**.
- При **`TELEGRAM_LOG_INCOMING=1`** — строка `tg_in` на каждое входящее сообщение (проверка, что Telegram дошёл до бота).

---

## Pinecone

- При отсутствии индекса возможно **автосоздание** serverless с `PINECONE_*`.
- У существующего индекса размерность должна совпадать с эмбеддингами.
- Разные чаты разделяются **`chat_id`** в метаданных.

---

## Зависимости

**docling** в `requirements.txt` не обязателен для `run_bot.py` (ноутбуки / другие сценарии). Для бота достаточно: `pyTelegramBotAPI`, `python-dotenv`, `openai`, `haystack-ai`, `pinecone-haystack`, `pinecone`.

---

## Устранение неполадок

| Симптом | Что проверить |
|--------|----------------|
| Нет ответов, бот «мёртвый» | Запущен ли `run_bot.py`; нет ли в логе `polling exited`. |
| В группе нет реакции на текст | Так задумано без `@бота`; для ответа — упоминание или reply на бота. |
| Пустой контекст / «нет сообщений» | **Group Privacy** выключен; сообщения писались **после** добавления бота и индексации. |
| Не видно, доходят ли сообщения | `TELEGRAM_LOG_INCOMING=1` и строки `tg_in` в логе. |
| Pinecone / размерность | `PINECONE_DIMENSION` и модель эмбеддингов; при смене модели часто нужен **новый** индекс. |
| 401 / ошибки LLM | `OPENAI_API_KEY`, `OPENAI_BASE_URL`. |
| Дорого / долго на mention | Два запроса чата + эмбеддинг на запрос; уменьшить `top_k` или упростить до `rag_answer` в коде. |

---

## Полезные ссылки

- [Haystack](https://docs.haystack.deepset.ai/)
- [Pinecone + Haystack](https://docs.haystack.deepset.ai/integrations/pinecone-document-store)
- [OpenAIChatGenerator](https://docs.haystack.deepset.ai/docs/openaichatgenerator)

---

## Лицензия и вклад

Не коммитьте `.env` с секретами; сохраняйте совместимость переменных окружения при изменениях.
