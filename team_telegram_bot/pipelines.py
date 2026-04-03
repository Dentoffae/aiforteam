"""
Haystack 2.x pipelines: OpenAI embeddings/chat (api_base_url for proxyapi.ru) + Pinecone.
Indexing: OpenAIDocumentEmbedder -> DocumentWriter
RAG (одним проходом): OpenAITextEmbedder -> PineconeEmbeddingRetriever -> ChatPromptBuilder -> OpenAIChatGenerator
Mention (@бот): retrieval → LLM-саммари контекста → финальный промпт → LLM-ответ (mention_answer_with_summary)
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from haystack import Document, Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

from team_telegram_bot.config import Settings

logger = logging.getLogger(__name__)


def _openai_secret(settings: Settings) -> Secret:
    return Secret.from_token(settings.openai_api_key)


def _pinecone_secret(settings: Settings) -> Secret:
    return Secret.from_token(settings.pinecone_api_key)


def build_document_store(settings: Settings) -> PineconeDocumentStore:
    return PineconeDocumentStore(
        api_key=_pinecone_secret(settings),
        index=settings.pinecone_index,
        namespace=settings.pinecone_namespace,
        dimension=settings.pinecone_dimension,
        spec=settings.pinecone_spec,
        metric="cosine",
    )


def build_indexing_pipeline(document_store: PineconeDocumentStore, settings: Settings) -> Pipeline:
    embedder = OpenAIDocumentEmbedder(
        api_key=_openai_secret(settings),
        model=settings.embedding_model,
        api_base_url=settings.openai_base_url,
        meta_fields_to_embed=["author_label", "source"],
        progress_bar=False,
    )
    writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE)
    pipe = Pipeline()
    pipe.add_component("embedder", embedder)
    pipe.add_component("writer", writer)
    pipe.connect("embedder", "writer")
    return pipe


def build_rag_pipeline(document_store: PineconeDocumentStore, settings: Settings) -> Pipeline:
    text_embedder = OpenAITextEmbedder(
        api_key=_openai_secret(settings),
        model=settings.embedding_model,
        api_base_url=settings.openai_base_url,
    )
    retriever = PineconeEmbeddingRetriever(document_store=document_store, top_k=12)

    template = [
        ChatMessage.from_user(
            """
Ты помощник в рабочем чате. По фрагментам переписки (в тексте уже есть автор и время) ответь на вопрос.
Цитируй конкретно: кто, когда, что сказал. Если данных мало — так и напиши.

Фрагменты из истории чата:
{% for document in documents %}
---
{{ document.content }}
{% endfor %}

Вопрос пользователя: {{ question }}

Ответ (на русском):
"""
        )
    ]
    prompt_builder = ChatPromptBuilder(template=template, required_variables="*")
    llm = OpenAIChatGenerator(
        api_key=_openai_secret(settings),
        model=settings.openai_model,
        api_base_url=settings.openai_base_url,
    )

    pipe = Pipeline()
    pipe.add_component("text_embedder", text_embedder)
    pipe.add_component("retriever", retriever)
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("llm", llm)
    pipe.connect("text_embedder.embedding", "retriever.query_embedding")
    pipe.connect("retriever", "prompt_builder")
    pipe.connect("prompt_builder.prompt", "llm.messages")
    return pipe


def chat_filter_equal(field: str, value: Any) -> dict[str, Any]:
    return {"field": field, "operator": "==", "value": value}


def _format_documents_block(documents: list[Document]) -> str:
    """Склеивает содержимое документов для промптов (в content уже автор/время)."""
    parts: list[str] = []
    for i, doc in enumerate(documents, 1):
        parts.append(f"--- Фрагмент {i} ---\n{doc.content}")
    return "\n\n".join(parts)


def _truncate_block(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


class TeamHaystackRuntime:
    """Holds document store and pipelines; indexing, RAG, mention (summary+answer), session summarize."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.document_store = build_document_store(settings)
        self._index_pipe = build_indexing_pipeline(self.document_store, settings)
        self._rag_pipe = build_rag_pipeline(self.document_store, settings)
        self._summarizer = OpenAIChatGenerator(
            api_key=_openai_secret(settings),
            model=settings.openai_model,
            api_base_url=settings.openai_base_url,
        )
        # Второй экземпляр для финального ответа по mention (можно отделить по смыслу от session summarize)
        self._mention_answer_llm = OpenAIChatGenerator(
            api_key=_openai_secret(settings),
            model=settings.openai_model,
            api_base_url=settings.openai_base_url,
        )

    def warm_up(self) -> None:
        try:
            n = self.document_store.count_documents()
            logger.info("Pinecone connected, documents in namespace (approx): %s", n)
        except Exception as e:
            logger.warning("Pinecone warm-up: %s", e)
        for name, component in self._index_pipe.walk():
            if hasattr(component, "warm_up"):
                try:
                    component.warm_up()
                except Exception as e:
                    logger.debug("warm_up %s: %s", name, e)
        for name, component in self._rag_pipe.walk():
            if hasattr(component, "warm_up"):
                try:
                    component.warm_up()
                except Exception as e:
                    logger.debug("warm_up %s: %s", name, e)
        for gen in (self._summarizer, self._mention_answer_llm):
            if hasattr(gen, "warm_up"):
                try:
                    gen.warm_up()
                except Exception as e:
                    logger.debug("warm_up generator: %s", e)

    def _retrieve_for_chat(self, *, chat_id: int, question: str, top_k: int | None = None) -> list[Document]:
        """Достаёт документы из Pinecone тем же эмбеддером/ретривером, что и в RAG-пайплайне."""
        filters = chat_filter_equal("chat_id", int(chat_id))
        text_embedder = self._rag_pipe.get_component("text_embedder")
        retriever = self._rag_pipe.get_component("retriever")
        emb_out = text_embedder.run(text=question)
        embedding = emb_out["embedding"]
        rkwargs: dict[str, Any] = {"query_embedding": embedding, "filters": filters}
        if top_k is not None:
            rkwargs["top_k"] = top_k
        ret_out = retriever.run(**rkwargs)
        return list(ret_out.get("documents") or [])

    def index_documents(self, documents: list[Document]) -> None:
        if not documents:
            return
        if len(documents) == 1:
            logger.debug("index_documents id=%s", documents[0].id)
        else:
            logger.info("index_documents batch count=%s", len(documents))
        self._index_pipe.run({"embedder": {"documents": documents}})

    def index_chat_message(
        self,
        *,
        chat_id: int,
        message_id: int,
        user_id: int,
        username: str | None,
        full_name: str,
        text: str,
        ts_iso: str,
        source: str = "message",
    ) -> None:
        author_label = full_name
        if username:
            author_label = f"{full_name} (@{username})"
        header = f"[{ts_iso}] {author_label} (msg {message_id})"
        content = f"{header}\n{text}"
        doc = Document(
            id=f"msg-{chat_id}-{message_id}",
            content=content,
            meta={
                "chat_id": int(chat_id),
                "message_id": int(message_id),
                "user_id": int(user_id),
                "username": username or "",
                "full_name": full_name,
                "author_label": author_label,
                "ts_iso": ts_iso,
                "source": source,
            },
        )
        self.index_documents([doc])

    def index_session_transcript(
        self,
        *,
        chat_id: int,
        session_id: str,
        chunks: list[str],
    ) -> None:
        docs: list[Document] = []
        for i, chunk in enumerate(chunks):
            docs.append(
                Document(
                    id=f"session-{chat_id}-{session_id}-{i}",
                    content=chunk,
                    meta={
                        "chat_id": int(chat_id),
                        "session_id": session_id,
                        "source": "session_transcript",
                        "chunk_index": i,
                        "author_label": f"Сессия {session_id[:8]}…",
                        "ts_iso": "",
                    },
                )
            )
        logger.info(
            "index_session_transcript chat_id=%s session_id=%s chunks=%s",
            chat_id,
            session_id[:8],
            len(docs),
        )
        self.index_documents(docs)

    def rag_answer(self, *, chat_id: int, question: str) -> str:
        filters = chat_filter_equal("chat_id", int(chat_id))
        out = self._rag_pipe.run(
            {
                "text_embedder": {"text": question},
                "retriever": {"filters": filters},
                "prompt_builder": {"question": question},
            }
        )
        retrieved = out.get("retriever", {}).get("documents") or []
        logger.debug("rag_answer retrieved chat_id=%s top_k_used=%s", chat_id, len(retrieved))
        replies = out["llm"]["replies"]
        if not replies:
            return "Не удалось получить ответ."
        return replies[0].text or ""

    def mention_answer_with_summary(
        self,
        *,
        chat_id: int,
        question: str,
        top_k: int = 12,
        excerpts_max_chars: int = 8000,
    ) -> str:
        """
        Сценарий обращения к боту по @упоминанию:
        1) векторный поиск по чату;
        2) отдельный вызов модели — саммари найденного контекста под вопрос;
        3) финальный промпт (саммари + усечённые фрагменты + вопрос) и ответ модели.
        """
        documents = self._retrieve_for_chat(chat_id=chat_id, question=question, top_k=top_k)
        logger.info(
            "mention pipeline chat_id=%s retrieved_docs=%s question_chars=%s",
            chat_id,
            len(documents),
            len(question),
        )
        if not documents:
            return (
                "В этой беседе пока нет проиндексированных сообщений для этого чата "
                "(или поиск ничего не нашёл). Напишите несколько сообщений в группе с выключенным Group Privacy."
            )

        fragments = _format_documents_block(documents)
        summary_prompt = f"""Пользователь обратился к ассистенту в рабочем чате (упоминание бота).

Вопрос пользователя:
{question}

Ниже фрагменты переписки, найденные по смыслу (в тексте уже указаны автор и время, где они были в сообщениях).

{fragments}

Сформируй нейтральное краткое саммари (примерно 5–12 предложений): о чём эти фрагменты в связи с вопросом, кто участвует, какие факты, договорённости или позиции важны. Не отвечай на вопрос пользователя — только сожми контекст для следующего шага.
Пиши на русском."""

        logger.debug("mention step1 summary prompt_chars=%s", len(summary_prompt))
        sum_out = self._summarizer.run([ChatMessage.from_user(summary_prompt)])
        sum_replies = sum_out.get("replies") or []
        if not sum_replies or not (sum_replies[0].text or "").strip():
            logger.warning("mention: пустое саммари, отвечаю по фрагментам без шага summary")
            context_summary = "(Саммари недоступно; используй только фрагменты ниже.)"
        else:
            context_summary = (sum_replies[0].text or "").strip()
        logger.info("mention step1 summary done chars=%s", len(context_summary))

        excerpts = _truncate_block(fragments, excerpts_max_chars)
        final_prompt = f"""Ты отвечаешь в рабочем чате: пользователь упомянул бота и задал вопрос.

## Саммари релевантного контекста (подготовлено отдельно)
{context_summary}

## Опорные фрагменты из истории (для точных цитат; автор и время указаны в тексте)
{excerpts}

## Вопрос пользователя
{question}

Дай развёрнутый ответ на русском. По возможности ссылайся на конкретных людей и формулировки из фрагментов. Если данных недостаточно — честно скажи об этом."""

        logger.debug("mention step2 final prompt_chars=%s", len(final_prompt))
        ans_out = self._mention_answer_llm.run([ChatMessage.from_user(final_prompt)])
        ans_replies = ans_out.get("replies") or []
        if not ans_replies:
            return "Не удалось сформировать ответ модели."
        return ans_replies[0].text or ""

    def summarize_session(self, transcript: str) -> str:
        prompt = f"""Проанализируй фрагмент обсуждения команды в рабочем чате (сообщения с указанием авторов и времени).

Текст обсуждения:
{transcript}

Сделай по шагам:
1) Краткое резюме сути обсуждения.
2) Определи тип: спор/дискуссия без чёткого итога ИЛИ обсуждение с принятием решения о действиях.
3) Если это спор — дай сбалансированное мнение: какие аргументы важны, на что обратить внимание (не объявляй «победителя»).
4) Если это решение о действиях — чётко сформулируй итоговое решение и предложи следующие шаги.

Отвечай на русском языке, структурированно."""

        logger.debug("summarize_session prompt_chars=%s", len(prompt))
        out = self._summarizer.run([ChatMessage.from_user(prompt)])
        replies = out["replies"]
        if not replies:
            return "Не удалось сформировать резюме."
        return replies[0].text or ""


def split_transcript(text: str, max_chars: int = 6000) -> list[str]:
    """Split long transcript into chunks for embedding (Pinecone / model limits)."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    lines = text.split("\n")
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for line in lines:
        line_len = len(line) + 1
        if size + line_len > max_chars and buf:
            chunks.append("\n".join(buf))
            buf = [line]
            size = line_len
        else:
            buf.append(line)
            size += line_len
    if buf:
        chunks.append("\n".join(buf))
    return chunks


def new_session_id() -> str:
    return str(uuid.uuid4())
