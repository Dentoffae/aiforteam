import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_base_url: str
    openai_model: str
    embedding_model: str
    pinecone_api_key: str
    pinecone_index: str
    pinecone_dimension: int
    pinecone_namespace: str
    pinecone_spec: dict
    telegram_token: str
    log_level: str
    log_file: str | None
    log_max_bytes: int
    log_backup_count: int
    telegram_log_incoming: bool

    @classmethod
    def from_env(cls) -> "Settings":
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")

        pc_key = os.environ.get("PINECONE_API_KEY", "").strip()
        if not pc_key:
            raise ValueError("PINECONE_API_KEY is not set")

        tg = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        if not tg:
            raise ValueError("TELEGRAM_BOT_TOKEN is not set")

        region = os.environ.get("PINECONE_REGION", "us-east-1").strip()
        cloud = os.environ.get("PINECONE_CLOUD", "aws").strip()
        spec = {"serverless": {"region": region, "cloud": cloud}}

        log_file_raw = os.environ.get("LOG_FILE", "").strip()
        log_max = int(os.environ.get("LOG_MAX_BYTES", str(10 * 1024 * 1024)))
        log_backups = int(os.environ.get("LOG_BACKUP_COUNT", "5"))
        log_incoming = os.environ.get("TELEGRAM_LOG_INCOMING", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )

        return cls(
            openai_api_key=api_key,
            openai_base_url=os.environ.get(
                "OPENAI_BASE_URL", "https://api.proxyapi.ru/openai/v1"
            ).rstrip("/"),
            openai_model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip(),
            embedding_model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small").strip(),
            pinecone_api_key=pc_key,
            pinecone_index=os.environ.get("PINECONE_INDEX_NAME", "aiforteam").strip(),
            pinecone_dimension=int(os.environ.get("PINECONE_DIMENSION", "1536")),
            pinecone_namespace=os.environ.get("PINECONE_NAMESPACE", "default").strip(),
            pinecone_spec=spec,
            telegram_token=tg,
            log_level=os.environ.get("LOG_LEVEL", "INFO").strip(),
            log_file=log_file_raw or None,
            log_max_bytes=max(log_max, 1024),
            log_backup_count=max(log_backups, 1),
            telegram_log_incoming=log_incoming,
        )
