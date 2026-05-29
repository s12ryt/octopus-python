from __future__ import annotations

import json
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

from sqlalchemy import (
    BigInteger,
    Boolean,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    delete,
    event,
    inspect,
    select,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker
from sqlalchemy.types import TypeDecorator


class JSONText(TypeDecorator):
    impl = Text
    cache_ok = True

    def process_bind_param(self, value: Any, dialect) -> str | None:  # type: ignore[override]
        if value is None:
            return None
        if isinstance(value, str):
            # Keep valid JSON strings as-is, wrap plain strings for safety.
            try:
                json.loads(value)
                return value
            except Exception:
                return json.dumps(value, ensure_ascii=False)
        return json.dumps(value, ensure_ascii=False)

    def process_result_value(self, value: str | None, dialect) -> Any:  # type: ignore[override]
        if value in (None, ""):
            return None
        try:
            return json.loads(value)
        except Exception:
            return value


class Base(DeclarativeBase):
    pass


class DictMixin:
    hidden_fields: set[str] = set()

    def to_dict(self, include_relationships: bool = True) -> dict[str, Any]:
        mapper = inspect(self.__class__)
        data: dict[str, Any] = {}
        for column in mapper.columns:
            if column.key in self.hidden_fields:
                continue
            data[column.key] = getattr(self, column.key)
        if include_relationships:
            for rel in mapper.relationships:
                if rel.key in self.hidden_fields:
                    continue
                value = getattr(self, rel.key, None)
                if value is None:
                    data[rel.key] = [] if rel.uselist else None
                elif rel.uselist:
                    data[rel.key] = [item.to_dict(False) if hasattr(item, "to_dict") else item for item in value]
                else:
                    data[rel.key] = value.to_dict(False) if hasattr(value, "to_dict") else value
        return data


class User(Base, DictMixin):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password: Mapped[str] = mapped_column(String(255), nullable=False)


class APIKey(Base, DictMixin):
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    api_key: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    expire_at: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    max_cost: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    supported_models: Mapped[str] = mapped_column(Text, nullable=False, default="")


class Channel(Base, DictMixin):
    __tablename__ = "channels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    type: Mapped[str] = mapped_column(String(100), nullable=False, default="openai/chat_completions")
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    base_urls: Mapped[Any] = mapped_column(JSONText, nullable=True, default=list)
    model: Mapped[str] = mapped_column(Text, nullable=False, default="")
    custom_model: Mapped[str] = mapped_column(Text, nullable=False, default="")
    proxy: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    auto_sync: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    auto_group: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    custom_header: Mapped[Any] = mapped_column(JSONText, nullable=True, default=list)
    param_override: Mapped[str] = mapped_column(Text, nullable=False, default="")
    channel_proxy: Mapped[str] = mapped_column(Text, nullable=False, default="")
    match_regex: Mapped[str] = mapped_column(Text, nullable=False, default="")

    keys: Mapped[list["ChannelKey"]] = relationship(
        back_populates="channel", cascade="all, delete-orphan", lazy="selectin"
    )
    stats: Mapped["StatsChannel"] = relationship(
        back_populates="channel", cascade="all, delete-orphan", uselist=False, lazy="selectin"
    )


class ChannelKey(Base, DictMixin):
    __tablename__ = "channel_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    channel_id: Mapped[int] = mapped_column(Integer, ForeignKey("channels.id", ondelete="CASCADE"), nullable=False, index=True)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    channel_key: Mapped[str] = mapped_column(Text, nullable=False, default="")
    status_code: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_use_time_stamp: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    total_cost: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    remark: Mapped[str] = mapped_column(Text, nullable=False, default="")

    channel: Mapped[Channel] = relationship(back_populates="keys")


class Group(Base, DictMixin):
    __tablename__ = "groups"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    mode: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    match_regex: Mapped[str] = mapped_column(Text, nullable=False, default="")
    first_token_time_out: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    session_keep_time: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    items: Mapped[list["GroupItem"]] = relationship(back_populates="group", cascade="all, delete-orphan", lazy="selectin")


class GroupItem(Base, DictMixin):
    __tablename__ = "group_items"
    __table_args__ = (UniqueConstraint("group_id", "channel_id", "model_name", name="uq_group_channel_model"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    group_id: Mapped[int] = mapped_column(Integer, ForeignKey("groups.id", ondelete="CASCADE"), nullable=False, index=True)
    channel_id: Mapped[int] = mapped_column(Integer, ForeignKey("channels.id", ondelete="CASCADE"), nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    weight: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    group: Mapped[Group] = relationship(back_populates="items")
    channel: Mapped[Channel] = relationship(lazy="selectin")


class LLMInfo(Base, DictMixin):
    __tablename__ = "llm_infos"

    name: Mapped[str] = mapped_column(String(255), primary_key=True)
    input: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    output: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    cache_read: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    cache_write: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)


class Setting(Base, DictMixin):
    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(255), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False, default="")


class StatsMixin(DictMixin):
    input_token: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    output_token: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    input_cost: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    output_cost: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    wait_time: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    request_success: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    request_failed: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)

    def add_metrics(self, metrics: dict[str, Any]) -> None:
        for key in [
            "input_token",
            "output_token",
            "input_cost",
            "output_cost",
            "wait_time",
            "request_success",
            "request_failed",
        ]:
            setattr(self, key, (getattr(self, key) or 0) + (metrics.get(key) or 0))


class StatsTotal(Base, StatsMixin):
    __tablename__ = "stats_totals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)


class StatsDaily(Base, StatsMixin):
    __tablename__ = "stats_dailies"

    date: Mapped[str] = mapped_column(String(20), primary_key=True)


class StatsHourly(Base, StatsMixin):
    __tablename__ = "stats_hourlies"

    hour: Mapped[int] = mapped_column(Integer, primary_key=True)
    date: Mapped[str] = mapped_column(String(20), nullable=False, default="")


class StatsModel(Base, StatsMixin):
    __tablename__ = "stats_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    channel_id: Mapped[int] = mapped_column(Integer, nullable=False, default=0, index=True)


class StatsChannel(Base, StatsMixin):
    __tablename__ = "stats_channels"

    channel_id: Mapped[int] = mapped_column(Integer, ForeignKey("channels.id", ondelete="CASCADE"), primary_key=True)
    channel: Mapped[Channel] = relationship(back_populates="stats")


class StatsAPIKey(Base, StatsMixin):
    __tablename__ = "stats_api_keys"

    api_key_id: Mapped[int] = mapped_column(Integer, primary_key=True)


class RelayLog(Base, DictMixin):
    __tablename__ = "relay_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    time: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    request_model_name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    request_api_key_name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    channel: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    channel_name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    actual_model_name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    input_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    ftut: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    use_time: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    cost: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    request_content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    response_content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    error: Mapped[str] = mapped_column(Text, nullable=False, default="")
    attempts: Mapped[Any] = mapped_column(JSONText, nullable=True, default=list)
    total_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class MigrationRecord(Base, DictMixin):
    __tablename__ = "migration_records"

    version: Mapped[str] = mapped_column(String(50), primary_key=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="success")


engine: Engine | None = None
SessionLocal: sessionmaker[Session] | None = None


def _mysql_url(path: str) -> str:
    # Go DSN: user:password@tcp(host:port)/dbname?params
    match = re.match(r"(?P<user>[^:]+):(?P<pwd>.*?)@tcp\((?P<host>[^)]+)\)/(?P<db>[^?]+)(?P<query>\?.*)?", path)
    if not match:
        return "mysql+pymysql://" + path
    query = match.group("query") or ""
    return f"mysql+pymysql://{match.group('user')}:{match.group('pwd')}@{match.group('host')}/{match.group('db')}{query}"


def make_database_url(db_type: str, path: str) -> str:
    db_type = (db_type or "sqlite").lower()
    if db_type == "sqlite":
        db_path = Path(path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return "sqlite:///" + str(db_path).replace("\\", "/")
    if db_type == "mysql":
        return _mysql_url(path)
    if db_type in {"postgres", "postgresql"}:
        if path.startswith("postgresql+"):
            return path
        return path.replace("postgresql://", "postgresql+psycopg://", 1)
    raise ValueError(f"unsupported database type: {db_type}")


def init_db(db_type: str, path: str) -> None:
    global engine, SessionLocal
    url = make_database_url(db_type, path)
    connect_args: dict[str, Any] = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    engine = create_engine(url, connect_args=connect_args, future=True)

    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, connection_record):  # type: ignore[no-untyped-def]
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA cache_size=10000")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA auto_vacuum=INCREMENTAL")
            cursor.close()

    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True)
    run_migrations()


def close_db() -> None:
    global engine, SessionLocal
    if engine is not None:
        engine.dispose()
    engine = None
    SessionLocal = None


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    if SessionLocal is None:
        raise RuntimeError("database is not initialized")
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Generator[Session, None, None]:
    with session_scope() as session:
        yield session


def run_migrations() -> None:
    """Best-effort compatibility migrations for legacy Go database snapshots."""
    if engine is None or SessionLocal is None:
        return
    with SessionLocal() as session:
        inspector = inspect(engine)
        columns = {c["name"] for c in inspector.get_columns("channels")} if inspector.has_table("channels") else set()
        if "base_url" in columns and "base_urls" in columns:
            rows = session.execute(text("SELECT id, base_url, base_urls FROM channels")).mappings().all()
            for row in rows:
                if row.get("base_url") and not row.get("base_urls"):
                    session.execute(
                        text("UPDATE channels SET base_urls=:base_urls WHERE id=:id"),
                        {"base_urls": json.dumps([{"url": row["base_url"], "delay": 0}]), "id": row["id"]},
                    )
        if "key" in columns:
            rows = session.execute(text("SELECT id, key FROM channels WHERE key IS NOT NULL AND key != ''")).mappings().all()
            for row in rows:
                exists = session.scalar(select(ChannelKey.id).where(ChannelKey.channel_id == row["id"]).limit(1))
                if not exists:
                    session.add(ChannelKey(channel_id=row["id"], enabled=True, channel_key=row["key"], remark="legacy"))
        # Convert numeric channel types if old data is present.
        mapping = {
            "0": "openai/chat_completions",
            "1": "openai/responses",
            "2": "anthropic/messages",
            "3": "gemini/contents",
            "4": "doubao",
            "5": "openai/embeddings",
        }
        for old, new in mapping.items():
            session.execute(text("UPDATE channels SET type=:new WHERE type=:old"), {"new": new, "old": old})
        session.merge(MigrationRecord(version="python-bootstrap", status="success"))
        session.commit()


def clear_all_runtime_tables(session: Session) -> None:
    for model in [RelayLog, StatsAPIKey, StatsChannel, StatsModel, StatsHourly, StatsDaily, StatsTotal]:
        session.execute(delete(model))
