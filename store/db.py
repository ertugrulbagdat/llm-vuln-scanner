"""Run Store bağlantı yönetimi — Spesifikasyon §9, §10.

Varsayılan hedef SQLite'tır (sıfır kurulum). Postgres'e geçiş yalnızca
`DEFAULT_DB_URL`'i ya da `create_db_engine(url=...)` argümanını değiştirmeyi
gerektirir; şema kodu aynı kalır.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from store.models import Base

__all__ = [
    "DEFAULT_DB_URL",
    "create_db_engine",
    "create_session_factory",
    "init_db",
    "session_scope",
]

DEFAULT_DB_URL = "sqlite:///llm_vuln_scanner.db"
"""Ham çıktılar yalnızca yerel dosya sisteminde tutulur, asla uzak bir servise
otomatik yüklenmez (§11)."""


def create_db_engine(url: str | None = None, *, echo: bool = False) -> Engine:
    """Bir SQLAlchemy `Engine` üretir.

    Args:
        url: Bağlantı dizesi. Verilmezse `LLM_SCANNER_DB_URL` ortam değişkeni,
            o da yoksa `DEFAULT_DB_URL` kullanılır.
        echo: SQL ifadelerini log'lar (hata ayıklama).
    """
    resolved = url or os.getenv("LLM_SCANNER_DB_URL") or DEFAULT_DB_URL
    engine = create_engine(resolved, echo=echo, future=True)

    if engine.dialect.name == "sqlite":
        _enable_sqlite_pragmas(engine)

    return engine


def _enable_sqlite_pragmas(engine: Engine) -> None:
    """SQLite'ta yabancı anahtarları ve eşzamanlı yazım için WAL'ı açar.

    `foreign_keys` SQLite'ta bağlantı başına varsayılan olarak KAPALIdır; açık
    olmadan `ondelete="CASCADE"` sessizce çalışmaz. WAL, motorun paralel
    worker'ları sonuç yazarken okuma/yazma çakışmasını azaltır (§5).

    `busy_timeout` olmadan, eş zamanlı worker'lar aynı anda sonuç yazmaya
    çalıştığında SQLite beklemek yerine anında "database is locked" fırlatır.
    """

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_connection: Any, _connection_record: Any) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=10000")
        finally:
            cursor.close()


def init_db(engine: Engine) -> None:
    """Eksik tabloları oluşturur. Var olan tablolara dokunmaz."""
    Base.metadata.create_all(engine)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Verilen engine'e bağlı bir oturum fabrikası döndürür."""
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """Başarıda commit, hatada rollback yapan oturum bağlamı."""
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
