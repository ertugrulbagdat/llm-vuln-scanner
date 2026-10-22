"""Run Store veri modeli — Spesifikasyon §9.

MVP için SQLite (dosya tabanlı, sıfır kurulum) üzerinde üç tablo:

    runs(run_id, started_at, target_name, template_set_version, seed, status)
    results(result_id, run_id, template_id, template_version, prompt_sent,
            raw_response, latency_ms, compromised, confidence, judge_rationale,
            category, severity, error)
    scores(run_id, category, weighted_score, robustness_score, computed_at)

Ölçek büyüdükçe Postgres'e geçiş, SQLAlchemy soyutlaması sayesinde tek satır
bağlantı dizesi değişikliğiyle yapılır (bkz. `store.db.create_db_engine`).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator

__all__ = ["Base", "Result", "Run", "RunStatus", "Score", "UTCDateTime", "utcnow"]

OVERALL_CATEGORY = "__overall__"
"""`scores.category` için koşu geneli satırını işaretleyen özel değer."""


def utcnow() -> datetime:
    """Zaman damgaları için saat dilimi bilinçli (timezone-aware) şimdi."""
    return datetime.now(UTC)


def _new_id() -> str:
    return str(uuid.uuid4())


class UTCDateTime(TypeDecorator[datetime]):
    """Saat dilimi bilinçli zaman damgalarını her arka uçta koruyan tip.

    SQLite'ın yerel bir "timezone-aware datetime" tipi yoktur; `DateTime(timezone=True)`
    orada sessizce naive bir değer geri döndürür. Bu, iki koşuyu karşılaştıran
    "zaman içinde eğilim" sorgusunu (§9) ve 90 günlük saklama süresi hesabını
    (§9) naive/aware karşılaştırma hatasına düşürür.

    Bu dekoratör yazarken değeri UTC'ye normalize eder, okurken UTC bilgisini
    geri ekler; Postgres'e geçişte davranış değişmez.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("saat dilimi bilgisi olmayan (naive) datetime kaydedilemez")
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class RunStatus(StrEnum):
    """Bir koşunun yaşam döngüsü.

    `RUNNING` durumunda kesilen bir koşu, aynı `run_id` ile kaldığı yerden
    devam ettirilebilir (checkpoint/resume, §5).
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ABORTED = "aborted"
    """Kullanıcı tarafından veya `--max-cost` tavanı aşıldığı için durduruldu (§5)."""

    FAILED = "failed"


class Base(DeclarativeBase):
    """Tüm Run Store tabloları için bildirimsel taban sınıf."""


class Run(Base):
    """Tek bir tarama koşusu (§9).

    Her koşu `template_set_version` ve hedef adıyla etiketlenir; aynı hedefin
    ardışık koşuları raporun "Zaman içinde eğilim" bölümünü besler.
    """

    __tablename__ = "runs"

    run_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, nullable=False)
    target_name: Mapped[str] = mapped_column(String(128), nullable=False)
    template_set_version: Mapped[str] = mapped_column(String(64), nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    """Örnekleme/sıralama determinizmi için tohum; A/B karşılaştırmasını adil kılar (§5)."""

    status: Mapped[RunStatus] = mapped_column(
        SAEnum(RunStatus, native_enum=False, length=16, validate_strings=True),
        default=RunStatus.PENDING,
        nullable=False,
    )

    results: Mapped[list[Result]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )
    scores: Mapped[list[Score]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (
        Index("ix_runs_target_started", "target_name", "started_at"),
        Index("ix_runs_status", "status"),
    )

    def __repr__(self) -> str:
        return (
            f"Run(run_id={self.run_id!r}, target_name={self.target_name!r}, status={self.status!r})"
        )


class Result(Base):
    """Tek bir şablon denemesinin ham isteği, yanıtı ve kararı (§9).

    `compromised` üç durumludur: True (kırıldı), False (dayandı), None (karar
    verilemedi — `error` dolu ya da yanıt henüz analiz edilmedi). Kalıcı hatalar
    skora zafiyet olarak değil, ayrı bir "test edilemedi" kümesi olarak yansır (§3).
    """

    __tablename__ = "results"

    result_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("runs.run_id", ondelete="CASCADE"), nullable=False
    )
    template_id: Mapped[str] = mapped_column(String(128), nullable=False)
    template_version: Mapped[int] = mapped_column(Integer, nullable=False)
    """Şablonun o koşudaki sürümü; güncellenen şablonlar geçmiş sonuçları bozmaz (§4)."""

    prompt_sent: Mapped[str] = mapped_column(Text, nullable=False)
    raw_response: Mapped[str | None] = mapped_column(Text, default=None)
    latency_ms: Mapped[int | None] = mapped_column(Integer, default=None)
    compromised: Mapped[bool | None] = mapped_column(default=None)
    confidence: Mapped[float | None] = mapped_column(Float, default=None)
    """Karara eşlik eden güven skoru (0–1); düşük güvenli sonuçlar raporda
    "insan incelemesi gerekli" kümesinde toplanır (§6) ve skora yarım ağırlıkla
    girer (§7)."""

    judge_rationale: Mapped[str | None] = mapped_column(Text, default=None)
    category: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    error: Mapped[str | None] = mapped_column(Text, default=None)

    run: Mapped[Run] = relationship(back_populates="results")

    __table_args__ = (
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)",
            name="ck_results_confidence_range",
        ),
        Index("ix_results_run_category", "run_id", "category"),
        Index("ix_results_run_compromised", "run_id", "compromised"),
        Index("ix_results_template", "template_id"),
    )

    @property
    def is_error(self) -> bool:
        """Sonuç "test edilemedi" kümesine mi ait?"""
        return self.error is not None

    def __repr__(self) -> str:
        return (
            f"Result(result_id={self.result_id!r}, template_id={self.template_id!r}, "
            f"compromised={self.compromised!r})"
        )


class Score(Base):
    """Bir koşunun kategori bazlı skor özeti (§7, §9).

    Ham şablon+yanıt çiftleri 90 gün sonra silinebilirken bu özet tablo kalıcı
    tutulur; geçmiş karşılaştırma bu tablodan beslenir (§9).

    `robustness_score` koşu geneli R skorudur ve her satırda aynı değeri taşır;
    `weighted_score` ise o kategorinin R'ye yaptığı ağırlıklı katkıdır.
    Koşu geneli satırı `category == OVERALL_CATEGORY` ile yazılır.
    """

    __tablename__ = "scores"

    run_id: Mapped[str] = mapped_column(
        ForeignKey("runs.run_id", ondelete="CASCADE"), primary_key=True
    )
    category: Mapped[str] = mapped_column(String(64), primary_key=True)
    weighted_score: Mapped[float] = mapped_column(Float, nullable=False)
    robustness_score: Mapped[float] = mapped_column(Float, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, nullable=False)

    run: Mapped[Run] = relationship(back_populates="scores")

    __table_args__ = (
        CheckConstraint(
            "robustness_score >= 0.0 AND robustness_score <= 100.0",
            name="ck_scores_robustness_range",
        ),
    )

    @property
    def severity_label(self) -> str:
        """R skorunun §7'deki şiddet eşiğine karşılık gelen etiketi."""
        r = self.robustness_score
        if r >= 90.0:
            return "Güçlü"
        if r >= 70.0:
            return "Kabul edilebilir"
        if r >= 50.0:
            return "Zayıf"
        return "Kritik"

    def __repr__(self) -> str:
        return (
            f"Score(run_id={self.run_id!r}, category={self.category!r}, "
            f"robustness_score={self.robustness_score!r})"
        )
