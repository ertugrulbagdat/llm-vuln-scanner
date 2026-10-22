"""Hedef adaptör arayüzü — Spesifikasyon §3.

Tüm hedefe özgü mantık bu modüldeki `TargetAdapter` soyut sınıfının arkasına
gizlenir. Fuzzing motoru, analiz ve raporlama katmanları bu arayüzün ardındaki
somut uygulamadan habersizdir; yeni bir hedef eklemek yalnızca bu sınıfı
uygulayan bir sınıf yazmak demektir (§3).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from types import TracebackType
from typing import Any, Self

__all__ = [
    "PermanentTargetError",
    "RateLimitPolicy",
    "Role",
    "TargetAdapter",
    "TargetError",
    "TargetResponse",
    "TransientTargetError",
    "Turn",
]


class Role(StrEnum):
    """Bir konuşma turunun konuşmacısı."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class Turn:
    """Çok türlü (multi-turn) senaryolarda tek bir konuşma turu (§5)."""

    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class RateLimitPolicy:
    """Bir adaptörün hedefe uygulanmasını istediği oran sınırı (§3).

    Motor bu politikayı merkezi bir token-bucket ile uygular. Varsayılanlar
    kasıtlı olarak muhafazakârdır: araç bir DoS mekanizması değildir ve hedefin
    üretim trafiğini etkilememelidir (§11).
    """

    requests_per_second: float = 1.0
    max_concurrency: int = 2
    max_retries: int = 3
    backoff_base_seconds: float = 1.0
    backoff_max_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.requests_per_second <= 0:
            raise ValueError("requests_per_second pozitif olmalı")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency en az 1 olmalı")
        if self.max_retries < 0:
            raise ValueError("max_retries negatif olamaz")
        if self.backoff_base_seconds <= 0:
            raise ValueError("backoff_base_seconds pozitif olmalı")
        if self.backoff_max_seconds < self.backoff_base_seconds:
            raise ValueError("backoff_max_seconds, backoff_base_seconds'tan küçük olamaz")

    def backoff_for(self, attempt: int) -> float:
        """`attempt` (0 tabanlı) için üstel geri çekilme süresi, saniye (§3)."""
        if attempt < 0:
            raise ValueError("attempt negatif olamaz")
        return min(self.backoff_base_seconds * (2.0**attempt), self.backoff_max_seconds)


@dataclass(frozen=True, slots=True)
class TargetResponse:
    """Hedeften dönen tek bir yanıt (§3).

    Ham metni, gecikmeyi, (varsa) token sayısını ve izlenebilirlik için
    sağlayıcıya özgü ham yanıtı taşır.
    """

    text: str
    latency_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    model: str | None = None
    raw: Any = None
    """Sağlayıcıya özgü, işlenmemiş yanıt gövdesi — yalnızca izlenebilirlik için."""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int | None:
        """Toplam token; sağlayıcı token bildirmiyorsa None."""
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return (self.input_tokens or 0) + (self.output_tokens or 0)


class TargetError(Exception):
    """Hedefle iletişimde oluşan hata için temel sınıf."""


class TransientTargetError(TargetError):
    """Geçici hata (429, zaman aşımı, 5xx) — motor üstel geri çekilmeyle tekrar dener (§3).

    Sağlayıcı bir `Retry-After` başlığı döndürdüyse `retry_after` doldurulur ve
    motor üstel geri çekilme yerine bu süreyi kullanır.
    """

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class PermanentTargetError(TargetError):
    """Kalıcı hata (401, 400, geçersiz yapılandırma).

    Motor tekrar denemez; sonuç `results.error` olarak kaydedilir ve skora
    zafiyet olarak değil, ayrı bir "test edilemedi" kategorisi olarak yansır (§3).
    """


class TargetAdapter(ABC):
    """Hedefe özgü HTTP/SDK çağrısını soyutlayan arayüz (§2, §3).

    Somut adaptörler `name`, `send` ve `rate_limit`'i uygular; çok turlu
    senaryoları destekliyorlarsa `supports_multi_turn`'ü geçersiz kılar.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Koşu kaydına yazılan hedef adı (`runs.target_name`, §9)."""

    @abstractmethod
    def send(
        self,
        prompt: str,
        *,
        system: str | None = None,
        conversation: list[Turn] | None = None,
    ) -> TargetResponse:
        """Hedefe tek bir prompt gönderir ve yanıtı döndürür.

        Args:
            prompt: Gönderilecek saldırı promptu (şablondan üretilmiş hâli).
            system: Hedef destekliyorsa kullanılacak sistem promptu.
            conversation: Çok turlu senaryolarda bu prompttan önceki turlar.

        Raises:
            TransientTargetError: Tekrar denenebilir hata.
            PermanentTargetError: Tekrar denenmemesi gereken hata.
        """

    @abstractmethod
    def rate_limit(self) -> RateLimitPolicy:
        """Bu hedefe uygulanacak oran sınırı politikası."""

    def supports_multi_turn(self) -> bool:
        """Adaptör çok turlu konuşmayı destekliyor mu (§3)."""
        return False

    def close(self) -> None:  # noqa: B027 — bilinçli olarak soyut değil
        """Açık kaynakları (HTTP istemcisi, oturum) serbest bırakır.

        İsteğe bağlı kanca: kapatacak kaynağı olmayan adaptörler (ör. stub)
        bunu uygulamak zorunda kalmasın diye soyut bırakılmamıştır.
        """

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
