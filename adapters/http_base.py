"""HTTP tabanlı adaptörler için ortak taban — Spesifikasyon §3, §10.

Spesifikasyon §10 tüm adaptörlerin tek bir HTTP kütüphanesi (`httpx`) üzerinden
çalışmasını, sağlayıcı SDK'larının isteğe bağlı olmasını öngörür. Bu modül
istek gönderme, süre ölçümü ve — en önemlisi — HTTP hatalarının `Transient` /
`Permanent` ayrımına eşlenmesini tek yerde toplar.
"""

from __future__ import annotations

import os
import time
from abc import abstractmethod
from typing import Any

import httpx

from adapters.base import (
    PermanentTargetError,
    RateLimitPolicy,
    TargetAdapter,
    TransientTargetError,
)

__all__ = ["HTTPAdapterBase", "read_api_key"]

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
"""Motorun üstel geri çekilmeyle tekrar denemesi gereken durum kodları (§3)."""


def read_api_key(env_var: str, explicit: str | None = None) -> str | None:
    """API anahtarını okur: önce açık argüman, sonra ortam değişkeni.

    Anahtarlar yalnızca `.env`/ortam üzerinden okunur, koda gömülmez (§11).
    """
    return explicit if explicit is not None else os.getenv(env_var)


class HTTPAdapterBase(TargetAdapter):
    """`httpx` ile konuşan adaptörler için ortak davranış.

    Alt sınıflar `name`, `send` ve istek gövdesini kurar; bu sınıf istemci
    yaşam döngüsünü, gecikme ölçümünü ve hata sınıflandırmasını üstlenir.
    """

    def __init__(
        self,
        *,
        timeout: float = 60.0,
        rate_limit_policy: RateLimitPolicy | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        """
        Args:
            timeout: Tek bir isteğin saniye cinsinden zaman aşımı.
            rate_limit_policy: Bu hedefe uygulanacak politika; verilmezse
                muhafazakâr varsayılan kullanılır (§11).
            client: Hazır bir `httpx.Client`. Testlerde `httpx.MockTransport`
                enjekte etmek için kullanılır; verilirse `timeout` yok sayılır.
        """
        self._policy = rate_limit_policy or RateLimitPolicy()
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout)

    @property
    @abstractmethod
    def name(self) -> str: ...

    def rate_limit(self) -> RateLimitPolicy:
        return self._policy

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _post_json(
        self,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], int]:
        """JSON gövdeli bir POST atar; (ayrıştırılmış yanıt, gecikme_ms) döndürür.

        Raises:
            TransientTargetError: Zaman aşımı, bağlantı hatası, 429 veya 5xx.
            PermanentTargetError: Diğer 4xx'ler ve ayrıştırılamayan gövde.
        """
        started = time.perf_counter()
        try:
            response = self._client.post(url, headers=headers, json=payload)
        except httpx.TimeoutException as exc:
            raise TransientTargetError(f"{self.name}: zaman aşımı — {exc}") from exc
        except httpx.TransportError as exc:
            # Bağlantı reddi/DNS hatası da geçici sayılır: yerel bir Ollama
            # sunucusu model yüklerken kısa süre bağlantı reddedebilir.
            raise TransientTargetError(f"{self.name}: bağlantı hatası — {exc}") from exc

        latency_ms = int((time.perf_counter() - started) * 1000)
        self._raise_for_status(response)

        try:
            body = response.json()
        except ValueError as exc:
            raise PermanentTargetError(
                f"{self.name}: yanıt JSON olarak ayrıştırılamadı — {exc}"
            ) from exc

        if not isinstance(body, dict):
            raise PermanentTargetError(f"{self.name}: yanıt kök öğesi bir nesne değil")

        return body, latency_ms

    def _raise_for_status(self, response: httpx.Response) -> None:
        """HTTP durum kodunu geçici/kalıcı hata ayrımına eşler (§3)."""
        if response.is_success:
            return

        detail = self._error_detail(response)

        if response.status_code in RETRYABLE_STATUS:
            raise TransientTargetError(
                f"{self.name}: HTTP {response.status_code} — {detail}",
                retry_after=self._retry_after(response),
            )

        raise PermanentTargetError(f"{self.name}: HTTP {response.status_code} — {detail}")

    @staticmethod
    def _retry_after(response: httpx.Response) -> float | None:
        """`Retry-After` başlığını saniyeye çevirir; yoksa veya bozuksa None."""
        raw = response.headers.get("retry-after")
        if raw is None:
            return None
        try:
            return max(0.0, float(raw))
        except ValueError:
            return None  # HTTP-date biçimi; üstel geri çekilmeye düşülür

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        """Hata gövdesinden kısa, kimlik bilgisi içermeyen bir özet çıkarır."""
        try:
            body = response.json()
        except ValueError:
            return response.text[:200]

        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                return str(error.get("message") or error.get("type") or error)[:200]
            if error is not None:
                return str(error)[:200]
        return str(body)[:200]
