"""Anthropic Messages API adaptörü — Spesifikasyon §3.

Sağlayıcı SDK'sı yerine ham HTTP (`httpx`) kullanılır: §10, tüm adaptörlerin
tek bir kütüphaneyle çalışmasını ve sağlayıcı SDK'larının isteğe bağlı
kalmasını şart koşar.
"""

from __future__ import annotations

from typing import Any

import httpx

from adapters.base import PermanentTargetError, RateLimitPolicy, TargetResponse, Turn
from adapters.http_base import HTTPAdapterBase, read_api_key

__all__ = ["ANTHROPIC_VERSION", "DEFAULT_ANTHROPIC_MODEL", "AnthropicAdapter"]

DEFAULT_BASE_URL = "https://api.anthropic.com"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5"
ANTHROPIC_VERSION = "2023-06-01"
API_KEY_ENV = "ANTHROPIC_API_KEY"

DEFAULT_MAX_TOKENS = 2048
"""Tarama yanıtları kısadır ve koşu başına yüzlerce istek atılır; düşük tavan
bilinçli bir maliyet kontrolü tercihidir (§5)."""


class AnthropicAdapter(HTTPAdapterBase):
    """Anthropic Messages API'sini hedefleyen adaptör.

    Örnek:
        >>> adapter = AnthropicAdapter(model="claude-opus-5")   # doctest: +SKIP
        >>> adapter.send("Merhaba").text                        # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: float = 60.0,
        rate_limit_policy: RateLimitPolicy | None = None,
        extra_body: dict[str, Any] | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        """
        Args:
            model: Hedeflenecek model kimliği.
            api_key: Verilmezse `ANTHROPIC_API_KEY` ortam değişkeninden okunur (§11).
            extra_body: İstek gövdesine eklenecek sağlayıcıya özgü alanlar
                (ör. `{"thinking": {"type": "disabled"}}`, `{"output_config": ...}`).

        Raises:
            ValueError: API anahtarı bulunamazsa.
        """
        super().__init__(timeout=timeout, rate_limit_policy=rate_limit_policy, client=client)

        key = read_api_key(API_KEY_ENV, api_key)
        if not key:
            raise ValueError(
                f"Anthropic API anahtarı bulunamadı. {API_KEY_ENV} ortam değişkenini "
                f".env üzerinden ayarlayın (§11)."
            )

        self._api_key = key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._max_tokens = max_tokens
        self._extra_body = dict(extra_body or {})

    @property
    def name(self) -> str:
        return f"anthropic:{self._model}"

    def supports_multi_turn(self) -> bool:
        return True

    def send(
        self,
        prompt: str,
        *,
        system: str | None = None,
        conversation: list[Turn] | None = None,
    ) -> TargetResponse:
        messages: list[dict[str, str]] = [
            {"role": turn.role.value, "content": turn.content}
            for turn in (conversation or ())
            if turn.role.value != "system"
        ]
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": messages,
            **self._extra_body,
        }
        if system is not None:
            payload["system"] = system

        body, latency_ms = self._post_json(
            f"{self._base_url}/v1/messages",
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            payload=payload,
        )

        usage = body.get("usage") or {}
        stop_reason = body.get("stop_reason")

        return TargetResponse(
            text=self._extract_text(body),
            latency_ms=latency_ms,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            model=body.get("model", self._model),
            raw=body,
            metadata={"stop_reason": stop_reason} if stop_reason else {},
        )

    @staticmethod
    def _extract_text(body: dict[str, Any]) -> str:
        """Yanıttaki `text` bloklarını birleştirir.

        `content` bir blok listesidir; `thinking` gibi metin dışı bloklar
        atlanır. Güvenlik sınıflandırıcısı isteği reddettiğinde (`stop_reason`
        `"refusal"`) liste boş gelebilir — bu durumda boş metin döner ve
        reddetme bilgisi `metadata["stop_reason"]` üzerinden analiz katmanına
        taşınır.
        """
        content = body.get("content")
        if not isinstance(content, list):
            raise PermanentTargetError("Anthropic yanıtında 'content' listesi yok")

        return "".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "text" and "text" in block
        )
