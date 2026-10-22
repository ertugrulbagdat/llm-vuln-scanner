"""OpenAI-uyumlu `/chat/completions` adaptörü — Spesifikasyon §3.

Tek bir `base_url` değişikliğiyle OpenAI, Azure OpenAI ve yerel uç noktaları
(Ollama, vLLM, LM Studio, llama.cpp) hedefler. Yerel kurulumlar için
`OpenAICompatAdapter.for_ollama()` kısayolu vardır.
"""

from __future__ import annotations

from typing import Any

import httpx

from adapters.base import PermanentTargetError, RateLimitPolicy, TargetResponse, Turn
from adapters.http_base import HTTPAdapterBase, read_api_key

__all__ = ["DEFAULT_OLLAMA_BASE_URL", "OpenAICompatAdapter"]

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
API_KEY_ENV = "OPENAI_API_KEY"

DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.0
"""Sıfır sıcaklık varsayılanı determinizm içindir: aynı `seed` ile aynı koşunun
tekrarlanabilmesi ve iki sistem promptunun adil karşılaştırılması (§5, §7)."""


class OpenAICompatAdapter(HTTPAdapterBase):
    """OpenAI `/chat/completions` şemasını konuşan her uç noktayı hedefler.

    Örnek — yerel Ollama:
        >>> adapter = OpenAICompatAdapter.for_ollama("llama3")   # doctest: +SKIP
        >>> adapter.send("Merhaba").text                         # doctest: +SKIP

    Örnek — OpenAI:
        >>> adapter = OpenAICompatAdapter(model="gpt-4o-mini")   # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        require_api_key: bool = False,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: float = 120.0,
        rate_limit_policy: RateLimitPolicy | None = None,
        extra_body: dict[str, Any] | None = None,
        adapter_name: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        """
        Args:
            model: Model kimliği (ör. `llama3`, `gpt-4o-mini`, `qwen2.5:7b`).
            base_url: `/chat/completions`'ın ekleneceği kök; sondaki `/` önemsiz.
            api_key: Verilmezse `OPENAI_API_KEY`'den okunur. Yerel uç noktalar
                anahtar istemediği için bulunamaması hata değildir; bu durumda
                `Authorization` başlığı hiç gönderilmez.
            require_api_key: True ise anahtar bulunamadığında hata verir.
            timeout: Yerel modeller ilk istekte diske/GPU'ya yüklenirken uzun
                sürebildiği için varsayılan cömerttir.

        Raises:
            ValueError: `require_api_key=True` ve anahtar bulunamadıysa.
        """
        super().__init__(timeout=timeout, rate_limit_policy=rate_limit_policy, client=client)

        key = read_api_key(API_KEY_ENV, api_key)
        if require_api_key and not key:
            raise ValueError(
                f"API anahtarı bulunamadı. {API_KEY_ENV} ortam değişkenini .env "
                f"üzerinden ayarlayın (§11)."
            )

        self._api_key = key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._extra_body = dict(extra_body or {})
        self._adapter_name = adapter_name

    @classmethod
    def for_ollama(
        cls,
        model: str,
        *,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        **kwargs: Any,
    ) -> OpenAICompatAdapter:
        """Yerel Ollama için hazır yapılandırma.

        Anahtar gerekmez ve hedef kendi makineniz olduğu için oran sınırı
        uzak sağlayıcılara göre gevşetilir; Ollama istekleri zaten sıraya
        aldığından eş zamanlılık düşük tutulur.
        """
        kwargs.setdefault(
            "rate_limit_policy", RateLimitPolicy(requests_per_second=4.0, max_concurrency=2)
        )
        kwargs.setdefault("adapter_name", f"ollama:{model}")
        return cls(model=model, base_url=base_url, require_api_key=False, **kwargs)

    @property
    def name(self) -> str:
        return self._adapter_name or f"openai-compat:{self._model}"

    def supports_multi_turn(self) -> bool:
        return True

    def send(
        self,
        prompt: str,
        *,
        system: str | None = None,
        conversation: list[Turn] | None = None,
    ) -> TargetResponse:
        messages: list[dict[str, str]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.extend(
            {"role": turn.role.value, "content": turn.content} for turn in (conversation or ())
        )
        messages.append({"role": "user", "content": prompt})

        headers = {"content-type": "application/json"}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"

        body, latency_ms = self._post_json(
            f"{self._base_url}/chat/completions",
            headers=headers,
            payload={
                "model": self._model,
                "messages": messages,
                "max_tokens": self._max_tokens,
                "temperature": self._temperature,
                "stream": False,
                **self._extra_body,
            },
        )

        message, finish_reason = self._extract_choice(body)
        usage = body.get("usage") or {}

        return TargetResponse(
            text=message,
            latency_ms=latency_ms,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            model=body.get("model", self._model),
            raw=body,
            metadata={"finish_reason": finish_reason} if finish_reason else {},
        )

    @staticmethod
    def _extract_choice(body: dict[str, Any]) -> tuple[str, str | None]:
        """İlk `choice`'tan metni ve bitiş sebebini çıkarır."""
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise PermanentTargetError("Yanıtta 'choices' listesi boş veya yok")

        first = choices[0]
        if not isinstance(first, dict):
            raise PermanentTargetError("'choices[0]' bir nesne değil")

        message = first.get("message")
        if not isinstance(message, dict):
            raise PermanentTargetError("'choices[0].message' bir nesne değil")

        # İçerik filtresi devreye girdiğinde content null gelebilir.
        content = message.get("content")
        return (content if isinstance(content, str) else ""), first.get("finish_reason")
