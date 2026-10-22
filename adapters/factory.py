"""Hedef adaptörü fabrikası — Spesifikasyon §3.

CLI ve notebook'un tek bir yapılandırma nesnesinden adaptör kurmasını sağlar.
Spesifikasyon §3 uzun vadede giriş noktası (entry point) tabanlı kayıt öngörür;
Faz 1'de yerleşik adaptörler için açık bir fabrika yeterlidir.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from adapters.anthropic import AnthropicAdapter
from adapters.base import RateLimitPolicy, TargetAdapter
from adapters.http_generic import HTTPGenericAdapter
from adapters.openai_compat import DEFAULT_OLLAMA_BASE_URL, OpenAICompatAdapter
from adapters.stub import StubAdapter

__all__ = ["TARGET_KINDS", "TargetOptions", "build_adapter"]

TARGET_KINDS = ("anthropic", "openai-compat", "ollama", "http-generic", "stub")


@dataclass(frozen=True, slots=True)
class TargetOptions:
    """Bir hedefi tanımlayan, taşıyıcıdan bağımsız seçenekler."""

    kind: str
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    config_path: str | None = None
    timeout: float | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    requests_per_second: float | None = None
    max_concurrency: int | None = None

    def rate_limit_policy(self) -> RateLimitPolicy | None:
        """Oran sınırı geçersiz kılmaları verildiyse politikayı kurar."""
        if self.requests_per_second is None and self.max_concurrency is None:
            return None
        default = RateLimitPolicy()
        return RateLimitPolicy(
            requests_per_second=self.requests_per_second or default.requests_per_second,
            max_concurrency=self.max_concurrency or default.max_concurrency,
        )


def build_adapter(options: TargetOptions) -> TargetAdapter:
    """`TargetOptions`'tan somut bir adaptör üretir.

    Raises:
        ValueError: Bilinmeyen hedef türü veya eksik zorunlu seçenek.
    """
    policy = options.rate_limit_policy()

    match options.kind:
        case "anthropic":
            return AnthropicAdapter(
                **_optional(
                    model=options.model,
                    api_key=options.api_key,
                    base_url=options.base_url,
                    max_tokens=options.max_tokens,
                    timeout=options.timeout,
                    rate_limit_policy=policy,
                )
            )

        case "openai-compat" | "ollama":
            if not options.model:
                raise ValueError(f"{options.kind} hedefi için --model zorunludur")

            common = _optional(
                api_key=options.api_key,
                max_tokens=options.max_tokens,
                temperature=options.temperature,
                timeout=options.timeout,
                rate_limit_policy=policy,
            )
            if options.kind == "ollama":
                return OpenAICompatAdapter.for_ollama(
                    options.model,
                    base_url=options.base_url or DEFAULT_OLLAMA_BASE_URL,
                    **common,
                )
            return OpenAICompatAdapter(
                model=options.model,
                **_optional(base_url=options.base_url),
                **common,
            )

        case "http-generic":
            if not options.config_path:
                raise ValueError("http-generic hedefi için --config zorunludur")
            return HTTPGenericAdapter.from_config_file(
                options.config_path, **_optional(timeout=options.timeout)
            )

        case "stub":
            return StubAdapter()

        case _:
            raise ValueError(
                f"bilinmeyen hedef türü: {options.kind!r} (geçerli: {', '.join(TARGET_KINDS)})"
            )


def _optional(**kwargs: Any) -> dict[str, Any]:
    """None olmayan anahtar-değer çiftlerini süzer.

    Böylece verilmeyen CLI bayrakları adaptörün kendi varsayılanını ezmez.
    """
    return {key: value for key, value in kwargs.items() if value is not None}
