"""Hedef adaptörleri — Spesifikasyon §3.

Faz 1 kapsamı: gerçek adaptörler (`AnthropicAdapter`, `OpenAICompatAdapter`,
`HTTPGenericAdapter`) ve bunları tek bir yapılandırmadan kuran fabrika.
`EtkinAgentAdapter` Faz 5'e bırakılmıştır.
"""

from adapters.anthropic import DEFAULT_ANTHROPIC_MODEL, AnthropicAdapter
from adapters.base import (
    PermanentTargetError,
    RateLimitPolicy,
    Role,
    TargetAdapter,
    TargetError,
    TargetResponse,
    TransientTargetError,
    Turn,
)
from adapters.factory import TARGET_KINDS, TargetOptions, build_adapter
from adapters.http_base import HTTPAdapterBase
from adapters.http_generic import HTTPGenericAdapter
from adapters.openai_compat import DEFAULT_OLLAMA_BASE_URL, OpenAICompatAdapter
from adapters.stub import StubAdapter, StubBehavior

__all__ = [
    "DEFAULT_ANTHROPIC_MODEL",
    "DEFAULT_OLLAMA_BASE_URL",
    "HTTPAdapterBase",
    "HTTPGenericAdapter",
    "PermanentTargetError",
    "RateLimitPolicy",
    "Role",
    "StubAdapter",
    "StubBehavior",
    "TARGET_KINDS",
    "TargetAdapter",
    "TargetError",
    "TargetOptions",
    "TargetResponse",
    "TransientTargetError",
    "Turn",
    "AnthropicAdapter",
    "OpenAICompatAdapter",
    "build_adapter",
]
