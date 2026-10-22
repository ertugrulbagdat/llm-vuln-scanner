"""Yapılandırılabilir genel HTTP adaptörü — Spesifikasyon §3.

Herhangi bir REST uç noktasını, istek şemasını ve yanıt yolunu bir
yapılandırma dosyasında tanımlayarak hedefler. Kod yazmadan yeni bir hedef
eklemenin yolu budur.

Örnek yapılandırma (YAML):

    name: musteri-asistani
    url: https://api.ornek.com/v1/chat
    method: POST
    headers:
      authorization: "Bearer ${MUSTERI_API_KEY}"
      content-type: application/json
    body_template:
      session_id: "tarama"
      system_prompt: "{{system}}"
      message: "{{prompt}}"
    response_path: data.reply.text
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import httpx
import yaml

from adapters.base import PermanentTargetError, RateLimitPolicy, TargetResponse, Turn
from adapters.http_base import HTTPAdapterBase

__all__ = ["HTTPGenericAdapter", "resolve_path"]

PLACEHOLDER_RE = re.compile(r"\{\{\s*(prompt|system)\s*\}\}")
ENV_RE = re.compile(r"\$\{(\w+)\}")
_PATH_SEGMENT_RE = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


def resolve_path(data: Any, path: str) -> Any:
    """Nokta/köşeli parantez yoluyla iç içe JSON'dan değer çeker.

    `"choices[0].message.content"` gibi yolları destekler.

    Raises:
        PermanentTargetError: Yol yanıt gövdesinde bulunamazsa.
    """
    current = data
    for name, index in _PATH_SEGMENT_RE.findall(path):
        try:
            current = current[int(index)] if index else current[name]
        except (KeyError, IndexError, TypeError) as exc:
            raise PermanentTargetError(
                f"yanıt yolu {path!r} çözümlenemedi ({name or index!r} adımında): {exc}"
            ) from exc
    return current


def _expand_env(value: str) -> str:
    """`${DEGISKEN}` kalıplarını ortam değişkenleriyle değiştirir (§11)."""

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        resolved = os.getenv(name)
        if resolved is None:
            raise ValueError(f"yapılandırmada geçen {name} ortam değişkeni tanımlı değil")
        return resolved

    return ENV_RE.sub(replace, value)


class HTTPGenericAdapter(HTTPAdapterBase):
    """İstek şeması ve yanıt yolu yapılandırılabilen REST adaptörü."""

    def __init__(
        self,
        *,
        name: str,
        url: str,
        body_template: dict[str, Any],
        response_path: str,
        method: str = "POST",
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
        rate_limit_policy: RateLimitPolicy | None = None,
        multi_turn: bool = False,
        client: httpx.Client | None = None,
    ) -> None:
        """
        Args:
            body_template: `{{prompt}}` ve `{{system}}` yer tutucularını içeren,
                iç içe olabilen istek gövdesi.
            response_path: Yanıt metninin bulunduğu yol, ör. `data.reply.text`.
            headers: Değerlerinde `${ORTAM_DEGISKENI}` kullanılabilir.

        Raises:
            ValueError: Gövde şablonu `{{prompt}}` içermiyorsa veya bir ortam
                değişkeni tanımsızsa.
        """
        super().__init__(timeout=timeout, rate_limit_policy=rate_limit_policy, client=client)

        if not self._contains_prompt(body_template):
            raise ValueError("body_template en az bir {{prompt}} yer tutucusu içermeli")

        self._adapter_name = name
        self._url = url
        self._method = method.upper()
        self._headers = {k: _expand_env(v) for k, v in (headers or {}).items()}
        self._body_template = body_template
        self._response_path = response_path
        self._multi_turn = multi_turn

    @classmethod
    def from_config_file(
        cls, path: str | Path, *, client: httpx.Client | None = None, **overrides: Any
    ) -> HTTPGenericAdapter:
        """YAML veya JSON yapılandırma dosyasından adaptör kurar.

        Raises:
            ValueError: Dosya ayrıştırılamazsa veya zorunlu alanlar eksikse.
        """
        config_path = Path(path)
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"{config_path}: yapılandırma ayrıştırılamadı — {exc}") from exc

        if not isinstance(raw, dict):
            raise ValueError(f"{config_path}: kök öğe bir sözlük olmalı")

        missing = {"name", "url", "body_template", "response_path"} - raw.keys()
        if missing:
            raise ValueError(f"{config_path}: zorunlu alan(lar) eksik: {sorted(missing)}")

        rate_limit = raw.pop("rate_limit", None)
        if rate_limit is not None:
            raw["rate_limit_policy"] = RateLimitPolicy(**rate_limit)

        return cls(client=client, **{**raw, **overrides})

    @property
    def name(self) -> str:
        return self._adapter_name

    def supports_multi_turn(self) -> bool:
        return self._multi_turn

    def send(
        self,
        prompt: str,
        *,
        system: str | None = None,
        conversation: list[Turn] | None = None,
    ) -> TargetResponse:
        if conversation and not self.supports_multi_turn():
            raise PermanentTargetError(
                f"{self.name}: çok turlu konuşma desteklenmiyor (multi_turn=False)"
            )

        payload = self._render(self._body_template, prompt=prompt, system=system or "")
        if not isinstance(payload, dict):  # pragma: no cover — yapıcı doğruluyor
            raise PermanentTargetError(f"{self.name}: gövde şablonu bir nesneye çözülmedi")

        body, latency_ms = self._post_json(self._url, headers=self._headers, payload=payload)
        text = resolve_path(body, self._response_path)

        if not isinstance(text, str):
            raise PermanentTargetError(
                f"{self.name}: {self._response_path!r} bir metin değil ({type(text).__name__})"
            )

        return TargetResponse(text=text, latency_ms=latency_ms, model=self.name, raw=body)

    @classmethod
    def _render(cls, node: Any, *, prompt: str, system: str) -> Any:
        """Şablon ağacındaki yer tutucuları özyinelemeli olarak doldurur."""
        match node:
            case str():
                values = {"prompt": prompt, "system": system}
                return PLACEHOLDER_RE.sub(lambda m: values[m.group(1)], node)
            case dict():
                return {k: cls._render(v, prompt=prompt, system=system) for k, v in node.items()}
            case list():
                return [cls._render(v, prompt=prompt, system=system) for v in node]
            case _:
                return node

    @staticmethod
    def _contains_prompt(template: dict[str, Any]) -> bool:
        """Şablonun bir yerinde `{{prompt}}` geçip geçmediğini denetler."""
        return "prompt" in {
            match.group(1) for match in PLACEHOLDER_RE.finditer(json.dumps(template))
        }
