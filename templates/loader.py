"""Şablon kütüphanesi yükleyici — Spesifikasyon §2 (Template Library), §4.

Dizin yapısı: `templates/<category>/<id>.yaml`. Yükleyici tüm ağacı tarar,
her dosyayı `Template` şemasına göre doğrular ve kategori/şiddet/etikete göre
filtrelenebilir bir liste döndürür.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import yaml

from templates.schema import Category, Severity, Template, TemplateValidationError

__all__ = ["LIBRARY_ROOT", "filter_templates", "load_template_file", "load_library"]

LIBRARY_ROOT = Path(__file__).parent
"""YAML kütüphanesinin kökü — bu paketin kendi dizini."""


def load_template_file(path: Path) -> Template:
    """Tek bir YAML dosyasını yükleyip doğrular.

    Raises:
        TemplateValidationError: Dosya ayrıştırılamazsa veya şemaya uymazsa.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise TemplateValidationError(f"{path}: YAML ayrıştırılamadı — {exc}") from exc

    if not isinstance(raw, dict):
        raise TemplateValidationError(f"{path}: kök öğe bir sözlük olmalı")

    return Template.from_dict(raw, origin=str(path))


def load_library(root: Path | None = None) -> list[Template]:
    """Kütüphanedeki tüm şablonları yükler, `id`'ye göre sıralı döndürür.

    Raises:
        TemplateValidationError: Herhangi bir dosya geçersizse veya aynı `id`
            iki dosyada birden tanımlanmışsa.
    """
    search_root = root or LIBRARY_ROOT
    by_id: dict[str, tuple[Template, Path]] = {}

    for path in sorted(search_root.rglob("*.yaml")):
        template = load_template_file(path)
        if existing := by_id.get(template.id):
            raise TemplateValidationError(
                f"{path}: {template.id!r} kimliği {existing[1]} dosyasında zaten tanımlı"
            )
        by_id[template.id] = (template, path)

    return [template for template, _ in by_id.values()]


def filter_templates(
    templates: Iterable[Template],
    *,
    categories: Sequence[Category] | None = None,
    severities: Sequence[Severity] | None = None,
    tags: Sequence[str] | None = None,
    exclude_tags: Sequence[str] | None = None,
) -> list[Template]:
    """Şablonları kategoriye, şiddete ve etikete göre süzer (§4).

    Birden çok ölçüt verildiğinde hepsi birden sağlanmalıdır (VE); `tags` için
    şablonun etiketlerinden en az biri eşleşmelidir (VEYA). `exclude_tags`
    etiketlerinden herhangi birini taşıyan şablon, diğer ölçütlere uysa bile
    elenir (ör. `automated_import` → yalnız çekirdek Faz 1 kütüphanesi).
    """
    wanted_categories = set(categories) if categories else None
    wanted_severities = set(severities) if severities else None
    wanted_tags = {tag.strip().lower() for tag in tags} if tags else None
    excluded_tags = {tag.strip().lower() for tag in exclude_tags} if exclude_tags else set()

    return [
        template
        for template in templates
        if (wanted_categories is None or template.category in wanted_categories)
        and (wanted_severities is None or template.severity in wanted_severities)
        and (wanted_tags is None or wanted_tags & set(template.tags))
        and not excluded_tags & set(template.tags)
    ]
