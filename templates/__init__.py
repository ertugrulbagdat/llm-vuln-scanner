"""Şablon kütüphanesi — şablonları yükler, kategoriye ve etiketlere göre filtreler (§2, §4).

Bu paket hem Pydantic şemasını hem de `<category>/<id>.yaml` düzenindeki YAML
şablon kütüphanesinin kökünü barındırır.
"""

from templates.loader import LIBRARY_ROOT, filter_templates, load_library, load_template_file
from templates.schema import Category, Severity, SuccessSignal, Template, TemplateValidationError

__all__ = [
    "Category",
    "LIBRARY_ROOT",
    "Severity",
    "SuccessSignal",
    "Template",
    "TemplateValidationError",
    "filter_templates",
    "load_library",
    "load_template_file",
]
