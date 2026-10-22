"""Raporlama — HTML/PDF rapor üretir (§2, §8).

Faz 1: jinja2 ile gömülü CSS içeren, harici bağımlılığı olmayan tek dosyalık
HTML rapor. Faz 4'te weasyprint ile PDF export eklenecek.
"""

from reports.html_generator import (
    Finding,
    HistoryPoint,
    export_report,
    generate_html,
    load_run,
    load_run_history,
    redact_text,
)

__all__ = [
    "Finding",
    "HistoryPoint",
    "export_report",
    "generate_html",
    "load_run",
    "load_run_history",
    "redact_text",
]
