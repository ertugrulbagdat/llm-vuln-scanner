"""Tek dosyalık HTML rapor üreticisi — Spesifikasyon §8.

Rapor yalnızca Run Store'dan okur; bu sayede yarıda kesilen bir koşudan da
kısmi rapor üretilebilir (§2).

Hassas içerik yönetimi (§8, §11): başarılı jailbreak çıktıları varsayılan
olarak kırpılır ve kanarya belirteci maskelenir. Tam metne yalnızca yerel Run
Store üzerinden erişilir.
"""

from __future__ import annotations

import csv
import io
import json
import urllib.parse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from sqlalchemy import select
from sqlalchemy.orm import Session

from analysis.rule_based import LOW_CONFIDENCE_THRESHOLD
from scoring.robustness import RunScore, score_results, sorted_by_risk
from store.models import OVERALL_CATEGORY, Result, Run, Score, utcnow

__all__ = [
    "Finding",
    "HistoryPoint",
    "export_report",
    "generate_html",
    "load_run_history",
    "redact_text",
]

TEMPLATE_DIR = Path(__file__).parent / "templates"
TEMPLATE_NAME = "report.html.j2"

MAX_EXCERPT_CHARS = 600
"""Redakte modda prompt/yanıt alıntılarının azami uzunluğu."""

MAX_FINDINGS = 15
MAX_REVIEW_ITEMS = 10

SEVERITY_CLASSES = {
    "Güçlü": "s-strong",
    "Kabul edilebilir": "s-ok",
    "Zayıf": "s-weak",
    "Kritik": "s-critical",
}

_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


@dataclass(frozen=True, slots=True)
class Finding:
    """Raporda gösterilen tek bir bulgu."""

    template_id: str
    category: str
    severity: str
    confidence: float
    prompt: str
    response: str
    rationale: str


@dataclass(frozen=True, slots=True)
class HistoryPoint:
    """Aynı hedefin geçmiş bir koşusundaki skoru (§9)."""

    run_id: str
    computed_at: datetime
    robustness_score: float


def redact_text(text: str | None, *, redact: bool, canary: str | None = None) -> str:
    """Hassas çıktıyı raporlanabilir hâle getirir (§8, §11).

    Redakte modda metin kırpılır ve kanarya belirteci maskelenir — aksi hâlde
    paylaşılan bir rapor, gizli belirteci başka bir taramada işe yaramaz hâle
    getirirdi.
    """
    if text is None:
        return "(yanıt yok)"

    cleaned = text
    if canary:
        cleaned = cleaned.replace(canary, "«KANARYA-MASKELENDİ»")

    if not redact:
        return cleaned

    if len(cleaned) > MAX_EXCERPT_CHARS:
        trimmed = len(cleaned) - MAX_EXCERPT_CHARS
        return f"{cleaned[:MAX_EXCERPT_CHARS]}\n… [{trimmed} karakter kırpıldı]"
    return cleaned


def _bar_color(resilience: float) -> str:
    """Dayanıklılık oranını §7 eşiklerinin rengine eşler."""
    if resilience >= 0.9:
        return "#059669"
    if resilience >= 0.7:
        return "#65a30d"
    if resilience >= 0.5:
        return "#d97706"
    return "#dc2626"


def load_run_history(session: Session, target_name: str, *, limit: int = 10) -> list[HistoryPoint]:
    """Aynı hedefin son koşularının genel skorlarını eskiden yeniye döndürür (§9)."""
    rows = session.execute(
        select(Score.run_id, Score.computed_at, Score.robustness_score)
        .join(Run, Run.run_id == Score.run_id)
        .where(Run.target_name == target_name, Score.category == OVERALL_CATEGORY)
        .order_by(Score.computed_at.desc())
        .limit(limit)
    ).all()

    return [
        HistoryPoint(run_id=run_id, computed_at=computed_at, robustness_score=robustness)
        for run_id, computed_at, robustness in reversed(rows)
    ]


def _collect_findings(
    results: Sequence[Result], *, redact: bool, canary: str | None
) -> tuple[list[Finding], list[Finding]]:
    """Sonuçları "kritik bulgular" ve "insan incelemesi" kümelerine ayırır (§6, §8)."""
    findings: list[Finding] = []
    review: list[Finding] = []

    for result in results:
        if result.error is not None:
            continue

        confidence = result.confidence or 0.0
        finding = Finding(
            template_id=result.template_id,
            category=result.category,
            severity=result.severity,
            confidence=confidence,
            prompt=redact_text(result.prompt_sent, redact=redact, canary=canary),
            response=redact_text(result.raw_response, redact=redact, canary=canary),
            rationale=result.judge_rationale or "",
        )

        if result.compromised is None or confidence < LOW_CONFIDENCE_THRESHOLD:
            review.append(finding)
        elif result.compromised:
            findings.append(finding)

    # §8: en yüksek şiddetli, en yüksek güvenli kırılmalar önce gelir.
    findings.sort(key=lambda f: (_SEVERITY_ORDER.get(f.severity, 9), -f.confidence))
    review.sort(key=lambda f: -f.confidence)

    return findings[:MAX_FINDINGS], review[:MAX_REVIEW_ITEMS]


def _export_uris(results: Sequence[Result], *, redact: bool, canary: str | None) -> tuple[str, str]:
    """Ham veri ekini `data:` URI'leri olarak gömer — rapor tek dosya kalır (§8)."""
    rows = [
        {
            "result_id": r.result_id,
            "template_id": r.template_id,
            "template_version": r.template_version,
            "category": r.category,
            "severity": r.severity,
            "compromised": r.compromised,
            "confidence": r.confidence,
            "latency_ms": r.latency_ms,
            "error": r.error,
            "prompt_sent": redact_text(r.prompt_sent, redact=redact, canary=canary),
            "raw_response": redact_text(r.raw_response, redact=redact, canary=canary),
            "judge_rationale": r.judge_rationale,
        }
        for r in results
    ]

    json_payload = json.dumps(rows, ensure_ascii=False, indent=2)

    buffer = io.StringIO()
    if rows:
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def to_uri(payload: str, mime: str) -> str:
        return f"data:{mime};charset=utf-8,{urllib.parse.quote(payload)}"

    return to_uri(json_payload, "application/json"), to_uri(buffer.getvalue(), "text/csv")


def generate_html(
    run: Run,
    results: Sequence[Result],
    score: RunScore | None = None,
    *,
    history: Sequence[HistoryPoint] = (),
    redact: bool = True,
    canary: str | None = None,
) -> str:
    """Koşudan tek dosyalık HTML rapor üretir.

    Args:
        run: Raporlanacak koşu.
        results: Koşunun sonuçları.
        score: Önceden hesaplanmış skor; verilmezse sonuçlardan hesaplanır.
        history: Aynı hedefin geçmiş koşuları ("Zaman içinde eğilim", §8).
        redact: Hassas çıktıyı kırp ve kanaryayı maskele (varsayılan: evet).
        canary: Maskelenecek kanarya belirteci.
    """
    run_score = score if score is not None else score_results(results)
    findings, review_items = _collect_findings(results, redact=redact, canary=canary)
    json_uri, csv_uri = _export_uris(results, redact=redact, canary=canary)

    previous = history[-2].robustness_score if len(history) > 1 else None
    delta = None if previous is None else round(run_score.robustness_score - previous, 2)

    environment = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(default_for_string=True, default=True),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    return environment.get_template(TEMPLATE_NAME).render(
        run=run,
        score=run_score,
        categories=sorted_by_risk(run_score.categories),
        findings=findings,
        review_items=review_items,
        history=list(history),
        delta=delta,
        redacted=redact,
        canary_planted=canary is not None,
        low_confidence_threshold=LOW_CONFIDENCE_THRESHOLD,
        severity_class=SEVERITY_CLASSES.get(run_score.label, "s-weak"),
        bar_color=_bar_color,
        json_uri=json_uri,
        csv_uri=csv_uri,
        generated_at=utcnow(),
    )


def export_report(
    path: str | Path,
    run: Run,
    results: Sequence[Result],
    score: RunScore | None = None,
    **kwargs: object,
) -> Path:
    """Raporu diske yazar ve yolunu döndürür.

    Üst dizin yoksa oluşturulur.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    html = generate_html(run, results, score, **kwargs)  # type: ignore[arg-type]
    output.write_text(html, encoding="utf-8")
    return output


def load_run(session: Session, run_id: str) -> tuple[Run, list[Result]]:
    """Bir koşuyu ve sonuçlarını Run Store'dan yükler.

    Raises:
        ValueError: Koşu bulunamazsa.
    """
    run = session.get(Run, run_id)
    if run is None:
        raise ValueError(f"koşu bulunamadı: {run_id!r}")

    results = list(session.scalars(select(Result).where(Result.run_id == run_id)))
    return run, results


def iter_categories(score: RunScore) -> Iterable[str]:
    """Skorda geçen kategori adları (notebook kolaylığı)."""
    return (category.category for category in score.categories)
