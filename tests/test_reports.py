"""HTML rapor üreticisi testleri (§8)."""

from __future__ import annotations

from datetime import timedelta

import pytest

from reports.html_generator import (
    MAX_EXCERPT_CHARS,
    export_report,
    generate_html,
    load_run,
    load_run_history,
    redact_text,
)
from scoring.robustness import build_score_rows, score_results
from store.db import create_db_engine, create_session_factory, init_db, session_scope
from store.models import Result, Run, RunStatus, utcnow

CANARY = "CANARY-GIZLI-9988"


@pytest.fixture
def sessions(tmp_path):
    engine = create_db_engine(f"sqlite:///{tmp_path / 'report.db'}")
    init_db(engine)
    return create_session_factory(engine)


def make_run(run_id: str = "run-1", target: str = "ollama:llama3") -> Run:
    return Run(
        run_id=run_id,
        target_name=target,
        template_set_version="0.1.0",
        seed=42,
        status=RunStatus.COMPLETED,
        started_at=utcnow(),
    )


def make_result(
    *,
    template_id: str = "t-001",
    category: str = "prompt_injection",
    severity: str = "high",
    compromised: bool | None = True,
    confidence: float | None = 0.9,
    response: str | None = "Tabii, işte istediğin şey.",
    prompt: str = "saldırı promptu",
    error: str | None = None,
    run_id: str = "run-1",
) -> Result:
    return Result(
        run_id=run_id,
        template_id=template_id,
        template_version=1,
        prompt_sent=prompt,
        raw_response=response,
        latency_ms=120,
        compromised=compromised,
        confidence=confidence,
        judge_rationale="kural tabanlı gerekçe",
        category=category,
        severity=severity,
        error=error,
    )


class TestRedactText:
    def test_uzun_metin_kirpilir(self):
        redacted = redact_text("x" * 2000, redact=True)

        assert len(redacted) < 2000
        assert "kırpıldı" in redacted

    def test_kisa_metin_kirpilmaz(self):
        assert redact_text("kısa yanıt", redact=True) == "kısa yanıt"

    def test_redakte_kapaliyken_tam_metin(self):
        text = "y" * (MAX_EXCERPT_CHARS + 500)
        assert redact_text(text, redact=False) == text

    def test_kanarya_her_zaman_maskelenir(self):
        """Paylaşılan bir rapor kanaryayı sızdırmamalı — redakte kapalıyken bile."""
        text = f"sistem promptum: {CANARY}"

        assert CANARY not in redact_text(text, redact=True, canary=CANARY)
        assert CANARY not in redact_text(text, redact=False, canary=CANARY)

    def test_yanit_yoksa_yer_tutucu(self):
        assert redact_text(None, redact=True) == "(yanıt yok)"


class TestGenerateHTML:
    def test_temel_yapi_ve_skor(self):
        results = [make_result(), make_result(template_id="t-002", compromised=False)]
        html = generate_html(make_run(), results)

        assert html.startswith("<!DOCTYPE html>")
        assert "Dayanıklılık Skoru" in html
        assert "ollama:llama3" in html
        assert "Yönetici Özeti" in html
        assert "Kategori Kırılımı" in html
        assert "Metodoloji" in html

    def test_tek_dosya_harici_bagimlilik_yok(self):
        """§8: rapor tek başına paylaşılabilir bir dosya olmalı."""
        html = generate_html(make_run(), [make_result()])

        assert "<script src=" not in html
        assert "https://cdn" not in html
        assert "<style>" in html

    def test_kirilmalar_bulgu_olarak_listelenir(self):
        html = generate_html(make_run(), [make_result(template_id="kirik-001")])

        assert "kirik-001" in html
        assert "En Kritik Bulgular" in html

    def test_kirilma_yoksa_bilgilendirme(self):
        html = generate_html(make_run(), [make_result(compromised=False)])
        assert "kırılma tespit edilmedi" in html

    def test_dusuk_guvenli_sonuc_inceleme_kumesine_gider(self):
        """§6: düşük güvenli karar otomatik zafiyet sayılmaz, ayrı kümede gösterilir."""
        results = [make_result(template_id="zayif-001", compromised=True, confidence=0.4)]
        html = generate_html(make_run(), results)

        assert "İnsan İncelemesi Gerekli" in html
        assert "zayif-001" in html

    def test_hatali_sonuclar_bulgu_olarak_gosterilmez(self):
        """Hatalı sonuç bulgu kartı üretmez; yalnızca ham veri ekinde yer alır (§3, §8)."""
        html = generate_html(make_run(), [make_result(template_id="hatali-001", error="timeout")])
        body = html.split('<a href="data:')[0]

        assert "hatali-001" not in body
        assert "kırılma tespit edilmedi" in body

    def test_kanarya_raporda_maskelenir(self):
        results = [make_result(response=f"işte: {CANARY}")]
        html = generate_html(make_run(), results, canary=CANARY)

        assert CANARY not in html
        assert "KANARYA-MASKELEND" in html

    def test_ham_veri_eki_gomulur(self):
        html = generate_html(make_run(), [make_result()])

        assert "data:application/json" in html
        assert "data:text/csv" in html

    def test_yarida_kesilen_kosu_uyarisi(self):
        run = make_run()
        run.status = RunStatus.ABORTED
        html = generate_html(run, [make_result()])

        assert "aborted" in html

    def test_skor_disaridan_verilebilir(self):
        results = [make_result(compromised=False)]
        score = score_results(results)
        html = generate_html(make_run(), results, score)

        assert "100.0" in html


class TestXSSKorumasi:
    """Rapora saldırgan kontrolündeki model çıktısı gömülür; autoescape şart."""

    def test_yanittaki_html_kacirilir(self):
        payload = "<script>alert('xss')</script>"
        html = generate_html(make_run(), [make_result(response=payload)])

        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html

    def test_promptaki_html_kacirilir(self):
        payload = '<img src=x onerror="alert(1)">'
        html = generate_html(make_run(), [make_result(prompt=payload)])

        # Etiketin tamamı kaçırıldığı için `onerror` düz metne dönüşür,
        # tarayıcı tarafından öznitelik olarak yorumlanmaz.
        assert "<img" not in html
        assert "&lt;img src=x onerror=&#34;alert(1)&#34;&gt;" in html

    def test_sablon_kimligindeki_html_kacirilir(self):
        html = generate_html(make_run(), [make_result(template_id="<b>kalın</b>")])
        assert "<b>kalın</b>" not in html

    def test_hedef_adindaki_html_kacirilir(self):
        html = generate_html(make_run(target="<script>evil</script>"), [make_result()])
        assert "<script>evil" not in html


class TestGecmisVeEgilim:
    def test_tek_kosuda_egilim_gosterilmez(self):
        html = generate_html(make_run(), [make_result()], history=[])
        assert "yeterli geçmiş koşu yok" in html

    def test_gecmis_kosular_tabloda_gosterilir(self, sessions):
        with session_scope(sessions) as session:
            for index in range(3):
                run = make_run(run_id=f"run-{index}")
                run.started_at = utcnow() - timedelta(days=3 - index)
                session.add(run)
                session.flush()
                score = score_results([make_result(run_id=run.run_id, compromised=index == 0)])
                session.add_all(build_score_rows(run.run_id, score))

        with session_scope(sessions) as session:
            history = load_run_history(session, "ollama:llama3")

        assert len(history) == 3
        assert history[0].computed_at <= history[-1].computed_at  # eskiden yeniye

        html = generate_html(make_run(run_id="run-2"), [make_result()], history=history)
        assert "Zaman İçinde Eğilim" in html
        assert "run-0"[:8] in html

    def test_onceki_kosuya_gore_degisim(self, sessions):
        from reports.html_generator import HistoryPoint

        history = [
            HistoryPoint(run_id="eski", computed_at=utcnow(), robustness_score=60.0),
            HistoryPoint(run_id="yeni", computed_at=utcnow(), robustness_score=80.0),
        ]
        html = generate_html(make_run(), [make_result(compromised=False)], history=history)

        assert "Bir önceki koşuya göre" in html

    def test_farkli_hedefin_gecmisi_karismaz(self, sessions):
        with session_scope(sessions) as session:
            for target in ("hedef-a", "hedef-b"):
                run = make_run(run_id=f"run-{target}", target=target)
                session.add(run)
                session.flush()
                session.add_all(
                    build_score_rows(run.run_id, score_results([make_result(run_id=run.run_id)]))
                )

        with session_scope(sessions) as session:
            assert len(load_run_history(session, "hedef-a")) == 1


class TestDiskeYazma:
    def test_rapor_dosyaya_yazilir(self, tmp_path):
        path = export_report(tmp_path / "alt" / "rapor.html", make_run(), [make_result()])

        assert path.exists()
        assert path.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")

    def test_store_dan_yuklenip_raporlanir(self, sessions, tmp_path):
        with session_scope(sessions) as session:
            session.add(make_run())
            session.add(make_result())

        with session_scope(sessions) as session:
            run, results = load_run(session, "run-1")
            path = export_report(tmp_path / "r.html", run, results)

        assert "t-001" in path.read_text(encoding="utf-8")

    def test_olmayan_kosu_hata(self, sessions):
        with session_scope(sessions) as session, pytest.raises(ValueError, match="bulunamadı"):
            load_run(session, "yok-boyle")
