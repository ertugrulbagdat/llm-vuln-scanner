"""Fuzzing motoru testleri (§5)."""

from __future__ import annotations

import threading
import time

import pytest
from sqlalchemy import select

from adapters.base import RateLimitPolicy
from adapters.stub import DEFAULT_CANARY, StubAdapter, StubBehavior
from engine.ratelimit import TokenBucket
from engine.runner import CANARY_INSTRUCTION, ScanConfig, ScanRunner, default_canary
from store.db import create_db_engine, create_session_factory, init_db, session_scope
from store.models import Result, Run, RunStatus, Score
from templates.schema import Template

FAST_POLICY = RateLimitPolicy(requests_per_second=1000.0, max_concurrency=4)


@pytest.fixture
def sessions(tmp_path):
    engine = create_db_engine(f"sqlite:///{tmp_path / 'engine.db'}")
    init_db(engine)
    return create_session_factory(engine)


def make_template(template_id: str, *, category: str = "prompt_injection", variants: int = 1):
    payload: dict[str, object] = {
        "id": template_id,
        "category": category,
        "subcategory": "test",
        "severity": "high",
        "template": "saldırı {{payload}}" if variants > 1 else "saldırı",
        "success_signal": "compliance",
    }
    if variants > 1:
        payload["variables"] = {"payload": [f"v{i}" for i in range(variants)]}
    return Template.model_validate(payload)


def runner(adapter, sessions) -> ScanRunner:
    """Geri çekilme uykusunu devre dışı bırakan hızlı motor."""
    return ScanRunner(adapter, sessions, sleep=lambda _: None)


def fast_stub(**kwargs) -> StubAdapter:
    kwargs.setdefault("rate_limit_policy", FAST_POLICY)
    return StubAdapter(**kwargs)


class TestTokenBucket:
    def test_gecersiz_hiz(self):
        with pytest.raises(ValueError, match="rate"):
            TokenBucket(0)

    def test_kapasiteyi_asan_istek(self):
        with pytest.raises(ValueError, match="kapasite"):
            TokenBucket(1.0, capacity=1.0).acquire(5.0)

    def test_ilk_istekler_beklemez(self):
        bucket = TokenBucket(10.0, capacity=3.0)
        assert bucket.acquire() == 0.0
        assert bucket.acquire() == 0.0

    def test_kapasite_bitince_beklenir(self):
        bucket = TokenBucket(20.0, capacity=1.0)
        bucket.acquire()
        assert bucket.acquire() > 0.0

    def test_es_zamanli_kullanim_hizi_asmaz(self):
        bucket = TokenBucket(50.0, capacity=1.0)
        started = time.monotonic()

        def worker() -> None:
            for _ in range(5):
                bucket.acquire()

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        elapsed = time.monotonic() - started
        # 20 istek / 50 rps ≈ 0.4 sn (ilk token bedava): ölçüm gevşek tutuldu.
        assert elapsed >= 0.25


class TestTemelKosu:
    def test_sonuclar_diske_yazilir(self, sessions):
        templates = [make_template("t-001"), make_template("t-002")]
        outcome = runner(fast_stub(), sessions).run(templates)

        with session_scope(sessions) as session:
            results = list(session.scalars(select(Result)))

        assert outcome.executed == 2
        assert len(results) == 2
        assert {r.template_id for r in results} == {"t-001", "t-002"}
        assert all(r.prompt_sent for r in results)
        assert all(r.latency_ms is not None for r in results)

    def test_kosu_satiri_olusur_ve_tamamlanir(self, sessions):
        adapter = fast_stub()
        outcome = runner(adapter, sessions).run([make_template("t-001")])

        with session_scope(sessions) as session:
            run = session.get(Run, outcome.run_id)

        assert run is not None
        assert run.status is RunStatus.COMPLETED
        assert run.target_name == adapter.name
        assert run.seed == 42

    def test_skorlar_yazilir(self, sessions):
        outcome = runner(fast_stub(), sessions).run([make_template("t-001")])

        with session_scope(sessions) as session:
            scores = list(session.scalars(select(Score).where(Score.run_id == outcome.run_id)))

        assert scores
        assert outcome.score.robustness_score == 100.0  # stub reddediyor

    def test_sablon_genisletmesi_is_uretir(self, sessions):
        outcome = runner(fast_stub(), sessions).run([make_template("t-001", variants=3)])
        assert outcome.executed == 3

    def test_bos_sablon_listesi_hata(self, sessions):
        with pytest.raises(ValueError, match="şablon yok"):
            runner(fast_stub(), sessions).run([])

    def test_ilerleme_geri_cagrisi(self, sessions):
        seen: list[tuple[int, int]] = []
        runner(fast_stub(), sessions).run(
            [make_template(f"t-{i:03}") for i in range(3)],
            on_progress=lambda done, total, _: seen.append((done, total)),
        )

        assert len(seen) == 3
        assert {total for _, total in seen} == {3}
        assert sorted(done for done, _ in seen) == [1, 2, 3]


class TestKirilmaTespiti:
    def test_sizinti_kirilma_olarak_kaydedilir(self, sessions):
        adapter = fast_stub(default_behavior=StubBehavior.LEAK)
        config = ScanConfig(system_prompt="sen bir asistansın", canary=DEFAULT_CANARY)

        outcome = runner(adapter, sessions).run([make_template("t-001")], config)

        with session_scope(sessions) as session:
            result = session.scalars(select(Result)).one()

        assert result.compromised is True
        assert result.confidence is not None
        assert result.confidence >= 0.95
        assert outcome.score.robustness_score == 0.0

    def test_kanarya_sistem_promptuna_eklenir(self, sessions):
        adapter = fast_stub()
        canary = default_canary()
        config = ScanConfig(system_prompt="temel prompt", canary=canary)

        runner(adapter, sessions).run([make_template("t-001")], config)

        sent_system = adapter.calls[0].system
        assert sent_system is not None
        assert sent_system.startswith("temel prompt")
        assert canary in sent_system
        assert CANARY_INSTRUCTION.format(canary=canary) in sent_system

    def test_sistem_promptu_yoksa_kanarya_ekilmez(self, sessions):
        config = ScanConfig(system_prompt=None, canary=default_canary())

        assert config.effective_system_prompt() is None
        assert config.active_canary() is None


class TestHataYonetimi:
    def test_gecici_hata_tekrar_denenir(self, sessions):
        adapter = fast_stub(transient_failures=2)
        outcome = runner(adapter, sessions).run([make_template("t-001")])

        with session_scope(sessions) as session:
            result = session.scalars(select(Result)).one()

        assert result.error is None
        assert len(adapter.calls) == 3  # 2 hata + 1 başarı
        assert outcome.requests_sent == 3

    def test_tum_denemeler_basarisizsa_hata_kaydedilir(self, sessions):
        policy = RateLimitPolicy(requests_per_second=1000.0, max_concurrency=4, max_retries=2)
        adapter = StubAdapter(transient_failures=99, rate_limit_policy=policy)

        outcome = runner(adapter, sessions).run([make_template("t-001")])

        with session_scope(sessions) as session:
            result = session.scalars(select(Result)).one()

        assert result.error is not None
        assert result.compromised is None
        assert len(adapter.calls) == 3  # 1 + max_retries
        assert outcome.score.errors == 1

    def test_kalici_hata_tekrar_denenmez(self, sessions):
        adapter = fast_stub(permanent_failure="401 Unauthorized")
        runner(adapter, sessions).run([make_template("t-001")])

        with session_scope(sessions) as session:
            result = session.scalars(select(Result)).one()

        assert result.error is not None
        assert "401" in result.error
        assert len(adapter.calls) == 1

    def test_hatalar_skoru_dusurmez(self, sessions):
        adapter = fast_stub(permanent_failure="400 Bad Request")
        outcome = runner(adapter, sessions).run([make_template("t-001")])

        assert outcome.score.robustness_score == 100.0
        assert outcome.score.errors == 1


class TestDeterminizmVeOrnekleme:
    def test_ayni_seed_ayni_sira(self, sessions):
        templates = [make_template(f"t-{i:03}") for i in range(12)]
        jobs_a = ScanRunner._build_jobs(templates, ScanConfig(seed=7))
        jobs_b = ScanRunner._build_jobs(templates, ScanConfig(seed=7))

        assert [j.template.id for j in jobs_a] == [j.template.id for j in jobs_b]

    def test_farkli_seed_farkli_sira(self, sessions):
        templates = [make_template(f"t-{i:03}") for i in range(12)]
        jobs_a = ScanRunner._build_jobs(templates, ScanConfig(seed=1))
        jobs_b = ScanRunner._build_jobs(templates, ScanConfig(seed=2))

        assert [j.template.id for j in jobs_a] != [j.template.id for j in jobs_b]

    def test_kategori_basina_ornekleme(self, sessions):
        templates = [
            *[make_template(f"pi-{i}", category="prompt_injection") for i in range(5)],
            *[make_template(f"jb-{i}", category="jailbreak") for i in range(5)],
        ]
        jobs = ScanRunner._build_jobs(templates, ScanConfig(seed=3, sample_per_category=2))

        categories = [j.template.category.value for j in jobs]
        assert len(jobs) == 4
        assert categories.count("prompt_injection") == 2
        assert categories.count("jailbreak") == 2

    def test_ornekleme_mevcuttan_fazlasini_istemez(self, sessions):
        templates = [make_template("t-001")]
        jobs = ScanRunner._build_jobs(templates, ScanConfig(sample_per_category=10))
        assert len(jobs) == 1

    def test_ornekleme_deterministik(self, sessions):
        templates = [make_template(f"t-{i:03}") for i in range(10)]
        a = ScanRunner._build_jobs(templates, ScanConfig(seed=9, sample_per_category=3))
        b = ScanRunner._build_jobs(templates, ScanConfig(seed=9, sample_per_category=3))

        assert [j.prompt for j in a] == [j.prompt for j in b]


class TestMaliyetVeResume:
    def test_istek_tavani_kosuyu_durdurur(self, sessions):
        templates = [make_template(f"t-{i:03}") for i in range(10)]
        outcome = runner(fast_stub(), sessions).run(
            templates, ScanConfig(max_requests=3, workers=1)
        )

        assert outcome.requests_sent <= 3
        assert outcome.status is RunStatus.ABORTED

    def test_resume_tamamlanmislari_atlar(self, sessions):
        templates = [make_template(f"t-{i:03}") for i in range(6)]
        first = runner(fast_stub(), sessions).run(templates, ScanConfig(max_requests=2, workers=1))
        assert first.status is RunStatus.ABORTED

        adapter = fast_stub()
        second = runner(adapter, sessions).run(
            templates, ScanConfig(run_id=first.run_id, resume=True)
        )

        assert second.run_id == first.run_id
        assert second.skipped == first.executed
        assert second.executed == 6 - first.executed
        assert second.status is RunStatus.COMPLETED

        with session_scope(sessions) as session:
            results = list(session.scalars(select(Result).where(Result.run_id == first.run_id)))
        assert len(results) == 6

    def test_resume_run_id_zorunlu(self, sessions):
        with pytest.raises(ValueError, match="run_id zorunlu"):
            runner(fast_stub(), sessions).run([make_template("t-1")], ScanConfig(resume=True))

    def test_olmayan_kosuya_devam_edilemez(self, sessions):
        config = ScanConfig(run_id="yok-boyle", resume=True)
        with pytest.raises(ValueError, match="bulunamadı"):
            runner(fast_stub(), sessions).run([make_template("t-1")], config)


class TestEsZamanlilik:
    def test_worker_sayisi_adaptor_tavaniyla_sinirlanir(self, sessions):
        """Adaptörün max_concurrency'si, kullanıcının --workers isteğini ezer (§3)."""
        active = 0
        peak = 0
        lock = threading.Lock()

        def responder(_: str) -> str:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return "Üzgünüm, yapamam."

        adapter = StubAdapter(
            responder=responder,
            rate_limit_policy=RateLimitPolicy(requests_per_second=1000.0, max_concurrency=2),
        )
        templates = [make_template(f"t-{i:03}") for i in range(8)]
        runner(adapter, sessions).run(templates, ScanConfig(workers=8))

        assert peak <= 2

    def test_es_zamanli_yazma_veritabanini_kilitlemez(self, sessions):
        """SQLite busy_timeout olmadan bu test 'database is locked' ile düşerdi."""
        templates = [make_template(f"t-{i:03}") for i in range(20)]
        outcome = runner(fast_stub(), sessions).run(templates, ScanConfig(workers=4))

        assert outcome.executed == 20
        with session_scope(sessions) as session:
            assert len(list(session.scalars(select(Result)))) == 20
