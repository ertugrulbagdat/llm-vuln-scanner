"""Kuru prova: model indirmeden tüm deney hattını sahte (stub) hedeflerle çalıştırır.

Gerçek modelleri indirmeden (her biri ~8 GB) önce notebook'un, analiz
modülünün ve veritabanı yollarının çalıştığını doğrulamak için kullanılır.
Ağa çıkmaz; yanıtlar içerik taşımayan yer tutucu metinlerdir.

    python experiments/heretic_karsilastirma/0_kuru_prova.py

Çıktı: `sonuclar/kuru_prova/deney_kural.db`. Notebook'ta
`KURU_PROVA = True` yapıldığında bu dosya okunur.
"""

from __future__ import annotations

import random
import sys
from collections.abc import Callable
from pathlib import Path

PROJE_KOKU = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJE_KOKU))

from adapters.stub import StubAdapter  # noqa: E402
from cli import DEFAULT_SYSTEM_PROMPT  # noqa: E402
from engine.runner import ScanConfig, ScanRunner, default_canary  # noqa: E402
from store.db import create_db_engine, create_session_factory, init_db  # noqa: E402
from templates.loader import load_library  # noqa: E402

SISTEM_PROMPTU_DOSYASI = Path(__file__).resolve().parent / "sistem_promptu.txt"
SISTEM_PROMPTU = (
    SISTEM_PROMPTU_DOSYASI.read_text(encoding="utf-8")
    if SISTEM_PROMPTU_DOSYASI.exists()
    else DEFAULT_SYSTEM_PROMPT
)

CIKTI = Path(__file__).resolve().parent / "sonuclar" / "kuru_prova" / "deney_kural.db"

RED = "Üzgünüm, bu isteğe yardımcı olamam."
UYUM = "Tabii, işte istediğin bilgi: [SİMÜLE EDİLMİŞ YANIT — içerik yok]"
NOTR = "Siparişinizle ilgili başka bir konuda yardımcı olabilirim."


def _cevaplayici(red_olasiligi: float, uyum_olasiligi: float, seed: int) -> Callable[[str], str]:
    """Belirli oranlarda reddeden / uyan / nötr kalan sahte bir model."""
    rng = random.Random(seed)

    def cevapla(_prompt: str) -> str:
        x = rng.random()
        if x < red_olasiligi:
            return RED
        if x < red_olasiligi + uyum_olasiligi:
            return UYUM
        return NOTR

    return cevapla


def main() -> None:
    CIKTI.parent.mkdir(parents=True, exist_ok=True)
    CIKTI.unlink(missing_ok=True)
    engine = create_db_engine(f"sqlite:///{CIKTI.as_posix()}")
    init_db(engine)
    sessions = create_session_factory(engine)
    sablonlar = load_library()

    # Orijinal: çoğunlukla reddeder. Heretic: reddetme büyük ölçüde kaybolmuş.
    profiller = {
        "stub:qwen3-4b-orijinal": (0.85, 0.05),
        "stub:qwen3-4b-heretic": (0.10, 0.60),
    }
    for ad, (red, uyum) in profiller.items():
        hedef = StubAdapter(adapter_name=ad, responder=_cevaplayici(red, uyum, seed=7))
        sonuc = ScanRunner(hedef, sessions).run(
            sablonlar,
            ScanConfig(
                seed=42,
                workers=1,
                system_prompt=SISTEM_PROMPTU,
                canary=default_canary(),
            ),
        )
        print(f"{ad:<28} R={sonuc.score.robustness_score:5.1f}  ({sonuc.score.label})")

    engine.dispose()
    print(f"\nKuru prova veritabanı: {CIKTI}")


if __name__ == "__main__":
    main()
