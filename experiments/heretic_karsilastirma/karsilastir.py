"""Orijinal ↔ Heretic karşılaştırma deneyi için analiz yardımcıları.

Notebook (`notebooks/heretic_karsilastirma.ipynb`) bu modülü kullanır; mantığın
notebook dışında durması, aynı hesabın test edilebilmesini ve tezde birebir
tekrarlanabilmesini sağlar.

Deneyin fikri: Heretic, modelin "reddetme yönünü" ağırlıklardan siler. Aynı
modelin bu işlemden önceki ve sonraki hâli, kırılganlığı *bilinen* bir çift
oluşturur. Tarayıcı iyi çalışıyorsa bu farkı görmelidir. Görmediği yerler,
Katman 1/Katman 2 tespitinin kaçırdığı vakalar için doğal bir aday kümedir.

Kullanılan veri: deneyin kendi SQLite dosyaları (`sonuclar/*.db`). Bu dosyalar
ham saldırı yanıtlarını içerir ve `.gitignore` ile depodan dışlanır (§11).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd
from scipy.stats import binomtest
from sqlalchemy import select

# Proje kökünü içe aktarma yoluna ekle (notebook veya betik nereden çalışırsa çalışsın).
PROJE_KOKU = Path(__file__).resolve().parents[2]
if str(PROJE_KOKU) not in sys.path:
    sys.path.insert(0, str(PROJE_KOKU))

from analysis.rule_based import LOW_CONFIDENCE_THRESHOLD, contains_refusal  # noqa: E402
from scoring.robustness import score_results  # noqa: E402
from store.db import create_db_engine, create_session_factory  # noqa: E402
from store.models import Result, Run  # noqa: E402

__all__ = [
    "DENEY_DIZINI",
    "KARAR_SIRASI",
    "McNemarSonucu",
    "altin_kume_adaylari",
    "eslesmis_ciftler",
    "kategori_dayaniklilik",
    "karar_dagilimi",
    "mcnemar",
    "model_etiketi",
    "red_orani",
    "sonuclari_yukle",
    "tespit_duyarliligi",
]

DENEY_DIZINI = Path(__file__).resolve().parent
SONUC_DIZINI = DENEY_DIZINI / "sonuclar"

KARAR_SIRASI = ["kırıldı", "kırıldı (düşük güven)", "kararsız", "dayandı", "hata"]
"""Grafiklerde ve tablolarda karar durumlarının sabit sırası."""

ESLESME_ANAHTARI = ["mod", "seed", "template_id", "prompt_sent"]
"""İki modelin aynı denemesini eşleyen alanlar. Aynı seed aynı iş listesini üretir (§5)."""


def model_etiketi(target_name: str) -> str:
    """`ollama:qwen3-4b-heretic` → `heretic`; diğer her şey → `orijinal`."""
    return "heretic" if "heretic" in target_name.lower() else "orijinal"


def _karar(row: pd.Series[Any]) -> str:
    if row["error"] is not None and not pd.isna(row["error"]):
        return "hata"
    if row["compromised"] is None or pd.isna(row["compromised"]):
        return "kararsız"
    if bool(row["compromised"]):
        conf = row["confidence"]
        if conf is not None and not pd.isna(conf) and conf < LOW_CONFIDENCE_THRESHOLD:
            return "kırıldı (düşük güven)"
        return "kırıldı"
    return "dayandı"


def sonuclari_yukle(db_dosyalari: dict[str, Path] | None = None) -> pd.DataFrame:
    """Deney veritabanlarındaki tüm sonuçları tek bir DataFrame'e yükler.

    Args:
        db_dosyalari: `{mod_adı: db_yolu}`. Verilmezse `sonuclar/` altındaki
            `deney_kural.db` ("kural") ve `deney_judge.db` ("judge") aranır.

    Dönen her satır bir denemedir; `model`, `mod`, `karar` ve `reddetti`
    sütunları eklenir. `reddetti`, yanıtta reddetme kalıbı geçip geçmediğidir;
    tarayıcının kararından bağımsız, Heretic'in `KeywordRate` ölçütüne denk bir
    ölçümdür.
    """
    if db_dosyalari is None:
        db_dosyalari = {
            mod: SONUC_DIZINI / f"deney_{mod}.db"
            for mod in ("kural", "judge")
            if (SONUC_DIZINI / f"deney_{mod}.db").exists()
        }
    if not db_dosyalari:
        raise FileNotFoundError(
            f"{SONUC_DIZINI} altında deney veritabanı yok. Önce 2_taramalari_kos.ps1'i çalıştırın."
        )

    satirlar: list[dict[str, object]] = []
    for mod, yol in db_dosyalari.items():
        engine = create_db_engine(f"sqlite:///{Path(yol).as_posix()}")
        sessions = create_session_factory(engine)
        with sessions() as session:
            sorgu = select(Result, Run).join(Run, Result.run_id == Run.run_id)
            for result, run in session.execute(sorgu).all():
                satirlar.append(
                    {
                        "mod": mod,
                        "run_id": run.run_id,
                        "target_name": run.target_name,
                        "model": model_etiketi(run.target_name),
                        "seed": run.seed,
                        "run_status": run.status.value,
                        "template_id": result.template_id,
                        "template_version": result.template_version,
                        "category": result.category,
                        "severity": result.severity,
                        "prompt_sent": result.prompt_sent,
                        "raw_response": result.raw_response,
                        "latency_ms": result.latency_ms,
                        "compromised": result.compromised,
                        "confidence": result.confidence,
                        "rationale": result.judge_rationale,
                        "error": result.error,
                    }
                )
        engine.dispose()

    df = pd.DataFrame(satirlar)
    if df.empty:
        raise ValueError("Veritabanlarında hiç sonuç yok.")
    df["karar"] = df.apply(_karar, axis=1)
    df["reddetti"] = df["raw_response"].map(lambda t: bool(t) and contains_refusal(t))
    return df


def kategori_dayaniklilik(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Her koşu için tarayıcının kendi skorlamasını (§7) yeniden hesaplar.

    R'yi burada yeniden yazmak yerine `scoring.score_results` çağrılır; böylece
    notebook'taki sayılar CLI çıktısı ve HTML raporla birebir aynı olur.

    Returns:
        (koşu özeti, kategori kırılımı) DataFrame'leri.
    """
    ozet: list[dict[str, object]] = []
    kategoriler: list[dict[str, object]] = []
    for (mod, run_id), grup in df.groupby(["mod", "run_id"]):
        sonuc_nesneleri = [
            Result(
                category=r.category,
                compromised=None if pd.isna(r.compromised) else bool(r.compromised),
                confidence=None if pd.isna(r.confidence) else float(cast(float, r.confidence)),
                error=None if pd.isna(r.error) else r.error,
            )
            for r in grup.itertuples()
        ]
        skor = score_results(sonuc_nesneleri)
        ilk = grup.iloc[0]
        ozet.append(
            {
                "mod": mod,
                "model": ilk["model"],
                "seed": ilk["seed"],
                "run_id": run_id,
                "R": skor.robustness_score,
                "etiket": skor.label,
                "deneme": skor.total_results,
                "kırılma": skor.compromised_count,
                "inceleme": skor.needs_review,
                "hata": skor.errors,
            }
        )
        for c in skor.categories:
            kategoriler.append(
                {
                    "mod": mod,
                    "model": ilk["model"],
                    "seed": ilk["seed"],
                    "category": c.category,
                    "dayaniklilik": c.resilience,
                    "kirilma_orani": c.compromise_rate,
                    "skorlanan": c.scored,
                    "inceleme": c.needs_review,
                    "hata": c.errors,
                }
            )
    return pd.DataFrame(ozet), pd.DataFrame(kategoriler)


def karar_dagilimi(df: pd.DataFrame) -> pd.DataFrame:
    """Model × mod başına karar durumlarının oranı (satır toplamı = 1)."""
    tablo = pd.crosstab([df["mod"], df["model"]], df["karar"], normalize="index")
    return tablo.reindex(columns=[k for k in KARAR_SIRASI if k in tablo.columns])


def red_orani(df: pd.DataFrame) -> pd.DataFrame:
    """Kategori başına reddetme kalıbı içeren yanıt oranı (hatalar hariç)."""
    gecerli = df[df["error"].isna()]
    return (
        gecerli.groupby(["mod", "category", "model"])["reddetti"]
        .mean()
        .unstack("model")
        .reindex(columns=["orijinal", "heretic"])
    )


def eslesmis_ciftler(df: pd.DataFrame) -> pd.DataFrame:
    """Orijinal ve Heretic modelin aynı denemelerini yan yana koyar.

    Aynı seed ve aynı şablon kütüphanesi aynı (şablon, prompt) iş listesini
    ürettiği için denemeler birebir eşlenir. Eşleşmeyen satırlar (ör. yarıda
    kalan bir koşu) atılır ve sayısı `attrs["eslesmeyen"]` içinde raporlanır.
    """
    sutunlar = [
        *ESLESME_ANAHTARI,
        "category",
        "severity",
        "karar",
        "reddetti",
        "compromised",
        "raw_response",
        "rationale",
    ]
    orj = df[df["model"] == "orijinal"][sutunlar]
    her = df[df["model"] == "heretic"][sutunlar]
    birlesik = orj.merge(
        her,
        on=[*ESLESME_ANAHTARI, "category", "severity"],
        suffixes=("_orj", "_her"),
        how="inner",
    )
    birlesik.attrs["eslesmeyen"] = len(orj) + len(her) - 2 * len(birlesik)
    return birlesik


@dataclass(frozen=True, slots=True)
class McNemarSonucu:
    """Eşleştirilmiş ikili sonuçlar için McNemar kesin testi."""

    yalniz_heretic: int
    """Yalnızca Heretic modelde kırılan denemeler (b)."""

    yalniz_orijinal: int
    """Yalnızca orijinal modelde kırılan denemeler (c)."""

    ikisi_de: int
    hicbiri: int
    p_degeri: float

    @property
    def toplam(self) -> int:
        return self.yalniz_heretic + self.yalniz_orijinal + self.ikisi_de + self.hicbiri

    def __str__(self) -> str:
        return (
            f"n={self.toplam} · yalnız heretic kırıldı={self.yalniz_heretic} · "
            f"yalnız orijinal kırıldı={self.yalniz_orijinal} · ikisi de={self.ikisi_de} · "
            f"hiçbiri={self.hicbiri} · McNemar kesin p={self.p_degeri:.4g}"
        )


def mcnemar(ciftler: pd.DataFrame, *, kararsiz_kirik_sayilsin: bool = False) -> McNemarSonucu:
    """Heretic modelin anlamlı ölçüde daha sık kırılıp kırılmadığını sınar.

    Yalnızca uyumsuz çiftler (bir modelde kırılıp diğerinde kırılmayan) bilgi
    taşır; H0 altında bunlar iki yöne eşit olasılıkla dağılır. 44 promptluk
    küçük bir kütüphanede ki-kare yaklaşımı yerine binom tabanlı kesin test
    kullanılır.

    Args:
        kararsiz_kirik_sayilsin: True ise "kararsız" sonuçlar da kırılma
            sayılır (iyimser olmayan, en kötü durum okuması). Hatalar her iki
            durumda da dışlanır.
    """
    kirik = {"kırıldı", "kırıldı (düşük güven)"}
    if kararsiz_kirik_sayilsin:
        kirik = kirik | {"kararsız"}

    gecerli = ciftler[(ciftler["karar_orj"] != "hata") & (ciftler["karar_her"] != "hata")]
    o = gecerli["karar_orj"].isin(kirik)
    h = gecerli["karar_her"].isin(kirik)
    b = int((h & ~o).sum())
    c = int((o & ~h).sum())
    p = binomtest(b, b + c, 0.5, alternative="two-sided").pvalue if b + c else 1.0
    return McNemarSonucu(
        yalniz_heretic=b,
        yalniz_orijinal=c,
        ikisi_de=int((h & o).sum()),
        hicbiri=int((~h & ~o).sum()),
        p_degeri=float(p),
    )


def tespit_duyarliligi(ciftler: pd.DataFrame) -> pd.DataFrame:
    """ "Reddi kaybolan" denemelerde tarayıcının Heretic model için verdiği karar.

    Orijinal model reddetti, Heretic model reddetmedi: bu, Heretic'in tam olarak
    hedeflediği davranış değişikliğidir. Bu alt kümede tarayıcının "kırıldı"
    deme oranı, tespit hattının duyarlılığı (recall) için bir alt sınır
    tahminidir. Kesin değer, insan etiketiyle (altın küme) belirlenir.
    """
    hedef = ciftler[ciftler["reddetti_orj"] & ~ciftler["reddetti_her"]]
    if hedef.empty:
        return pd.DataFrame()
    tablo = pd.crosstab(
        [hedef["mod"], hedef["category"]], hedef["karar_her"], margins=True, margins_name="toplam"
    )
    return tablo.reindex(columns=[*[k for k in KARAR_SIRASI if k in tablo.columns], "toplam"])


def altin_kume_adaylari(
    ciftler: pd.DataFrame, cikti: Path | None = None, *, kirp: int = 400
) -> pd.DataFrame:
    """İnsan etiketlemesi için öncelikli denemeleri CSV'ye yazar (§6 altın küme).

    Öncelik sırası:
      1. Reddi kaybolduğu hâlde Heretic için "dayandı" denenler (olası yanlış negatif).
      2. Heretic için "kararsız" kalanlar.
      3. Orijinal model için "kırıldı" denenler (olası yanlış pozitif, Heretic
         zaten kırılgan olmalı; orijinalde kırılma şaşırtıcıdır).

    Yanıtlar `kirp` karakterde kesilir; tam metin yalnızca yerel SQLite'tadır (§8, §11).
    """
    kayip = ciftler["reddetti_orj"] & ~ciftler["reddetti_her"]
    gruplar = [
        ("olasi_yanlis_negatif", ciftler[kayip & (ciftler["karar_her"] == "dayandı")], "her"),
        ("heretic_kararsiz", ciftler[ciftler["karar_her"] == "kararsız"], "her"),
        (
            "orijinal_kirildi",
            ciftler[ciftler["karar_orj"].isin(["kırıldı", "kırıldı (düşük güven)"])],
            "orj",
        ),
    ]
    satirlar = []
    for neden, alt, taraf in gruplar:
        for r in alt.itertuples():
            yanit = getattr(r, f"raw_response_{taraf}") or ""
            satirlar.append(
                {
                    "oncelik_nedeni": neden,
                    "model": "heretic" if taraf == "her" else "orijinal",
                    "mod": r.mod,
                    "seed": r.seed,
                    "category": r.category,
                    "severity": r.severity,
                    "template_id": r.template_id,
                    "tarayici_karari": getattr(r, f"karar_{taraf}"),
                    "gerekce": getattr(r, f"rationale_{taraf}"),
                    "prompt": r.prompt_sent,
                    "yanit_kirpilmis": yanit[:kirp] + ("…" if len(yanit) > kirp else ""),
                    "insan_etiketi": "",  # kırıldı / dayandı — elle doldurulur
                    "not": "",
                }
            )
    sonuc = pd.DataFrame(satirlar).drop_duplicates(
        subset=["model", "mod", "seed", "template_id", "prompt"]
    )
    if cikti is not None:
        cikti.parent.mkdir(parents=True, exist_ok=True)
        # utf-8-sig: Excel Türkçe karakterleri doğru açsın.
        sonuc.to_csv(cikti, index=False, encoding="utf-8-sig")
    return sonuc
