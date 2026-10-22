"""Adım 2: İki modeli aynı koşullarda tarar.

Her koşu `cli.py scan` ile yapılır; yani deney, tarayıcının kendisini olduğu
gibi kullanır. Değişmeyen koşullar:
  * aynı şablon kütüphanesi ve aynı seed -> aynı (şablon, prompt) iş listesi
  * aynı sistem promptu (`sistem_promptu.txt`), kanarya her koşuda yeniden üretilir
  * sıcaklık 0, tek worker (yerel GPU'da sıralı istek, gecikme ölçümü temiz kalır)

Modlar:
  * kural  -> yalnızca Katman 1          -> sonuclar/deney_kural.db
  * judge  -> Katman 1 + LLM-judge       -> sonuclar/deney_judge.db
    (judge yalnızca Katman 1'in kararsız / düşük güvenli bıraktıklarına çağrılır)

Kullanım (proje kökünden):
    python experiments/heretic_karsilastirma/2_taramalari_kos.py --yalnizca-yazdir
    python experiments/heretic_karsilastirma/2_taramalari_kos.py
    python experiments/heretic_karsilastirma/2_taramalari_kos.py --judge-model gemma3:4b
    python experiments/heretic_karsilastirma/2_taramalari_kos.py --tekrar 3 --sicaklik 0.7
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

DENEY_DIZINI = Path(__file__).resolve().parent
PROJE_KOKU = DENEY_DIZINI.parents[1]
SONUC_DIZINI = DENEY_DIZINI / "sonuclar"
SISTEM_PROMPTU = DENEY_DIZINI / "sistem_promptu.txt"
MANIFEST = SONUC_DIZINI / "kosular.jsonl"

HEDEFLER = ["qwen3-4b-orijinal", "qwen3-4b-heretic"]
KOSU_RE = re.compile(r"Koşu:\s*([0-9a-f-]{36})")
SKOR_RE = re.compile(r"Dayanıklılık Skoru:\s*([\d.]+)")


def komut_olustur(
    model: str, seed: int, args: argparse.Namespace, mod: str
) -> tuple[list[str], Path]:
    db = SONUC_DIZINI / f"deney_{mod}.db"
    rapor = SONUC_DIZINI / f"rapor_{model}_{mod}_s{seed}.html"
    komut = [
        sys.executable, "cli.py", "scan",
        "--target", "ollama",
        "--model", model,
        "--seed", str(seed),
        "--temperature", str(args.sicaklik),
        "--workers", "1",
        "--system-prompt-file", str(SISTEM_PROMPTU),
        "--db-url", f"sqlite:///{db.as_posix()}",
        "--report", str(rapor),
        "--quiet",
        "--i-own-this-target",
    ]  # fmt: skip
    if args.base_url:
        komut += ["--base-url", args.base_url]
    if args.sample:
        komut += ["--sample", str(args.sample)]
    if args.judge_model:
        komut += ["--judge", "ollama", "--judge-model", args.judge_model]
        if args.base_url:
            komut += ["--judge-base-url", args.base_url]
    return komut, rapor


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--seed", type=int, default=42, help="İlk seed (varsayılan 42).")
    ap.add_argument(
        "--tekrar", type=int, default=1, help="Model başına koşu sayısı (seed, seed+1, …)."
    )
    ap.add_argument(
        "--sicaklik", type=float, default=0.0, help="Örnekleme sıcaklığı (varsayılan 0)."
    )
    ap.add_argument("--judge-model", help="Verilirse judge modunda koşar (ör. gemma3:4b).")
    ap.add_argument("--base-url", help="Ollama başka makinedeyse, ör. http://192.168.1.50:11434/v1")
    ap.add_argument("--sample", type=int, help="Duman testi: her kategoriden N şablon.")
    ap.add_argument("--yalnizca-yazdir", action="store_true", help="Komutları yazdır, çalıştırma.")
    args = ap.parse_args()

    if args.judge_model and args.judge_model in HEDEFLER:
        sys.exit("Judge, test edilen modellerden biri olamaz (önyargı riski, §6).")
    if args.tekrar > 1 and args.sicaklik == 0:
        print(
            "Not: sıcaklık 0'da tekrarlar neredeyse aynı yanıtları üretir. Yanıt değişkenliğini\n"
            "ölçmek istiyorsanız --sicaklik 0.7 gibi bir değer verin.\n"
        )

    mod = "judge" if args.judge_model else "kural"
    SONUC_DIZINI.mkdir(parents=True, exist_ok=True)

    # Seed dışta, model içte: iki modelin koşuları zaman içinde iç içe geçer,
    # böylece makinedeki yük değişimi (ısınma, arka plan işleri) bir modeli kayırmaz.
    isler = [
        (seed, model) for seed in range(args.seed, args.seed + args.tekrar) for model in HEDEFLER
    ]

    for sira, (seed, model) in enumerate(isler, start=1):
        komut, rapor = komut_olustur(model, seed, args, mod)
        print(f"\n[{sira}/{len(isler)}] {model} · mod={mod} · seed={seed}")
        print("  $", subprocess.list2cmdline(komut))
        if args.yalnizca_yazdir:
            continue

        baslangic = datetime.now(UTC)
        # Windows'ta yönlendirilen çıktı varsayılan olarak cp1254 ile kodlanır;
        # alt süreci UTF-8'e zorlayıp öyle okuyoruz.
        ortam = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
        sonuc = subprocess.run(
            komut, cwd=PROJE_KOKU, env=ortam, text=True, encoding="utf-8", errors="replace",
            capture_output=True, check=False,
        )  # fmt: skip
        cikti = sonuc.stdout + sonuc.stderr
        kosu = KOSU_RE.search(cikti)
        skor = SKOR_RE.search(cikti)

        if sonuc.returncode not in (0, 3) or kosu is None:
            print(cikti)
            sys.exit(f"Koşu başarısız oldu (çıkış kodu {sonuc.returncode}).")

        durum = "tamamlandı" if sonuc.returncode == 0 else "maliyet tavanında durdu"
        print(f"  R = {skor.group(1) if skor else '?'} · run_id = {kosu.group(1)} · {durum}")
        print(f"  rapor: {rapor}")

        with MANIFEST.open("a", encoding="utf-8") as f:
            kayit = {
                "zaman": baslangic.isoformat(),
                "run_id": kosu.group(1),
                "model": model,
                "mod": mod,
                "seed": seed,
                "sicaklik": args.sicaklik,
                "judge_model": args.judge_model,
                "sample": args.sample,
                "R": float(skor.group(1)) if skor else None,
                "cikis_kodu": sonuc.returncode,
            }
            f.write(json.dumps(kayit, ensure_ascii=False) + "\n")

    if not args.yalnizca_yazdir:
        print(
            f"\nBitti. Analiz için: notebooks/heretic_karsilastirma.ipynb\nKoşu kaydı: {MANIFEST}"
        )


if __name__ == "__main__":
    main()
