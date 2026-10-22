"""Adım 1: Orijinal ve Heretic modeli indirip Ollama'ya aynı yöntemle yükler.

Adil karşılaştırmanın şartı, iki modelin aradaki tek farkın Heretic işlemi
olacak şekilde hazırlanmasıdır. Bu yüzden ikisi de:
  * Hugging Face'teki orijinal safetensors ağırlıklarından,
  * aynı kuantizasyonla (varsayılan q4_K_M),
  * aynı Modelfile parametreleriyle
Ollama'ya alınır. Hazır GGUF dosyaları farklı kişilerce, farklı ayarlarla
üretildiği için kullanılmaz.

Kullanım (proje kökünden):
    pip install -U huggingface_hub
    python experiments/heretic_karsilastirma/1_modelleri_hazirla.py

Disk: indirme ~16 GB (2 × ~8 GB bf16), Ollama'daki q4_K_M kopyaları ~2 × 2.5 GB.
İndirme bittikten sonra `modeller/` klasörü silinebilir.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DENEY_DIZINI = Path(__file__).resolve().parent
MODEL_DIZINI = DENEY_DIZINI / "modeller"

MODELLER = {
    # ollama adı: (Hugging Face deposu, yerel klasör)
    "qwen3-4b-orijinal": ("Qwen/Qwen3-4B-Instruct-2507", "orijinal"),
    "qwen3-4b-heretic": ("p-e-w/Qwen3-4B-Instruct-2507-heretic", "heretic"),
}

MODELFILE = """\
# Bu dosya 1_modelleri_hazirla.py tarafından üretilir; elle düzenlemeyin.
# İki model için birebir aynıdır — yalnızca FROM satırı farklıdır.
FROM ./modeller/{klasor}
PARAMETER num_ctx 4096
"""

INDIRILECEKLER = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"]
"""Ağırlıklar + tokenizer + sohbet şablonu. README, görseller vb. indirilmez."""


def _calistir(komut: list[str]) -> subprocess.CompletedProcess[str]:
    print("  $", " ".join(komut))
    return subprocess.run(komut, cwd=DENEY_DIZINI, text=True, capture_output=True, check=False)


def _ollama_kontrol() -> None:
    if shutil.which("ollama") is None:
        sys.exit(
            "Ollama bulunamadı. https://ollama.com/download adresinden kurun ve tekrar deneyin."
        )
    sonuc = _calistir(["ollama", "list"])
    if sonuc.returncode != 0:
        sys.exit(f"Ollama sunucusuna ulaşılamadı. 'ollama serve' çalışıyor mu?\n{sonuc.stderr}")


def indir(depo: str, hedef: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub yüklü değil: pip install -U huggingface_hub")

    print(f"\n[indir] {depo} -> {hedef}")
    snapshot_download(repo_id=depo, local_dir=hedef, allow_patterns=INDIRILECEKLER)


def ollamaya_yukle(ad: str, klasor: str, kuantizasyon: str) -> None:
    modelfile = DENEY_DIZINI / f"Modelfile.{klasor}"
    modelfile.write_text(MODELFILE.format(klasor=klasor), encoding="utf-8")

    print(f"\n[ollama] {ad} oluşturuluyor ({kuantizasyon})…")
    sonuc = _calistir(["ollama", "create", ad, "-f", modelfile.name, "--quantize", kuantizasyon])
    if sonuc.returncode != 0:
        sys.exit(
            f"'ollama create' başarısız oldu:\n{sonuc.stderr or sonuc.stdout}\n"
            "Ollama sürümünüz bu mimariyi safetensors'tan içe aktaramıyor olabilir. "
            "README'deki 'B planı: llama.cpp ile GGUF' bölümüne bakın."
        )
    print(f"  {ad} hazır.")


def sablonlari_karsilastir() -> None:
    """İki modelin Ollama'daki sohbet şablonu aynı mı? Farklıysa karşılaştırma adil değildir."""
    sablonlar = {}
    for ad in MODELLER:
        sonuc = _calistir(["ollama", "show", ad, "--template"])
        sablonlar[ad] = sonuc.stdout.strip()

    if len(set(sablonlar.values())) == 1 and next(iter(sablonlar.values())):
        print("\n[kontrol] Sohbet şablonları birebir aynı. [OK]")
    else:
        print(
            "\n[UYARI] İki modelin sohbet şablonu farklı ya da boş. Karşılaştırma adil olmaz.\n"
            "        README'deki 'Sohbet şablonu' notuna bakın."
        )
        for ad, sablon in sablonlar.items():
            print(f"--- {ad} ---\n{sablon[:600]}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--kuantizasyon", default="q4_K_M", help="İki model için ortak (varsayılan q4_K_M)."
    )
    ap.add_argument("--indirmeyi-atla", action="store_true", help="modeller/ zaten doluysa.")
    args = ap.parse_args()

    _ollama_kontrol()
    for ad, (depo, klasor) in MODELLER.items():
        hedef = MODEL_DIZINI / klasor
        if not args.indirmeyi_atla:
            indir(depo, hedef)
        elif not any(hedef.glob("*.safetensors")):
            sys.exit(f"{hedef} içinde safetensors yok; --indirmeyi-atla olmadan çalıştırın.")
        ollamaya_yukle(ad, klasor, args.kuantizasyon)

    sablonlari_karsilastir()
    print(
        "\nTamam. Sonraki adım:\n"
        "  python experiments/heretic_karsilastirma/2_taramalari_kos.py --yalnizca-yazdir\n"
        "  python experiments/heretic_karsilastirma/2_taramalari_kos.py"
    )


if __name__ == "__main__":
    main()
