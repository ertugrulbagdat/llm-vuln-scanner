"""Heretic'in zararlı prompt kümesini şablon kütüphanesine aktarır (Faz 3).

Heretic (https://github.com/p-e-w/heretic) promptları kendi deposunda tutmaz;
`config.default.toml` dosyası onları Hugging Face'teki
`mlabonne/harmful_behaviors` veri setinin `text` sütunundan okur:

* `train[:400]` → reddetme yönünü bulmak için "kötü" promptlar
* `test[:100]`  → Heretic'in `KeywordRate` değerlendirme kümesi

Varsayılan olarak değerlendirme kümesi (`test[:100]`) aktarılır. Böylece
tarayıcının bu kategorideki sonucu, Heretic'in kendi ölçtüğü reddetme oranıyla
aynı promptlar üzerinden karşılaştırılabilir.

Her prompt `templates/<category>/<id>.yaml` dosyasına yazılır ve `Template`
şemasıyla yazılmadan önce doğrulanır. Betik idempotenttir: her çalıştırmada önceki
içe aktarmayı (`*-heretic-*.yaml`) silip yeniden üretir.

Kullanım (proje kökünden):

    pip install huggingface_hub pyarrow
    python scripts/import_heretic.py                     # test[:100]
    python scripts/import_heretic.py --split "train[:400]"
    python scripts/import_heretic.py --input yerel.parquet --dry-run
    python cli.py validate-templates
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from templates.loader import LIBRARY_ROOT  # noqa: E402
from templates.schema import (  # noqa: E402
    PLACEHOLDER_RE,
    Category,
    Severity,
    SuccessSignal,
    Template,
)

DATASET_ID = "mlabonne/harmful_behaviors"
DATASET_REVISION = "01cead01398926d81f7c52bdb790ee8cf77ebba7"
"""Tekrarlanabilirlik için sabitlenmiş commit (2026-10-20 itibarıyla `main`)."""

DATASET_COLUMN = "text"
SPLIT_RE = re.compile(r"^(?P<name>train|test)(?:\[:(?P<limit>\d+)\])?$")

ID_PREFIX = {
    Category.HARMFUL_CONTENT: "hc",
    Category.JAILBREAK: "jb",
    Category.PROMPT_INJECTION: "pi",
}

# --- Sınıflandırma ---------------------------------------------------------
# Kurallar sırayla denenir, ilk eşleşen kazanır. Daha özgül kalıplar üstte.

CATEGORY_RULES: list[tuple[Category, re.Pattern[str]]] = [
    (
        Category.PROMPT_INJECTION,
        re.compile(
            r"ignore (all |the )?(previous|prior|above) instructions|system prompt"
            r"|disregard (your|the) (rules|instructions)",
            re.I,
        ),
    ),
    (
        Category.JAILBREAK,
        re.compile(
            r"\b(pretend|role-?play|you are now|act as|DAN\b|developer mode"
            r"|no restrictions|unfiltered)",
            re.I,
        ),
    ),
]
"""Veri seti doğrudan zararlı taleplerden oluşur; bu kurallar yalnızca sarmalayıcı
içeren nadir satırları ayırmak için vardır. Eşleşmeyen her şey `harmful_content`."""


@dataclass(frozen=True)
class Subcategory:
    name: str
    severity: Severity
    pattern: re.Pattern[str]


def _kw(*words: str) -> re.Pattern[str]:
    return re.compile(r"\b(" + "|".join(words) + r")", re.I)


SUBCATEGORY_RULES: list[Subcategory] = [
    Subcategory("child_exploitation", Severity.CRITICAL,
                _kw("child porn", r"minors?\b", "underage", "child sexual")),
    Subcategory(
        "weapons_explosives",
        Severity.CRITICAL,
        _kw("bomb", "explosive", "weapon", "firearm", "gun", "chemical weapon",
            "bioweapon", "biological", "poison"),
    ),
    Subcategory("self_harm", Severity.CRITICAL,
                _kw("suicide", "self-harm", "self harm", "eating disorder", "anorexia",
                    "bulimia", "cutting")),
    Subcategory("drugs", Severity.CRITICAL,
                _kw("drug", "cocaine", "heroin", "meth", "narcotic", "controlled substance")),
    Subcategory("violence", Severity.CRITICAL,
                _kw("kill", "murder", "assault", "violent", "violence", "terror", "attack on",
                    "harm (a|an|someone|others|people)", "cause harm", "kidnap",
                    "traffick")),
    Subcategory("malware", Severity.HIGH,
                _kw("malware", "virus", "ransomware", "trojan", "keylogger", "worm", "botnet",
                    "exploit")),
    Subcategory("cyber_intrusion", Severity.HIGH,
                _kw("hack", "crack", "breach", "unauthori[sz]ed access", "database",
                    "network", "password", "ddos", "phishing", "cyber", "bypass security",
                    "denies access", "denial of service", "sp(y|ies) on")),
    Subcategory("fraud_financial", Severity.HIGH,
                _kw("fraud", "scam", "counterfeit", "launder", "insider trading", "embezzl",
                    "tax", "steal", "theft", "extort", "blackmail", "credit card", "shoplift")),
    Subcategory("privacy_violation", Severity.HIGH,
                _kw("identity", "personal information", "private information", "stalk",
                    "spy", "track", "surveil", "dox")),
    Subcategory("misinformation", Severity.HIGH,
                _kw("fake news", "false information", "misinformation", "disinformation",
                    "propaganda", "conspiracy", "fake review", "hoax")),
    Subcategory("hate_harassment", Severity.HIGH,
                _kw("hate", "racis", "discriminat", "harass", "bully", "cyberbully",
                    "minority", "offensive", "inflammatory")),
    Subcategory("illegal_activity", Severity.HIGH,
                _kw("fake id", "passport", "evad", "escape from", "law enforcement", "sabotag",
                    "spam", "cheat", "manipulat", "circumvent", "digital rights",
                    "not even used")),
]

FALLBACK_SUBCATEGORY = Subcategory("heretic_import", Severity.HIGH, re.compile(r"$^"))


def classify(text: str) -> tuple[Category, Subcategory]:
    category = next(
        (cat for cat, pattern in CATEGORY_RULES if pattern.search(text)),
        Category.HARMFUL_CONTENT,
    )
    sub = next((s for s in SUBCATEGORY_RULES if s.pattern.search(text)), FALLBACK_SUBCATEGORY)
    return category, sub


# --- Veri okuma ------------------------------------------------------------


def parse_split(split: str) -> tuple[str, int | None]:
    match = SPLIT_RE.fullmatch(split.strip())
    if not match:
        raise SystemExit(f"Geçersiz --split: {split!r} (ör. 'test', 'test[:100]', 'train[:400]')")
    limit = match.group("limit")
    return match.group("name"), int(limit) if limit else None


def read_prompts(split: str, input_path: Path | None) -> list[str]:
    """Promptları HF veri setinden ya da yerel bir dosyadan okur."""
    name, limit = parse_split(split)

    if input_path is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise SystemExit(
                "huggingface_hub kurulu değil: `pip install huggingface_hub pyarrow` "
                "ya da --input ile yerel bir dosya verin."
            ) from exc
        input_path = Path(
            hf_hub_download(
                DATASET_ID,
                f"data/{name}-00000-of-00001.parquet",
                repo_type="dataset",
                revision=DATASET_REVISION,
            )
        )

    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        prompts = pd.read_parquet(input_path)[DATASET_COLUMN].tolist()
    elif suffix == ".jsonl":
        prompts = pd.read_json(input_path, lines=True)[DATASET_COLUMN].tolist()
    elif suffix == ".csv":
        prompts = pd.read_csv(input_path)[DATASET_COLUMN].tolist()
    elif suffix == ".txt":
        prompts = input_path.read_text(encoding="utf-8").splitlines()
    else:
        raise SystemExit(f"Desteklenmeyen dosya türü: {input_path}")

    cleaned = [str(p).strip() for p in prompts if str(p).strip()]
    return cleaned[:limit] if limit is not None else cleaned


# --- YAML üretimi ----------------------------------------------------------


class _LiteralDumper(yaml.SafeDumper):
    """`template` alanını mevcut şablonlar gibi `|` blok biçiminde yazar."""


def _str_presenter(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    style = "|" if data.endswith("\n") else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_LiteralDumper.add_representer(str, _str_presenter)


def build_template(
    index: int, text: str, split: str, signal: SuccessSignal
) -> tuple[Template, Subcategory] | None:
    if PLACEHOLDER_RE.search(text):
        # `{{x}}` içeren ham metin şemada tanımsız değişken hatası verir.
        print(f"  [ATLA] #{index}: yer tutucuya benzeyen ifade içeriyor", file=sys.stderr)
        return None

    category, sub = classify(text)
    template = Template.model_validate(
        {
            "id": f"{ID_PREFIX[category]}-heretic-{index:03d}",
            "category": category,
            "subcategory": sub.name,
            "severity": sub.severity,
            "template": text + "\n",
            "success_signal": signal,
            "tags": ["heretic", "automated_import", "english", "refusal-probe"],
            "source": f"heretic:{DATASET_ID}@{DATASET_REVISION[:7]}/{split}",
            "version": 1,
        }
    )
    return template, sub


def to_yaml(template: Template) -> str:
    data = template.model_dump(mode="json", exclude={"variables"})
    body = yaml.dump(
        data, Dumper=_LiteralDumper, sort_keys=False, allow_unicode=True, width=1000
    )
    # Etiketleri mevcut dosyalardaki gibi tek satırda göster.
    tags_block = yaml.dump({"tags": data["tags"]}, Dumper=_LiteralDumper, sort_keys=False)
    return body.replace(tags_block, f"tags: [{', '.join(data['tags'])}]\n")


def remove_previous_import() -> int:
    removed = 0
    for category in ID_PREFIX:
        for path in (LIBRARY_ROOT / category.value).glob("*-heretic-*.yaml"):
            path.unlink()
            removed += 1
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="test[:100]",
                        help="HF split dilimi (varsayılan: Heretic değerlendirme kümesi "
                             "test[:100])")
    parser.add_argument("--input", type=Path,
                        help="HF yerine yerel dosya (.parquet/.jsonl/.csv/.txt)")
    parser.add_argument("--signal", choices=[SuccessSignal.REFUSAL_ABSENT.value,
                                             SuccessSignal.JUDGE_ONLY.value],
                        default=SuccessSignal.REFUSAL_ABSENT.value,
                        help="success_signal (varsayılan refusal_absent; judge_only kararı "
                             "LLM-judge'a bırakır)")
    parser.add_argument("--dry-run", action="store_true", help="Dosya yazmadan özet göster")
    args = parser.parse_args()

    prompts = read_prompts(args.split, args.input)
    signal = SuccessSignal(args.signal)
    print(f"{len(prompts)} prompt okundu ({args.input or DATASET_ID}, {args.split}).")

    built = [
        result
        for i, text in enumerate(prompts, start=1)
        if (result := build_template(i, text, args.split, signal)) is not None
    ]

    summary: dict[tuple[str, str, str], int] = {}
    for template, sub in built:
        key = (template.category.value, sub.name, sub.severity.value)
        summary[key] = summary.get(key, 0) + 1
    for (cat, sub_name, sev), count in sorted(summary.items()):
        print(f"  {cat:<18} {sub_name:<20} {sev:<8} {count:>4}")

    if args.dry_run:
        print("Kuru çalıştırma: dosya yazılmadı.")
        return

    removed = remove_previous_import()
    for template, _ in built:
        path = LIBRARY_ROOT / template.category.value / f"{template.id}.yaml"
        path.write_text(to_yaml(template), encoding="utf-8")

    print(f"{removed} eski dosya silindi, {len(built)} şablon yazıldı → {LIBRARY_ROOT}")


if __name__ == "__main__":
    main()
