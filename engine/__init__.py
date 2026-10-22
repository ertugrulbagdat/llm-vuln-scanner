"""Fuzzing motoru — test koşumunu yönetir: sıraya alma, paralellik, yeniden deneme (§2, §5).

Faz 1'de uygulanan yetenekler: eş zamanlı worker havuzu, merkezi token-bucket
oran sınırlama, üstel geri çekilmeli yeniden deneme, checkpoint/resume,
kategori bazlı örnekleme, `seed` ile determinizm ve istek tavanıyla maliyet
kontrolü.
"""

from engine.ratelimit import TokenBucket
from engine.runner import Job, ScanConfig, ScanResult, ScanRunner, default_canary

__all__ = [
    "Job",
    "ScanConfig",
    "ScanResult",
    "ScanRunner",
    "TokenBucket",
    "default_canary",
]
