"""Merkezi oran sınırlayıcı — Spesifikasyon §3, §5.

Her adaptör kendi `RateLimitPolicy`'sini bildirir; motor bunu tüm worker'ların
paylaştığı tek bir token-bucket ile uygular. Böylece eş zamanlılık arttığında
hedefe giden istek hızı politikayı aşmaz (§11: araç bir DoS mekanizması değildir).
"""

from __future__ import annotations

import threading
import time

__all__ = ["TokenBucket"]


class TokenBucket:
    """İş parçacığı güvenli token-bucket oran sınırlayıcı.

    Kova `rate` token/saniye hızıyla `capacity` tavanına kadar dolar. `acquire()`
    yeterli token birikene dek bloklar, böylece kısa patlamalara (burst) izin
    verilirken uzun vadeli ortalama hız `rate`'i aşmaz.
    """

    def __init__(self, rate: float, *, capacity: float | None = None) -> None:
        """
        Args:
            rate: Saniyedeki izin verilen istek sayısı.
            capacity: Biriktirilebilecek azami token; verilmezse `rate` (bir
                saniyelik patlama) kullanılır.

        Raises:
            ValueError: `rate` pozitif değilse.
        """
        if rate <= 0:
            raise ValueError("rate pozitif olmalı")

        self._rate = rate
        self._capacity = capacity if capacity is not None else max(rate, 1.0)
        self._tokens = self._capacity
        self._updated_at = time.monotonic()
        self._lock = threading.Lock()

    @property
    def rate(self) -> float:
        return self._rate

    def acquire(self, tokens: float = 1.0) -> float:
        """Token ayrılana kadar bloklar; beklenen süreyi saniye olarak döndürür.

        Raises:
            ValueError: `tokens` kova kapasitesini aşıyorsa (asla dolmaz).
        """
        if tokens > self._capacity:
            raise ValueError(f"{tokens} token kova kapasitesini ({self._capacity}) aşıyor")

        waited = 0.0
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return waited
                deficit = tokens - self._tokens
                sleep_for = deficit / self._rate

            # Uyku kilit DIŞINDA yapılır; aksi hâlde bir worker beklerken
            # diğerleri kovaya hiç erişemez ve eş zamanlılık anlamsızlaşır.
            time.sleep(sleep_for)
            waited += sleep_for

    def _refill(self) -> None:
        """Geçen süreye göre token ekler. Çağıran kilidi tutuyor olmalıdır."""
        now = time.monotonic()
        elapsed = now - self._updated_at
        if elapsed > 0:
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
            self._updated_at = now
