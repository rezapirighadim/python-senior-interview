"""
=============================================================================
LLD 03: URL SHORTENER
=============================================================================
Design a URL shortener like bit.ly.

KEY CONCEPTS:
  - Base62 encoding for short codes
  - Repository pattern (swap storage easily)
  - TTL / expiration support
=============================================================================
"""
import string
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Protocol


# --- Base62 encoding ---
CHARSET = string.digits + string.ascii_lowercase + string.ascii_uppercase  # 0-9a-zA-Z


def base62_encode(num: int) -> str:
    if num == 0:
        return CHARSET[0]
    result = []
    while num > 0:
        result.append(CHARSET[num % 62])
        num //= 62
    return "".join(reversed(result))


def base62_decode(code: str) -> int:
    result = 0
    for char in code:
        result = result * 62 + CHARSET.index(char)
    return result


# --- Models ---
@dataclass
class ShortURL:
    code: str
    original_url: str
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None
    click_count: int = 0

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


# --- Storage (Repository pattern) ---
class URLStore(Protocol):
    def save(self, short_url: ShortURL) -> None: ...
    def get_by_code(self, code: str) -> ShortURL | None: ...
    def get_by_url(self, original_url: str) -> ShortURL | None: ...


class InMemoryURLStore:
    def __init__(self):
        self._by_code: dict[str, ShortURL] = {}
        self._by_url: dict[str, ShortURL] = {}

    def save(self, short_url: ShortURL) -> None:
        self._by_code[short_url.code] = short_url
        self._by_url[short_url.original_url] = short_url

    def get_by_code(self, code: str) -> ShortURL | None:
        return self._by_code.get(code)

    def get_by_url(self, original_url: str) -> ShortURL | None:
        return self._by_url.get(original_url)


# --- URL Shortener Service ---
class URLShortener:
    BASE_URL = "https://short.ly/"

    def __init__(self, store: URLStore):
        self.store = store
        self._counter = 100000  # start from 100000 to get 3+ char codes

    def shorten(self, original_url: str, ttl_hours: int | None = None) -> str:
        # Check if already shortened
        existing = self.store.get_by_url(original_url)
        if existing and not existing.is_expired:
            return self.BASE_URL + existing.code

        # Generate new short code
        self._counter += 1
        code = base62_encode(self._counter)

        expires_at = None
        if ttl_hours:
            expires_at = datetime.now() + timedelta(hours=ttl_hours)

        short_url = ShortURL(
            code=code,
            original_url=original_url,
            expires_at=expires_at,
        )
        self.store.save(short_url)
        return self.BASE_URL + code

    def resolve(self, short_url: str) -> str:
        code = short_url.replace(self.BASE_URL, "")
        entry = self.store.get_by_code(code)

        if entry is None:
            raise ValueError(f"Short URL not found: {code}")

        if entry.is_expired:
            raise ValueError(f"Short URL expired: {code}")

        entry.click_count += 1
        return entry.original_url

    def get_stats(self, short_url: str) -> dict:
        code = short_url.replace(self.BASE_URL, "")
        entry = self.store.get_by_code(code)
        if entry is None:
            raise ValueError(f"Not found: {code}")

        return {
            "code": entry.code,
            "original_url": entry.original_url,
            "clicks": entry.click_count,
            "created_at": entry.created_at.isoformat(),
            "expired": entry.is_expired,
        }


# --- Demo ---
if __name__ == "__main__":
    store = InMemoryURLStore()
    shortener = URLShortener(store)

    # Shorten URLs
    url1 = shortener.shorten("https://www.example.com/very/long/path/to/some/page")
    url2 = shortener.shorten("https://docs.python.org/3/library/dataclasses.html")
    url3 = shortener.shorten("https://temp.com/offer", ttl_hours=24)

    print(f"Short 1: {url1}")
    print(f"Short 2: {url2}")
    print(f"Short 3: {url3} (expires in 24h)")

    # Same URL returns same short code
    url1_again = shortener.shorten("https://www.example.com/very/long/path/to/some/page")
    print(f"\nSame URL again: {url1_again}")
    print(f"Same code? {url1 == url1_again}")

    # Resolve
    original = shortener.resolve(url1)
    print(f"\nResolve {url1} -> {original}")

    # Click a few times
    shortener.resolve(url1)
    shortener.resolve(url1)

    # Stats
    stats = shortener.get_stats(url1)
    print(f"Stats: {stats}")
