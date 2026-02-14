"""
=============================================================================
LLD 05: RATE LIMITER
=============================================================================
Design a rate limiter that supports multiple algorithms and per-user limits.

KEY CONCEPTS:
  - Strategy pattern (swap algorithms)
  - Token Bucket and Sliding Window algorithms
  - Decorator pattern for easy integration
=============================================================================
"""
import time
import functools
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol


# --- Strategy interface ---
class RateLimitAlgorithm(Protocol):
    def allow_request(self, key: str) -> bool: ...


# --- Algorithm 1: Token Bucket ---
@dataclass
class TokenBucketState:
    tokens: float
    last_refill: float


class TokenBucket:
    """
    Tokens are added at a fixed rate. Each request costs one token.
    Allows bursts up to `capacity`, then rate-limited.
    """
    def __init__(self, rate: float, capacity: int):
        self.rate = rate          # tokens per second
        self.capacity = capacity  # max burst size
        self._buckets: dict[str, TokenBucketState] = {}

    def allow_request(self, key: str) -> bool:
        now = time.time()

        if key not in self._buckets:
            self._buckets[key] = TokenBucketState(self.capacity, now)

        bucket = self._buckets[key]

        # Refill tokens
        elapsed = now - bucket.last_refill
        bucket.tokens = min(self.capacity, bucket.tokens + elapsed * self.rate)
        bucket.last_refill = now

        if bucket.tokens >= 1:
            bucket.tokens -= 1
            return True
        return False


# --- Algorithm 2: Sliding Window Log ---
class SlidingWindowLog:
    """
    Tracks timestamps of requests in a sliding window.
    Precise but uses more memory.
    """
    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._logs: dict[str, deque[float]] = {}

    def allow_request(self, key: str) -> bool:
        now = time.time()

        if key not in self._logs:
            self._logs[key] = deque()

        log = self._logs[key]

        # Remove expired timestamps
        cutoff = now - self.window_seconds
        while log and log[0] < cutoff:
            log.popleft()

        if len(log) < self.max_requests:
            log.append(now)
            return True
        return False


# --- Algorithm 3: Fixed Window Counter ---
class FixedWindowCounter:
    """
    Counts requests in fixed time windows.
    Simple but can allow 2x burst at window boundaries.
    """
    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._windows: dict[str, tuple[int, int]] = {}  # key -> (window_id, count)

    def allow_request(self, key: str) -> bool:
        current_window = int(time.time() // self.window_seconds)

        if key not in self._windows:
            self._windows[key] = (current_window, 0)

        window_id, count = self._windows[key]

        # New window? Reset counter.
        if window_id != current_window:
            self._windows[key] = (current_window, 1)
            return True

        if count < self.max_requests:
            self._windows[key] = (window_id, count + 1)
            return True
        return False


# --- Rate Limiter Service ---
class RateLimiter:
    def __init__(self, algorithm: RateLimitAlgorithm):
        self.algorithm = algorithm

    def is_allowed(self, key: str) -> bool:
        return self.algorithm.allow_request(key)

    def check_or_raise(self, key: str) -> None:
        if not self.is_allowed(key):
            raise PermissionError(f"Rate limit exceeded for {key}")


# --- Decorator for easy use ---
def rate_limit(limiter: RateLimiter, key_func=None):
    """Decorator that rate-limits a function."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = key_func(*args, **kwargs) if key_func else "global"
            limiter.check_or_raise(key)
            return func(*args, **kwargs)
        return wrapper
    return decorator


# --- Demo ---
if __name__ == "__main__":
    print("=" * 50)
    print("Token Bucket: rate=5/sec, burst=3")
    print("=" * 50)

    tb = RateLimiter(TokenBucket(rate=5, capacity=3))
    for i in range(8):
        allowed = tb.is_allowed("user-1")
        print(f"  Request {i+1}: {'ALLOW' if allowed else 'DENY'}")

    print(f"\n{'=' * 50}")
    print("Sliding Window: 3 requests per 1 second")
    print("=" * 50)

    sw = RateLimiter(SlidingWindowLog(max_requests=3, window_seconds=1.0))
    for i in range(5):
        allowed = sw.is_allowed("user-1")
        print(f"  Request {i+1}: {'ALLOW' if allowed else 'DENY'}")

    time.sleep(1.1)
    print("  (1 second later...)")
    print(f"  Request 6: {'ALLOW' if sw.is_allowed('user-1') else 'DENY'}")

    print(f"\n{'=' * 50}")
    print("Decorator usage")
    print("=" * 50)

    api_limiter = RateLimiter(TokenBucket(rate=2, capacity=2))

    @rate_limit(api_limiter, key_func=lambda user_id: f"user:{user_id}")
    def get_profile(user_id: str):
        return {"user": user_id, "name": "Reza"}

    for i in range(4):
        try:
            result = get_profile("user-1")
            print(f"  Call {i+1}: {result}")
        except PermissionError as e:
            print(f"  Call {i+1}: BLOCKED - {e}")
