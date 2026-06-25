"""
5bit Rate Limiter — In-Memory Sliding Window + Optional Redis
===============================================================
Wire into any HTTP handler. Per-IP or per-user (JWT sub) buckets.
Drop-in Redis option for multi-process deployments.

Usage:
  limiter = RateLimiter(max_req=100, window=60)
  if not limiter.allow(ip):
      return 429, {'error': 'Rate limit exceeded', 'retryAfter': 42}
"""
import time, threading
from collections import defaultdict
from typing import Optional


class RateLimiter:
    """Sliding window rate limiter. In-memory by default, optional Redis."""

    def __init__(self, max_req: int = 100, window: int = 60,
                 redis_url: Optional[str] = None):
        self.max = max_req
        self.window = window
        self.buckets: defaultdict = defaultdict(list)
        self.lock = threading.Lock()
        self.redis = None

        if redis_url:
            try:
                import redis
                self.redis = redis.from_url(redis_url)
            except ImportError:
                pass  # Fall back to in-memory

    def allow(self, key: str) -> bool:
        """Check if request is allowed. Returns True if under limit."""
        if self.redis:
            return self._allow_redis(key)

        now = time.time()
        with self.lock:
            bucket = self.buckets[key]
            # Slide window: drop expired entries
            bucket[:] = [t for t in bucket if now - t < self.window]
            if len(bucket) >= self.max:
                return False
            bucket.append(now)
            return True

    def retry_after(self, key: str) -> int:
        """Seconds until the oldest request expires."""
        with self.lock:
            bucket = self.buckets.get(key, [])
            if bucket:
                return max(0, int(self.window - (time.time() - bucket[0])))
        return 0

    # ── Redis (multi-process) ──────────────────────────────────────────

    def _allow_redis(self, key: str) -> bool:
        """Redis-backed sliding window using sorted sets."""
        now = time.time()
        rkey = f"5bit:rl:{key}"
        cutoff = now - self.window

        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(rkey, 0, cutoff)       # Remove expired
        pipe.zcard(rkey)                                # Count remaining
        pipe.zadd(rkey, {str(now): now})                # Add current
        pipe.expire(rkey, self.window + 10)             # Auto-cleanup
        _, count, _, _ = pipe.execute()

        return count < self.max

    def reset(self, key: str):
        """Reset a key's bucket (for testing)."""
        if self.redis:
            self.redis.delete(f"5bit:rl:{key}")
        else:
            with self.lock:
                self.buckets.pop(key, None)


# ── Convenience: per-IP + per-user ────────────────────────────────────

class APIRateLimiter:
    """Dual-bucket: IP (100/60s) + authenticated user (1000/60s)."""

    def __init__(self, redis_url: Optional[str] = None):
        self.ip_limiter = RateLimiter(max_req=100, window=60, redis_url=redis_url)
        self.user_limiter = RateLimiter(max_req=1000, window=60, redis_url=redis_url)

    def check(self, ip: str, user_id: Optional[int] = None) -> tuple:
        """Returns (allowed: bool, retry_after: int)."""
        if user_id:
            key = f"user:{user_id}"
            if not self.user_limiter.allow(key):
                return False, self.user_limiter.retry_after(key)

        if not self.ip_limiter.allow(ip):
            return False, self.ip_limiter.retry_after(ip)

        return True, 0
