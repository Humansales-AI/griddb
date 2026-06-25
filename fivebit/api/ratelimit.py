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
            bucket[:] = [t for t in bucket if now - t < self.window]
            if not bucket:
                del self.buckets[key]  # Fix #1: prevent unbounded dict growth
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
        pipe.zremrangebyscore(rkey, 0, cutoff)
        pipe.zcard(rkey)
        _, count = pipe.execute()  # Count BEFORE deciding

        if count >= self.max:
            return False
        # Only record on allow — matches in-memory behavior
        self.redis.zadd(rkey, {str(now): now})
        self.redis.expire(rkey, self.window + 10)
        return True

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
        """Returns (allowed: bool, retry_after: int). IP first (cheaper), user second."""
        # Check IP first — cheaper, prevent quota waste if IP blocked
        if user_id:
            ip_ok = self.ip_limiter.allow(ip)
            if not ip_ok:
                return False, self.ip_limiter.retry_after(ip)
            # IP passed, now check user
            user_ok = self.user_limiter.allow(f"user:{user_id}")
            if not user_ok:
                # Roll back the IP slot — user blocked doesn't mean IP should be penalized
                self.ip_limiter.buckets[ip].pop()
                return False, self.user_limiter.retry_after(f"user:{user_id}")
            return True, 0

        if not self.ip_limiter.allow(ip):
            return False, self.ip_limiter.retry_after(ip)
        return True, 0
