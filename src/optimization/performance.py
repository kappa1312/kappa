"""
Performance Optimization System for Kappa OS

Provides caching, pooling, and tuning capabilities.
"""

import asyncio
import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any

from loguru import logger


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""

    key: str
    value: Any
    created_at: float
    expires_at: float | None = None
    hits: int = 0


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""

    cache_hits: int = 0
    cache_misses: int = 0
    cache_size: int = 0
    avg_response_time_ms: float = 0.0
    total_requests: int = 0
    session_pool_size: int = 0
    active_sessions: int = 0


class CacheManager:
    """
    In-memory cache for frequently accessed data.

    Usage:
        cache = CacheManager(max_size=1000, default_ttl=300)
        cache.set("key", value)
        value = cache.get("key")
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: int | None = 300,  # 5 minutes
    ):
        self._cache: dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        """Get value from cache."""
        entry = self._cache.get(key)

        if entry is None:
            self._misses += 1
            return None

        # Check expiration
        if entry.expires_at and time.time() > entry.expires_at:
            del self._cache[key]
            self._misses += 1
            return None

        entry.hits += 1
        self._hits += 1
        return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ):
        """Set value in cache."""
        # Evict if at capacity
        if len(self._cache) >= self.max_size:
            self._evict()

        expires_at = None
        effective_ttl = ttl if ttl is not None else self.default_ttl
        if effective_ttl is not None and effective_ttl >= 0:
            expires_at = time.time() + effective_ttl

        self._cache[key] = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            expires_at=expires_at,
        )

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self):
        """Clear entire cache."""
        self._cache.clear()

    def has(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        entry = self._cache.get(key)
        if entry is None:
            return False
        if entry.expires_at and time.time() > entry.expires_at:
            del self._cache[key]
            return False
        return True

    def _evict(self):
        """Evict least recently used entries."""
        if not self._cache:
            return

        # Sort by hits (LFU) and created_at (LRU hybrid)
        entries = sorted(
            self._cache.items(),
            key=lambda x: (x[1].hits, x[1].created_at),
        )

        # Remove bottom 10%
        to_remove = max(1, len(entries) // 10)
        for key, _ in entries[:to_remove]:
            del self._cache[key]

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0,
        }


class SessionPool:
    """
    Pool of reusable sessions for efficient resource management.
    """

    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self._available: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self._in_use: dict[str, Any] = {}
        self._created = 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> Any:
        """Acquire a session from the pool."""
        try:
            # Try to get from pool first
            session = self._available.get_nowait()
            logger.debug("Reusing session from pool")
        except asyncio.QueueEmpty:
            # Create new session if under limit
            async with self._lock:
                if self._created < self.max_size:
                    session = await self._create_session()
                    self._created += 1
                    logger.debug(f"Created new session ({self._created}/{self.max_size})")
                else:
                    # Wait for available session
                    session = await self._available.get()
                    logger.debug("Waited for available session")

        session_id = str(id(session))
        self._in_use[session_id] = session
        return session

    async def release(self, session: Any):
        """Release session back to pool."""
        session_id = str(id(session))
        if session_id in self._in_use:
            del self._in_use[session_id]
            try:
                self._available.put_nowait(session)
                logger.debug("Session returned to pool")
            except asyncio.QueueFull:
                # Pool is full, destroy session
                await self._destroy_session(session)

    async def _create_session(self) -> Any:
        """Create a new session."""
        return {
            "id": f"session-{self._created}",
            "created_at": time.time(),
            "status": "ready",
        }

    async def _destroy_session(self, session: Any):
        """Destroy a session."""
        session_id = session.get("id", "unknown") if isinstance(session, dict) else str(id(session))
        logger.debug(f"Destroying session: {session_id}")

    def get_stats(self) -> dict:
        """Get pool statistics."""
        return {
            "max_size": self.max_size,
            "created": self._created,
            "available": self._available.qsize(),
            "in_use": len(self._in_use),
        }

    async def shutdown(self):
        """Shutdown the pool and destroy all sessions."""
        # Destroy in-use sessions
        for session in list(self._in_use.values()):
            await self._destroy_session(session)
        self._in_use.clear()

        # Destroy available sessions
        while not self._available.empty():
            try:
                session = self._available.get_nowait()
                await self._destroy_session(session)
            except asyncio.QueueEmpty:
                break

        self._created = 0
        logger.info("Session pool shutdown complete")


class TimerContext:
    """Context manager for timing operations."""

    def __init__(self, optimizer: "PerformanceOptimizer", name: str):
        self.optimizer = optimizer
        self.name = name
        self.start_time: float | None = None
        self.duration_ms: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            self.duration_ms = (time.perf_counter() - self.start_time) * 1000
            self.optimizer.record_time(self.name, self.duration_ms)

    async def __aenter__(self):
        self.start_time = time.perf_counter()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            self.duration_ms = (time.perf_counter() - self.start_time) * 1000
            self.optimizer.record_time(self.name, self.duration_ms)


class PerformanceOptimizer:
    """
    Central performance optimization manager.

    Usage:
        optimizer = PerformanceOptimizer()

        # Cache context
        @optimizer.cached("context", ttl=60)
        async def get_context(project_id):
            ...

        # Time operations
        with optimizer.timer("operation"):
            ...
    """

    def __init__(
        self,
        cache_max_size: int = 1000,
        cache_default_ttl: int = 300,
        pool_max_size: int = 10,
    ):
        self.cache = CacheManager(
            max_size=cache_max_size,
            default_ttl=cache_default_ttl,
        )
        self.session_pool = SessionPool(max_size=pool_max_size)
        self._timers: dict[str, list[float]] = {}
        self._start_time = time.time()

    def cached(
        self,
        prefix: str,
        ttl: int | None = None,
    ) -> Callable:
        """Decorator for caching function results."""

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Generate cache key
                key = self._generate_cache_key(prefix, args, kwargs)

                # Check cache
                cached_value = self.cache.get(key)
                if cached_value is not None:
                    return cached_value

                # Execute function
                result = await func(*args, **kwargs)

                # Cache result
                self.cache.set(key, result, ttl)

                return result

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Generate cache key
                key = self._generate_cache_key(prefix, args, kwargs)

                # Check cache
                cached_value = self.cache.get(key)
                if cached_value is not None:
                    return cached_value

                # Execute function
                result = func(*args, **kwargs)

                # Cache result
                self.cache.set(key, result, ttl)

                return result

            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper

        return decorator

    def timer(self, name: str) -> TimerContext:
        """Context manager for timing operations."""
        return TimerContext(self, name)

    def record_time(self, name: str, duration_ms: float):
        """Record a timing measurement."""
        if name not in self._timers:
            self._timers[name] = []
        self._timers[name].append(duration_ms)

        # Keep only last 1000 measurements
        if len(self._timers[name]) > 1000:
            self._timers[name] = self._timers[name][-1000:]

    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        cache_stats = self.cache.get_stats()
        pool_stats = self.session_pool.get_stats()

        # Calculate average response times
        all_times = []
        for times in self._timers.values():
            all_times.extend(times)

        avg_time = sum(all_times) / len(all_times) if all_times else 0

        return PerformanceMetrics(
            cache_hits=cache_stats["hits"],
            cache_misses=cache_stats["misses"],
            cache_size=cache_stats["size"],
            avg_response_time_ms=avg_time,
            total_requests=len(all_times),
            session_pool_size=pool_stats["max_size"],
            active_sessions=pool_stats["in_use"],
        )

    def get_timer_stats(self, name: str) -> dict:
        """Get statistics for specific timer."""
        times = self._timers.get(name, [])

        if not times:
            return {"count": 0}

        sorted_times = sorted(times)
        count = len(times)

        return {
            "count": count,
            "min": min(times),
            "max": max(times),
            "avg": sum(times) / count,
            "p50": sorted_times[count // 2],
            "p95": sorted_times[int(count * 0.95)] if count > 20 else max(times),
            "p99": sorted_times[int(count * 0.99)] if count > 100 else max(times),
        }

    def get_all_timer_stats(self) -> dict[str, dict]:
        """Get statistics for all timers."""
        return {name: self.get_timer_stats(name) for name in self._timers}

    def _generate_cache_key(
        self,
        prefix: str,
        args: tuple,
        kwargs: dict,
    ) -> str:
        """Generate cache key from function arguments."""
        key_parts = [prefix]

        # Add args
        for arg in args:
            try:
                key_parts.append(str(arg))
            except Exception:
                key_parts.append(str(id(arg)))

        # Add sorted kwargs
        for k, v in sorted(kwargs.items()):
            try:
                key_parts.append(f"{k}={v}")
            except Exception:
                key_parts.append(f"{k}={id(v)}")

        key_string = ":".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def reset(self):
        """Reset all metrics and caches."""
        self.cache.clear()
        self._timers.clear()
        logger.info("Performance optimizer reset")

    async def shutdown(self):
        """Shutdown the optimizer and release resources."""
        await self.session_pool.shutdown()
        self.cache.clear()
        self._timers.clear()
        logger.info("Performance optimizer shutdown complete")

    def get_uptime_seconds(self) -> float:
        """Get uptime in seconds."""
        return time.time() - self._start_time
