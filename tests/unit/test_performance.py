"""
Unit tests for the Performance Optimization module.

Tests the performance optimization system including:
- CacheManager
- SessionPool
- PerformanceOptimizer
- TimerContext
"""

import asyncio
import time

import pytest

from src.optimization.performance import (
    CacheEntry,
    CacheManager,
    PerformanceMetrics,
    PerformanceOptimizer,
    SessionPool,
)

# =============================================================================
# TEST CacheEntry DATACLASS
# =============================================================================


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_create_cache_entry(self):
        """Test creating a CacheEntry."""
        entry = CacheEntry(
            key="test-key",
            value={"data": "test"},
            created_at=time.time(),
        )

        assert entry.key == "test-key"
        assert entry.value == {"data": "test"}
        assert entry.hits == 0

    def test_cache_entry_with_expiry(self):
        """Test CacheEntry with expiration."""
        now = time.time()
        entry = CacheEntry(
            key="test",
            value="value",
            created_at=now,
            expires_at=now + 300,
        )

        assert entry.expires_at is not None


# =============================================================================
# TEST CacheManager CLASS
# =============================================================================


class TestCacheManager:
    """Tests for CacheManager class."""

    @pytest.fixture
    def cache(self):
        return CacheManager(max_size=100, default_ttl=300)

    def test_cache_init(self, cache):
        """Test CacheManager initialization."""
        assert cache.max_size == 100
        assert cache.default_ttl == 300

    def test_cache_set_and_get(self, cache):
        """Test setting and getting values."""
        cache.set("key1", "value1")
        result = cache.get("key1")

        assert result == "value1"

    def test_cache_miss(self, cache):
        """Test cache miss returns None."""
        result = cache.get("nonexistent")

        assert result is None

    def test_cache_delete(self, cache):
        """Test deleting cached value."""
        cache.set("key1", "value1")
        deleted = cache.delete("key1")

        assert deleted is True
        assert cache.get("key1") is None

    def test_cache_has(self, cache):
        """Test checking if key exists."""
        cache.set("key1", "value1")

        assert cache.has("key1") is True
        assert cache.has("key2") is False

    def test_cache_clear(self, cache):
        """Test clearing all cached values."""
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_cache_expiry(self):
        """Test cache entry expiration."""
        cache = CacheManager(default_ttl=0)  # Immediate expiry
        cache.set("key1", "value1", ttl=0)

        # Entry should be expired
        time.sleep(0.01)
        result = cache.get("key1")

        assert result is None

    def test_cache_eviction(self):
        """Test cache eviction when full."""
        cache = CacheManager(max_size=2)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Should trigger eviction

        # At least key3 should be present
        assert cache.get("key3") == "value3"

    def test_cache_stats(self, cache):
        """Test getting cache statistics."""
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        stats = cache.get_stats()

        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1


# =============================================================================
# TEST SessionPool CLASS
# =============================================================================


@pytest.mark.asyncio
class TestSessionPool:
    """Tests for SessionPool class."""

    async def test_pool_init(self):
        """Test SessionPool initialization."""
        pool = SessionPool(max_size=5)

        assert pool.max_size == 5
        assert pool._created == 0

    async def test_pool_acquire(self):
        """Test acquiring a session from pool."""
        pool = SessionPool(max_size=5)

        session = await pool.acquire()

        assert session is not None
        assert pool._created == 1

    async def test_pool_release(self):
        """Test releasing a session back to pool."""
        pool = SessionPool(max_size=5)

        session = await pool.acquire()
        await pool.release(session)

        stats = pool.get_stats()
        assert stats["available"] == 1

    async def test_pool_reuse(self):
        """Test session reuse from pool."""
        pool = SessionPool(max_size=5)

        session1 = await pool.acquire()
        await pool.release(session1)
        session2 = await pool.acquire()

        # Should reuse the same session
        assert pool._created == 1

    async def test_pool_stats(self):
        """Test getting pool statistics."""
        pool = SessionPool(max_size=5)

        await pool.acquire()
        stats = pool.get_stats()

        assert stats["max_size"] == 5
        assert stats["created"] == 1
        assert stats["in_use"] == 1

    async def test_pool_shutdown(self):
        """Test shutting down the pool."""
        pool = SessionPool(max_size=5)

        await pool.acquire()
        await pool.shutdown()

        assert pool._created == 0


# =============================================================================
# TEST PerformanceOptimizer CLASS
# =============================================================================


class TestPerformanceOptimizer:
    """Tests for PerformanceOptimizer class."""

    @pytest.fixture
    def optimizer(self):
        return PerformanceOptimizer(
            cache_max_size=100,
            cache_default_ttl=300,
            pool_max_size=5,
        )

    def test_optimizer_init(self, optimizer):
        """Test PerformanceOptimizer initialization."""
        assert optimizer.cache is not None
        assert optimizer.session_pool is not None

    def test_optimizer_timer(self, optimizer):
        """Test timer context manager."""
        with optimizer.timer("test_operation") as timer:
            time.sleep(0.01)

        assert timer.duration_ms > 0

    @pytest.mark.asyncio
    async def test_optimizer_async_timer(self, optimizer):
        """Test async timer context manager."""
        async with optimizer.timer("test_async_operation") as timer:
            await asyncio.sleep(0.01)

        assert timer.duration_ms > 0

    def test_optimizer_record_time(self, optimizer):
        """Test recording timing measurements."""
        optimizer.record_time("operation", 100.0)
        optimizer.record_time("operation", 150.0)

        stats = optimizer.get_timer_stats("operation")

        assert stats["count"] == 2
        assert stats["avg"] == 125.0

    def test_optimizer_get_metrics(self, optimizer):
        """Test getting performance metrics."""
        optimizer.record_time("test", 100.0)

        metrics = optimizer.get_metrics()

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_requests == 1

    def test_optimizer_get_all_timer_stats(self, optimizer):
        """Test getting all timer statistics."""
        optimizer.record_time("op1", 100.0)
        optimizer.record_time("op2", 200.0)

        all_stats = optimizer.get_all_timer_stats()

        assert "op1" in all_stats
        assert "op2" in all_stats

    def test_optimizer_reset(self, optimizer):
        """Test resetting the optimizer."""
        optimizer.cache.set("key", "value")
        optimizer.record_time("test", 100.0)
        optimizer.reset()

        assert optimizer.cache.get("key") is None
        assert optimizer.get_timer_stats("test")["count"] == 0

    @pytest.mark.asyncio
    async def test_optimizer_shutdown(self, optimizer):
        """Test shutting down the optimizer."""
        await optimizer.session_pool.acquire()
        await optimizer.shutdown()

        assert optimizer.session_pool._created == 0


# =============================================================================
# TEST CACHING DECORATOR
# =============================================================================


class TestCachingDecorator:
    """Tests for the caching decorator."""

    @pytest.fixture
    def optimizer(self):
        return PerformanceOptimizer()

    @pytest.mark.asyncio
    async def test_cached_async_function(self, optimizer):
        """Test caching async function results."""
        call_count = 0

        @optimizer.cached("test_prefix")
        async def expensive_operation(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        result1 = await expensive_operation(5)
        result2 = await expensive_operation(5)

        assert result1 == 10
        assert result2 == 10
        assert call_count == 1  # Only called once

    def test_cached_sync_function(self, optimizer):
        """Test caching sync function results."""
        call_count = 0

        @optimizer.cached("sync_prefix")
        def simple_operation(x):
            nonlocal call_count
            call_count += 1
            return x + 1

        result1 = simple_operation(5)
        result2 = simple_operation(5)

        assert result1 == 6
        assert result2 == 6
        assert call_count == 1


# =============================================================================
# TEST PerformanceMetrics DATACLASS
# =============================================================================


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics dataclass."""

    def test_metrics_defaults(self):
        """Test PerformanceMetrics default values."""
        metrics = PerformanceMetrics()

        assert metrics.cache_hits == 0
        assert metrics.cache_misses == 0
        assert metrics.cache_size == 0
        assert metrics.avg_response_time_ms == 0.0
        assert metrics.total_requests == 0

    def test_metrics_with_values(self):
        """Test PerformanceMetrics with values."""
        metrics = PerformanceMetrics(
            cache_hits=100,
            cache_misses=20,
            cache_size=50,
            avg_response_time_ms=45.5,
            total_requests=120,
        )

        assert metrics.cache_hits == 100
        assert metrics.cache_misses == 20
