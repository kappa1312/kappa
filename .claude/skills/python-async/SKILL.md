# Python Async Patterns Skill

## Activation

This skill activates when working on:
- Async function implementation
- Concurrent task execution
- Database async operations
- Session management

## Core Async Patterns

### Basic Async/Await

```python
import anyio

async def fetch_data(url: str) -> dict:
    """Basic async function."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()

# Run from sync context
def main():
    result = anyio.run(fetch_data, "https://api.example.com")
```

### Task Groups (Parallel Execution)

```python
import anyio

async def process_items(items: list[str]) -> list[dict]:
    """Process multiple items in parallel."""
    results: list[dict] = []

    async def process_one(item: str) -> None:
        result = await expensive_operation(item)
        results.append(result)

    async with anyio.create_task_group() as tg:
        for item in items:
            tg.start_soon(process_one, item)

    return results
```

### Semaphore (Concurrency Limiting)

```python
import anyio

async def rate_limited_fetch(
    urls: list[str],
    max_concurrent: int = 5
) -> list[dict]:
    """Fetch URLs with concurrency limit."""
    semaphore = anyio.Semaphore(max_concurrent)
    results: list[dict] = []

    async def fetch_with_limit(url: str) -> None:
        async with semaphore:
            result = await fetch_url(url)
            results.append(result)

    async with anyio.create_task_group() as tg:
        for url in urls:
            tg.start_soon(fetch_with_limit, url)

    return results
```

### Timeouts

```python
import anyio

async def with_timeout(coro, timeout_seconds: float):
    """Execute coroutine with timeout."""
    with anyio.fail_after(timeout_seconds):
        return await coro

# Or with graceful handling
async def with_optional_timeout(coro, timeout_seconds: float):
    """Execute with timeout, return None on timeout."""
    with anyio.move_on_after(timeout_seconds) as cancel_scope:
        return await coro
    if cancel_scope.cancelled_caught:
        return None
```

### Retry Pattern

```python
import anyio
from typing import TypeVar, Callable, Awaitable

T = TypeVar("T")

async def with_retry(
    func: Callable[[], Awaitable[T]],
    max_attempts: int = 3,
    backoff_base: float = 2.0,
    exceptions: tuple = (Exception,),
) -> T:
    """Execute async function with exponential backoff retry."""
    last_exception: Exception | None = None

    for attempt in range(max_attempts):
        try:
            return await func()
        except exceptions as e:
            last_exception = e
            if attempt < max_attempts - 1:
                delay = backoff_base ** attempt
                await anyio.sleep(delay)

    raise last_exception  # type: ignore
```

### Context Managers

```python
from contextlib import asynccontextmanager
from typing import AsyncGenerator

@asynccontextmanager
async def managed_session() -> AsyncGenerator[Session, None]:
    """Async context manager for session lifecycle."""
    session = await create_session()
    try:
        yield session
    finally:
        await session.close()

# Usage
async def do_work():
    async with managed_session() as session:
        result = await session.execute(query)
```

### Producer-Consumer Pattern

```python
import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

async def producer(send_stream: MemoryObjectSendStream[str]) -> None:
    """Produce items to stream."""
    async with send_stream:
        for i in range(100):
            await send_stream.send(f"item-{i}")

async def consumer(
    receive_stream: MemoryObjectReceiveStream[str],
    results: list[str]
) -> None:
    """Consume items from stream."""
    async with receive_stream:
        async for item in receive_stream:
            processed = await process(item)
            results.append(processed)

async def run_pipeline() -> list[str]:
    """Run producer-consumer pipeline."""
    results: list[str] = []
    send_stream, receive_stream = anyio.create_memory_object_stream[str](
        max_buffer_size=10
    )

    async with anyio.create_task_group() as tg:
        tg.start_soon(producer, send_stream)
        tg.start_soon(consumer, receive_stream, results)

    return results
```

## Database Async Patterns

### Connection Pool

```python
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# Engine with connection pooling
engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost/db",
    pool_size=10,          # Maintained connections
    max_overflow=20,       # Additional connections allowed
    pool_timeout=30,       # Seconds to wait for connection
    pool_recycle=3600,     # Recycle connections after 1 hour
    pool_pre_ping=True,    # Verify connections before use
)

# Session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Don't expire objects after commit
)
```

### Transactional Operations

```python
async def transfer_funds(
    from_account: str,
    to_account: str,
    amount: float
) -> None:
    """Atomic transfer with transaction."""
    async with async_session_maker() as session:
        async with session.begin():  # Automatic commit/rollback
            # Debit
            from_acc = await session.get(Account, from_account)
            from_acc.balance -= amount

            # Credit
            to_acc = await session.get(Account, to_account)
            to_acc.balance += amount

            # Both succeed or both fail
```

### Bulk Operations

```python
from sqlalchemy.dialects.postgresql import insert

async def bulk_upsert(
    session: AsyncSession,
    items: list[dict]
) -> None:
    """Efficient bulk insert/update."""
    stmt = insert(MyModel).values(items)
    stmt = stmt.on_conflict_do_update(
        index_elements=["id"],
        set_={
            "name": stmt.excluded.name,
            "updated_at": func.now(),
        }
    )
    await session.execute(stmt)
```

## Claude Agent SDK Async Patterns

### Query Pattern

```python
from claude_agent_sdk import query, ClaudeAgentOptions

async def ask_claude(prompt: str) -> str:
    """Simple async query to Claude."""
    messages: list[str] = []

    async for message in query(prompt=prompt):
        messages.append(str(message))

    return "\n".join(messages)
```

### Session Pattern

```python
from claude_agent_sdk import ClaudeSDKClient

async def interactive_session(task: Task) -> Result:
    """Manage interactive Claude session."""
    client = ClaudeSDKClient()

    try:
        # Start conversation
        response = await client.send(task.prompt)

        # Continue if needed
        while not response.is_complete:
            response = await client.send(response.next_prompt)

        return Result(success=True, output=response.final_output)

    except Exception as e:
        return Result(success=False, error=str(e))

    finally:
        await client.close()
```

### Parallel Sessions

```python
async def run_parallel_sessions(
    tasks: list[Task],
    max_concurrent: int = 5
) -> list[Result]:
    """Run multiple Claude sessions in parallel."""
    semaphore = anyio.Semaphore(max_concurrent)
    results: list[Result] = []

    async def run_one(task: Task) -> None:
        async with semaphore:
            result = await interactive_session(task)
            results.append(result)

    async with anyio.create_task_group() as tg:
        for task in tasks:
            tg.start_soon(run_one, task)

    return results
```

## Error Handling Patterns

### Graceful Degradation

```python
async def fetch_with_fallback(
    primary_url: str,
    fallback_url: str
) -> dict:
    """Try primary, fall back to secondary."""
    try:
        return await fetch_url(primary_url)
    except httpx.HTTPError:
        return await fetch_url(fallback_url)
```

### Error Collection

```python
async def process_all(items: list[str]) -> tuple[list[dict], list[Exception]]:
    """Process all items, collect errors instead of failing fast."""
    results: list[dict] = []
    errors: list[Exception] = []

    async def process_one(item: str) -> None:
        try:
            result = await process(item)
            results.append(result)
        except Exception as e:
            errors.append(e)

    async with anyio.create_task_group() as tg:
        for item in items:
            tg.start_soon(process_one, item)

    return results, errors
```

## Best Practices

1. **Use anyio over asyncio** - Better cancellation, structured concurrency
2. **Always use task groups** - Ensures proper cleanup on errors
3. **Limit concurrency** - Use semaphores to prevent resource exhaustion
4. **Handle timeouts** - All external calls should have timeouts
5. **Use context managers** - Ensure proper resource cleanup
6. **Don't block the event loop** - Use `anyio.to_thread.run_sync()` for sync ops
7. **Test async code** - Use `pytest-asyncio` with `asyncio_mode = "auto"`
