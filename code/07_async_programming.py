"""
=============================================================================
FILE 07: ASYNC PROGRAMMING — asyncio, aiohttp, Async Patterns
=============================================================================
Async is CRITICAL for AI companies — API calls, model serving, real-time
systems. This is the #1 topic that separates mid-level from senior.

HOW ASYNC WORKS (mental model):
  → Imagine a restaurant with ONE waiter (event loop)
  → The waiter takes order from Table 1, sends to kitchen (I/O operation)
  → Instead of WAITING at Table 1, waiter goes to Table 2
  → When kitchen signals food ready, waiter picks it up
  → ONE waiter handles MANY tables efficiently

KEY RULE: async is for I/O-bound tasks (network, disk, database)
          NOT for CPU-bound tasks (use multiprocessing for that)
=============================================================================
"""
import asyncio
import time
from typing import Any


# =============================================================================
# 1. BASICS — async/await
# =============================================================================

# A coroutine is defined with `async def`
async def fetch_data(url: str) -> dict:
    """Simulates an API call."""
    print(f"  Fetching {url}...")
    await asyncio.sleep(1)  # Simulates network I/O (non-blocking!)
    return {"url": url, "status": 200}

# IMPORTANT: calling fetch_data() returns a COROUTINE OBJECT, not the result!
# You must `await` it or schedule it as a task.

async def basic_example():
    """Sequential awaiting — each waits for the previous."""
    result1 = await fetch_data("https://api.example.com/users")
    result2 = await fetch_data("https://api.example.com/orders")
    # Total time: ~2 seconds (1+1) — NOT concurrent!
    return [result1, result2]


# =============================================================================
# 2. CONCURRENT EXECUTION — The Whole Point
# =============================================================================

async def concurrent_example():
    """Run multiple coroutines concurrently with gather."""
    # asyncio.gather runs all coroutines concurrently
    results = await asyncio.gather(
        fetch_data("https://api.example.com/users"),
        fetch_data("https://api.example.com/orders"),
        fetch_data("https://api.example.com/products"),
    )
    # Total time: ~1 second (all run at the same time!)
    return results


async def concurrent_with_tasks():
    """Using Tasks for more control."""
    # Create tasks — they start running immediately!
    task1 = asyncio.create_task(fetch_data("https://api.example.com/users"))
    task2 = asyncio.create_task(fetch_data("https://api.example.com/orders"))

    # Do other work while tasks run in the background
    print("  Tasks are running in background...")

    # Await results when needed
    result1 = await task1
    result2 = await task2
    return [result1, result2]


# =============================================================================
# 3. ERROR HANDLING IN ASYNC
# =============================================================================

async def risky_fetch(url: str) -> dict:
    """Simulates a fetch that might fail."""
    await asyncio.sleep(0.5)
    if "error" in url:
        raise ConnectionError(f"Failed to connect to {url}")
    return {"url": url, "status": 200}


async def gather_with_error_handling():
    """gather with return_exceptions=True — don't let one failure kill all."""
    results = await asyncio.gather(
        risky_fetch("https://api.example.com/ok"),
        risky_fetch("https://api.example.com/error"),
        risky_fetch("https://api.example.com/also-ok"),
        return_exceptions=True,  # Returns exceptions instead of raising
    )
    for result in results:
        if isinstance(result, Exception):
            print(f"  Error: {result}")
        else:
            print(f"  Success: {result}")


# =============================================================================
# 4. TASKGROUP — Modern Error Handling (Python 3.11+)
# =============================================================================

async def taskgroup_example():
    """TaskGroup: structured concurrency — if one fails, all are cancelled."""
    try:
        async with asyncio.TaskGroup() as tg:
            task1 = tg.create_task(fetch_data("https://api.example.com/a"))
            task2 = tg.create_task(fetch_data("https://api.example.com/b"))
        # All tasks are guaranteed complete here
        print(f"  Results: {task1.result()}, {task2.result()}")
    except* ConnectionError as eg:
        # ExceptionGroup handling (Python 3.11+)
        for exc in eg.exceptions:
            print(f"  Connection failed: {exc}")


# =============================================================================
# 5. ASYNC GENERATORS — Streaming Data
# =============================================================================

async def stream_data(count: int):
    """Async generator — yields data over time."""
    for i in range(count):
        await asyncio.sleep(0.1)  # Simulates waiting for next chunk
        yield {"chunk": i, "data": f"item-{i}"}

async def consume_stream():
    """Consume an async generator."""
    async for item in stream_data(5):
        print(f"  Received: {item}")


# =============================================================================
# 6. ASYNC CONTEXT MANAGERS
# =============================================================================

class AsyncDatabasePool:
    """Async context manager for database connections."""

    def __init__(self, url: str, pool_size: int = 5):
        self.url = url
        self.pool_size = pool_size

    async def __aenter__(self):
        """Async setup — called with `async with`."""
        print(f"  Connecting to {self.url} (pool_size={self.pool_size})...")
        await asyncio.sleep(0.1)  # Simulate connection time
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async cleanup."""
        print(f"  Closing connection pool...")
        await asyncio.sleep(0.1)

    async def query(self, sql: str) -> list[dict]:
        """Execute a query."""
        await asyncio.sleep(0.1)
        return [{"id": 1, "result": f"data for: {sql}"}]


async def db_example():
    async with AsyncDatabasePool("postgres://localhost/mydb") as db:
        results = await db.query("SELECT * FROM users")
        print(f"  Query results: {results}")


# =============================================================================
# 7. SEMAPHORE — Limit Concurrency
# =============================================================================

async def rate_limited_fetch(sem: asyncio.Semaphore, url: str) -> dict:
    """Respects concurrency limit using semaphore."""
    async with sem:  # Only N coroutines can enter at once
        print(f"  Fetching {url}...")
        await asyncio.sleep(1)
        return {"url": url}


async def semaphore_example():
    """Limit to 3 concurrent requests."""
    sem = asyncio.Semaphore(3)
    urls = [f"https://api.example.com/item/{i}" for i in range(10)]

    results = await asyncio.gather(
        *[rate_limited_fetch(sem, url) for url in urls]
    )
    print(f"  Fetched {len(results)} items (max 3 at a time)")


# =============================================================================
# 8. ASYNC QUEUE — Producer/Consumer Pattern
# =============================================================================

async def producer(queue: asyncio.Queue, name: str, count: int):
    """Produces items and puts them in the queue."""
    for i in range(count):
        item = f"{name}-item-{i}"
        await queue.put(item)
        print(f"  Produced: {item}")
        await asyncio.sleep(0.1)


async def consumer(queue: asyncio.Queue, name: str):
    """Consumes items from the queue."""
    while True:
        item = await queue.get()
        print(f"  {name} consumed: {item}")
        await asyncio.sleep(0.2)  # Simulate processing
        queue.task_done()


async def producer_consumer_example():
    """Producer-consumer with async queue."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=5)

    # Create producers and consumers
    producers = [
        asyncio.create_task(producer(queue, "P1", 3)),
        asyncio.create_task(producer(queue, "P2", 3)),
    ]
    consumers = [
        asyncio.create_task(consumer(queue, "C1")),
        asyncio.create_task(consumer(queue, "C2")),
    ]

    # Wait for all items to be produced
    await asyncio.gather(*producers)
    # Wait for all items to be processed
    await queue.join()

    # Cancel consumers (they run forever)
    for c in consumers:
        c.cancel()


# =============================================================================
# 9. TIMEOUTS AND CANCELLATION
# =============================================================================

async def slow_operation():
    await asyncio.sleep(10)
    return "Done"


async def timeout_example():
    """Set a timeout on an async operation."""
    try:
        # Method 1: asyncio.wait_for
        result = await asyncio.wait_for(slow_operation(), timeout=2.0)
    except asyncio.TimeoutError:
        print("  Operation timed out!")

    # Method 2: asyncio.timeout (Python 3.11+)
    try:
        async with asyncio.timeout(2.0):
            result = await slow_operation()
    except TimeoutError:
        print("  Operation timed out (context manager)!")


# =============================================================================
# 10. RUNNING BLOCKING CODE IN ASYNC
# =============================================================================

def cpu_heavy_task(n: int) -> int:
    """A blocking CPU-bound function."""
    total = 0
    for i in range(n):
        total += i * i
    return total


async def run_blocking_in_async():
    """Run blocking code without blocking the event loop."""
    loop = asyncio.get_event_loop()

    # Run in thread pool (for I/O-bound blocking code)
    result = await loop.run_in_executor(None, cpu_heavy_task, 1_000_000)
    print(f"  Thread pool result: {result}")

    # For CPU-bound: use ProcessPoolExecutor
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, cpu_heavy_task, 1_000_000)
        print(f"  Process pool result: {result}")


# =============================================================================
# 11. REAL-WORLD PATTERN: Async HTTP Client
# =============================================================================
"""
# Using aiohttp (pip install aiohttp)
import aiohttp

async def fetch_multiple_apis():
    async with aiohttp.ClientSession() as session:
        tasks = []
        urls = [
            "https://api.github.com/users/python",
            "https://api.github.com/users/django",
            "https://api.github.com/users/fastapi",
        ]
        for url in urls:
            tasks.append(fetch_url(session, url))
        results = await asyncio.gather(*tasks)
        return results

async def fetch_url(session: aiohttp.ClientSession, url: str) -> dict:
    async with session.get(url) as response:
        return await response.json()
"""


# =============================================================================
# 12. REAL-WORLD PATTERN: Async Retry with Backoff
# =============================================================================

async def async_retry(
    func,
    max_retries: int = 3,
    base_delay: float = 1.0,
    *args,
    **kwargs,
):
    """Retry an async function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"  Retry {attempt+1}/{max_retries} in {delay}s: {e}")
            await asyncio.sleep(delay)


# =============================================================================
# INTERVIEW QUESTIONS
# =============================================================================
"""
Q: What's the difference between threading and asyncio?
A: Threading uses OS threads (preemptive, limited by GIL for CPU tasks).
   Asyncio uses a single thread with cooperative multitasking (coroutines).
   Asyncio is better for I/O-bound tasks (less overhead, no race conditions).
   Threading can be better for blocking I/O libraries that don't support async.

Q: Can you mix sync and async code?
A: Yes. Use `asyncio.run()` to run async from sync.
   Use `loop.run_in_executor()` to run sync from async.
   Libraries like `asgiref` provide sync_to_async and async_to_sync.

Q: What happens if you call a blocking function in async code?
A: It BLOCKS the entire event loop! All other coroutines stop.
   Always use run_in_executor for blocking calls.

Q: What's the difference between gather and TaskGroup?
A: gather: returns results in order, can return_exceptions
   TaskGroup: structured concurrency, cancels all on first failure (3.11+)
   TaskGroup is generally preferred in modern Python.

Q: When should you NOT use async?
A: CPU-bound tasks (use multiprocessing)
   Simple scripts that don't do I/O
   When all libraries you use are synchronous
"""


# =============================================================================
# RUNNING THE EXAMPLES
# =============================================================================
async def main():
    print("=" * 60)
    print("FILE 07: Async Programming")
    print("=" * 60)

    print("\n--- Sequential (slow) ---")
    start = time.perf_counter()
    await basic_example()
    print(f"  Time: {time.perf_counter() - start:.2f}s")

    print("\n--- Concurrent with gather (fast!) ---")
    start = time.perf_counter()
    results = await concurrent_example()
    print(f"  Time: {time.perf_counter() - start:.2f}s")
    print(f"  Got {len(results)} results")

    print("\n--- Error handling ---")
    await gather_with_error_handling()

    print("\n--- Async generator stream ---")
    await consume_stream()

    print("\n--- Async context manager ---")
    await db_example()

    print("\n--- Timeout ---")
    await timeout_example()

    print("\n--- Producer/Consumer ---")
    await producer_consumer_example()

    print("\n✓ File 07 complete. Move to 08_concurrency_and_parallelism.py")


if __name__ == "__main__":
    asyncio.run(main())
