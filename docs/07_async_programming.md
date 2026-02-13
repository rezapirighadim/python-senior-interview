# 07 — Async Programming

## Mental Model

One waiter (event loop) serving many tables (coroutines). Instead of waiting at one table for food, the waiter takes orders from other tables. When food is ready, the waiter picks it up.

**Key rule:** async is for I/O-bound tasks. Use multiprocessing for CPU-bound.

## Basics

```python
async def fetch_data(url):
    await asyncio.sleep(1)  # non-blocking I/O
    return {"url": url}

# MUST await or schedule as task
result = await fetch_data("https://api.example.com")
```

## Concurrent Execution

```python
# asyncio.gather — run concurrently
results = await asyncio.gather(
    fetch_data("/users"),
    fetch_data("/orders"),
    fetch_data("/products"),
)
# Total: ~1 second (not 3!)

# Tasks — start immediately, await later
task1 = asyncio.create_task(fetch_data("/users"))
task2 = asyncio.create_task(fetch_data("/orders"))
result1 = await task1
result2 = await task2
```

## Error Handling

```python
# gather with return_exceptions
results = await asyncio.gather(*tasks, return_exceptions=True)
for r in results:
    if isinstance(r, Exception):
        handle_error(r)

# TaskGroup (3.11+) — structured concurrency
async with asyncio.TaskGroup() as tg:
    task1 = tg.create_task(fetch("/a"))
    task2 = tg.create_task(fetch("/b"))
```

## Semaphore — Limit Concurrency

```python
sem = asyncio.Semaphore(3)  # max 3 concurrent
async def limited_fetch(url):
    async with sem:
        return await fetch(url)
```

## Async Queue — Producer/Consumer

```python
queue = asyncio.Queue(maxsize=10)
await queue.put(item)       # producer
item = await queue.get()    # consumer
queue.task_done()
await queue.join()           # wait for all processed
```

## Timeouts

```python
# wait_for
result = await asyncio.wait_for(slow_op(), timeout=5.0)

# context manager (3.11+)
async with asyncio.timeout(5.0):
    result = await slow_op()
```

## Running Blocking Code

```python
# NEVER call blocking functions directly in async!
# Use run_in_executor
loop = asyncio.get_event_loop()
result = await loop.run_in_executor(None, blocking_function, arg)
```

## Interview Q&A

| Question | Answer |
|----------|--------|
| Threading vs asyncio? | Threading: OS threads, preemptive. Asyncio: single thread, cooperative. Asyncio better for I/O. |
| Mix sync and async? | `asyncio.run()` from sync. `run_in_executor()` from async. |
| Blocking in async code? | Blocks entire event loop. Use `run_in_executor`. |
| gather vs TaskGroup? | gather: returns exceptions. TaskGroup: cancels all on failure (3.11+). |
