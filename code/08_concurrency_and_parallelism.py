"""
=============================================================================
FILE 08: CONCURRENCY & PARALLELISM — Threading, Multiprocessing, GIL
=============================================================================
Understanding the GIL and when to use threads vs processes vs async is
a KEY senior-level topic. Get this wrong = instant rejection.

MENTAL MODEL:
  Threading    = Multiple cooks sharing ONE kitchen (GIL = one can cook at a time)
  Multiprocess = Multiple cooks, each with their OWN kitchen
  Asyncio      = One cook who juggles multiple dishes efficiently

WHEN TO USE WHAT:
  I/O-bound (network, disk, DB) → asyncio (best) or threading
  CPU-bound (math, image processing, ML) → multiprocessing
  Simple parallelism → concurrent.futures (ThreadPool or ProcessPool)
=============================================================================
"""
import multiprocessing
import os
import queue
import threading
import time
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
)


# =============================================================================
# 1. THE GIL (Global Interpreter Lock) — MUST KNOW
# =============================================================================
"""
WHAT: The GIL is a mutex that protects Python object access.
      Only ONE thread can execute Python bytecode at a time.

WHY IT EXISTS:
  → CPython uses reference counting for memory management
  → Without GIL, two threads could modify ref count simultaneously → crash
  → GIL makes single-threaded code faster (no locking overhead)

CONSEQUENCES:
  → CPU-bound threads don't run in parallel (GIL serializes them)
  → I/O-bound threads DO benefit (GIL is released during I/O)
  → multiprocessing bypasses GIL (separate processes = separate GILs)

PYTHON 3.13+ (Free-threaded CPython):
  → Experimental build without GIL (--disable-gil)
  → True parallel threading for CPU-bound code
  → Not yet production-ready, but the future direction
"""


# =============================================================================
# 2. THREADING — For I/O-Bound Tasks
# =============================================================================

def download_page(url: str) -> str:
    """Simulates downloading a web page."""
    print(f"  [{threading.current_thread().name}] Downloading {url}...")
    time.sleep(1)  # Simulates network I/O — GIL is released here!
    return f"Content of {url}"


# --- Basic threading ---
def threading_basic():
    threads = []
    results = []

    def worker(url: str):
        result = download_page(url)
        results.append(result)  # Thread-safe for list.append (GIL)

    urls = [f"https://example.com/page/{i}" for i in range(5)]

    for url in urls:
        t = threading.Thread(target=worker, args=(url,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()  # Wait for all threads to complete

    return results


# --- Thread synchronization ---
class ThreadSafeCounter:
    """Counter that can be safely used from multiple threads."""

    def __init__(self):
        self._count = 0
        self._lock = threading.Lock()  # Mutex lock

    def increment(self):
        with self._lock:  # Acquire lock, auto-release on exit
            self._count += 1

    @property
    def count(self):
        return self._count


def thread_safe_demo():
    counter = ThreadSafeCounter()
    threads = []

    def worker():
        for _ in range(10000):
            counter.increment()

    for _ in range(10):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print(f"  Counter: {counter.count} (expected: 100000)")


# --- Thread-safe queue ---
def producer_consumer_threads():
    """Classic producer-consumer with thread-safe queue."""
    q: queue.Queue = queue.Queue(maxsize=10)
    results = []

    def producer():
        for i in range(20):
            q.put(f"item-{i}")
            time.sleep(0.01)
        q.put(None)  # Sentinel to signal "done"

    def consumer():
        while True:
            item = q.get()
            if item is None:
                break
            results.append(item)
            q.task_done()

    t1 = threading.Thread(target=producer)
    t2 = threading.Thread(target=consumer)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    return results


# =============================================================================
# 3. MULTIPROCESSING — For CPU-Bound Tasks
# =============================================================================

def cpu_intensive(n: int) -> int:
    """CPU-bound task: calculate sum of squares."""
    return sum(i * i for i in range(n))


def multiprocessing_basic():
    """Use multiple processes for CPU-bound work."""
    numbers = [5_000_000] * 4  # 4 tasks

    # Sequential (uses 1 CPU core)
    start = time.perf_counter()
    sequential_results = [cpu_intensive(n) for n in numbers]
    seq_time = time.perf_counter() - start

    # Parallel (uses multiple CPU cores)
    start = time.perf_counter()
    with multiprocessing.Pool(processes=4) as pool:
        parallel_results = pool.map(cpu_intensive, numbers)
    par_time = time.perf_counter() - start

    print(f"  Sequential: {seq_time:.2f}s")
    print(f"  Parallel:   {par_time:.2f}s")
    print(f"  Speedup:    {seq_time/par_time:.1f}x")
    return sequential_results == parallel_results


# --- Shared state between processes ---
def shared_state_demo():
    """Sharing data between processes."""

    def worker(shared_array, shared_value, index):
        shared_array[index] = index * index
        with shared_value.get_lock():
            shared_value.value += 1

    # Shared memory (no pickling overhead)
    shared_array = multiprocessing.Array("i", 5)   # Shared int array
    shared_value = multiprocessing.Value("i", 0)    # Shared int

    processes = []
    for i in range(5):
        p = multiprocessing.Process(
            target=worker,
            args=(shared_array, shared_value, i)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print(f"  Array: {list(shared_array)}")
    print(f"  Value: {shared_value.value}")


# =============================================================================
# 4. CONCURRENT.FUTURES — The Easy Way (Recommended!)
# =============================================================================
# High-level API that works the same for threads and processes

def fetch_url(url: str) -> dict:
    """Simulates fetching a URL."""
    time.sleep(0.5)
    return {"url": url, "status": 200}


def concurrent_futures_demo():
    urls = [f"https://api.example.com/item/{i}" for i in range(10)]

    # --- ThreadPoolExecutor (for I/O-bound) ---
    print("  ThreadPoolExecutor:")
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Method 1: map — ordered results
        results = list(executor.map(fetch_url, urls))
    print(f"  Fetched {len(results)} URLs in {time.perf_counter()-start:.2f}s")

    # --- ProcessPoolExecutor (for CPU-bound) ---
    print("\n  ProcessPoolExecutor:")
    numbers = [5_000_000] * 4
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(cpu_intensive, numbers))
    print(f"  Computed {len(results)} results in {time.perf_counter()-start:.2f}s")


def futures_advanced():
    """Using submit() for more control."""
    urls = [f"https://api.example.com/item/{i}" for i in range(5)]

    with ThreadPoolExecutor(max_workers=3) as executor:
        # submit() returns Future objects
        future_to_url = {
            executor.submit(fetch_url, url): url
            for url in urls
        }

        # as_completed — process results as they arrive (not in order!)
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                result = future.result()  # Get result (or raises exception)
                print(f"  {url} → {result['status']}")
            except Exception as e:
                print(f"  {url} → Error: {e}")


# =============================================================================
# 5. COMPARISON TABLE
# =============================================================================
"""
┌──────────────────┬───────────────┬────────────────┬──────────────┐
│                  │  Threading    │ Multiprocessing│   Asyncio    │
├──────────────────┼───────────────┼────────────────┼──────────────┤
│ GIL Impact       │ YES (limited) │ NO (separate)  │ N/A (1 thrd) │
│ Memory           │ Shared        │ Separate*      │ Shared       │
│ Overhead         │ Low           │ High (fork)    │ Very Low     │
│ Best For         │ I/O-bound     │ CPU-bound      │ I/O-bound    │
│ Scalability      │ ~100 threads  │ ~CPU cores     │ ~10K+ tasks  │
│ Race Conditions  │ YES           │ Less likely    │ NO**         │
│ Communication    │ Shared memory │ Queue/Pipe     │ Queue        │
│ Error Handling   │ Complex       │ Complex        │ Simple       │
│ Code Complexity  │ Medium        │ Medium         │ Low-Medium   │
└──────────────────┴───────────────┴────────────────┴──────────────┘

* Can use shared memory via multiprocessing.Value/Array
** No race conditions because only one coroutine runs at a time

DECISION GUIDE:
  1. Need to call many APIs/do network I/O? → asyncio
  2. Need to process CPU-heavy tasks?       → multiprocessing
  3. Using blocking I/O libraries?          → threading
  4. Want simple parallel execution?        → concurrent.futures
"""


# =============================================================================
# 6. THREAD-LOCAL STORAGE
# =============================================================================
# Each thread gets its own copy of the data

thread_local = threading.local()

def process_request(request_id: str):
    thread_local.request_id = request_id
    # Any function called from this thread can access thread_local.request_id
    handle_request()

def handle_request():
    # Access thread-local data without passing it as an argument
    print(f"  Handling request: {thread_local.request_id}")


# =============================================================================
# 7. REAL-WORLD PATTERNS
# =============================================================================

# --- Worker pool with graceful shutdown ---
class WorkerPool:
    """Production-ready worker pool using concurrent.futures."""

    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures = []

    def submit(self, fn, *args, **kwargs):
        future = self.executor.submit(fn, *args, **kwargs)
        self.futures.append(future)
        return future

    def wait_all(self, timeout: float | None = None):
        done, not_done = wait(self.futures, timeout=timeout)
        results = []
        for f in done:
            try:
                results.append(f.result())
            except Exception as e:
                results.append(e)
        return results, not_done

    def shutdown(self, wait: bool = True):
        self.executor.shutdown(wait=wait)


# --- Batch processor ---
def batch_process(items: list, batch_size: int, processor, max_workers: int = 4):
    """Process items in parallel batches."""
    results = []

    # Split into batches
    batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(processor, batch) for batch in batches]
        for future in as_completed(futures):
            results.extend(future.result())

    return results


# =============================================================================
# INTERVIEW QUESTIONS
# =============================================================================
"""
Q: Explain the GIL. Is Python truly multithreaded?
A: The GIL allows only one thread to execute Python bytecode at a time.
   Python IS multithreaded, but CPU-bound threads can't run in parallel.
   I/O-bound threads CAN run in parallel (GIL released during I/O).
   For true CPU parallelism, use multiprocessing or C extensions.

Q: When would you use threading over asyncio?
A: When using libraries that don't support async (blocking I/O).
   When you need preemptive multitasking (thread scheduling).
   When integrating with C extensions that release the GIL.

Q: How do you avoid race conditions in multithreaded Python?
A: Use locks (threading.Lock), queues (queue.Queue),
   thread-local storage, or immutable data.
   Better yet: use asyncio which avoids race conditions by design.

Q: What's the overhead of multiprocessing?
A: Process creation (fork/spawn), data serialization (pickle),
   inter-process communication. Use for tasks >100ms to amortize overhead.
"""


if __name__ == "__main__":
    print("=" * 60)
    print("FILE 08: Concurrency & Parallelism")
    print("=" * 60)

    print("\n--- Threading (I/O-bound) ---")
    start = time.perf_counter()
    results = threading_basic()
    print(f"  Fetched {len(results)} pages in {time.perf_counter()-start:.2f}s")

    print("\n--- Thread-safe counter ---")
    thread_safe_demo()

    print("\n--- Multiprocessing (CPU-bound) ---")
    multiprocessing_basic()

    print("\n--- concurrent.futures ---")
    concurrent_futures_demo()

    print("\n--- Futures advanced (as_completed) ---")
    futures_advanced()

    print("\n✓ File 08 complete. Move to 09_celery_and_scalability.py")
