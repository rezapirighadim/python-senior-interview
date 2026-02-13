# 08 — Concurrency & Parallelism

## The GIL

The Global Interpreter Lock allows only **one thread** to execute Python bytecode at a time.

- CPU-bound threads: **no parallel execution** (GIL serializes them)
- I/O-bound threads: **do benefit** (GIL released during I/O)
- Multiprocessing: **bypasses GIL** (separate processes)
- Python 3.13+: experimental free-threaded builds

## Threading — For I/O-Bound

```python
import threading

def worker(url):
    result = download(url)
    results.append(result)

threads = [threading.Thread(target=worker, args=(url,)) for url in urls]
for t in threads: t.start()
for t in threads: t.join()
```

Thread safety: use `threading.Lock()` for shared mutable state.

## Multiprocessing — For CPU-Bound

```python
from multiprocessing import Pool

with Pool(processes=4) as pool:
    results = pool.map(cpu_intensive, data)
```

## concurrent.futures — The Easy Way

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# I/O-bound
with ThreadPoolExecutor(max_workers=5) as executor:
    results = list(executor.map(fetch_url, urls))

# CPU-bound
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(cpu_task, data))

# Process results as they complete
future_to_url = {executor.submit(fetch, url): url for url in urls}
for future in as_completed(future_to_url):
    result = future.result()
```

## Comparison

|  | Threading | Multiprocessing | Asyncio |
|--|-----------|----------------|---------|
| GIL Impact | Yes | No | N/A |
| Memory | Shared | Separate | Shared |
| Overhead | Low | High | Very Low |
| Best For | I/O-bound | CPU-bound | I/O-bound |
| Scalability | ~100 threads | ~CPU cores | ~10K+ tasks |
| Race Conditions | Yes | Less likely | No |

## Decision Guide

1. Many API calls / network I/O? **asyncio**
2. CPU-heavy processing? **multiprocessing**
3. Blocking I/O libraries? **threading**
4. Simple parallel execution? **concurrent.futures**
