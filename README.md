# Python Senior Interview Prep

A comprehensive, hands-on guide to prepare for **Senior Python Developer** interviews at AI companies. Covers everything from fundamentals to system design, with runnable code examples, real-world patterns, and problem-solving strategies.

## Who Is This For?

- Developers returning to Python after a break
- Mid-level devs preparing for senior-level interviews
- Anyone targeting Python roles at AI/ML companies

## Structure

```
python-senior-interview/
├── code/           # Runnable Python files (start here)
│   ├── 01_python_fundamentals.py
│   ├── 02_advanced_python.py
│   ├── ...
│   └── 16_interview_qa.py
├── docs/           # Markdown reference guides
│   ├── 01_python_fundamentals.md
│   ├── ...
│   └── 16_interview_qa.md
├── lld/            # Low-Level Design problems with solutions
│   ├── 01_parking_lot.py
│   ├── ...
│   └── 10_elevator_system.py
├── hld/            # High-Level Design (system design) write-ups
│   ├── 01_url_shortener.md
│   ├── ...
│   └── 10_key_value_store.md
├── PRACTICE_90MIN.md
└── README.md
```

## Topics Covered

| # | Topic | Key Concepts |
|---|-------|-------------|
| 01 | **Python Fundamentals** | Data types, strings, lists, dicts, sets, comprehensions, gotchas |
| 02 | **Advanced Python** | Decorators, generators, context managers, dunder methods, metaclasses |
| 03 | **Type Hints & Modern Python** | Type annotations, dataclasses, protocols, pattern matching, Pydantic |
| 04 | **OOP & SOLID Principles** | Single responsibility, open/closed, Liskov, interface segregation, DI |
| 05 | **Design Patterns** | Singleton, factory, strategy, observer, decorator, builder, repository |
| 06 | **Functional Programming** | Higher-order functions, functools, itertools, composition, immutability |
| 07 | **Async Programming** | asyncio, gather, TaskGroup, semaphores, async generators, queues |
| 08 | **Concurrency & Parallelism** | GIL, threading, multiprocessing, concurrent.futures |
| 09 | **Celery & Scalability** | Distributed tasks, Redis, caching, rate limiting, message queues |
| 10 | **FastAPI & Web** | REST APIs, Pydantic, dependency injection, middleware, streaming |
| 11 | **Testing & Quality** | pytest, mocking, fixtures, property-based testing, code quality tools |
| 12 | **Data Structures & Algorithms** | Hash maps, linked lists, trees, graphs, heaps, tries, sorting |
| 13 | **LeetCode Patterns** | Two pointers, sliding window, binary search, DP, backtracking |
| 14 | **Python for AI** | NumPy, pandas, embeddings, vector DBs, RAG, LangChain, model serving |
| 15 | **System Design** | Architecture patterns, circuit breaker, caching, microservices, API design |
| 16 | **Interview Q&A** | 29 real interview questions, behavioral tips, 6-week study plan |

## Low-Level Design (LLD)

The `lld/` folder contains 10 classic LLD interview problems, each solved with clean Python and real design patterns. Every file is runnable.

| # | Problem | Patterns Used |
|---|---------|--------------|
| 01 | **Parking Lot** | Enum, Strategy (pricing), SRP |
| 02 | **Library Management** | Repository, Enum (status), data modeling |
| 03 | **URL Shortener** | Base62 encoding, Repository, TTL/expiration |
| 04 | **Task Scheduler** | Heap (priority queue), State machine, recurring tasks |
| 05 | **Rate Limiter** | Strategy (3 algorithms: Token Bucket, Sliding Window, Fixed Window) |
| 06 | **Notification Service** | Strategy (channels), Observer, user preferences |
| 07 | **LRU Cache** | Doubly linked list + hash map, OrderedDict alternative |
| 08 | **Event Bus** | Observer/pub-sub, wildcard topics, middleware |
| 09 | **In-Memory File System** | Composite pattern, tree traversal, path parsing |
| 10 | **Elevator System** | State machine, Strategy (dispatch), queue-based scheduling |

```bash
cd lld
python 01_parking_lot.py
python 07_lru_cache.py
```

## High-Level Design (HLD)

The `hld/` folder contains 10 system design problems with full write-ups — the same format you'd present in an interview: requirements, estimates, architecture diagrams, API design, scaling strategy, and trade-offs.

| # | Problem | Key Topics |
|---|---------|-----------|
| 01 | **URL Shortener** | Base62, caching, 301 vs 302, read-heavy scaling |
| 02 | **Chat System** | WebSocket, Kafka, Cassandra, presence, fan-out |
| 03 | **News Feed** | Fan-out on write vs read, hybrid model, feed ranking |
| 04 | **Rate Limiter** | Token bucket, sliding window, Redis Lua scripts, distributed |
| 05 | **Notification System** | Multi-channel, templates, Kafka workers, retry/DLQ |
| 06 | **File Storage** | Presigned URLs, chunked upload, versioning, S3, sync |
| 07 | **Search Autocomplete** | Trie, pre-computed top-K, Redis sorted sets, debounce |
| 08 | **Web Crawler** | URL frontier, politeness, Bloom filter, dedup, LSM |
| 09 | **AI Inference Service** | GPU serving, dynamic batching, model registry, A/B, streaming |
| 10 | **Key-Value Store** | Consistent hashing, quorum, LSM tree, vector clocks, gossip |

Each file follows the **4-step interview framework**: Clarify → Estimate → Design → Deep Dive.

## How to Use

### Run the code files

Every `.py` file is self-contained and runnable:

```bash
cd code
python 01_python_fundamentals.py
python 02_advanced_python.py
# ...and so on
```

### Read the docs

The `docs/` folder has the same content in Markdown format for quick reference and review.

### Suggested study order

**Week 1** — Python core (files 01-03)
**Week 2** — OOP & design (files 04-06)
**Week 3** — Async & scalability (files 07-10)
**Week 4** — DSA & LeetCode (files 12-13)
**Week 5** — System design & AI (files 14-15)
**Week 6** — Mock interviews (files 11, 16)

### Daily routine

- Morning: 1 hour LeetCode (2-3 problems)
- Afternoon: 1 hour reading/coding Python concepts
- Evening: 30 min system design practice

## Requirements

- Python 3.10+ (some features use modern syntax)
- No external dependencies for core files
- Optional: `pip install fastapi uvicorn celery redis pydantic pytest` for web/scalability topics

## Quick Reference

### Big-O Cheat Sheet

| Complexity | Name | Example |
|-----------|------|---------|
| O(1) | Constant | Dict lookup, list append |
| O(log n) | Logarithmic | Binary search |
| O(n) | Linear | List traversal |
| O(n log n) | Linearithmic | Sorting |
| O(n²) | Quadratic | Nested loops |

### Pattern Recognition

| Problem Type | Pattern |
|-------------|---------|
| Sorted array + find pair | Two Pointers |
| Subarray of size k | Sliding Window |
| Find min/max with condition | Binary Search on Answer |
| Count ways / optimize | Dynamic Programming |
| Generate all combinations | Backtracking |
| Overlapping ranges | Merge Intervals |
| Dependencies / ordering | Topological Sort |
| Next greater element | Monotonic Stack |

### When to Use What (Concurrency)

| Task Type | Use |
|----------|-----|
| I/O-bound (API calls, DB) | `asyncio` |
| CPU-bound (math, ML) | `multiprocessing` |
| Blocking I/O libraries | `threading` |
| Background jobs | Celery + Redis |

## License

MIT
