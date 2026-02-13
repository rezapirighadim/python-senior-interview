# 16 — Interview Q&A & Study Plan

## Python Core

**Q: What is the GIL?**
Only one thread executes Python bytecode at a time. CPU-bound threads can't run in parallel. I/O threads benefit (GIL released during I/O). Use multiprocessing for CPU parallelism.

**Q: deepcopy vs shallow copy?**
Shallow copies share nested objects. Deep copies are fully independent.

**Q: What are decorators?**
Functions that wrap other functions. Real examples: `@retry`, `@cache`, `@app.route`, `@login_required`.

**Q: How does garbage collection work?**
Reference counting (primary) + cyclic garbage collector (for circular references).

**Q: Generators — when to use?**
Large files (lazy iteration), infinite sequences, data pipelines. Compute on demand, save memory.

## OOP & Design

**Q: SOLID principles?**
S: one responsibility. O: extend, don't modify. L: subtypes are substitutable. I: small interfaces. D: depend on abstractions.

**Q: Composition vs inheritance?**
Prefer composition (has-a). More flexible, avoids deep hierarchies. Use inheritance only for clear "is-a" with shallow depth.

**Q: What design patterns do you use?**
Decorator, Strategy, Repository, Observer, Factory. Module-level singletons.

## Async & Concurrency

**Q: asyncio vs threading vs multiprocessing?**
asyncio: I/O-bound, high concurrency. threading: blocking I/O libraries. multiprocessing: CPU-bound.

**Q: Blocking function in async code?**
Blocks the entire event loop. Use `run_in_executor()`.

## System Design

**Q: Design an AI inference API?**
FastAPI, load model at startup, cache predictions, rate limit, streaming for LLMs, horizontal scaling with load balancer.

**Q: How to handle background tasks?**
Simple: FastAPI BackgroundTasks. Distributed: Celery + Redis. Periodic: Celery Beat.

**Q: CAP theorem?**
Distributed systems: pick 2 of Consistency, Availability, Partition tolerance. Since partitions happen, choose CP or AP.

## AI/ML

**Q: What is RAG?**
Retrieve relevant documents from vector DB, include as context in LLM prompt. Grounds responses in your data.

**Q: Fine-tuning vs RAG?**
RAG first (cheaper, instant updates). Fine-tune only for style/domain language.

## Behavioral (STAR Framework)

- **Situation** → context
- **Task** → your responsibility
- **Action** → what you did (technical details)
- **Result** → outcome with metrics

## Coding Interview Tips

1. Don't jump into coding — clarify first
2. Think out loud
3. Start with brute force, then optimize
4. Use meaningful variable names
5. Test with examples + edge cases
6. State time and space complexity

## 6-Week Study Plan

| Week | Focus | Files |
|------|-------|-------|
| 1 | Python core | 01-03 |
| 2 | OOP & design | 04-06 |
| 3 | Async & scalability | 07-10 |
| 4 | DSA & LeetCode | 12-13 |
| 5 | System design & AI | 14-15 |
| 6 | Mock interviews | 11, 16 |

**Daily:** LeetCode (1hr) + concepts (1hr) + system design (30min)
