"""
=============================================================================
FILE 16: INTERVIEW Q&A — Common Questions & How to Answer Them
=============================================================================
This is your final review. Real interview questions organized by category.
Practice explaining these OUT LOUD — interviews are verbal.
=============================================================================
"""


# =============================================================================
# CATEGORY 1: PYTHON CORE
# =============================================================================
"""
Q1: What is the GIL and how does it affect multithreading?
A: The Global Interpreter Lock (GIL) allows only one thread to execute
   Python bytecode at a time. This means CPU-bound threads can't run in
   parallel. However, I/O-bound threads benefit because the GIL is released
   during I/O operations. For CPU parallelism, use multiprocessing.
   Python 3.13+ has experimental free-threaded builds without the GIL.

Q2: Explain the difference between deepcopy and shallow copy.
A: Shallow copy creates a new object but references the same nested objects.
   Deep copy creates a new object AND recursively copies all nested objects.
   Example: if you shallow copy a list of lists, modifying a nested list
   in the copy also affects the original.

Q3: What are Python decorators? Give a real-world example.
A: A decorator is a function that wraps another function to add behavior.
   Real examples: @retry for automatic retries, @cache for memoization,
   @app.route in Flask/FastAPI for URL routing, @login_required for auth.
   Under the hood: @deco def func → func = deco(func)

Q4: How does Python's garbage collection work?
A: Python uses reference counting as the primary mechanism — when an
   object's reference count drops to 0, it's immediately freed. For
   circular references (A→B→A), Python has a cyclic garbage collector
   that periodically detects and collects unreachable cycles.

Q5: What's the difference between `is` and `==`?
A: `==` checks VALUE equality (calls __eq__).
   `is` checks IDENTITY (same object in memory).
   Use `is` only for None, True, False: `if x is None`.
   Python caches small integers (-5 to 256), so `is` may work for those
   but you should NEVER rely on it.

Q6: Explain *args and **kwargs.
A: *args collects extra positional arguments into a tuple.
   **kwargs collects extra keyword arguments into a dict.
   Used for flexible function signatures.
   def func(*args, **kwargs): args is tuple, kwargs is dict

Q7: What are generators and when would you use them?
A: Generators are functions that yield values lazily instead of returning
   all at once. Use when: processing large files (don't load entire file),
   infinite sequences, data pipelines. They save memory because values
   are computed on demand.

Q8: What is a context manager?
A: An object that defines __enter__ and __exit__ methods, used with `with`.
   Guarantees cleanup even if exceptions occur. Common use: file handling,
   database connections, locks. Use @contextmanager decorator for simple cases.
"""


# =============================================================================
# CATEGORY 2: OOP & DESIGN
# =============================================================================
"""
Q9: Explain SOLID principles with examples.
A: S — Single Responsibility: Each class does one thing
   O — Open/Closed: Open for extension, closed for modification (use polymorphism)
   L — Liskov Substitution: Subtypes must be substitutable for base types
   I — Interface Segregation: Small focused interfaces, not fat ones
   D — Dependency Inversion: Depend on abstractions, not implementations

Q10: When would you use composition over inheritance?
A: Almost always. Composition ("has-a") is more flexible than inheritance ("is-a").
    Use composition when: you want to swap behaviors, avoid deep hierarchies,
    combine behaviors from multiple sources. Use inheritance when: there's a
    clear "is-a" relationship and the hierarchy is shallow (2-3 levels max).

Q11: What design patterns do you use most in Python?
A: Decorator pattern (Python's @ syntax), Strategy (swap algorithms via
    dependency injection), Repository (abstract data access), Observer
    (event-driven systems), and Factory (create objects based on conditions).
    I also heavily use the "module as singleton" pattern — Python modules
    are cached after first import.

Q12: What is dependency injection and why is it important?
A: DI means passing dependencies to a class rather than creating them inside.
    Benefits: testability (inject mocks), flexibility (swap implementations),
    separation of concerns. FastAPI has built-in DI with Depends().
"""


# =============================================================================
# CATEGORY 3: ASYNC & CONCURRENCY
# =============================================================================
"""
Q13: When would you use asyncio vs threading vs multiprocessing?
A: asyncio: I/O-bound tasks, high concurrency (1000s of connections),
    when using async-compatible libraries.
    threading: I/O-bound tasks with blocking libraries that don't support async.
    multiprocessing: CPU-bound tasks (math, image processing, ML training).
    concurrent.futures provides a simple API for both threads and processes.

Q14: What happens if you call a blocking function in async code?
A: It blocks the ENTIRE event loop. All other coroutines stop until the
    blocking call completes. To avoid this, use loop.run_in_executor()
    to run blocking code in a thread pool.

Q15: How does the asyncio event loop work?
A: The event loop is a single-threaded loop that manages coroutines.
    When a coroutine hits `await`, it suspends and the event loop runs
    other ready coroutines. When the awaited I/O completes, the coroutine
    is resumed. This cooperative multitasking handles thousands of
    concurrent connections with minimal overhead.
"""


# =============================================================================
# CATEGORY 4: SYSTEM DESIGN & ARCHITECTURE
# =============================================================================
"""
Q16: How would you design a REST API for an AI inference service?
A: FastAPI with Pydantic for validation. Load model at startup (lifespan).
    POST /predict endpoint with typed request/response.
    Add caching for repeated predictions (Redis).
    Background tasks for heavy processing.
    Rate limiting per user.
    Streaming endpoint for LLM output (Server-Sent Events).
    Health check endpoint.
    Horizontal scaling with load balancer.

Q17: How do you handle background tasks in Python?
A: For simple cases: FastAPI BackgroundTasks (runs after response).
    For distributed: Celery with Redis/RabbitMQ broker.
    For periodic: Celery Beat or APScheduler.
    Key principles: tasks must be idempotent, have time limits,
    proper error handling and retry logic.

Q18: How would you design a scalable data pipeline?
A: Extract-Transform-Load (ETL) pattern.
    Orchestration: Airflow or Prefect.
    Processing: Celery for Python tasks, Spark for big data.
    Storage: S3 for raw data, PostgreSQL for processed data.
    Monitoring: track pipeline health, data quality, latency.
    Idempotent steps so failed runs can be retried safely.

Q19: Explain CAP theorem.
A: In a distributed system, you can only guarantee 2 of 3:
    Consistency: all nodes see the same data at the same time
    Availability: every request gets a response
    Partition tolerance: system works despite network failures
    Since partitions always happen, you choose between CP (consistent
    but might be unavailable) or AP (available but might be stale).
"""


# =============================================================================
# CATEGORY 5: TESTING & QUALITY
# =============================================================================
"""
Q20: How do you write testable code?
A: 1. Dependency injection (don't hardcode dependencies)
    2. Single responsibility (small, focused functions)
    3. Pure functions where possible (same input → same output)
    4. Avoid global state
    5. Use interfaces/protocols for dependencies

Q21: What's the difference between unit, integration, and e2e tests?
A: Unit: test individual functions/classes in isolation (mock dependencies)
    Integration: test how components work together (real DB, real APIs)
    E2E: test complete user flows (browser automation, full stack)
    Pyramid: many unit tests, some integration, few e2e.

Q22: How do you approach debugging a production issue?
A: 1. Check logs and error messages
    2. Reproduce the issue (locally if possible)
    3. Check recent deployments (git blame, deploy history)
    4. Add targeted logging if needed
    5. Use monitoring tools (metrics, traces)
    6. Isolate the component (is it DB? Network? Code?)
    7. Fix, test, deploy, verify
"""


# =============================================================================
# CATEGORY 6: PYTHON FOR AI/ML
# =============================================================================
"""
Q23: What is RAG and how would you implement it?
A: Retrieval-Augmented Generation. Instead of relying on LLM's training data,
    we retrieve relevant documents from our own data and include them in the
    prompt. Implementation:
    1. Chunk documents → generate embeddings → store in vector DB
    2. On query: embed question → search vector DB → get top-k documents
    3. Construct prompt: "Context: {docs}\n\nQuestion: {query}"
    4. Send to LLM → get grounded response

Q24: How do you serve ML models in production?
A: Load model at startup (not per-request). Use FastAPI for the API layer.
    GPU inference with batching for throughput. Cache predictions for
    repeated inputs. Monitor model latency and accuracy drift.
    Use model registry (MLflow) for versioning. Rolling deployments
    for model updates. Health checks on GPU workers.

Q25: What's the difference between fine-tuning and RAG?
A: Fine-tuning: retrain the model on your data. Expensive, slow to update,
    but great for teaching the model a new style or domain.
    RAG: keep the model as-is, just give it your data as context. Cheaper,
    easy to update (just update the vector DB), great for factual data.
    Use RAG first (cheaper, faster), fine-tune only if RAG isn't enough.
"""


# =============================================================================
# CATEGORY 7: BEHAVIORAL QUESTIONS
# =============================================================================
"""
Q26: Tell me about a challenging technical problem you solved.
FRAMEWORK (STAR):
  Situation: Describe the context
  Task: What was your responsibility?
  Action: What did you DO (technical details!)
  Result: What was the outcome (metrics!)

Example: "Our API response times degraded to 5 seconds under load.
I profiled the application and found N+1 queries in our user listing
endpoint. I implemented eager loading with SQLAlchemy's joinedload
and added Redis caching for frequently accessed data. Response times
dropped to 200ms and we handled 10x more concurrent users."

Q27: How do you handle disagreements with team members?
A: "I focus on data and trade-offs, not opinions. When I disagreed
about using microservices, I created a comparison document outlining
complexity, deployment costs, and team size requirements. We agreed
to start monolith and extract services when we had clear scaling needs.
The key is to understand their perspective, present facts, and find
common ground."

Q28: How do you approach learning a new technology?
A: "I build something real with it. For example, when learning FastAPI,
I built a small service for our team. I read the official docs first
(not blog posts), build a prototype, then gradually add production
concerns like testing, error handling, and monitoring."

Q29: How do you mentor junior developers?
A: "Code reviews are my primary tool. I don't just point out issues —
I explain WHY something is a problem and link to resources. I pair
program for complex tasks. I encourage them to propose solutions
before asking me, which builds critical thinking."
"""


# =============================================================================
# CATEGORY 8: CODING EXERCISE TIPS
# =============================================================================
"""
DURING THE CODING INTERVIEW:

1. DON'T jump into coding immediately
   → Ask clarifying questions (5 min)
   → Discuss examples and edge cases
   → State your approach before coding

2. THINK OUT LOUD
   → "I'm thinking of using a hash map because..."
   → "The brute force would be O(n²), but I can optimize with..."
   → "Let me handle the edge case where..."

3. START SIMPLE
   → Write brute force first if you're stuck
   → Then optimize
   → Interviewers prefer working brute force over broken optimal

4. CODE CLEANLY
   → Use meaningful variable names (not i, j, k everywhere)
   → Write helper functions for clarity
   → Add brief comments for non-obvious logic

5. TEST YOUR CODE
   → Walk through with an example
   → Check edge cases: empty input, single element, duplicates
   → State the time and space complexity

6. PYTHON-SPECIFIC TIPS
   → Use collections (Counter, defaultdict, deque)
   → Use enumerate instead of range(len())
   → Use zip for parallel iteration
   → Use list comprehensions for clean transforms
   → Use sorted with key parameter
   → Use set for O(1) lookups
"""


# =============================================================================
# STUDY PLAN — How to Prepare
# =============================================================================
"""
WEEK 1: Python Fundamentals (Files 01-03)
  → Run each file, understand every example
  → Practice typing the code yourself (not copy-paste)
  → Focus on: data types, comprehensions, generators, type hints

WEEK 2: OOP & Design (Files 04-06)
  → SOLID principles — practice explaining each one
  → Implement one design pattern per day
  → Focus on: when to use what pattern

WEEK 3: Async & Scalability (Files 07-10)
  → Build a small FastAPI app
  → Set up Celery with Redis locally
  → Focus on: understanding async mental model

WEEK 4: DSA & LeetCode (Files 12-13)
  → 2-3 problems per day on LeetCode
  → Focus on PATTERNS, not memorizing solutions
  → Practice: Two Pointers, Sliding Window, Binary Search, DP
  → Time yourself (45 min per problem max)

WEEK 5: System Design & AI (Files 14-15)
  → Practice designing systems on a whiteboard
  → Build a simple RAG system
  → Focus on: trade-offs and communication

WEEK 6: Mock Interviews (File 16)
  → Practice explaining concepts OUT LOUD
  → Time yourself on coding problems
  → Record yourself and review
  → Do mock interviews with friends or online platforms

DAILY ROUTINE:
  → Morning: 1 hour LeetCode (2-3 problems)
  → Afternoon: 1 hour reading/coding Python concepts
  → Evening: 30 min system design practice
  → Before bed: Review flashcards of key concepts
"""


if __name__ == "__main__":
    print("=" * 60)
    print("FILE 16: Interview Q&A & Study Plan")
    print("=" * 60)
    print("""
CONGRATULATIONS! You've completed the Python Senior Interview Prep!

FILES OVERVIEW:
  01 — Python Fundamentals (basics, data types, gotchas)
  02 — Advanced Python (decorators, generators, context managers)
  03 — Type Hints & Modern Python (types, dataclasses, protocols)
  04 — OOP & SOLID Principles (clean design)
  05 — Design Patterns (factory, strategy, observer, etc.)
  06 — Functional Programming (map/filter/reduce, itertools)
  07 — Async Programming (asyncio, gather, queues)
  08 — Concurrency & Parallelism (threading, multiprocessing, GIL)
  09 — Celery & Scalability (background tasks, Redis, caching)
  10 — FastAPI & Web (API development, dependency injection)
  11 — Testing & Quality (pytest, mocking, code quality)
  12 — Data Structures & Algorithms (implementations)
  13 — LeetCode Patterns (problem-solving strategies)
  14 — Python for AI (embeddings, RAG, LangChain)
  15 — System Design (architecture, reliability patterns)
  16 — Interview Q&A (this file — study plan)

NEXT STEPS:
  1. Read each file IN ORDER
  2. RUN every file: python 01_python_fundamentals.py
  3. MODIFY the code — experiment!
  4. Practice explaining concepts OUT LOUD
  5. Do 2-3 LeetCode problems DAILY
  6. Build a small project combining what you've learned
     (e.g., FastAPI + Celery + Redis + RAG system)
    """)
