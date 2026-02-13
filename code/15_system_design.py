"""
=============================================================================
FILE 15: SYSTEM DESIGN — Architecture for AI Systems
=============================================================================
Senior developers are expected to design systems, not just code.
This file teaches you how to THINK about system design problems.

SYSTEM DESIGN INTERVIEW FRAMEWORK (4 steps):
  1. CLARIFY requirements (5 min)
  2. ESTIMATE scale (5 min)
  3. DESIGN high-level architecture (15 min)
  4. DEEP DIVE into key components (15 min)
=============================================================================
"""


# =============================================================================
# 1. THE FRAMEWORK — Apply This to Every Problem
# =============================================================================
"""
STEP 1: CLARIFY
  Ask questions! Don't assume.
  - Who are the users? How many?
  - What are the core features?
  - What are the non-functional requirements?
    → Latency? Throughput? Availability? Consistency?
  - Any constraints? (budget, tech stack, team size)

STEP 2: ESTIMATE
  Back-of-envelope calculations:
  - Users: 10M DAU
  - Reads vs writes: 100:1 ratio
  - QPS: 10M / 86400 ≈ 116 req/sec (avg), 580 req/sec (peak = 5x)
  - Storage: 10M users × 1KB = 10GB
  - Bandwidth: 580 req/sec × 1KB = 580KB/sec

STEP 3: HIGH-LEVEL DESIGN
  Draw boxes and arrows:
  Client → Load Balancer → API Servers → Database
                                      → Cache
                                      → Queue → Workers

STEP 4: DEEP DIVE
  Pick 2-3 components and go deep:
  - Database schema and indexing
  - Caching strategy
  - How to handle failures
"""


# =============================================================================
# 2. COMMON BUILDING BLOCKS
# =============================================================================
"""
LOAD BALANCER
  → Distributes traffic across servers
  → Algorithms: Round Robin, Least Connections, IP Hash
  → Tools: Nginx, HAProxy, AWS ALB

API GATEWAY
  → Single entry point for clients
  → Handles: auth, rate limiting, routing, SSL
  → Tools: Kong, AWS API Gateway, Nginx

APPLICATION SERVERS
  → Run your Python code (FastAPI, Django)
  → Stateless (scale horizontally by adding more)
  → Run behind Gunicorn/Uvicorn with multiple workers

DATABASE
  → SQL (PostgreSQL): Structured data, transactions, joins
  → NoSQL (MongoDB): Flexible schema, horizontal scaling
  → Choose based on data model, not hype

CACHE
  → Redis: In-memory key-value store
  → Cache-Aside: App checks cache → miss → query DB → store in cache
  → Write-Through: Write to cache AND DB simultaneously
  → TTL (Time-To-Live): Auto-expire cached data

MESSAGE QUEUE
  → Decouple producers from consumers
  → Redis, RabbitMQ, Kafka
  → Use for: background jobs, event processing, cross-service communication

CDN (Content Delivery Network)
  → Cache static files close to users
  → CloudFront, Cloudflare, Akamai

SEARCH ENGINE
  → Elasticsearch: Full-text search, analytics
  → Use when DB queries are too slow for search
"""


# =============================================================================
# 3. DESIGN PATTERNS FOR SCALABILITY
# =============================================================================

# --- Pattern: Service Layer Architecture ---
from dataclasses import dataclass
from typing import Protocol


class Repository(Protocol):
    def get(self, id: str) -> dict | None: ...
    def save(self, entity: dict) -> None: ...
    def delete(self, id: str) -> None: ...


class Cache(Protocol):
    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str, ttl: int = 300) -> None: ...
    def delete(self, key: str) -> None: ...


class EventBus(Protocol):
    def publish(self, event: str, data: dict) -> None: ...


class UserService:
    """
    Service layer — contains business logic.
    Dependencies are injected (testable, swappable).
    """
    def __init__(self, repo: Repository, cache: Cache, events: EventBus):
        self.repo = repo
        self.cache = cache
        self.events = events

    def get_user(self, user_id: str) -> dict | None:
        # Check cache first
        import json
        cached = self.cache.get(f"user:{user_id}")
        if cached:
            return json.loads(cached)

        # Cache miss → query DB
        user = self.repo.get(user_id)
        if user:
            self.cache.set(f"user:{user_id}", json.dumps(user), ttl=300)
        return user

    def create_user(self, data: dict) -> dict:
        # Validate
        if not data.get("email"):
            raise ValueError("Email required")

        # Save
        self.repo.save(data)

        # Cache
        import json
        self.cache.set(f"user:{data['id']}", json.dumps(data))

        # Publish event (other services react)
        self.events.publish("user.created", data)

        return data

    def delete_user(self, user_id: str) -> None:
        self.repo.delete(user_id)
        self.cache.delete(f"user:{user_id}")
        self.events.publish("user.deleted", {"id": user_id})


# =============================================================================
# 4. DESIGN EXAMPLE: AI Inference Service
# =============================================================================
"""
PROBLEM: Design a system that serves ML model predictions at scale.
  - 1000 requests/sec
  - Latency < 200ms (p99)
  - Models are 1-5GB
  - Models updated weekly

ARCHITECTURE:

Client → API Gateway → Load Balancer → Inference Service (GPU pods)
                                              ↓
                                        Model Registry ← Training Pipeline
                                              ↓
                                        Model Cache (Redis)
                                              ↓
                           ┌──────────────────┼──────────────────┐
                           ↓                  ↓                  ↓
                     GPU Worker 1       GPU Worker 2       GPU Worker 3
                           ↓
                     Result Cache (Redis)

KEY DECISIONS:

1. MODEL LOADING:
   - Load models at startup, not per-request
   - Keep models in GPU memory
   - Use model registry (MLflow) for versioning
   - Rolling deployment for model updates

2. SCALING:
   - Horizontal: add more GPU pods
   - Vertical: larger GPUs
   - Auto-scale based on queue depth
   - Batch requests for throughput (dynamic batching)

3. CACHING:
   - Cache predictions for identical inputs (Redis)
   - Cache model artifacts (prevent re-downloading)

4. RELIABILITY:
   - Health checks on GPU pods
   - Graceful degradation (fallback to CPU model)
   - Circuit breaker for model failures
   - Request timeout (kill long-running predictions)

5. MONITORING:
   - Latency (p50, p95, p99)
   - Throughput (requests/sec)
   - GPU utilization
   - Model accuracy drift
   - Error rates
"""


# =============================================================================
# 5. DESIGN EXAMPLE: Real-Time Chat with AI
# =============================================================================
"""
PROBLEM: Design a ChatGPT-like system.
  - Real-time streaming responses
  - Conversation history
  - Multiple concurrent users

ARCHITECTURE:

Client (WebSocket) → API Gateway → Chat Service
                                      ↓
                                 Conversation Store (PostgreSQL)
                                      ↓
                                 LLM Service
                                      ↓
                            ┌─────────┼──────────┐
                            ↓                     ↓
                    OpenAI/Anthropic API     Self-hosted Model
                            ↓
                    Token Stream → WebSocket → Client

KEY COMPONENTS:

1. WebSocket Connection Manager:
   - Maintains persistent connections
   - Handles reconnection
   - Connection pooling

2. Conversation Store:
   - PostgreSQL for conversation history
   - Redis for active session state
   - Message format: {role, content, timestamp, tokens}

3. LLM Service:
   - Async streaming (yield tokens)
   - Rate limiting per user
   - Token counting and billing
   - Retry with fallback models

4. RAG Integration:
   - Vector DB for knowledge base
   - Retrieve relevant context before LLM call
   - Source attribution in responses
"""


# =============================================================================
# 6. DATABASE DESIGN — Common Interview Topics
# =============================================================================
"""
INDEXING:
  → B-Tree index: Default, good for range queries
  → Hash index: O(1) lookup, exact match only
  → Composite index: (col1, col2) — order matters!
  → Rule: Index columns in WHERE, JOIN, ORDER BY clauses

SHARDING:
  → Split data across multiple databases
  → Shard key: user_id, region, date
  → Challenges: cross-shard queries, rebalancing
  → Use when single DB can't handle the load

REPLICATION:
  → Primary-Replica: Primary for writes, replicas for reads
  → Increases read throughput and availability
  → Trade-off: eventual consistency (replica lag)

SQL vs NoSQL:
  SQL (PostgreSQL):
    → ACID transactions
    → Complex queries and joins
    → Structured data
    → Use for: users, orders, financial data

  NoSQL (MongoDB, DynamoDB):
    → Flexible schema
    → Horizontal scaling
    → High write throughput
    → Use for: logs, events, real-time data, unstructured content
"""


# =============================================================================
# 7. MICROSERVICES vs MONOLITH
# =============================================================================
"""
MONOLITH:
  ✓ Simple to develop and deploy
  ✓ Easy to debug (one process)
  ✓ No network overhead between components
  ✗ Hard to scale specific components
  ✗ One bad deploy affects everything
  ✗ Hard for large teams to work on

MICROSERVICES:
  ✓ Scale individual services independently
  ✓ Teams work independently
  ✓ Technology diversity (different languages per service)
  ✗ Network complexity (latency, failures)
  ✗ Distributed debugging is hard
  ✗ Data consistency across services

WHEN TO USE MICROSERVICES:
  → Large team (>20 developers)
  → Different scaling needs per component
  → Independent deployment is critical
  → Components have different tech requirements

COMMON PATTERN:
  Start monolith → Extract services as needed
  "Don't start with microservices" — Martin Fowler
"""


# =============================================================================
# 8. API DESIGN BEST PRACTICES
# =============================================================================
"""
REST API:
  GET    /users          → List users
  POST   /users          → Create user
  GET    /users/{id}     → Get specific user
  PATCH  /users/{id}     → Update user
  DELETE /users/{id}     → Delete user

  GET    /users/{id}/orders  → User's orders (nested resource)

RULES:
  → Use nouns, not verbs: /users not /getUsers
  → Use plural: /users not /user
  → Use HTTP status codes properly:
    200 OK, 201 Created, 204 No Content
    400 Bad Request, 401 Unauthorized, 403 Forbidden, 404 Not Found
    500 Internal Server Error
  → Version your API: /v1/users
  → Use pagination: ?page=1&limit=20
  → Use filtering: ?status=active&sort=-created_at

gRPC (for internal services):
  → Binary protocol (faster than JSON)
  → Strongly typed (protobuf)
  → Streaming support
  → Use between microservices
"""


# =============================================================================
# 9. RELIABILITY PATTERNS
# =============================================================================

import time
from typing import Callable, Any


# --- Circuit Breaker ---
class CircuitBreaker:
    """
    Prevents cascading failures. If a service fails too many times,
    stop calling it for a while (let it recover).

    States: CLOSED (normal) → OPEN (failing) → HALF_OPEN (testing)
    """
    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"

    def call(self, func: Callable, *args, **kwargs) -> Any:
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit is OPEN — service unavailable")

        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            raise


# --- Retry with Exponential Backoff ---
def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> Any:
    """Retry a function with exponential backoff and jitter."""
    import random

    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, delay * 0.1)
            time.sleep(delay + jitter)


# --- Health Check ---
@dataclass
class HealthStatus:
    service: str
    healthy: bool
    latency_ms: float
    details: str = ""

def check_health(services: dict[str, Callable]) -> list[HealthStatus]:
    """Check health of all dependencies."""
    results = []
    for name, check_fn in services.items():
        start = time.perf_counter()
        try:
            check_fn()
            latency = (time.perf_counter() - start) * 1000
            results.append(HealthStatus(name, True, latency))
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            results.append(HealthStatus(name, False, latency, str(e)))
    return results


# =============================================================================
# 10. INTERVIEW CHEAT SHEET
# =============================================================================
"""
WHEN THEY ASK "Design X":

1. ALWAYS clarify scale and requirements first
2. Start with the simplest architecture that works
3. Then add complexity as needed (caching, queues, sharding)
4. Talk about trade-offs at every decision point
5. Show you know how to monitor and operate the system

COMMON QUESTIONS:
  → Design a URL shortener
  → Design a rate limiter
  → Design a chat application
  → Design a news feed
  → Design an AI inference service
  → Design a data pipeline
  → Design a notification system

TRADE-OFFS TO DISCUSS:
  → Consistency vs Availability (CAP theorem)
  → Latency vs Throughput
  → SQL vs NoSQL
  → Monolith vs Microservices
  → Push vs Pull
  → Cache freshness vs Performance
  → Build vs Buy
"""


if __name__ == "__main__":
    print("=" * 60)
    print("FILE 15: System Design")
    print("=" * 60)

    print("\n--- Circuit Breaker Demo ---")
    cb = CircuitBreaker(failure_threshold=3, reset_timeout=5)
    call_count = 0

    def unreliable_service():
        nonlocal call_count
        call_count += 1
        if call_count <= 4:
            raise ConnectionError("Service down")
        return "Success!"

    for i in range(6):
        try:
            result = cb.call(unreliable_service)
            print(f"  Call {i+1}: {result} (state={cb.state})")
        except Exception as e:
            print(f"  Call {i+1}: {e} (state={cb.state})")

    print("\n--- Health Check ---")
    health = check_health({
        "database": lambda: time.sleep(0.01),
        "cache": lambda: time.sleep(0.001),
        "broken_service": lambda: (_ for _ in ()).throw(ConnectionError("down")),
    })
    for h in health:
        status = "✓" if h.healthy else "✗"
        print(f"  {status} {h.service}: {h.latency_ms:.1f}ms {h.details}")

    print("\n✓ File 15 complete. Move to 16_interview_qa.py")
