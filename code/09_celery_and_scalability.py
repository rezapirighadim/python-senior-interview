"""
=============================================================================
FILE 09: CELERY & SCALABILITY — Distributed Task Processing
=============================================================================
This is what makes Python apps handle MILLIONS of users.
Celery, Redis, RabbitMQ, caching, rate limiting, background jobs.

MENTAL MODEL:
  → Your web server is a RESTAURANT
  → Celery workers are the KITCHEN STAFF
  → Redis/RabbitMQ is the ORDER TICKET SYSTEM
  → You don't make the customer wait while you cook!

pip install celery redis
=============================================================================
"""


# =============================================================================
# 1. CELERY BASICS — What & Why
# =============================================================================
"""
WHAT IS CELERY?
  → Distributed task queue for Python
  → Lets you run heavy tasks in the BACKGROUND
  → Scales by adding more worker processes

ARCHITECTURE:
  ┌─────────┐     ┌─────────────┐     ┌──────────┐
  │ Web App  │────→│  Broker      │────→│ Worker 1 │
  │ (FastAPI)│     │  (Redis/     │     │ Worker 2 │
  │          │     │  RabbitMQ)   │     │ Worker 3 │
  └─────────┘     └─────────────┘     └──────────┘
                         │
                  ┌──────┴──────┐
                  │ Result Store │
                  │ (Redis/DB)   │
                  └─────────────┘

USE CASES:
  → Sending emails/notifications
  → Image/video processing
  → Report generation
  → Data pipeline processing
  → ML model training/inference
  → Periodic scheduled tasks (cron replacement)
"""


# =============================================================================
# 2. CELERY SETUP & CONFIGURATION
# =============================================================================

# --- celery_app.py ---
"""
from celery import Celery

# Create Celery app
app = Celery(
    "myapp",
    broker="redis://localhost:6379/0",       # Message broker
    backend="redis://localhost:6379/1",       # Result backend
)

# Configuration
app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",

    # Timezone
    timezone="UTC",
    enable_utc=True,

    # Task execution
    task_acks_late=True,              # Ack AFTER task completes (reliability)
    task_reject_on_worker_lost=True,  # Requeue if worker dies
    worker_prefetch_multiplier=1,     # Don't prefetch too many tasks

    # Retry
    task_default_retry_delay=60,      # 60 seconds between retries
    task_max_retries=3,

    # Concurrency
    worker_concurrency=4,             # Number of concurrent workers
)
"""


# =============================================================================
# 3. DEFINING CELERY TASKS
# =============================================================================

"""
from celery_app import app

# --- Basic task ---
@app.task
def send_email(to: str, subject: str, body: str) -> dict:
    # Simulate sending email
    import time
    time.sleep(2)  # This runs in the background!
    return {"status": "sent", "to": to}


# --- Task with retry ---
@app.task(bind=True, max_retries=3, default_retry_delay=60)
def process_payment(self, order_id: str, amount: float) -> dict:
    try:
        # Call payment API
        result = call_payment_api(order_id, amount)
        return result
    except PaymentError as exc:
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)


# --- Task with rate limiting ---
@app.task(rate_limit="10/m")  # Max 10 calls per minute
def call_external_api(data: dict) -> dict:
    return external_api.post(data)


# --- Task with time limit ---
@app.task(soft_time_limit=300, time_limit=600)  # 5min soft, 10min hard
def generate_report(report_id: str) -> str:
    from celery.exceptions import SoftTimeLimitExceeded
    try:
        # Long-running task
        return create_report(report_id)
    except SoftTimeLimitExceeded:
        # Graceful cleanup before hard limit kills us
        save_partial_report(report_id)
        raise
"""


# =============================================================================
# 4. CALLING CELERY TASKS
# =============================================================================

"""
# --- Fire and forget ---
send_email.delay("user@example.com", "Welcome!", "Hello there")

# --- Get result later ---
result = send_email.apply_async(
    args=["user@example.com", "Welcome!", "Hello there"],
    countdown=60,            # Delay execution by 60 seconds
    expires=3600,            # Task expires after 1 hour
    queue="emails",          # Route to specific queue
)

# Check result
print(result.id)             # Task ID
print(result.status)         # PENDING, STARTED, SUCCESS, FAILURE
print(result.ready())        # True if complete
print(result.get(timeout=10))  # Block until result (with timeout)

# --- Chain tasks (pipeline) ---
from celery import chain, group, chord

# Sequential: task1 → task2 → task3
pipeline = chain(
    download_data.s("dataset-1"),    # .s() = signature (lazy task)
    process_data.s(),                # Result of previous is first arg
    store_results.s(),
)
pipeline.apply_async()

# Parallel: run all at once
batch = group(
    process_item.s(item_id) for item_id in item_ids
)
batch.apply_async()

# Chord: parallel + callback when all complete
callback = generate_summary.s()
chord(
    [process_item.s(i) for i in range(10)],
    callback,
).apply_async()
"""


# =============================================================================
# 5. PERIODIC TASKS (Celery Beat)
# =============================================================================

"""
from celery.schedules import crontab

app.conf.beat_schedule = {
    # Run every 5 minutes
    "cleanup-expired-sessions": {
        "task": "tasks.cleanup_sessions",
        "schedule": 300.0,  # seconds
    },

    # Run daily at midnight
    "generate-daily-report": {
        "task": "tasks.daily_report",
        "schedule": crontab(hour=0, minute=0),
    },

    # Run every Monday at 9am
    "weekly-digest": {
        "task": "tasks.send_weekly_digest",
        "schedule": crontab(hour=9, minute=0, day_of_week=1),
    },

    # Run first day of every month
    "monthly-billing": {
        "task": "tasks.process_monthly_billing",
        "schedule": crontab(day_of_month=1, hour=0, minute=0),
    },
}

# Run with: celery -A celery_app beat --loglevel=info
"""


# =============================================================================
# 6. REDIS — In-Memory Data Store
# =============================================================================

"""
import redis

# Connection
r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

# Basic operations
r.set("user:1:name", "Reza")              # SET
r.get("user:1:name")                       # GET → "Reza"
r.set("session:abc", "data", ex=3600)      # SET with expiry (1 hour)
r.delete("user:1:name")                    # DELETE
r.exists("user:1:name")                    # EXISTS → 0

# Hash (like a dict)
r.hset("user:1", mapping={"name": "Reza", "age": "30", "city": "Toronto"})
r.hget("user:1", "name")                  # "Reza"
r.hgetall("user:1")                        # {"name": "Reza", "age": "30", ...}

# List (queue)
r.lpush("queue:tasks", "task1", "task2")   # Push to left
r.rpop("queue:tasks")                      # Pop from right → "task1"

# Set
r.sadd("online_users", "user1", "user2")
r.sismember("online_users", "user1")       # True
r.smembers("online_users")                 # {"user1", "user2"}

# Sorted set (leaderboard)
r.zadd("leaderboard", {"alice": 100, "bob": 85, "charlie": 92})
r.zrevrange("leaderboard", 0, 2, withscores=True)  # Top 3

# Pub/Sub
# Publisher:
r.publish("notifications", "New order received!")
# Subscriber:
pubsub = r.pubsub()
pubsub.subscribe("notifications")
for message in pubsub.listen():
    print(message)
"""


# =============================================================================
# 7. CACHING PATTERNS
# =============================================================================

# --- Simple in-memory cache with TTL ---
import time
from functools import wraps


def ttl_cache(seconds: int = 300):
    """Cache function results with time-to-live."""
    def decorator(func):
        cache = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.time()

            if key in cache:
                result, timestamp = cache[key]
                if now - timestamp < seconds:
                    return result

            result = func(*args, **kwargs)
            cache[key] = (result, now)
            return result

        wrapper.cache_clear = lambda: cache.clear()
        return wrapper
    return decorator


@ttl_cache(seconds=60)
def get_user_profile(user_id: int) -> dict:
    """Expensive database query — cached for 60 seconds."""
    time.sleep(0.5)  # Simulate DB query
    return {"id": user_id, "name": f"User-{user_id}"}


# --- Redis caching pattern ---
"""
import json
import redis

r = redis.Redis(decode_responses=True)

def get_user_cached(user_id: int) -> dict:
    # Try cache first
    cache_key = f"user:{user_id}"
    cached = r.get(cache_key)

    if cached:
        return json.loads(cached)  # Cache HIT

    # Cache MISS — fetch from DB
    user = db.query(f"SELECT * FROM users WHERE id = {user_id}")

    # Store in cache with 5 minute TTL
    r.set(cache_key, json.dumps(user), ex=300)
    return user

def invalidate_user_cache(user_id: int):
    r.delete(f"user:{user_id}")
"""


# =============================================================================
# 8. RATE LIMITING
# =============================================================================

class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, rate: float, capacity: int):
        """
        rate: tokens added per second
        capacity: maximum tokens (burst size)
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = __import__("threading").Lock()

    def allow(self) -> bool:
        """Returns True if request is allowed."""
        with self._lock:
            now = time.time()
            # Refill tokens
            elapsed = now - self.last_refill
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.rate
            )
            self.last_refill = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False


# Usage
limiter = RateLimiter(rate=10, capacity=20)  # 10 requests/sec, burst of 20


# =============================================================================
# 9. MESSAGE QUEUES — Beyond Celery
# =============================================================================
"""
WHEN TO USE WHAT:

Redis as broker:
  ✓ Fast, simple setup
  ✓ Good for small-medium workloads
  ✗ Messages can be lost if Redis crashes (unless using Redis Streams)

RabbitMQ as broker:
  ✓ Message persistence and acknowledgment
  ✓ Complex routing (exchanges, bindings)
  ✓ Better for mission-critical workloads
  ✗ More complex to operate

Redis Streams (modern):
  ✓ Persistent, consumer groups, message acknowledgment
  ✓ Built into Redis (no extra infrastructure)
  ✓ Good balance of speed and reliability

Kafka:
  ✓ Massive throughput (millions of messages/sec)
  ✓ Message replay (consumers can re-read)
  ✓ Best for event sourcing and data pipelines
  ✗ Overkill for simple task queues
"""


# =============================================================================
# 10. PRODUCTION CHECKLIST
# =============================================================================
"""
□ Celery tasks are idempotent (safe to retry)
□ Tasks have time limits (soft + hard)
□ Failed tasks are logged and alerted on
□ Use separate queues for different priority levels
□ Monitor queue depth (alert if growing too fast)
□ Use connection pooling for Redis/database
□ Cache invalidation strategy defined
□ Rate limiting on external API calls
□ Graceful worker shutdown (finish current task, don't accept new)
□ Dead letter queue for permanently failed tasks
□ Periodic cleanup of old task results
□ Health check endpoint for workers
"""


# =============================================================================
# 11. COMPLETE EXAMPLE — Putting It All Together
# =============================================================================

# This is a realistic task structure for an AI company

class TaskSimulator:
    """Simulates a Celery-like task system for demonstration."""

    def __init__(self):
        self.completed: list[dict] = []

    def process_image(self, image_id: str) -> dict:
        """Simulates image processing task."""
        time.sleep(0.1)
        result = {
            "image_id": image_id,
            "status": "processed",
            "features_extracted": 512,
        }
        self.completed.append(result)
        return result

    def run_inference(self, model_id: str, data: list) -> dict:
        """Simulates ML inference task."""
        time.sleep(0.1)
        return {
            "model_id": model_id,
            "predictions": [0.85, 0.12, 0.03],
            "latency_ms": 42,
        }

    def generate_embeddings(self, texts: list[str]) -> dict:
        """Simulates embedding generation."""
        time.sleep(0.05)
        return {
            "count": len(texts),
            "dimensions": 768,
            "embeddings": [[0.1] * 768 for _ in texts],
        }


if __name__ == "__main__":
    print("=" * 60)
    print("FILE 09: Celery & Scalability")
    print("=" * 60)

    print("\n--- TTL Cache ---")
    start = time.perf_counter()
    p1 = get_user_profile(1)
    t1 = time.perf_counter() - start

    start = time.perf_counter()
    p2 = get_user_profile(1)
    t2 = time.perf_counter() - start

    print(f"  First call:  {t1:.3f}s (cache miss)")
    print(f"  Second call: {t2:.6f}s (cache hit)")

    print("\n--- Rate Limiter ---")
    limiter = RateLimiter(rate=5, capacity=3)
    for i in range(6):
        allowed = limiter.allow()
        print(f"  Request {i+1}: {'✓ Allowed' if allowed else '✗ Denied'}")

    print("\n--- Task Simulator ---")
    sim = TaskSimulator()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(sim.process_image, f"img-{i}")
            for i in range(10)
        ]
        for f in futures:
            f.result()
    print(f"  Processed {len(sim.completed)} images")

    inference = sim.run_inference("gpt-4", [1, 2, 3])
    print(f"  Inference: {inference}")

    print("\n✓ File 09 complete. Move to 10_fastapi_and_web.py")
