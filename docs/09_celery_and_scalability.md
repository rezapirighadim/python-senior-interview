# 09 — Celery & Scalability

## What Is Celery?

Distributed task queue. Run heavy tasks in the **background** while your API responds instantly.

```
Web App → Broker (Redis/RabbitMQ) → Workers
                    ↓
              Result Store
```

## Setup

```python
from celery import Celery

app = Celery("myapp", broker="redis://localhost:6379/0", backend="redis://localhost:6379/1")

app.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
)
```

## Defining Tasks

```python
@app.task
def send_email(to, subject, body):
    return {"status": "sent", "to": to}

@app.task(bind=True, max_retries=3)
def process_payment(self, order_id, amount):
    try:
        return call_api(order_id, amount)
    except Exception as exc:
        raise self.retry(exc=exc, countdown=2**self.request.retries)
```

## Calling Tasks

```python
send_email.delay("user@x.com", "Hi", "Hello")           # fire and forget
result = send_email.apply_async(args=[...], countdown=60) # delayed
result.get(timeout=10)                                     # wait for result

# Chain (sequential)
chain(download.s("data"), process.s(), store.s())()

# Group (parallel)
group(process.s(i) for i in items)()

# Chord (parallel + callback)
chord([process.s(i) for i in items], summarize.s())()
```

## Periodic Tasks (Celery Beat)

```python
app.conf.beat_schedule = {
    "daily-report": {
        "task": "tasks.daily_report",
        "schedule": crontab(hour=0, minute=0),
    },
}
```

## Redis Quick Reference

```python
r.set("key", "value", ex=3600)       # with TTL
r.hset("user:1", mapping={...})      # hash
r.lpush("queue", "item")             # list
r.sadd("online_users", "user1")      # set
r.zadd("leaderboard", {"alice": 100})# sorted set
```

## Caching Patterns

```python
# Cache-aside
def get_user(id):
    cached = cache.get(f"user:{id}")
    if cached: return cached
    user = db.query(id)
    cache.set(f"user:{id}", user, ttl=300)
    return user
```

## Rate Limiting (Token Bucket)

```python
class RateLimiter:
    def __init__(self, rate, capacity):
        self.rate = rate          # tokens per second
        self.capacity = capacity  # burst size
        self.tokens = capacity

    def allow(self):
        self.refill()
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False
```

## Broker Comparison

| Broker | Pros | Cons |
|--------|------|------|
| Redis | Fast, simple | Messages can be lost |
| RabbitMQ | Persistent, reliable | More complex |
| Kafka | Massive throughput, replay | Overkill for simple tasks |
