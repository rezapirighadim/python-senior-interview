# HLD 04 — Distributed Rate Limiter

## Requirements

**Functional:**
- Limit API requests per user/IP/API key
- Support different limits for different endpoints
- Return proper HTTP 429 (Too Many Requests)
- Configurable rules (100 req/min for /api, 10 req/min for /login)

**Non-Functional:**
- Distributed (works across multiple app servers)
- Low latency (< 5ms overhead per request)
- Accurate under high concurrency
- Fault tolerant (if rate limiter fails, allow traffic — fail open)

## High-Level Design

```
Client → API Gateway → Rate Limiter → App Server
                           │
                           ▼
                    ┌──────────────┐
                    │ Redis Cluster │
                    └──────────────┘
```

Rate limiter sits at the API Gateway level (middleware), before reaching application code.

## Algorithms

### 1. Token Bucket (most common)

```
Each user has a bucket:
  - Capacity: 100 tokens
  - Refill rate: 10 tokens/second

Request arrives:
  - If tokens >= 1 → allow, tokens -= 1
  - If tokens < 1  → deny (429)
  - Tokens refill continuously (up to capacity)
```

**Pros:** Allows bursts, smooth rate limiting
**Cons:** Two parameters to tune

### 2. Sliding Window Log

```
Track timestamps of all requests in a window:
  - Window: 60 seconds, limit: 100
  - On request: remove entries older than 60s, count remaining
  - If count < 100 → allow
  - If count >= 100 → deny
```

**Pros:** Very accurate
**Cons:** High memory (stores all timestamps)

### 3. Sliding Window Counter

```
Combine fixed window counters with weighted overlap:
  - Current window count: 40 (30s into 60s window)
  - Previous window count: 80
  - Weighted: 80 × (30/60) + 40 = 80
  - If 80 < 100 → allow
```

**Pros:** Low memory, reasonably accurate
**Cons:** Approximate

### 4. Fixed Window Counter

```
Divide time into fixed windows (e.g., per minute):
  - Key: "user:123:window:2024-01-15T10:30"
  - Increment counter on each request
  - If counter > limit → deny
```

**Pros:** Simplest, lowest memory
**Cons:** Boundary burst (2x limit at window edge)

## Redis Implementation

### Token Bucket with Lua Script

```lua
-- Atomic token bucket in Redis (Lua script)
local key = KEYS[1]
local rate = tonumber(ARGV[1])       -- tokens per second
local capacity = tonumber(ARGV[2])
local now = tonumber(ARGV[3])

local data = redis.call('HMGET', key, 'tokens', 'last_refill')
local tokens = tonumber(data[1]) or capacity
local last_refill = tonumber(data[2]) or now

-- Refill
local elapsed = now - last_refill
tokens = math.min(capacity, tokens + elapsed * rate)

-- Try consume
local allowed = 0
if tokens >= 1 then
    tokens = tokens - 1
    allowed = 1
end

redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
redis.call('EXPIRE', key, math.ceil(capacity / rate) * 2)

return allowed
```

### Sliding Window with Redis Sorted Set

```
Key: rate:{user_id}:{endpoint}
Score: timestamp
Member: request_id

On request:
  ZREMRANGEBYSCORE key 0 (now - window)   -- remove old
  ZCARD key                                -- count current
  if count < limit:
      ZADD key now request_id              -- add new
      return ALLOW
  else:
      return DENY
```

## Configuration

```yaml
rate_limits:
  - endpoint: "/api/*"
    limits:
      - window: 60       # seconds
        max_requests: 100
        key: "user_id"

  - endpoint: "/api/login"
    limits:
      - window: 300
        max_requests: 5
        key: "ip"

  - endpoint: "/api/upload"
    limits:
      - window: 3600
        max_requests: 20
        key: "user_id"
```

## HTTP Response

```
# When rate limited:
HTTP 429 Too Many Requests
Headers:
  X-RateLimit-Limit: 100
  X-RateLimit-Remaining: 0
  X-RateLimit-Reset: 1705312800     # Unix timestamp when limit resets
  Retry-After: 30                    # seconds until retry
```

## Distributed Challenges

### Problem: Multiple app servers, one user

```
Server A: reads counter = 99    (allows)
Server B: reads counter = 99    (allows)
Both increment → counter = 101  (over limit!)
```

### Solution: Redis atomic operations

- All servers share one Redis
- Lua scripts are atomic in Redis
- No race condition

### Problem: Redis goes down

**Fail open:** allow all traffic (availability > rate limiting)
**Local fallback:** each server keeps a local approximate counter

## Scaling

| Component | Strategy |
|-----------|----------|
| Redis | Cluster mode, shard by user_id hash |
| Rules | Store in config service, cache locally |
| Monitoring | Track 429 rates, alert on spikes |
| Multi-region | Rate limit per region + global aggregate |

## Trade-offs

| Decision | Choice | Why |
|----------|--------|-----|
| Algorithm | Token Bucket | Allows bursts, smooth limiting |
| Storage | Redis | Atomic, fast, distributed |
| Failure mode | Fail open | Availability over strictness |
| Scope | Per-user + per-IP | Prevents abuse while allowing legitimate bursts |
