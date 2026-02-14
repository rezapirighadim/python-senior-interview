# HLD 01 — URL Shortener (bit.ly)

## Requirements

**Functional:**
- Shorten a long URL → short URL
- Redirect short URL → original URL
- Custom aliases (optional)
- Link expiration (TTL)
- Analytics (click count, referrer, geo)

**Non-Functional:**
- 100M URLs created/month
- 10:1 read/write ratio → 1B redirects/month
- Low latency redirects (< 50ms)
- High availability (99.99%)
- Short URLs should be as short as possible

## Estimates

```
Writes:  100M / month ≈ 40 / sec
Reads:   1B / month   ≈ 400 / sec (peak: 2000/sec)
Storage: 100M × 1KB   ≈ 100GB / year
```

## High-Level Design

```
Client
  │
  ▼
┌──────────────┐
│  API Gateway  │  (rate limiting, auth)
└──────┬───────┘
       │
       ▼
┌──────────────┐     ┌───────────┐
│  App Servers  │────▶│   Cache    │  (Redis — hot URLs)
│  (FastAPI)    │     └─────┬─────┘
└──────┬───────┘           │ miss
       │                   ▼
       │            ┌───────────┐
       └───────────▶│  Database  │  (PostgreSQL)
                    └───────────┘
```

## API Design

```
POST /api/v1/shorten
  Body: { "url": "https://...", "custom_alias": "my-link", "ttl_hours": 72 }
  Response: { "short_url": "https://short.ly/Ab3xK", "expires_at": "..." }

GET /{short_code}
  Response: 301 Redirect to original URL

GET /api/v1/stats/{short_code}
  Response: { "clicks": 12345, "created_at": "...", "original_url": "..." }
```

## Database Schema

```sql
CREATE TABLE urls (
    id          BIGSERIAL PRIMARY KEY,
    short_code  VARCHAR(10) UNIQUE NOT NULL,
    original_url TEXT NOT NULL,
    created_at  TIMESTAMP DEFAULT NOW(),
    expires_at  TIMESTAMP,
    click_count BIGINT DEFAULT 0
);

CREATE INDEX idx_short_code ON urls(short_code);
CREATE INDEX idx_expires ON urls(expires_at) WHERE expires_at IS NOT NULL;
```

## Short Code Generation

**Option A — Counter + Base62 (recommended):**
- Auto-increment ID → Base62 encode → "Ab3xK"
- 6 chars = 62^6 = 56 billion combinations
- No collisions, predictable

**Option B — Random hash:**
- MD5(url)[:7] → check for collision → retry
- Unpredictable but collision risk

**Option C — Pre-generated pool:**
- Background worker generates codes ahead of time
- App server pops from pool — no collision check needed

## Key Design Decisions

### Caching (Redis)
```
Read flow:
  1. Check Redis for short_code → original_url
  2. Cache HIT → redirect (95% of cases)
  3. Cache MISS → query DB → store in Redis (TTL 24h) → redirect
```

### 301 vs 302 Redirect
- **301 (Permanent):** Browser caches it. Faster. Less analytics.
- **302 (Temporary):** Browser always hits server. Better for analytics.
- Use **302** if you need click tracking.

### Handling Expiration
- Background cron job cleans expired URLs every hour
- On read: check `expires_at`, return 404 if expired
- Redis TTL auto-evicts expired cached entries

## Scaling

```
                    ┌──────────┐
                    │   CDN    │  (cache 301 redirects at edge)
                    └────┬─────┘
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
       ┌─────────┐ ┌─────────┐ ┌─────────┐
       │ App 1   │ │ App 2   │ │ App 3   │
       └────┬────┘ └────┬────┘ └────┬────┘
            │            │            │
            ▼            ▼            ▼
       ┌─────────────────────────────────┐
       │         Redis Cluster           │
       └────────────────┬────────────────┘
                        │
               ┌────────┴────────┐
               ▼                 ▼
          ┌─────────┐      ┌─────────┐
          │ DB Primary│     │DB Replica│  (read replicas)
          └─────────┘      └─────────┘
```

- **App servers:** stateless, scale horizontally behind load balancer
- **Redis cluster:** cache hot URLs, 95%+ hit rate
- **DB read replicas:** offload read queries
- **CDN:** cache 301 redirects at edge for global latency

## Trade-offs Discussed

| Decision | Option A | Option B |
|----------|----------|----------|
| Code gen | Counter + Base62 (simple, no collision) | Random hash (unpredictable) |
| Redirect | 301 (fast, cached) | 302 (analytics-friendly) |
| DB | SQL (ACID, joins for analytics) | NoSQL (scale, but analytics harder) |
| Cache eviction | TTL-based | LRU |
