# 15 — System Design

## The 4-Step Framework

1. **Clarify** (5 min) — users, features, scale, constraints
2. **Estimate** (5 min) — QPS, storage, bandwidth
3. **High-level design** (15 min) — boxes and arrows
4. **Deep dive** (15 min) — pick 2-3 components

## Building Blocks

| Component | Purpose | Tools |
|-----------|---------|-------|
| Load Balancer | Distribute traffic | Nginx, HAProxy, AWS ALB |
| API Gateway | Auth, rate limit, routing | Kong, AWS API Gateway |
| App Servers | Run your code | FastAPI + Uvicorn |
| Database | Persist data | PostgreSQL, MongoDB |
| Cache | Speed up reads | Redis |
| Message Queue | Decouple services | Redis, RabbitMQ, Kafka |
| CDN | Static files near users | CloudFront, Cloudflare |
| Search | Full-text search | Elasticsearch |

## Caching Strategy

- **Cache-aside:** app checks cache → miss → query DB → store in cache
- **Write-through:** write to cache AND DB simultaneously
- **TTL:** auto-expire after N seconds

## Database

| SQL (PostgreSQL) | NoSQL (MongoDB) |
|-----------------|-----------------|
| ACID transactions | Flexible schema |
| Complex joins | Horizontal scaling |
| Structured data | High write throughput |
| Users, orders, finance | Logs, events, unstructured |

**Indexing:** B-Tree (default), composite indexes, index WHERE/JOIN/ORDER BY columns.

**Sharding:** split data across DBs. Challenges: cross-shard queries, rebalancing.

## Microservices vs Monolith

Start monolith, extract services as needed. Microservices add complexity (network, debugging, consistency). Only worth it for large teams (20+) with different scaling needs.

## API Design

```
GET    /users          # list
POST   /users          # create
GET    /users/{id}     # read
PATCH  /users/{id}     # update
DELETE /users/{id}     # delete
```

Use nouns, plurals, proper HTTP status codes, versioning, pagination.

## Reliability Patterns

### Circuit Breaker

If a service fails repeatedly, stop calling it temporarily:

```
CLOSED (normal) → OPEN (failing, skip calls) → HALF_OPEN (test one call)
```

### Retry with Backoff

```python
delay = base_delay * (2 ** attempt)  # exponential
delay += random.uniform(0, delay * 0.1)  # jitter
```

### Health Checks

Check all dependencies, report status and latency.

## Design Example: AI Inference Service

```
Client → API Gateway → Load Balancer → GPU Workers
                                           ↓
                                     Model Registry
                                           ↓
                                     Result Cache (Redis)
```

Key decisions: load models at startup, batch requests, cache predictions, auto-scale on queue depth, fallback to CPU.

## Trade-offs to Discuss

- Consistency vs Availability (CAP theorem)
- Latency vs Throughput
- SQL vs NoSQL
- Monolith vs Microservices
- Cache freshness vs Performance
- Build vs Buy
