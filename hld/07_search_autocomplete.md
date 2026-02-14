# HLD 07 — Search Autocomplete (Typeahead)

## Requirements

**Functional:**
- As user types, show top 5-10 suggestions
- Ranked by popularity / relevance
- Personalized (recent searches)
- Support multiple languages
- Update suggestions as new data arrives

**Non-Functional:**
- 200ms response time (must feel instant)
- 500M queries/day
- Suggestions updated within hours (not real-time)
- High availability

## Estimates

```
QPS:           500M / 86400 ≈ 5800/sec
Peak QPS:      ~20K/sec
Characters/query: avg 5 chars → 5 requests per query
Actual QPS:    ~100K/sec (each keystroke triggers a request)
Data:          5M unique search terms, updated daily
```

## High-Level Design

```
┌──────────┐     ┌──────────────┐     ┌───────────────┐
│  Client   │────▶│ API Gateway   │────▶│ Autocomplete  │
└──────────┘     └──────────────┘     │ Service        │
                                      └───────┬───────┘
                                              │
                              ┌────────────────┼───────────────┐
                              ▼                │               ▼
                       ┌────────────┐   ┌──────┴─────┐  ┌──────────┐
                       │ Trie Cache  │   │ Personalized│  │ Analytics │
                       │ (Redis)     │   │ (Redis)     │  │ Pipeline │
                       └────────────┘   └────────────┘  └──────────┘
                                                              │
                                                              ▼
                                                        ┌──────────┐
                                                        │ Search    │
                                                        │ Logs DB   │
                                                        └──────────┘
```

## API Design

```
GET /api/v1/autocomplete?q=pyt&user_id=u123
Response:
{
    "suggestions": [
        { "text": "python tutorial", "score": 95000 },
        { "text": "python download", "score": 82000 },
        { "text": "python for beginners", "score": 71000 },
        { "text": "pytorch documentation", "score": 45000 },
        { "text": "python interview questions", "score": 38000 }
    ]
}
```

## Core Data Structure: Trie

```
            root
           / | \
          p  j  ...
         /
        y
       /
      t
     / \
    h   o
   /     \
  o       r
 /         \
n           c
|            \
(python:95K)  h
               \
               (pytorch:45K)
```

Each node stores:
- Children map
- Top K suggestions for this prefix (pre-computed!)

### Pre-computing top K at each node

```
At node "pyt":
  top_5 = ["python tutorial", "python download", "python for beginners",
           "pytorch documentation", "python interview questions"]

At node "pyth":
  top_5 = ["python tutorial", "python download", "python for beginners",
           "python interview questions", "python 3.12"]
```

This makes query time O(prefix_length) — just walk the trie and return pre-computed results.

## Data Pipeline

```
┌───────────┐     ┌──────────────┐     ┌──────────────┐     ┌───────────┐
│Search Logs │────▶│ Aggregation  │────▶│ Trie Builder  │────▶│ Redis     │
│(Kafka)     │     │ (Spark/daily)│     │ (Weekly job)  │     │ Cluster   │
└───────────┘     └──────────────┘     └──────────────┘     └───────────┘

1. Every search query is logged to Kafka
2. Daily Spark job aggregates: { term → count } for last 30 days
3. Weekly job rebuilds trie from top 5M terms
4. New trie is loaded into Redis (swap: blue-green)
```

## Storage in Redis

### Option A: Sorted Sets per prefix

```
Key: autocomplete:py
Value: Sorted Set { "python tutorial": 95000, "pytorch": 45000, ... }

Key: autocomplete:pyt
Value: Sorted Set { "python tutorial": 95000, "python download": 82000, ... }
```

Query: `ZREVRANGE autocomplete:{prefix} 0 4` → top 5 by score

### Option B: Hash with serialized trie

Serialize trie nodes as JSON, store in Redis hash.

**Option A is simpler** and works well up to 5M terms.

## Personalization

```
Per-user recent searches (Redis list):
  Key: recent:{user_id}
  Value: ["python async", "fastapi tutorial", ...]

On query:
  1. Get global suggestions (top 5 from trie)
  2. Get personal recent searches matching prefix
  3. Merge: personal first, then global (deduplicated)
  4. Return top 5
```

## Client-Side Optimizations

```
1. Debounce: wait 200ms after last keystroke before requesting
2. Cache: store prefix → results locally
   "p" → results, "py" → results (don't re-request "p" when typing "py")
3. If "pyth" returns 5 results that all start with "python",
   client can filter locally for "pytho" without new request
```

## Filtering Bad Content

- Maintain a blocklist of offensive terms
- Filter at trie build time (don't include blocked terms)
- Real-time block: add to Redis set, check before returning

## Scaling

| Component | Strategy |
|-----------|----------|
| Redis | Cluster, sharded by prefix hash |
| Trie builder | Offline Spark job, runs weekly |
| API servers | Stateless, cached responses (CDN for popular prefixes) |
| Search logs | Kafka → S3 → Spark |

## Trade-offs

| Decision | Choice | Why |
|----------|--------|-----|
| Data structure | Trie with pre-computed top-K | O(prefix_len) query, fast |
| Storage | Redis sorted sets | Fast, supports ranking |
| Update frequency | Weekly rebuild | Balance freshness vs compute cost |
| Personalization | Separate Redis list per user | Simple, fast merge |
| Client | Debounce + local cache | Reduces server load by ~80% |
