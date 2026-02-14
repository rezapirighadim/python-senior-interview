# HLD 08 — Web Crawler

## Requirements

**Functional:**
- Crawl web pages starting from seed URLs
- Extract and follow links
- Store page content for indexing
- Respect robots.txt
- Handle different content types (HTML, PDF)

**Non-Functional:**
- Crawl 1 billion pages/month
- Politeness: don't overload any single domain
- Scalable: add more workers to crawl faster
- Fault tolerant: resume after failures
- Avoid duplicate pages

## Estimates

```
Pages/month:  1B
Pages/sec:    1B / (30 × 86400) ≈ 385/sec
Avg page:     500KB
Storage:      1B × 500KB = 500TB/month
Bandwidth:    385 × 500KB = 190MB/sec
```

## High-Level Design

```
┌───────────┐     ┌──────────────┐     ┌──────────────┐
│ Seed URLs  │────▶│ URL Frontier  │────▶│ Fetcher      │
└───────────┘     │ (Priority Q)  │     │ Workers      │
                  └───────┬──────┘     └──────┬───────┘
                          ▲                    │
                          │                    ▼
                  ┌───────┴──────┐     ┌──────────────┐
                  │ URL Filter    │     │ HTML Parser   │
                  │ (Dedup/Rules) │     └──────┬───────┘
                  └──────────────┘            │
                                              ├────────────────┐
                                              ▼                ▼
                                       ┌────────────┐  ┌────────────┐
                                       │ Content    │  │ Link       │
                                       │ Store (S3) │  │ Extractor  │
                                       └────────────┘  └──────┬─────┘
                                                              │
                                                              ▼
                                                       new URLs → URL Frontier
```

## Core Components

### 1. URL Frontier (Priority Queue)

Not a simple queue — it manages:
- **Priority:** important domains crawled first
- **Politeness:** max 1 request/second per domain
- **Freshness:** recrawl popular pages more often

```
Structure:
  Front queues (priority-based):
    High priority:   [cnn.com/..., bbc.com/...]
    Medium priority: [blog1.com/..., wiki.org/...]
    Low priority:    [random-site.com/...]

  Back queues (one per domain — politeness):
    queue_cnn.com:   [url1, url2, url3]
    queue_bbc.com:   [url1, url2]

  Domain delay tracker:
    cnn.com → last_fetched: 10:00:01, min_delay: 1s
```

### 2. Fetcher Workers

```
Worker loop:
  1. Get next URL from frontier (respects politeness)
  2. Check robots.txt (cache per domain)
  3. HTTP GET with timeout (30s)
  4. Handle redirects (max 5 hops)
  5. Pass response to parser
  6. On failure: retry with backoff (max 3)
```

### 3. HTML Parser + Link Extractor

```
1. Parse HTML (extract title, text, metadata)
2. Extract all <a href="..."> links
3. Normalize URLs:
   - Resolve relative paths
   - Remove fragments (#section)
   - Lowercase domain
   - Remove tracking params (utm_source, etc.)
4. Filter URLs:
   - Same domain only? Or follow external?
   - Skip: .jpg, .pdf, .zip (or handle separately)
   - Skip: URLs matching blocklist patterns
5. Send clean URLs to URL Filter
```

### 4. URL Filter (Deduplication)

```
Before adding URL to frontier, check:
  1. Bloom filter: "Have we seen this URL before?"
     - If NO → definitely new → add to frontier
     - If YES → might be duplicate → check DB
  2. Content hash: even different URLs can have same content
     - Compute SimHash of page content
     - Compare with existing hashes → detect near-duplicates
```

**Bloom Filter:**
- Space efficient: 1B URLs ≈ 1GB (10 bits per URL)
- False positive rate: ~1% (acceptable — we just re-check)
- Zero false negatives

### 5. Content Store

```
S3 storage:
  Key: domain/path/hash
  Value: { url, html, text, title, timestamp, headers }

Metadata in DB:
  url, content_hash, last_crawled, status, s3_key
```

## Politeness & robots.txt

```
# robots.txt example (https://example.com/robots.txt)
User-agent: *
Disallow: /private/
Disallow: /admin/
Crawl-delay: 2

User-agent: MyBot
Allow: /api/public/
Disallow: /

# Cache robots.txt per domain (TTL: 24 hours)
# Always respect Crawl-delay
# Default: 1 request per second per domain
```

## Handling Scale

### URL Frontier Distribution

```
Consistent hashing by domain:
  hash("cnn.com") → Frontier Server 1
  hash("bbc.com") → Frontier Server 2

Each frontier server manages politeness for its assigned domains.
Workers pull from any frontier server.
```

### Fetcher Scaling

```
100 fetcher workers × 1 page/sec = 100 pages/sec
Need 385 pages/sec → ~400 workers
Each worker handles: DNS, HTTP, redirect following
Use async HTTP (aiohttp) → each worker handles multiple concurrent fetches
50 async workers × 10 concurrent = 500 pages/sec
```

## Recrawling Strategy

```
Priority-based recrawl:
  News sites:     every 1 hour
  Popular blogs:  every 24 hours
  Static pages:   every 7 days
  Low-traffic:    every 30 days

Detect change rate:
  If page changed on last 5 crawls → increase frequency
  If page unchanged for 3 crawls → decrease frequency
```

## Scaling Summary

| Component | Strategy |
|-----------|----------|
| Frontier | Distributed by domain (consistent hashing) |
| Fetchers | Horizontal scale, async HTTP |
| Parser | Stateless workers, scale with fetchers |
| Dedup | Bloom filter in Redis, content hash in DB |
| Storage | S3 (unlimited), metadata in sharded DB |
| DNS | Local DNS cache (DNS is a bottleneck at scale) |
