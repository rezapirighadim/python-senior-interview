# HLD 03 — News Feed System (Twitter / Instagram Feed)

## Requirements

**Functional:**
- User creates a post (text, image, video)
- User sees a feed of posts from people they follow
- Feed is ranked (not just chronological)
- Like, comment, share
- Push notifications for new posts from close friends

**Non-Functional:**
- 200M DAU
- Average user follows 300 people
- Feed loads in < 500ms
- New post appears in followers' feeds within 5 seconds
- High availability

## Estimates

```
Posts/day:     200M users × 2 posts/day = 400M posts
Feed reads:    200M users × 10 opens/day = 2B feed reads
Feed QPS:      2B / 86400 ≈ 23K/sec
Post writes:   400M / 86400 ≈ 4600/sec
Storage/day:   400M × 1KB = 400GB
```

## High-Level Design

```
┌──────────┐     ┌──────────────┐
│  Client   │────▶│ API Gateway   │
└──────────┘     └──────┬───────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
  ┌───────────┐  ┌────────────┐  ┌────────────┐
  │ Post       │  │ Feed        │  │ User       │
  │ Service    │  │ Service     │  │ Service    │
  └─────┬─────┘  └──────┬─────┘  └────────────┘
        │               │
        ▼               ▼
  ┌───────────┐  ┌────────────┐
  │ Post DB   │  │ Feed Cache │
  │(Postgres) │  │  (Redis)   │
  └───────────┘  └────────────┘
        │
        ▼
  ┌───────────────┐
  │ Fan-out Service│ (Kafka → workers)
  └───────────────┘
```

## Feed Generation — The Core Problem

### Option A: Fan-out on Write (Push Model)

```
When user posts:
  1. Save post to Post DB
  2. Get all followers of this user
  3. For each follower: insert post_id into their feed cache (Redis list)

When user reads feed:
  1. Read pre-built feed from Redis → fast!
```

**Pros:** Feed read is instant (O(1) from cache)
**Cons:** Celebrity problem — user with 10M followers = 10M writes per post

### Option B: Fan-out on Read (Pull Model)

```
When user reads feed:
  1. Get list of people they follow
  2. Fetch recent posts from each
  3. Merge + rank → return top N
```

**Pros:** No write amplification
**Cons:** Slow reads, especially for users following many people

### Option C: Hybrid (Best — used in production)

```
Regular users (< 10K followers) → Fan-out on Write
Celebrities (> 10K followers)   → Fan-out on Read

When reading feed:
  1. Get pre-built feed from cache (regular users' posts)
  2. Fetch recent posts from followed celebrities
  3. Merge + rank
```

## Feed Ranking

```
score = (
    engagement_weight × (likes + comments × 2 + shares × 3)
    + recency_weight × time_decay(created_at)
    + relationship_weight × closeness_score(user, author)
    + content_weight × content_quality_score
)
```

For AI companies: this is where ML models come in.
- Feature store for user-post features
- Real-time model serving for ranking
- A/B testing different ranking algorithms

## Data Model

```sql
-- Posts
CREATE TABLE posts (
    post_id     BIGSERIAL PRIMARY KEY,
    user_id     BIGINT NOT NULL,
    content     TEXT,
    media_url   TEXT,
    created_at  TIMESTAMP DEFAULT NOW()
);

-- User follows
CREATE TABLE follows (
    follower_id BIGINT,
    followee_id BIGINT,
    created_at  TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (follower_id, followee_id)
);

-- Feed cache in Redis (per user)
-- Key: feed:{user_id}
-- Value: Sorted Set (score = timestamp, member = post_id)
-- Keep last 500 posts per user
```

## Post Creation Flow

```
1. Client → API Gateway → Post Service
2. Post Service:
   a. Validate + save to Post DB
   b. Upload media to S3 → get CDN URL
   c. Publish event to Kafka: "post.created"
3. Fan-out Worker consumes event:
   a. Get follower list
   b. If followers < 10K: push post_id to each follower's Redis feed
   c. If followers >= 10K: skip (pull on read)
4. Notification Worker:
   a. Send push notification to close friends
```

## Scaling

| Component | Strategy |
|-----------|----------|
| Post DB | Shard by user_id, read replicas |
| Feed Cache | Redis Cluster, sorted sets per user |
| Fan-out | Kafka partitions + worker pool (scale independently) |
| Media | S3 + CloudFront CDN |
| Search | Elasticsearch for post search |
| Ranking | ML model served via separate inference service |

## Trade-offs

| Decision | Choice | Why |
|----------|--------|-----|
| Feed model | Hybrid push/pull | Balances write amplification vs read latency |
| Feed storage | Redis sorted set | O(log n) insert, O(1) range read, auto-eviction |
| Post DB | PostgreSQL | Structured, ACID, good for analytics |
| Queue | Kafka | Durable, handles bursty celebrity posts |
| Ranking | ML model | Personalized, A/B testable |
