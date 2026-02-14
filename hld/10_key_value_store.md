# HLD 10 — Distributed Key-Value Store (Redis / DynamoDB)

## Requirements

**Functional:**
- put(key, value)
- get(key) → value
- delete(key)
- TTL support (auto-expire)
- Configurable consistency (strong / eventual)

**Non-Functional:**
- 100K read/sec, 50K write/sec
- < 10ms latency (p99)
- Data size: 10TB+
- Automatic failover
- Horizontal scaling (add nodes to handle more data)

## Estimates

```
Operations:   150K/sec total
Data:         10TB across cluster
Replication:  3 copies → 30TB raw storage
Nodes:        10TB / 1TB per node = 10 nodes (+ replicas = 30)
```

## High-Level Design

```
┌──────────┐     ┌──────────────┐
│  Client   │────▶│ Coordinator   │
└──────────┘     │ (any node)    │
                 └──────┬───────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │  Node 1   │  │  Node 2   │  │  Node 3   │
    │ (Primary) │  │ (Primary) │  │ (Primary) │
    │  ↓        │  │  ↓        │  │  ↓        │
    │ Replica A │  │ Replica A │  │ Replica A │
    │ Replica B │  │ Replica B │  │ Replica B │
    └──────────┘  └──────────┘  └──────────┘
```

## Data Partitioning — Consistent Hashing

```
Problem: How to distribute keys across N nodes?

Simple hash: node = hash(key) % N
  Problem: when N changes (add/remove node), ALL keys remap!

Consistent Hashing:
  - Nodes and keys are placed on a hash ring (0 to 2^128)
  - Key is assigned to the next node clockwise on the ring
  - When a node is added/removed, only keys near it move (~1/N keys)

    Node A          Node B
      ●───────────────●
     / ↑               \
    /  key1             \
   /                     \
  ●───────────────────────●
 Node D                 Node C
                ↑
               key2

Virtual nodes: each physical node maps to 100+ virtual positions
  - Ensures even distribution
  - Hot nodes can have fewer virtual nodes
```

## Replication

```
Replication factor: N = 3 (each key stored on 3 nodes)

For key K:
  - Primary: Node determined by consistent hashing
  - Replica 1: next node clockwise
  - Replica 2: next-next node clockwise

Write path (quorum: W = 2):
  1. Coordinator sends write to all 3 replicas
  2. Wait for 2 ACKs → return success to client
  3. Third replica catches up asynchronously

Read path (quorum: R = 2):
  1. Coordinator sends read to all 3 replicas
  2. Wait for 2 responses → return latest version

Quorum formula: R + W > N ensures consistency
  Strong consistency: R=2, W=2, N=3 (overlap guaranteed)
  Eventual consistency: R=1, W=1 (fast but may read stale)
```

## Write Path (Single Node)

```
1. Write to Write-Ahead Log (WAL) — durability
2. Write to MemTable (sorted in-memory structure)
3. Return ACK to client

Background:
4. When MemTable is full → flush to SSTable on disk
5. Periodically compact SSTables (merge + remove deleted keys)
```

This is the **LSM Tree** (Log-Structured Merge Tree) architecture, used by Cassandra, RocksDB, LevelDB.

## Read Path (Single Node)

```
1. Check MemTable (most recent data)
2. Check Bloom filter for each SSTable:
   - "Is key possibly in this SSTable?"
   - If NO → skip (saves disk read)
   - If YES → read SSTable
3. Return most recent value found

Bloom filter: ~1% false positive rate, prevents 99% of unnecessary disk reads
```

## Conflict Resolution

```
When replicas disagree (concurrent writes):

Option A: Last-Write-Wins (LWW)
  - Each write has a timestamp
  - Highest timestamp wins
  - Simple but can lose data

Option B: Vector Clocks
  - Track causal ordering: {node1: 3, node2: 1, node3: 2}
  - Detect conflicts (neither version is newer)
  - Return both versions → client resolves

Option C: CRDTs (Conflict-free Replicated Data Types)
  - Data structures that auto-merge
  - Counters, sets, registers
  - No conflicts by design
```

## Failure Handling

### Node Failure Detection
```
Gossip protocol:
  - Each node pings random peers every 1 second
  - If node doesn't respond to 3 pings → marked suspicious
  - If multiple nodes agree → marked dead
  - Dead node's data served by replicas
```

### Temporary Failure — Hinted Handoff
```
Node C is down, write intended for C:
  1. Write to Node D instead (hint: "this belongs to C")
  2. When C comes back → D forwards the write to C
  3. D deletes the hint
```

### Permanent Failure — Anti-entropy
```
Merkle trees:
  - Each node computes hash tree of its data
  - Periodically compare trees between replicas
  - Only sync the branches that differ
  - Efficient: compare O(log N) hashes instead of N keys
```

## Data Model

```
Storage per key:
  key:        string (max 256 bytes)
  value:      bytes (max 1MB)
  version:    vector clock or timestamp
  ttl:        optional expiration time
  metadata:   content type, flags

On-disk format (SSTable):
  [key1|value1|meta1][key2|value2|meta2]...
  + Index: key → offset in file
  + Bloom filter for quick existence check
```

## API

```
PUT /api/v1/kv/{key}
  Body: { "value": "...", "ttl": 3600 }
  Headers: Consistency: strong | eventual

GET /api/v1/kv/{key}
  Headers: Consistency: strong | eventual
  → { "value": "...", "version": "v3" }

DELETE /api/v1/kv/{key}
  → 204 No Content (tombstone written)
```

## Scaling

| Component | Strategy |
|-----------|----------|
| Add capacity | Add nodes → consistent hashing rebalances automatically |
| Hot keys | Read replicas, caching layer, key splitting |
| Cross-region | Async replication between data centers |
| Compaction | Background, rate-limited to avoid I/O spikes |

## Trade-offs Summary

| Decision | Choice | Why |
|----------|--------|-----|
| Partitioning | Consistent hashing + virtual nodes | Minimal rebalancing |
| Replication | Quorum (R+W>N) | Tunable consistency |
| Conflict | LWW (simple) or Vector Clocks (correct) | Depends on use case |
| Storage | LSM Tree | Write-optimized, good for high throughput |
| Failure | Gossip + hinted handoff + anti-entropy | Decentralized, self-healing |
