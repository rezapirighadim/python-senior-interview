# HLD 02 — Real-Time Chat System (WhatsApp / Slack)

## Requirements

**Functional:**
- 1-on-1 messaging
- Group chats (up to 500 members)
- Online/offline status
- Message delivery status (sent, delivered, read)
- Message history / search
- File/image sharing

**Non-Functional:**
- 50M daily active users
- Average user sends 40 messages/day
- Low latency (< 200ms delivery)
- Messages must not be lost
- Eventual consistency is OK for read receipts

## Estimates

```
Messages/day: 50M × 40 = 2B messages/day
Messages/sec: 2B / 86400 ≈ 23K/sec (peak: ~100K/sec)
Storage/day:  2B × 100 bytes ≈ 200GB/day
Storage/year: ~73TB
```

## High-Level Design

```
┌──────────┐          ┌──────────────────┐
│ Client A  │◀═══WS══▶│   Chat Server 1   │
└──────────┘          └────────┬─────────┘
                               │
                        ┌──────┴──────┐
                        │  Message     │
┌──────────┐            │  Queue       │
│ Client B  │◀═══WS══▶ │  (Kafka)     │
└──────────┘            └──────┬──────┘
      ▲                        │
      │                 ┌──────┴──────┐
      │                 │ Message      │
      └════WS══════════▶│ Service      │
                        └──────┬──────┘
                               │
                    ┌──────────┼──────────┐
                    ▼          ▼          ▼
              ┌──────────┐ ┌───────┐ ┌────────┐
              │ Message DB│ │ Cache │ │File CDN│
              │(Cassandra)│ │(Redis)│ │ (S3)   │
              └──────────┘ └───────┘ └────────┘
```

## Core Components

### 1. Connection Service (WebSocket)
- Maintains persistent WebSocket connections
- Each server handles ~50K concurrent connections
- Connection registry in Redis: `user_id → server_id`

### 2. Message Flow

```
1. Client A sends message via WebSocket
2. Chat Server receives message, assigns message_id + timestamp
3. Message pushed to Kafka (topic: per-user or per-chat)
4. Message Service consumes from Kafka:
   a. Store in Cassandra
   b. Check if recipient is online (Redis lookup)
   c. If online → push via WebSocket
   d. If offline → store in pending queue → push notification
5. Client B receives message, sends "delivered" ack
6. Client B opens chat, sends "read" ack
```

### 3. Message Storage

```
Messages table (Cassandra — optimized for write-heavy):
  partition_key: chat_id
  clustering_key: message_id (TimeUUID — sorted by time)
  columns: sender_id, content, type, created_at, status

Recent messages: Redis sorted set (score = timestamp)
  Key: chat:{chat_id}:recent
  Last 50 messages cached
```

### 4. Presence Service
```
- User connects → set key in Redis with TTL: online:{user_id} = server_id
- Heartbeat every 30s → refresh TTL
- User disconnects → key expires → offline
- Subscribe to friends' status via Redis pub/sub
```

### 5. Group Messaging
```
1. Sender sends message to group
2. Fan-out on write (small groups < 500):
   - For each member: enqueue message delivery
3. Message Service delivers to each online member
4. Offline members get it when they reconnect (pull unread)
```

## API Design

```
WebSocket: wss://chat.example.com/ws?token=...

# Send message
{ "type": "message", "chat_id": "c123", "content": "Hello!", "msg_type": "text" }

# Receive message
{ "type": "message", "msg_id": "m456", "chat_id": "c123", "sender": "u1", "content": "Hello!" }

# Typing indicator
{ "type": "typing", "chat_id": "c123" }

# Read receipt
{ "type": "read", "chat_id": "c123", "msg_id": "m456" }

# REST endpoints for history
GET /api/v1/chats/{chat_id}/messages?before={msg_id}&limit=50
GET /api/v1/chats                    # list user's chats
POST /api/v1/chats                   # create group chat
POST /api/v1/chats/{chat_id}/files   # upload file
```

## Scaling

| Component | Strategy |
|-----------|----------|
| WebSocket servers | Horizontal scale, each handles 50K connections |
| Kafka | Partitioned by chat_id, consumer groups per service |
| Cassandra | Partition by chat_id, natural horizontal scaling |
| Redis | Cluster mode, sharded by user_id |
| File storage | S3 + CDN for images/files |
| Push notifications | Separate service, FCM/APNs |

## Handling Edge Cases

- **Message ordering:** TimeUUID in Cassandra guarantees order within a chat
- **Duplicate messages:** Idempotency key (client-generated msg_id)
- **Network disconnect:** Client queues messages locally, resends on reconnect
- **Unread counts:** Redis counter per user per chat, reset on read

## Trade-offs

| Decision | Choice | Why |
|----------|--------|-----|
| Protocol | WebSocket | Full-duplex, low latency |
| Message DB | Cassandra | Write-heavy, time-series data, horizontal scale |
| Queue | Kafka | Durable, ordered, replay capability |
| Delivery | Fan-out on write | Simple for small groups (< 500) |
| Consistency | Eventual | Chat doesn't need strong consistency |
