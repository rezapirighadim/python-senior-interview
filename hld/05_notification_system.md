# HLD 05 — Notification System (Email, SMS, Push)

## Requirements

**Functional:**
- Send notifications via Email, SMS, Push
- Support scheduled / delayed notifications
- User notification preferences (opt-in/out per channel)
- Template-based messages
- Delivery tracking (sent, delivered, failed, read)

**Non-Functional:**
- 10M notifications/day
- Soft real-time (push < 1s, email < 30s, SMS < 10s)
- At-least-once delivery
- High availability
- No duplicate sends (idempotent)

## Estimates

```
Send rate:  10M / 86400 ≈ 115/sec (peak: 500/sec)
Push:       60% of notifications
Email:      30%
SMS:        10%
```

## High-Level Design

```
┌──────────────┐
│ Event Sources │  (API, Cron, Other Services)
└──────┬───────┘
       │
       ▼
┌──────────────┐     ┌───────────────┐
│ Notification  │────▶│ Message Queue  │ (Kafka / RabbitMQ)
│ Service       │     └───────┬───────┘
└──────────────┘              │
                    ┌─────────┼──────────┐
                    ▼         ▼          ▼
              ┌─────────┐ ┌───────┐ ┌────────┐
              │ Email    │ │ SMS   │ │ Push   │
              │ Worker   │ │Worker │ │ Worker │
              └────┬────┘ └───┬───┘ └───┬────┘
                   │          │         │
                   ▼          ▼         ▼
              ┌─────────┐ ┌───────┐ ┌────────┐
              │ SendGrid │ │Twilio │ │FCM/APNs│
              └─────────┘ └───────┘ └────────┘
                   │          │         │
                   ▼          ▼         ▼
              ┌──────────────────────────────┐
              │      Delivery Tracker DB      │
              └──────────────────────────────┘
```

## API Design

```
POST /api/v1/notifications
{
    "recipient_id": "u123",
    "template_id": "welcome_email",
    "channels": ["email", "push"],
    "data": { "name": "Reza", "plan": "Pro" },
    "scheduled_at": "2025-01-15T10:00:00Z"    // optional
}

GET /api/v1/notifications/{id}
→ { "status": "delivered", "channel": "email", "sent_at": "..." }

PUT /api/v1/users/{id}/preferences
{
    "email": true,
    "sms": false,
    "push": true,
    "quiet_hours": { "start": "22:00", "end": "08:00" }
}
```

## Core Components

### 1. Notification Service

Receives requests, validates, applies business rules:

```
1. Validate request (recipient exists, template exists)
2. Fetch user preferences → filter channels
3. Check quiet hours → defer if needed
4. Render template with data
5. Deduplicate (idempotency key check in Redis)
6. Publish to Kafka (one message per channel)
```

### 2. Template Engine

```
Templates table:
  template_id: "welcome_email"
  channel: "email"
  subject: "Welcome, {{name}}!"
  body: "Hi {{name}}, thanks for joining {{plan}} plan..."

Templates table:
  template_id: "welcome_email"
  channel: "push"
  title: "Welcome!"
  body: "Hi {{name}}, you're all set."
```

### 3. Channel Workers

Each worker pool handles one channel:

```
Email Worker:
  1. Consume from Kafka topic: notifications.email
  2. Call SendGrid API
  3. On success: update status → "delivered"
  4. On failure: retry with backoff (max 3 retries)
  5. On permanent failure: update status → "failed", alert

Push Worker:
  1. Consume from Kafka topic: notifications.push
  2. Lookup device tokens for user
  3. Call FCM (Android) / APNs (iOS)
  4. Handle invalid tokens (remove from DB)
```

### 4. Scheduled Notifications

```
Option A: Kafka delayed messages (not natively supported)
Option B: Scheduler table + polling

scheduled_notifications:
  id, notification_data, scheduled_at, status

Cron job every minute:
  SELECT * FROM scheduled_notifications
  WHERE scheduled_at <= NOW() AND status = 'pending'
  → publish to Kafka
  → mark as 'processing'
```

### 5. Delivery Tracking

```sql
CREATE TABLE delivery_log (
    notification_id  UUID PRIMARY KEY,
    recipient_id     VARCHAR NOT NULL,
    channel          VARCHAR NOT NULL,  -- email, sms, push
    template_id      VARCHAR NOT NULL,
    status           VARCHAR NOT NULL,  -- queued, sent, delivered, failed
    provider_id      VARCHAR,           -- SendGrid message ID
    created_at       TIMESTAMP,
    sent_at          TIMESTAMP,
    delivered_at     TIMESTAMP,
    error            TEXT
);
```

## Reliability

### At-least-once delivery
- Kafka consumer commits offset AFTER successful processing
- If worker crashes, message is redelivered

### Idempotency (no duplicate sends)
- Client provides idempotency_key
- Before sending: check Redis `SET idempotency:{key} NX EX 86400`
- If key exists → skip (already sent)

### Retry with backoff
```
Attempt 1: immediate
Attempt 2: after 30s
Attempt 3: after 2min
Attempt 4: after 10min → Dead Letter Queue
```

### Dead Letter Queue (DLQ)
- Permanently failed notifications go to DLQ
- Ops team reviews, fixes, replays

## Scaling

| Component | Strategy |
|-----------|----------|
| Notification Service | Stateless, horizontal scale |
| Kafka | Partition per channel, consumer groups |
| Workers | Auto-scale based on queue depth |
| DB | PostgreSQL with read replicas for tracking queries |
| Templates | Cache in Redis, invalidate on update |

## Monitoring

- Delivery rate per channel (sent/min)
- Failure rate per provider
- End-to-end latency (request → delivered)
- Queue depth (alert if growing)
- Provider API response times
