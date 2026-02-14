# HLD 06 — File Storage Service (Google Drive / Dropbox)

## Requirements

**Functional:**
- Upload / download files
- Create folders
- File versioning
- Share files with other users (view / edit permissions)
- Sync across devices
- Search files by name

**Non-Functional:**
- 50M users, 10M DAU
- Average file size: 500KB, max: 10GB
- Upload: 2 files/user/day = 20M uploads/day
- 99.9% availability
- Strong consistency for file metadata
- Eventual consistency OK for sync across devices

## Estimates

```
Uploads/sec:  20M / 86400 ≈ 230/sec
Storage/day:  20M × 500KB = 10TB/day
Storage/year: ~3.6PB
Read/Write:   5:1 ratio → 1150 reads/sec
```

## High-Level Design

```
┌──────────┐     ┌──────────────┐     ┌──────────────┐
│  Client   │────▶│ API Gateway   │────▶│ Metadata     │
│  (Web/App)│     └──────────────┘     │ Service      │
└──────────┘            │              └──────┬───────┘
      │                 │                     │
      │                 ▼                     ▼
      │          ┌──────────────┐      ┌───────────┐
      │          │ Upload       │      │ Metadata   │
      │          │ Service      │      │ DB (Postgres)│
      │          └──────┬───────┘      └───────────┘
      │                 │
      │                 ▼
      │          ┌──────────────┐
      └─────────▶│ Object Store  │  (S3 / MinIO)
   (direct upload)│              │
                 └──────────────┘
                        │
                        ▼
                 ┌──────────────┐
                 │     CDN       │  (download acceleration)
                 └──────────────┘
```

## Upload Flow

### Small files (< 5MB)

```
1. Client → API: POST /api/v1/files/upload { name, folder_id, size }
2. API returns presigned S3 URL
3. Client uploads directly to S3 (no app server bottleneck!)
4. S3 triggers event → Upload Service
5. Upload Service:
   a. Verify file integrity (checksum)
   b. Generate thumbnail (if image/video)
   c. Create metadata record in DB
   d. Notify Sync Service
```

### Large files (> 5MB — chunked upload)

```
1. Client: POST /api/v1/files/upload/init { name, size, chunk_count }
2. Server returns upload_id + presigned URLs for each chunk
3. Client uploads chunks in parallel to S3
4. Client: POST /api/v1/files/upload/complete { upload_id }
5. Server assembles chunks, creates metadata
```

## Data Model

```sql
-- Users
CREATE TABLE users (
    user_id     UUID PRIMARY KEY,
    email       VARCHAR UNIQUE,
    quota_bytes BIGINT DEFAULT 15000000000  -- 15GB
);

-- Files (metadata only — actual data in S3)
CREATE TABLE files (
    file_id         UUID PRIMARY KEY,
    name            VARCHAR NOT NULL,
    folder_id       UUID REFERENCES folders(folder_id),
    owner_id        UUID REFERENCES users(user_id),
    size_bytes       BIGINT,
    mime_type       VARCHAR,
    s3_key          VARCHAR NOT NULL,
    checksum        VARCHAR,
    current_version INT DEFAULT 1,
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW(),
    is_deleted      BOOLEAN DEFAULT FALSE
);

-- File versions
CREATE TABLE file_versions (
    version_id  UUID PRIMARY KEY,
    file_id     UUID REFERENCES files(file_id),
    version     INT,
    s3_key      VARCHAR NOT NULL,
    size_bytes  BIGINT,
    created_at  TIMESTAMP DEFAULT NOW()
);

-- Folders
CREATE TABLE folders (
    folder_id   UUID PRIMARY KEY,
    name        VARCHAR NOT NULL,
    parent_id   UUID REFERENCES folders(folder_id),
    owner_id    UUID REFERENCES users(user_id)
);

-- Sharing
CREATE TABLE shares (
    share_id    UUID PRIMARY KEY,
    file_id     UUID REFERENCES files(file_id),
    user_id     UUID REFERENCES users(user_id),
    permission  VARCHAR NOT NULL,  -- 'view', 'edit'
    created_at  TIMESTAMP DEFAULT NOW()
);
```

## Sync Service

```
How sync works:
1. Client keeps local state: { file_id: last_modified_at }
2. On app open or periodically:
   GET /api/v1/sync?since=2025-01-15T10:00:00Z
   → Returns list of changed files since timestamp
3. Client downloads changed files, uploads local changes
4. Conflict resolution:
   - If file changed on server AND locally → keep both versions
   - Or: last-write-wins (simpler, lossy)

Real-time sync:
  - WebSocket connection for instant notifications
  - Server pushes { "type": "file_changed", "file_id": "..." }
  - Client fetches new version
```

## Key Design Decisions

### Presigned URLs (direct upload to S3)
- Client uploads directly to S3, bypassing app servers
- App servers never handle file bytes → massive bandwidth savings
- Presigned URL expires after 15 minutes

### Deduplication
- Compute SHA-256 hash of file content
- If hash exists → point to existing S3 object
- Saves storage: many users upload same files (PDFs, images)

### File versioning
- Each update creates a new S3 object (old version preserved)
- Version limit: keep last 30 versions
- Background job cleans up old versions beyond limit

## Scaling

| Component | Strategy |
|-----------|----------|
| Metadata DB | PostgreSQL with read replicas, shard by user_id for >100M users |
| Object Storage | S3 (virtually unlimited, 99.999999999% durability) |
| Upload | Presigned URLs → S3 directly (app servers don't touch file data) |
| Download | CDN (CloudFront) for hot files |
| Search | Elasticsearch index on file metadata |
| Sync | WebSocket servers for real-time, polling fallback |
