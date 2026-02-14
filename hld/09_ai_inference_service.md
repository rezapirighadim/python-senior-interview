# HLD 09 — AI Model Inference Service

## Requirements

**Functional:**
- Serve ML model predictions via REST/gRPC API
- Support multiple models (text, image, embedding)
- Model versioning and A/B testing
- Streaming responses (for LLMs)
- Batch inference for offline processing

**Non-Functional:**
- 1000 requests/sec real-time inference
- Latency: p50 < 100ms, p99 < 500ms (non-LLM models)
- LLM: first token < 500ms, streaming
- 99.9% availability
- GPU utilization > 80%
- Zero-downtime model updates

## High-Level Design

```
┌──────────┐     ┌──────────────┐     ┌──────────────┐
│  Client   │────▶│ API Gateway   │────▶│ Router       │
└──────────┘     │(rate limit,   │     │ Service      │
                 │ auth)         │     └──────┬───────┘
                 └──────────────┘            │
                                    ┌────────┼────────┐
                                    ▼        ▼        ▼
                              ┌─────────┐ ┌───────┐ ┌────────┐
                              │ Model A │ │Model B│ │Model C │
                              │ (GPU)   │ │(GPU)  │ │(CPU)   │
                              └────┬────┘ └───┬───┘ └───┬────┘
                                   │          │         │
                                   ▼          ▼         ▼
                              ┌──────────────────────────────┐
                              │       Model Registry          │
                              │       (MLflow / S3)           │
                              └──────────────────────────────┘

                              ┌──────────────────────────────┐
                              │   Result Cache (Redis)        │
                              └──────────────────────────────┘

                              ┌──────────────────────────────┐
                              │   Batch Queue (Kafka)         │
                              └──────────────────────────────┘
```

## API Design

```
# Real-time inference
POST /api/v1/predict
{
    "model": "sentiment-v2",
    "input": { "text": "This product is amazing!" }
}
→ { "prediction": "positive", "confidence": 0.95, "latency_ms": 42 }

# Streaming (LLM)
POST /api/v1/generate
{
    "model": "llama-3-70b",
    "prompt": "Explain quantum computing",
    "max_tokens": 500,
    "stream": true
}
→ SSE stream: data: {"token": "Quantum"}\ndata: {"token": " computing"}\n...

# Embedding
POST /api/v1/embed
{
    "model": "text-embedding-v3",
    "inputs": ["Hello world", "How are you"]
}
→ { "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]], "dimensions": 768 }

# Batch
POST /api/v1/batch
{
    "model": "image-classifier-v1",
    "inputs": ["s3://bucket/img1.jpg", "s3://bucket/img2.jpg", ...],
    "callback_url": "https://myapp.com/webhook/batch-done"
}
→ { "batch_id": "b-123", "status": "queued", "estimated_time": "5min" }
```

## Core Components

### 1. Router Service

```
Responsibilities:
  - Route to correct model version
  - A/B test traffic splitting (90% v2, 10% v3)
  - Fallback to previous version on errors
  - Check result cache before inference

Routing logic:
  1. Check Redis cache: hash(model + input) → cached result
  2. If cache miss → select model version (A/B config)
  3. Route to healthy GPU pod running that version
  4. Return result + cache it
```

### 2. Model Serving Pods

```
Each pod:
  - Loads ONE model version into GPU memory
  - Runs inference server (FastAPI / Triton / vLLM)
  - Health check endpoint: /health
  - Metrics: latency, throughput, GPU utilization, errors

Lifecycle:
  1. Pod starts → downloads model from registry (S3/MLflow)
  2. Loads model into GPU memory (can take 1-5 min for large models)
  3. Readiness probe passes → starts receiving traffic
  4. On model update → new pods start with new version → old pods drain
```

### 3. Dynamic Batching

```
Problem: GPU is most efficient processing multiple inputs at once.
         But requests arrive one at a time.

Solution: Dynamic batching
  1. Incoming requests queue in a buffer
  2. When buffer has N requests OR timeout (10ms) → batch them
  3. Run single GPU inference on the batch
  4. Split results back to individual requests

Example:
  5 requests arrive within 10ms → batched into 1 GPU call
  GPU processes batch of 5 in 50ms (vs 5 × 40ms = 200ms sequential)
  Per-request latency: 60ms (10ms wait + 50ms inference)
```

### 4. Model Registry

```
MLflow or custom registry:
  - Store model artifacts (weights, config, metadata)
  - Version tracking: model-v1, model-v2, model-v3
  - Metadata: accuracy, training date, dataset version
  - Deployment config: GPU type, memory requirement, batch size

Model artifact stored in S3:
  s3://models/sentiment/v2/model.pt
  s3://models/sentiment/v2/config.json
  s3://models/sentiment/v2/tokenizer/
```

### 5. Result Cache

```
Redis cache for deterministic models:
  Key: cache:{model_name}:{hash(input)}
  Value: { prediction, confidence, model_version }
  TTL: 1 hour (or based on model type)

Not cacheable:
  - LLM with temperature > 0 (non-deterministic)
  - Models with real-time features (current time, user state)

Cache hit rate target: 30-50% (depending on use case)
```

## Model Update — Zero Downtime

```
Blue-Green Deployment:
  1. Current: 4 pods running model-v2 (blue)
  2. Deploy: 4 new pods with model-v3 (green)
  3. Green pods load model, pass health checks
  4. Router shifts traffic: 100% blue → 10% green (canary)
  5. Monitor metrics for 30 min
  6. If good → shift to 100% green
  7. If bad → rollback to 100% blue
  8. Drain and terminate blue pods
```

## GPU Resource Management

```
GPU sharing strategies:
  1. One model per GPU (simplest, wastes memory for small models)
  2. Multiple models per GPU (NVIDIA MPS / time-sharing)
  3. Model sharding across GPUs (for models > single GPU memory)

Resource allocation:
  Model          | GPU Memory | Instances | GPU Type
  sentiment-v2   | 2GB        | 2         | T4
  embedding-v3   | 4GB        | 4         | T4
  llama-3-70b    | 140GB      | 2         | 4×A100 (sharded)
```

## Scaling

| Component | Strategy |
|-----------|----------|
| Router | Stateless, horizontal auto-scale on CPU |
| GPU pods | Auto-scale on GPU utilization + queue depth |
| Cache | Redis Cluster |
| Batch queue | Kafka, scale workers on queue depth |
| Model storage | S3 (cached on pod local SSD) |

## Monitoring

```
Key metrics:
  - Request latency (p50, p95, p99) per model
  - Throughput (req/sec) per model
  - GPU utilization (%) — target > 80%
  - GPU memory usage
  - Cache hit rate
  - Error rate per model version
  - Model accuracy drift (compare predictions to ground truth)
  - Queue depth (batch jobs)

Alerts:
  - p99 latency > 500ms
  - Error rate > 1%
  - GPU utilization < 20% (wasting resources) or > 95% (overloaded)
  - Model accuracy drops below threshold
```
