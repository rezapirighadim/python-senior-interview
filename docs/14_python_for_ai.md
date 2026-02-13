# 14 — Python for AI

## NumPy

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.zeros((3, 4))
c = np.random.randn(3, 3)

# Vectorized (fast — no Python loops)
a + b           # element-wise
a @ b           # matrix multiplication
m[m > 5]        # boolean indexing
m.reshape(3, 4) # reshape
```

NumPy is ~100x faster than Python loops (runs in C, contiguous memory).

## Pandas

```python
import pandas as pd

df = pd.DataFrame({"name": [...], "age": [...], "salary": [...]})
df[df["age"] > 28]                          # filter
df.groupby("dept")["salary"].mean()         # aggregate
df.sort_values("salary", ascending=False)   # sort
df["bonus"] = df["salary"] * 0.1            # vectorized operation
```

## LLM APIs

```python
# OpenAI
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}],
)

# Anthropic
import anthropic
client = anthropic.Anthropic()
message = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    messages=[{"role": "user", "content": prompt}],
)
```

## Embeddings & Vector Databases

**Embeddings** convert text into numerical vectors. Similar content produces similar vectors.

**Vector databases** store millions of vectors for fast similarity search.

| DB | Best For |
|----|----------|
| ChromaDB | Local prototyping |
| Pinecone | Production, managed |
| Weaviate | Open-source, hybrid search |
| Milvus | Billions of vectors |

## RAG (Retrieval-Augmented Generation)

The most important pattern in production AI:

1. **Ingest:** chunk documents, generate embeddings, store in vector DB
2. **Query:** embed question, search vector DB, get relevant docs
3. **Generate:** feed context + question to LLM, get grounded response

**Why RAG?** LLMs don't know your data. RAG grounds responses in your documents. Cheaper than fine-tuning.

## Model Serving

```python
# Load at startup, not per-request
@app.on_event("startup")
async def load_model():
    app.state.model = load_model("model.pt")

@app.post("/predict")
async def predict(features: list[float]):
    return {"prediction": app.state.model.predict(features)}
```

## Key Libraries

| Category | Libraries |
|----------|-----------|
| ML/DL | scikit-learn, PyTorch, TensorFlow, XGBoost |
| LLM | openai, anthropic, langchain |
| Vector DB | chromadb, pinecone, weaviate |
| Data | numpy, pandas, polars |
| Serving | FastAPI, vLLM, Triton |
| Pipeline | Airflow, Prefect, dbt |

## Fine-tuning vs RAG

| | RAG | Fine-tuning |
|--|-----|------------|
| Cost | Low | High |
| Update speed | Instant | Hours/days |
| Best for | Factual data, Q&A | Style, domain language |
| Start with | This one first | Only if RAG isn't enough |
