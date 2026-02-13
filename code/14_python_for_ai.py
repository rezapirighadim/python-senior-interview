"""
=============================================================================
FILE 14: PYTHON FOR AI — ML Ecosystem, LangChain, Vector DBs
=============================================================================
For an AI company, you need to know the AI/ML Python ecosystem.
This file covers what a SENIOR PYTHON DEV (not ML engineer) needs to know.

You don't need to train models, but you MUST know:
  → How to serve models via APIs
  → How to work with embeddings and vector DBs
  → How to build LLM-powered applications
  → How to handle data pipelines
=============================================================================
"""


# =============================================================================
# 1. NUMPY — The Foundation of ALL Python AI
# =============================================================================

"""
import numpy as np

# Creating arrays
a = np.array([1, 2, 3, 4, 5])
b = np.zeros((3, 4))          # 3x4 matrix of zeros
c = np.ones((2, 3))           # 2x3 matrix of ones
d = np.random.randn(3, 3)     # 3x3 matrix of random normal values
e = np.arange(0, 10, 2)       # [0, 2, 4, 6, 8]
f = np.linspace(0, 1, 5)      # [0, 0.25, 0.5, 0.75, 1.0]

# Vectorized operations (FAST — no Python loops!)
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
a + b        # [5, 7, 9]    — element-wise addition
a * b        # [4, 10, 18]  — element-wise multiplication
np.dot(a, b) # 32           — dot product
a @ b        # 32           — matrix multiplication (same as dot for 1D)

# Reshaping
m = np.arange(12).reshape(3, 4)  # 3 rows, 4 columns

# Indexing & slicing
m[0, :]      # First row
m[:, 0]      # First column
m[m > 5]     # All elements > 5 (boolean indexing)

# Broadcasting — operate on different shapes
# [1, 2, 3] + [[10], [20], [30]] → [[11, 12, 13], [21, 22, 23], [31, 32, 33]]

# WHY NumPy is fast:
# → Operations run in C (not Python)
# → Vectorized: no Python loop overhead
# → Contiguous memory layout (cache-friendly)
# → A Python for loop over 1M elements: ~100ms
# → NumPy over 1M elements: ~1ms
"""


# =============================================================================
# 2. PANDAS — Data Manipulation
# =============================================================================

"""
import pandas as pd

# Creating DataFrames
df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie", "Diana"],
    "age": [25, 30, 35, 28],
    "salary": [70000, 85000, 120000, 65000],
    "department": ["Engineering", "Engineering", "Management", "Design"],
})

# Selection
df["name"]                     # Single column (Series)
df[["name", "age"]]            # Multiple columns (DataFrame)
df.loc[0]                      # Row by label
df.iloc[0:2]                   # Rows by position
df[df["age"] > 28]             # Filter rows

# Aggregation
df.groupby("department")["salary"].mean()
df.groupby("department").agg({"salary": ["mean", "max"], "age": "mean"})

# Common operations
df.sort_values("salary", ascending=False)
df.fillna(0)                   # Fill missing values
df.drop_duplicates()
df.merge(other_df, on="id")    # SQL-like join
df.apply(lambda row: row["salary"] * 1.1, axis=1)  # Apply function to each row

# Performance tip: Use .itertuples() not .iterrows()
# Even better: use vectorized operations!
df["bonus"] = df["salary"] * 0.1  # Vectorized — fast!
"""


# =============================================================================
# 3. WORKING WITH LLM APIs
# =============================================================================

"""
# --- OpenAI API ---
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

def chat_completion(prompt: str, model: str = "gpt-4") -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=1000,
    )
    return response.choices[0].message.content


# --- Anthropic API ---
import anthropic

client = anthropic.Anthropic(api_key="your-api-key")

def claude_completion(prompt: str) -> str:
    message = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )
    return message.content[0].text


# --- Streaming responses ---
async def stream_completion(prompt: str):
    '''Stream tokens as they're generated.'''
    stream = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
"""


# =============================================================================
# 4. EMBEDDINGS & VECTOR DATABASES
# =============================================================================

"""
WHAT ARE EMBEDDINGS?
  → Convert text/images into numerical vectors (list of floats)
  → Similar content → similar vectors (close in vector space)
  → "king" and "queen" are closer than "king" and "banana"

WHY VECTOR DATABASES?
  → Store millions of vectors
  → Fast similarity search (nearest neighbors)
  → Power RAG (Retrieval-Augmented Generation) systems

# --- Generate embeddings ---
from openai import OpenAI
client = OpenAI()

def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding  # Returns 1536-dim vector

# --- Cosine similarity (manual) ---
import numpy as np

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# --- ChromaDB (simple local vector DB) ---
# pip install chromadb
import chromadb

client = chromadb.Client()
collection = client.create_collection("my_docs")

# Add documents (embeddings are auto-generated!)
collection.add(
    documents=["Python is great", "Machine learning is fun", "I love pizza"],
    ids=["doc1", "doc2", "doc3"],
)

# Query — find similar documents
results = collection.query(
    query_texts=["programming languages"],
    n_results=2,
)
# Returns: doc1 ("Python is great") and doc2 ("Machine learning is fun")


# --- Pinecone (production vector DB) ---
from pinecone import Pinecone

pc = Pinecone(api_key="your-key")
index = pc.Index("my-index")

# Upsert vectors
index.upsert(vectors=[
    {"id": "vec1", "values": [0.1, 0.2, ...], "metadata": {"text": "hello"}},
    {"id": "vec2", "values": [0.3, 0.4, ...], "metadata": {"text": "world"}},
])

# Query
results = index.query(
    vector=[0.1, 0.2, ...],
    top_k=5,
    include_metadata=True,
)
"""


# =============================================================================
# 5. RAG — Retrieval-Augmented Generation
# =============================================================================

"""
RAG PATTERN (the most important pattern in production AI):

1. INGEST: Split documents → Generate embeddings → Store in vector DB
2. QUERY:  User question → Generate query embedding → Search vector DB
           → Get relevant documents → Feed to LLM with context

WHY RAG?
  → LLMs don't know your private data
  → LLMs can hallucinate
  → RAG grounds responses in your actual documents
  → Much cheaper than fine-tuning

IMPLEMENTATION:
"""

# Simplified RAG system (no external dependencies)
class SimpleRAG:
    """A minimal RAG system to understand the pattern."""

    def __init__(self):
        self.documents: list[dict] = []

    def ingest(self, text: str, metadata: dict | None = None):
        """In real system: chunk text, generate embedding, store in vector DB."""
        self.documents.append({
            "text": text,
            "metadata": metadata or {},
            "words": set(text.lower().split()),  # Simple word-based "embedding"
        })

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """In real system: embed query, vector similarity search."""
        query_words = set(query.lower().split())
        scored = []
        for doc in self.documents:
            # Simple word overlap score (real system uses cosine similarity)
            overlap = len(query_words & doc["words"])
            scored.append((overlap, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]

    def query(self, question: str) -> str:
        """In real system: search → build prompt with context → call LLM."""
        relevant_docs = self.search(question)
        context = "\n".join(doc["text"] for doc in relevant_docs)

        # In real system, this would be:
        # prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        # return llm.complete(prompt)
        return f"Based on {len(relevant_docs)} documents:\n{context}"


# =============================================================================
# 6. LANGCHAIN — LLM Application Framework
# =============================================================================

"""
LangChain provides abstractions for building LLM applications:

CORE CONCEPTS:
  → Models: LLM wrappers (OpenAI, Anthropic, etc.)
  → Prompts: Template-based prompt construction
  → Chains: Sequence of operations
  → Memory: Conversation history
  → Agents: LLM decides which tools to use
  → Retrievers: RAG integration

# Simple chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}"),
])

# LCEL (LangChain Expression Language) — pipe syntax
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"input": "What is Python?"})

# RAG chain
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA

vectorstore = Chroma.from_documents(documents, OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
)
answer = qa_chain.run("What is our refund policy?")
"""


# =============================================================================
# 7. ML MODEL SERVING
# =============================================================================

"""
# Serving a model with FastAPI

from fastapi import FastAPI
import numpy as np
import pickle

app = FastAPI()

# Load model at startup (not per-request!)
@app.on_event("startup")
async def load_model():
    with open("model.pkl", "rb") as f:
        app.state.model = pickle.load(f)

@app.post("/predict")
async def predict(features: list[float]):
    model = app.state.model
    prediction = model.predict(np.array([features]))
    return {"prediction": prediction.tolist()}

# For GPU models (PyTorch):
import torch

@app.on_event("startup")
async def load_model():
    model = torch.load("model.pt")
    model.eval()
    app.state.model = model
    app.state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(app.state.device)

@app.post("/predict")
async def predict(data: dict):
    with torch.no_grad():
        tensor = torch.tensor(data["features"]).to(app.state.device)
        output = app.state.model(tensor)
        return {"prediction": output.cpu().numpy().tolist()}
"""


# =============================================================================
# 8. DATA PIPELINE PATTERNS
# =============================================================================

# --- ETL Pipeline (Extract, Transform, Load) ---
class DataPipeline:
    """Simple but extensible data pipeline."""

    def __init__(self):
        self.steps: list = []

    def add_step(self, name: str, func):
        self.steps.append((name, func))
        return self

    def run(self, data):
        for name, func in self.steps:
            print(f"  Running step: {name}")
            data = func(data)
        return data


# Example usage
def extract(data: dict) -> list[dict]:
    """Extract raw data."""
    return data.get("records", [])

def transform(records: list[dict]) -> list[dict]:
    """Clean and transform data."""
    return [
        {**r, "name": r["name"].strip().title()}
        for r in records
        if r.get("name")
    ]

def load(records: list[dict]) -> dict:
    """Load into destination."""
    return {"loaded": len(records), "records": records}

pipeline = DataPipeline()
pipeline.add_step("extract", extract)
pipeline.add_step("transform", transform)
pipeline.add_step("load", load)


# =============================================================================
# 9. KEY LIBRARIES CHEAT SHEET
# =============================================================================
"""
DATA & ML:
  numpy         → Numerical computing (arrays, linear algebra)
  pandas        → Data manipulation (DataFrames, CSV/Excel)
  polars        → Fast alternative to pandas (Rust-based)
  scikit-learn  → Traditional ML (classification, regression, clustering)
  pytorch       → Deep learning (tensors, neural networks, GPU)
  tensorflow    → Deep learning (Google's framework)
  xgboost       → Gradient boosting (tabular data king)

AI / LLM:
  openai        → OpenAI API (GPT, DALL-E, embeddings)
  anthropic     → Anthropic API (Claude)
  langchain     → LLM application framework
  chromadb      → Local vector database
  pinecone      → Cloud vector database
  huggingface   → Pre-trained models, tokenizers, datasets

DATA PROCESSING:
  apache-airflow → Workflow orchestration
  prefect        → Modern workflow engine
  dbt            → Data transformation (SQL)
  great-expectations → Data validation

SERVING:
  fastapi       → API framework
  uvicorn       → ASGI server
  gunicorn      → WSGI server (with uvicorn workers)
  triton        → NVIDIA model serving
  vllm          → Fast LLM inference
"""


if __name__ == "__main__":
    print("=" * 60)
    print("FILE 14: Python for AI")
    print("=" * 60)

    print("\n--- Simple RAG System ---")
    rag = SimpleRAG()
    rag.ingest("Python is a programming language used for AI and web development")
    rag.ingest("Machine learning uses algorithms to learn patterns from data")
    rag.ingest("FastAPI is a modern web framework for building APIs with Python")
    rag.ingest("Vector databases store embeddings for similarity search")

    print(rag.query("What is Python used for?"))

    print("\n--- Data Pipeline ---")
    raw_data = {
        "records": [
            {"name": "  alice  ", "score": 95},
            {"name": "  bob  ", "score": 87},
            {"name": "", "score": 0},
            {"name": "  charlie  ", "score": 92},
        ]
    }
    result = pipeline.run(raw_data)
    print(f"  Result: {result}")

    print("""
INTERVIEW TIP FOR AI COMPANIES:
  → Know how to build a RAG system end-to-end
  → Understand embeddings and vector similarity
  → Know how to serve ML models via FastAPI
  → Understand data pipeline patterns (ETL)
  → Know when to use fine-tuning vs RAG vs prompt engineering
    """)

    print("✓ File 14 complete. Move to 15_system_design.py")
