"""
=============================================================================
FILE 10: FastAPI & WEB — Modern Python API Development
=============================================================================
FastAPI is THE framework for AI companies. It's async-native, type-safe,
and generates API docs automatically. You MUST know this for interviews.

pip install fastapi uvicorn pydantic
Run: uvicorn 10_fastapi_and_web:app --reload
=============================================================================
"""
from dataclasses import dataclass
from enum import Enum
from typing import Annotated


# =============================================================================
# 1. WHY FASTAPI? (vs Flask/Django)
# =============================================================================
"""
FastAPI advantages:
  → Async-native (built on Starlette ASGI)
  → 15K-20K req/sec vs Flask's 2K-3K
  → Automatic API documentation (Swagger UI + ReDoc)
  → Type validation via Pydantic (catches errors before your code runs)
  → Dependency injection built-in
  → Modern Python (type hints are the API spec)

When to use what:
  FastAPI → APIs, microservices, AI inference endpoints
  Django  → Full-featured web apps with admin, ORM, auth
  Flask   → Simple apps, prototypes (being replaced by FastAPI)
"""


# =============================================================================
# 2. BASIC FASTAPI APP
# =============================================================================

# In a real app, you'd import from fastapi
# We'll show the code structure — run with uvicorn to test

"""
from fastapi import FastAPI, HTTPException, Query, Path, Body, Depends
from pydantic import BaseModel, Field, EmailStr, field_validator
from typing import Annotated

app = FastAPI(
    title="Senior Python Interview API",
    description="Learning FastAPI for AI company interviews",
    version="1.0.0",
)


# --- Pydantic models (request/response schemas) ---
class UserCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, examples=["Reza"])
    email: EmailStr
    age: int = Field(..., ge=0, le=150)

    @field_validator("name")
    @classmethod
    def name_must_be_capitalized(cls, v: str) -> str:
        if not v[0].isupper():
            raise ValueError("Name must start with a capital letter")
        return v

class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    age: int

    model_config = {"from_attributes": True}  # Allows ORM model → Pydantic

class UserUpdate(BaseModel):
    name: str | None = None
    email: EmailStr | None = None
    age: int | None = Field(None, ge=0, le=150)


# --- In-memory "database" ---
fake_db: dict[int, dict] = {}
next_id = 1


# =============================================================================
# 3. CRUD ENDPOINTS
# =============================================================================

@app.post("/users/", response_model=UserResponse, status_code=201)
async def create_user(user: UserCreate):
    global next_id
    user_data = {"id": next_id, **user.model_dump()}
    fake_db[next_id] = user_data
    next_id += 1
    return user_data


@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int = Path(..., ge=1)):
    if user_id not in fake_db:
        raise HTTPException(status_code=404, detail="User not found")
    return fake_db[user_id]


@app.get("/users/", response_model=list[UserResponse])
async def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    search: str | None = Query(None, min_length=1),
):
    users = list(fake_db.values())
    if search:
        users = [u for u in users if search.lower() in u["name"].lower()]
    return users[skip : skip + limit]


@app.patch("/users/{user_id}", response_model=UserResponse)
async def update_user(user_id: int, user_update: UserUpdate):
    if user_id not in fake_db:
        raise HTTPException(status_code=404, detail="User not found")

    stored = fake_db[user_id]
    update_data = user_update.model_dump(exclude_unset=True)
    stored.update(update_data)
    return stored


@app.delete("/users/{user_id}", status_code=204)
async def delete_user(user_id: int):
    if user_id not in fake_db:
        raise HTTPException(status_code=404, detail="User not found")
    del fake_db[user_id]


# =============================================================================
# 4. DEPENDENCY INJECTION — FastAPI's Killer Feature
# =============================================================================

# Dependencies are functions that provide shared logic
async def get_current_user(token: str = Query(..., alias="token")):
    '''Simulates authentication.'''
    if token != "secret-token":
        raise HTTPException(status_code=401, detail="Invalid token")
    return {"user_id": 1, "role": "admin"}


async def require_admin(current_user: dict = Depends(get_current_user)):
    '''Requires admin role — chains dependencies!'''
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin required")
    return current_user


@app.get("/admin/dashboard")
async def admin_dashboard(admin: dict = Depends(require_admin)):
    return {"message": f"Welcome, admin {admin['user_id']}!"}


# --- Database session dependency ---
class FakeDB:
    def __init__(self):
        self.connected = True

    def query(self, sql: str):
        return [{"id": 1}]

    def close(self):
        self.connected = False


async def get_db():
    '''Dependency that provides a database session.'''
    db = FakeDB()
    try:
        yield db  # This is what the endpoint receives
    finally:
        db.close()  # Cleanup — always runs!


@app.get("/items/")
async def list_items(db: FakeDB = Depends(get_db)):
    return db.query("SELECT * FROM items")


# =============================================================================
# 5. MIDDLEWARE — Cross-Cutting Concerns
# =============================================================================
import time as time_module

@app.middleware("http")
async def add_timing_header(request, call_next):
    start = time_module.perf_counter()
    response = await call_next(request)
    elapsed = time_module.perf_counter() - start
    response.headers["X-Process-Time"] = f"{elapsed:.4f}"
    return response


# =============================================================================
# 6. BACKGROUND TASKS
# =============================================================================
from fastapi import BackgroundTasks

def send_notification(email: str, message: str):
    '''Runs in background after response is sent.'''
    time_module.sleep(2)
    print(f"Notification sent to {email}: {message}")

@app.post("/orders/")
async def create_order(background_tasks: BackgroundTasks):
    order = {"id": 1, "status": "created"}
    # This runs AFTER the response is returned to the client
    background_tasks.add_task(send_notification, "user@example.com", "Order created!")
    return order


# =============================================================================
# 7. ERROR HANDLING
# =============================================================================
from fastapi.responses import JSONResponse

class AppError(Exception):
    def __init__(self, message: str, code: int = 400):
        self.message = message
        self.code = code

@app.exception_handler(AppError)
async def app_error_handler(request, exc: AppError):
    return JSONResponse(
        status_code=exc.code,
        content={"error": exc.message, "type": "app_error"},
    )


# =============================================================================
# 8. AI-SPECIFIC ENDPOINTS
# =============================================================================

class InferenceRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    model: str = Field(default="gpt-4")
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=1000, ge=1, le=4096)

class InferenceResponse(BaseModel):
    text: str
    model: str
    tokens_used: int
    latency_ms: float

@app.post("/inference", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest):
    start = time_module.perf_counter()

    # In real code: call your ML model or external API
    result_text = f"Response to: {request.text[:50]}..."

    latency = (time_module.perf_counter() - start) * 1000
    return InferenceResponse(
        text=result_text,
        model=request.model,
        tokens_used=42,
        latency_ms=latency,
    )


# --- Streaming response (for LLM-like output) ---
from fastapi.responses import StreamingResponse
import asyncio

async def generate_stream(prompt: str):
    '''Yields tokens one by one (like ChatGPT).'''
    words = f"This is a response to: {prompt}".split()
    for word in words:
        yield f"data: {word}\n\n"
        await asyncio.sleep(0.1)
    yield "data: [DONE]\n\n"

@app.post("/inference/stream")
async def stream_inference(request: InferenceRequest):
    return StreamingResponse(
        generate_stream(request.text),
        media_type="text/event-stream",
    )


# =============================================================================
# 9. WEBSOCKETS
# =============================================================================
from fastapi import WebSocket, WebSocketDisconnect

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)

    async def broadcast(self, message: str):
        for ws in self.active:
            await ws.send_text(message)

ws_manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        while True:
            data = await ws.receive_text()
            await ws_manager.broadcast(f"User says: {data}")
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)


# =============================================================================
# 10. TESTING FASTAPI
# =============================================================================
# pytest + httpx (FastAPI's recommended test client)

# test_api.py:
'''
import pytest
from httpx import AsyncClient, ASGITransport
from main import app

@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

@pytest.mark.asyncio
async def test_create_user(client: AsyncClient):
    response = await client.post("/users/", json={
        "name": "Reza",
        "email": "reza@example.com",
        "age": 30,
    })
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Reza"
    assert "id" in data

@pytest.mark.asyncio
async def test_get_nonexistent_user(client: AsyncClient):
    response = await client.get("/users/99999")
    assert response.status_code == 404
'''
"""


# =============================================================================
# 11. PROJECT STRUCTURE — How a Real FastAPI App Looks
# =============================================================================
"""
myapp/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app factory
│   ├── config.py             # Settings (Pydantic BaseSettings)
│   ├── dependencies.py       # Shared dependencies
│   ├── models/               # SQLAlchemy/Pydantic models
│   │   ├── __init__.py
│   │   ├── user.py
│   │   └── order.py
│   ├── schemas/              # Pydantic request/response schemas
│   │   ├── __init__.py
│   │   ├── user.py
│   │   └── order.py
│   ├── routers/              # API routes (like Django views)
│   │   ├── __init__.py
│   │   ├── users.py
│   │   └── orders.py
│   ├── services/             # Business logic
│   │   ├── __init__.py
│   │   ├── user_service.py
│   │   └── email_service.py
│   ├── repositories/         # Data access layer
│   │   ├── __init__.py
│   │   └── user_repo.py
│   └── middleware/
│       ├── __init__.py
│       └── auth.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_users.py
│   └── test_orders.py
├── alembic/                  # Database migrations
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── README.md
"""


# =============================================================================
# PRINTABLE SUMMARY
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("FILE 10: FastAPI & Web Development")
    print("=" * 60)

    print("""
KEY CONCEPTS FOR INTERVIEW:

1. FastAPI is ASGI-based (async), Flask is WSGI (sync)
2. Pydantic validates request data automatically
3. Dependency injection for auth, DB sessions, shared logic
4. Background tasks for non-blocking operations
5. Streaming responses for LLM/AI output
6. WebSockets for real-time communication
7. Middleware for cross-cutting concerns (timing, CORS, auth)
8. Test with httpx.AsyncClient (not requests)

COMMON INTERVIEW QUESTIONS:
  Q: How does FastAPI achieve high performance?
  A: ASGI (async), Starlette underneath, Pydantic for fast validation,
     type hints compiled to validation code at startup.

  Q: How do you handle authentication?
  A: Dependency injection. Create a dependency that extracts/validates
     JWT token and inject it into protected endpoints.

  Q: How do you serve ML models with FastAPI?
  A: Load model at startup (lifespan event), create inference endpoint,
     use background tasks or Celery for heavy processing, stream for LLMs.
    """)

    print("✓ File 10 complete. Move to 11_testing_and_quality.py")
