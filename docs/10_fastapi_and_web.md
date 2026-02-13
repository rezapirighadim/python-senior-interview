# 10 — FastAPI & Web

## Why FastAPI?

- **Async-native** (ASGI, built on Starlette)
- **15K-20K req/sec** vs Flask's 2K-3K
- **Auto API docs** (Swagger + ReDoc)
- **Type validation** via Pydantic
- **Dependency injection** built-in

## Basic App

```python
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field, EmailStr

app = FastAPI()

class UserCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    age: int = Field(..., ge=0, le=150)

@app.post("/users/", status_code=201)
async def create_user(user: UserCreate):
    return {"id": 1, **user.model_dump()}

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    if user_id not in db:
        raise HTTPException(status_code=404, detail="Not found")
    return db[user_id]
```

## Dependency Injection

```python
async def get_current_user(token: str = Query(...)):
    if token != "valid":
        raise HTTPException(401)
    return {"user_id": 1}

async def require_admin(user=Depends(get_current_user)):
    if user["role"] != "admin":
        raise HTTPException(403)
    return user

@app.get("/admin")
async def admin(admin=Depends(require_admin)):
    return {"message": "Welcome, admin"}

# DB session with cleanup
async def get_db():
    db = Database()
    try:
        yield db
    finally:
        db.close()
```

## Middleware

```python
@app.middleware("http")
async def timing(request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{time.perf_counter()-start:.4f}"
    return response
```

## Streaming (for LLMs)

```python
from fastapi.responses import StreamingResponse

async def generate(prompt):
    for word in response_words:
        yield f"data: {word}\n\n"
        await asyncio.sleep(0.1)

@app.post("/stream")
async def stream(request: InferenceRequest):
    return StreamingResponse(generate(request.text), media_type="text/event-stream")
```

## WebSockets

```python
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    while True:
        data = await ws.receive_text()
        await ws.send_text(f"Echo: {data}")
```

## Testing

```python
from httpx import AsyncClient, ASGITransport

@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

@pytest.mark.asyncio
async def test_create(client):
    r = await client.post("/users/", json={...})
    assert r.status_code == 201
```

## Project Structure

```
app/
├── main.py           # FastAPI app
├── config.py         # Settings
├── models/           # SQLAlchemy models
├── schemas/          # Pydantic schemas
├── routers/          # API routes
├── services/         # Business logic
├── repositories/     # Data access
└── middleware/
```
