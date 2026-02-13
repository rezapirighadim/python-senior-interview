"""
=============================================================================
FILE 05: DESIGN PATTERNS — Real-World Python Examples
=============================================================================
The most asked design patterns in interviews with PRACTICAL examples.
Not textbook theory — real code you'd use in production.
=============================================================================
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol


# =============================================================================
# 1. SINGLETON — One Instance to Rule Them All
# =============================================================================
# Use case: Database connection pool, app configuration, logging

# Method 1: Using __new__ (most common in interviews)
class DatabasePool:
    _instance: DatabasePool | None = None
    _initialized: bool = False

    def __new__(cls) -> DatabasePool:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.connections: list[str] = []
            self.max_connections = 10
            self._initialized = True
            print("DatabasePool initialized (only once!)")

    def get_connection(self) -> str:
        return f"connection-{len(self.connections) + 1}"


# Method 2: Module-level (the Pythonic way — simplest)
# Just put your singleton in a module. Modules are cached after first import.
# config.py:
#   settings = Settings()  # Created once, reused everywhere via import


# =============================================================================
# 2. FACTORY — Create Objects Without Specifying Exact Class
# =============================================================================
# Use case: Creating different types of notifications, parsers, exporters

class Notification(Protocol):
    def send(self, message: str) -> str: ...


class EmailNotification:
    def __init__(self, recipient: str):
        self.recipient = recipient

    def send(self, message: str) -> str:
        return f"Email to {self.recipient}: {message}"


class SMSNotification:
    def __init__(self, phone: str):
        self.phone = phone

    def send(self, message: str) -> str:
        return f"SMS to {self.phone}: {message}"


class SlackNotification:
    def __init__(self, channel: str):
        self.channel = channel

    def send(self, message: str) -> str:
        return f"Slack #{self.channel}: {message}"


class NotificationFactory:
    """Factory that creates the right notification type."""

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, notification_type: str, notification_class: type):
        cls._registry[notification_type] = notification_class

    @classmethod
    def create(cls, notification_type: str, **kwargs) -> Notification:
        if notification_type not in cls._registry:
            raise ValueError(f"Unknown notification type: {notification_type}")
        return cls._registry[notification_type](**kwargs)


# Register notification types
NotificationFactory.register("email", EmailNotification)
NotificationFactory.register("sms", SMSNotification)
NotificationFactory.register("slack", SlackNotification)

# Usage — caller doesn't need to know the concrete class
notif = NotificationFactory.create("email", recipient="reza@example.com")
print(notif.send("Hello!"))


# =============================================================================
# 3. STRATEGY — Swap Algorithms at Runtime
# =============================================================================
# Use case: Different pricing, sorting, compression, auth strategies

class CompressionStrategy(Protocol):
    def compress(self, data: bytes) -> bytes: ...
    def decompress(self, data: bytes) -> bytes: ...


class GzipCompression:
    def compress(self, data: bytes) -> bytes:
        import gzip
        return gzip.compress(data)

    def decompress(self, data: bytes) -> bytes:
        import gzip
        return gzip.decompress(data)


class NoCompression:
    def compress(self, data: bytes) -> bytes:
        return data

    def decompress(self, data: bytes) -> bytes:
        return data


class FileStorage:
    """Uses strategy pattern for compression."""

    def __init__(self, compression: CompressionStrategy):
        self.compression = compression
        self._data: dict[str, bytes] = {}

    def save(self, filename: str, data: bytes) -> None:
        self._data[filename] = self.compression.compress(data)
        print(f"Saved {filename} ({len(data)} → {len(self._data[filename])} bytes)")

    def load(self, filename: str) -> bytes:
        return self.compression.decompress(self._data[filename])

# Switch strategies without changing FileStorage code:
storage = FileStorage(GzipCompression())
storage.save("test.txt", b"Hello " * 1000)

storage_fast = FileStorage(NoCompression())
storage_fast.save("test.txt", b"Hello " * 1000)


# =============================================================================
# 4. OBSERVER — Pub/Sub Event System
# =============================================================================
# Use case: Event-driven systems, UI updates, webhook notifications

class EventEmitter:
    """Simple but powerful event system."""

    def __init__(self):
        self._listeners: dict[str, list] = {}

    def on(self, event: str, callback) -> None:
        """Subscribe to an event."""
        self._listeners.setdefault(event, []).append(callback)

    def off(self, event: str, callback) -> None:
        """Unsubscribe from an event."""
        if event in self._listeners:
            self._listeners[event].remove(callback)

    def emit(self, event: str, **data) -> None:
        """Notify all listeners of an event."""
        for callback in self._listeners.get(event, []):
            callback(**data)


# Real-world example: Order system
class OrderSystem:
    def __init__(self):
        self.events = EventEmitter()

    def place_order(self, order_id: str, amount: float) -> None:
        print(f"Order {order_id} placed: ${amount}")
        self.events.emit("order_placed", order_id=order_id, amount=amount)

    def cancel_order(self, order_id: str) -> None:
        print(f"Order {order_id} cancelled")
        self.events.emit("order_cancelled", order_id=order_id)


# Listeners — completely decoupled from OrderSystem
def send_confirmation_email(order_id: str, amount: float):
    print(f"  → Email: Order {order_id} confirmed (${amount})")

def update_inventory(order_id: str, amount: float):
    print(f"  → Inventory updated for order {order_id}")

def notify_analytics(order_id: str, **kwargs):
    print(f"  → Analytics: tracking order {order_id}")

# Wire up
shop = OrderSystem()
shop.events.on("order_placed", send_confirmation_email)
shop.events.on("order_placed", update_inventory)
shop.events.on("order_placed", notify_analytics)


# =============================================================================
# 5. DECORATOR PATTERN — Add Behavior Without Modifying Original
# =============================================================================
# (Not the same as Python's @decorator syntax, though they're related!)
# Use case: Adding logging, caching, auth, rate limiting to services

class DataSource(Protocol):
    def read(self, key: str) -> str | None: ...
    def write(self, key: str, value: str) -> None: ...


class SimpleDatabase:
    def __init__(self):
        self._store: dict[str, str] = {}

    def read(self, key: str) -> str | None:
        return self._store.get(key)

    def write(self, key: str, value: str) -> None:
        self._store[key] = value


class LoggingDecorator:
    """Adds logging to any DataSource."""

    def __init__(self, wrapped: DataSource):
        self._wrapped = wrapped

    def read(self, key: str) -> str | None:
        result = self._wrapped.read(key)
        print(f"[LOG] read({key}) → {result}")
        return result

    def write(self, key: str, value: str) -> None:
        print(f"[LOG] write({key}, {value})")
        self._wrapped.write(key, value)


class CachingDecorator:
    """Adds caching layer to any DataSource."""

    def __init__(self, wrapped: DataSource):
        self._wrapped = wrapped
        self._cache: dict[str, str] = {}

    def read(self, key: str) -> str | None:
        if key in self._cache:
            print(f"[CACHE HIT] {key}")
            return self._cache[key]
        print(f"[CACHE MISS] {key}")
        value = self._wrapped.read(key)
        if value is not None:
            self._cache[key] = value
        return value

    def write(self, key: str, value: str) -> None:
        self._cache[key] = value
        self._wrapped.write(key, value)


# Stack decorators — each adds a layer
db: DataSource = SimpleDatabase()
db = LoggingDecorator(db)     # Add logging
db = CachingDecorator(db)     # Add caching on top
db.write("user:1", "Reza")
db.read("user:1")  # Cache hit
db.read("user:1")  # Cache hit again


# =============================================================================
# 6. BUILDER — Construct Complex Objects Step by Step
# =============================================================================
# Use case: Building queries, configurations, HTTP requests

@dataclass
class HttpRequest:
    method: str = "GET"
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    body: str | None = None
    timeout: int = 30


class HttpRequestBuilder:
    """Fluent builder for HTTP requests."""

    def __init__(self):
        self._request = HttpRequest()

    def method(self, method: str) -> HttpRequestBuilder:
        self._request.method = method
        return self  # Return self for chaining

    def url(self, url: str) -> HttpRequestBuilder:
        self._request.url = url
        return self

    def header(self, key: str, value: str) -> HttpRequestBuilder:
        self._request.headers[key] = value
        return self

    def body(self, body: str) -> HttpRequestBuilder:
        self._request.body = body
        return self

    def timeout(self, seconds: int) -> HttpRequestBuilder:
        self._request.timeout = seconds
        return self

    def build(self) -> HttpRequest:
        if not self._request.url:
            raise ValueError("URL is required")
        return self._request


# Fluent API — reads like English
request = (
    HttpRequestBuilder()
    .method("POST")
    .url("https://api.example.com/users")
    .header("Content-Type", "application/json")
    .header("Authorization", "Bearer token123")
    .body('{"name": "Reza"}')
    .timeout(10)
    .build()
)


# =============================================================================
# 7. REPOSITORY PATTERN — Abstract Data Access
# =============================================================================
# Use case: Decoupling business logic from database implementation

@dataclass
class User:
    id: str
    name: str
    email: str


class UserRepository(Protocol):
    """Abstract repository — defines the interface."""
    def get(self, user_id: str) -> User | None: ...
    def save(self, user: User) -> None: ...
    def delete(self, user_id: str) -> None: ...
    def find_by_email(self, email: str) -> User | None: ...


class InMemoryUserRepository:
    """In-memory implementation — great for testing."""

    def __init__(self):
        self._users: dict[str, User] = {}

    def get(self, user_id: str) -> User | None:
        return self._users.get(user_id)

    def save(self, user: User) -> None:
        self._users[user.id] = user

    def delete(self, user_id: str) -> None:
        self._users.pop(user_id, None)

    def find_by_email(self, email: str) -> User | None:
        for user in self._users.values():
            if user.email == email:
                return user
        return None


class SQLUserRepository:
    """SQL implementation — for production."""

    def __init__(self, connection_string: str):
        self.conn_str = connection_string

    def get(self, user_id: str) -> User | None:
        # In real code: SELECT * FROM users WHERE id = ?
        print(f"SQL: SELECT * FROM users WHERE id = '{user_id}'")
        return None

    def save(self, user: User) -> None:
        print(f"SQL: INSERT INTO users VALUES ('{user.id}', ...)")

    def delete(self, user_id: str) -> None:
        print(f"SQL: DELETE FROM users WHERE id = '{user_id}'")

    def find_by_email(self, email: str) -> User | None:
        print(f"SQL: SELECT * FROM users WHERE email = '{email}'")
        return None


# Business logic doesn't care about the implementation
class UserService:
    def __init__(self, repo: UserRepository):
        self.repo = repo

    def register(self, name: str, email: str) -> User:
        existing = self.repo.find_by_email(email)
        if existing:
            raise ValueError(f"Email {email} already registered")

        import uuid
        user = User(id=str(uuid.uuid4()), name=name, email=email)
        self.repo.save(user)
        return user


# =============================================================================
# 8. CHAIN OF RESPONSIBILITY — Pipeline Processing
# =============================================================================
# Use case: Middleware, request validation, data processing pipelines

class Handler(ABC):
    def __init__(self):
        self._next: Handler | None = None

    def set_next(self, handler: Handler) -> Handler:
        self._next = handler
        return handler

    def handle(self, request: dict) -> dict | None:
        if self._next:
            return self._next.handle(request)
        return request


class AuthenticationHandler(Handler):
    def handle(self, request: dict) -> dict | None:
        token = request.get("token")
        if not token:
            print("  ✗ No auth token")
            return None
        if token != "valid-token":
            print("  ✗ Invalid token")
            return None
        print("  ✓ Authenticated")
        return super().handle(request)


class RateLimitHandler(Handler):
    def __init__(self):
        super().__init__()
        self._request_count = 0
        self._limit = 100

    def handle(self, request: dict) -> dict | None:
        self._request_count += 1
        if self._request_count > self._limit:
            print("  ✗ Rate limit exceeded")
            return None
        print(f"  ✓ Rate limit OK ({self._request_count}/{self._limit})")
        return super().handle(request)


class ValidationHandler(Handler):
    def handle(self, request: dict) -> dict | None:
        if not request.get("body"):
            print("  ✗ Empty body")
            return None
        print("  ✓ Validation passed")
        return super().handle(request)


# Build the chain
auth = AuthenticationHandler()
rate_limit = RateLimitHandler()
validation = ValidationHandler()

auth.set_next(rate_limit).set_next(validation)


# =============================================================================
# 9. COMMAND PATTERN — Encapsulate Actions as Objects
# =============================================================================
# Use case: Undo/redo, task queues, macro recording

class Command(Protocol):
    def execute(self) -> None: ...
    def undo(self) -> None: ...


@dataclass
class TextEditor:
    content: str = ""

    def insert(self, text: str, position: int) -> None:
        self.content = self.content[:position] + text + self.content[position:]

    def delete(self, position: int, length: int) -> str:
        deleted = self.content[position:position + length]
        self.content = self.content[:position] + self.content[position + length:]
        return deleted


@dataclass
class InsertCommand:
    editor: TextEditor
    text: str
    position: int

    def execute(self) -> None:
        self.editor.insert(self.text, self.position)

    def undo(self) -> None:
        self.editor.delete(self.position, len(self.text))


@dataclass
class DeleteCommand:
    editor: TextEditor
    position: int
    length: int
    _deleted_text: str = ""

    def execute(self) -> None:
        self._deleted_text = self.editor.delete(self.position, self.length)

    def undo(self) -> None:
        self.editor.insert(self._deleted_text, self.position)


class CommandHistory:
    def __init__(self):
        self._history: list[Command] = []

    def execute(self, command: Command) -> None:
        command.execute()
        self._history.append(command)

    def undo(self) -> None:
        if self._history:
            command = self._history.pop()
            command.undo()


# =============================================================================
# INTERVIEW CHEAT SHEET — When to Use Which Pattern
# =============================================================================
"""
Singleton    → Need exactly one instance (config, connection pool, logger)
Factory      → Object creation depends on runtime conditions
Strategy     → Need to swap algorithms at runtime
Observer     → Decouple event producers from consumers
Decorator    → Add behavior without modifying existing code
Builder      → Complex object construction with many optional params
Repository   → Abstract database access for testability
Chain of Resp → Pipeline/middleware processing
Command      → Undo/redo, task queues, action recording

Python-specific patterns:
- Use Protocol instead of abstract interfaces (duck typing)
- Use @dataclass for simple data objects
- Use module-level singletons (simplest approach)
- Use context managers for resource management
- Use decorators (@ syntax) for cross-cutting concerns
"""


if __name__ == "__main__":
    print("=" * 60)
    print("FILE 05: Design Patterns")
    print("=" * 60)

    print("\n--- Singleton ---")
    db1 = DatabasePool()
    db2 = DatabasePool()
    print(f"Same instance? {db1 is db2}")

    print("\n--- Factory ---")
    for ntype, kwargs in [
        ("email", {"recipient": "a@b.com"}),
        ("sms", {"phone": "+1234567890"}),
        ("slack", {"channel": "general"}),
    ]:
        n = NotificationFactory.create(ntype, **kwargs)
        print(f"  {n.send('Hello!')}")

    print("\n--- Observer ---")
    shop.place_order("ORD-001", 99.99)

    print("\n--- Chain of Responsibility ---")
    print("Request with valid token:")
    auth.handle({"token": "valid-token", "body": "data"})
    print("Request with no token:")
    auth.handle({"body": "data"})

    print("\n--- Command (Undo/Redo) ---")
    editor = TextEditor()
    history = CommandHistory()
    history.execute(InsertCommand(editor, "Hello ", 0))
    history.execute(InsertCommand(editor, "World!", 6))
    print(f"  Content: '{editor.content}'")
    history.undo()
    print(f"  After undo: '{editor.content}'")
    history.undo()
    print(f"  After undo: '{editor.content}'")

    print("\n✓ File 05 complete. Move to 06_functional_programming.py")
