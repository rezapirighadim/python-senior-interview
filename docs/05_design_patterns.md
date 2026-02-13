# 05 — Design Patterns

## Singleton — One Instance

```python
class DatabasePool:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

Pythonic way: just use a module-level variable. Modules are cached after first import.

## Factory — Create Without Specifying Class

```python
class NotificationFactory:
    _registry = {}

    @classmethod
    def register(cls, type_name, klass):
        cls._registry[type_name] = klass

    @classmethod
    def create(cls, type_name, **kwargs):
        return cls._registry[type_name](**kwargs)
```

## Strategy — Swap Algorithms at Runtime

```python
class FileStorage:
    def __init__(self, compression: CompressionStrategy):
        self.compression = compression

    def save(self, filename, data):
        self._data[filename] = self.compression.compress(data)

# Swap strategies without changing FileStorage
storage = FileStorage(GzipCompression())
storage_fast = FileStorage(NoCompression())
```

## Observer — Event System

```python
class EventEmitter:
    def __init__(self):
        self._listeners = {}

    def on(self, event, callback):
        self._listeners.setdefault(event, []).append(callback)

    def emit(self, event, **data):
        for cb in self._listeners.get(event, []):
            cb(**data)
```

## Decorator Pattern — Add Behavior Layers

```python
db = SimpleDatabase()
db = LoggingDecorator(db)   # add logging
db = CachingDecorator(db)   # add caching
```

## Builder — Complex Object Construction

```python
request = (
    HttpRequestBuilder()
    .method("POST")
    .url("https://api.example.com/users")
    .header("Authorization", "Bearer token")
    .body('{"name": "Reza"}')
    .build()
)
```

## Repository — Abstract Data Access

```python
class UserRepository(Protocol):
    def get(self, user_id: str) -> User | None: ...
    def save(self, user: User) -> None: ...

# InMemoryUserRepository for tests, SQLUserRepository for production
```

## Chain of Responsibility — Pipeline

```python
auth.set_next(rate_limit).set_next(validation)
auth.handle(request)  # flows through the chain
```

## Command — Undo/Redo

```python
history = CommandHistory()
history.execute(InsertCommand(editor, "Hello", 0))
history.undo()  # reverses the insert
```

## Cheat Sheet

| Pattern | Use When |
|---------|----------|
| Singleton | One instance needed (config, pool) |
| Factory | Creation depends on runtime conditions |
| Strategy | Swap algorithms at runtime |
| Observer | Decouple event producers/consumers |
| Decorator | Add behavior without modifying code |
| Builder | Complex construction, many optional params |
| Repository | Abstract database access |
| Chain of Responsibility | Pipeline/middleware |
| Command | Undo/redo, task queues |
