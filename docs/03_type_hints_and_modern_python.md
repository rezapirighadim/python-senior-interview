# 03 — Type Hints & Modern Python

## Basic Type Hints

```python
name: str = "Reza"
scores: dict[str, float] = {"Alice": 95.5}    # lowercase (3.9+)
unique_ids: set[int] = {1, 2, 3}
coordinates: tuple[float, float] = (3.14, 2.71)

# Optional / Union (3.10+)
user_email: str | None = None       # modern
identifier: int | str = "abc-123"   # union
```

## Function Signatures

```python
def greet(name: str, excited: bool = False) -> str:
    return f"Hello, {name}{'!!!' if excited else ''}"

def log_message(msg: str) -> None:
    print(msg)
```

## Advanced Types

```python
from typing import TypeVar, Literal, overload
from collections.abc import Callable

T = TypeVar("T")

def first_element(items: list[T]) -> T | None:
    return items[0] if items else None

# Literal — restrict values
def set_mode(mode: Literal["read", "write", "append"]) -> None: ...

# Callable types
Adder = Callable[[int, int], int]
```

## Protocols — Structural Typing

Python's answer to Go interfaces. No inheritance needed:

```python
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> str: ...

class Circle:
    def draw(self) -> str:
        return "Drawing circle"

def render(shape: Drawable) -> None:
    print(shape.draw())

render(Circle())  # works because Circle has draw()
```

## Dataclasses

Auto-generates `__init__`, `__repr__`, `__eq__`:

```python
from dataclasses import dataclass, field

@dataclass
class User:
    name: str
    email: str
    age: int
    active: bool = True
    tags: list[str] = field(default_factory=list)  # mutable default!

    def __post_init__(self):
        if self.age < 0:
            raise ValueError("Age cannot be negative")

@dataclass(frozen=True)  # immutable
class Point:
    x: float
    y: float
```

## NamedTuple

Lightweight immutable data:

```python
from typing import NamedTuple

class Coordinate(NamedTuple):
    latitude: float
    longitude: float
    altitude: float = 0.0
```

## Enums

```python
from enum import Enum, auto

class Status(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    DELETED = "deleted"

class Priority(Enum):
    LOW = auto()
    HIGH = auto()
```

## Pattern Matching (3.10+)

```python
match response:
    case HttpResponse(status=200, body=body):
        return f"Success: {body}"
    case HttpResponse(status=404):
        return "Not found"
    case HttpResponse(status=s) if 500 <= s < 600:
        return f"Server error: {s}"
```

## Best Practices

- Use `str | None` over `Optional[str]` (3.10+)
- Use `list[str]` over `List[str]` (3.9+)
- Use Protocol for duck typing instead of ABC
- Use `@dataclass` for data containers
- Use Pydantic for external data validation
- Run mypy in CI/CD
