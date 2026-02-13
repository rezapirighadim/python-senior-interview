"""
=============================================================================
FILE 03: TYPE HINTS & MODERN PYTHON (3.10+)
=============================================================================
Type hints don't change runtime behavior — they're for tooling (mypy, IDE).
But they're EXPECTED from senior devs and critical in production codebases.
=============================================================================
"""
from __future__ import annotations  # Postponed evaluation of annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    ClassVar,
    Final,
    Literal,
    NamedTuple,
    Protocol,
    Self,
    TypeAlias,
    TypeVar,
    overload,
)


# =============================================================================
# 1. BASIC TYPE HINTS
# =============================================================================

# Simple types
name: str = "Reza"
age: int = 30
score: float = 95.5
active: bool = True

# Collections (Python 3.9+ — use lowercase, no need to import from typing)
names: list[str] = ["Alice", "Bob"]
scores: dict[str, float] = {"Alice": 95.5, "Bob": 87.3}
unique_ids: set[int] = {1, 2, 3}
coordinates: tuple[float, float] = (3.14, 2.71)  # Fixed length
values: tuple[int, ...] = (1, 2, 3, 4, 5)        # Variable length

# Optional — value can be the type OR None
# Python 3.10+ syntax:
user_email: str | None = None        # Modern way
# Old way: Optional[str] = None      # Still works

# Union types — Python 3.10+
identifier: int | str = "abc-123"    # Modern way
# Old way: Union[int, str]


# Function signatures
def greet(name: str, excited: bool = False) -> str:
    """Return type is annotated with ->"""
    if excited:
        return f"Hello, {name}!!!"
    return f"Hello, {name}"

# Function that returns nothing
def log_message(msg: str) -> None:
    print(msg)


# =============================================================================
# 2. ADVANCED TYPE HINTS
# =============================================================================

# --- TypeVar for generics ---
T = TypeVar("T")

def first_element(items: list[T]) -> T | None:
    """Returns the first element of any typed list."""
    return items[0] if items else None

# first_element([1, 2, 3])      → inferred as int
# first_element(["a", "b"])     → inferred as str


# --- Callable types ---
from collections.abc import Callable

# Function that takes (int, int) and returns int
Adder: TypeAlias = Callable[[int, int], int]

def apply_operation(a: int, b: int, op: Adder) -> int:
    return op(a, b)


# --- Literal types — restrict to specific values ---
def set_mode(mode: Literal["read", "write", "append"]) -> None:
    print(f"Mode set to: {mode}")

# set_mode("read")    ✓
# set_mode("delete")  ✗ mypy error


# --- TypeGuard for type narrowing ---
from typing import TypeGuard

def is_string_list(items: list[Any]) -> TypeGuard[list[str]]:
    return all(isinstance(item, str) for item in items)

def process(items: list[Any]) -> None:
    if is_string_list(items):
        # mypy now knows items is list[str] here
        print(", ".join(items))


# --- Overload — different return types for different inputs ---
@overload
def parse_value(value: str) -> str: ...
@overload
def parse_value(value: int) -> int: ...
@overload
def parse_value(value: float) -> float: ...

def parse_value(value: str | int | float) -> str | int | float:
    if isinstance(value, str):
        return value.strip()
    return value


# =============================================================================
# 3. PROTOCOLS — Structural Typing (Duck Typing with Type Safety)
# =============================================================================
# Instead of inheritance, define what methods/attributes an object must have.
# This is Python's answer to Go interfaces.

class Drawable(Protocol):
    """Any object with a draw() method satisfies this protocol."""
    def draw(self) -> str: ...

class Circle:
    def draw(self) -> str:
        return "Drawing circle"

class Square:
    def draw(self) -> str:
        return "Drawing square"

def render(shape: Drawable) -> None:
    """Accepts ANY object with a draw() method — no inheritance needed!"""
    print(shape.draw())

# Both work because they have draw() — this is structural typing
render(Circle())
render(Square())

# Runtime protocol checking
from typing import runtime_checkable

@runtime_checkable
class Printable(Protocol):
    def __str__(self) -> str: ...

print(isinstance("hello", Printable))  # True


# =============================================================================
# 4. DATACLASSES — The Modern Way to Define Data Containers
# =============================================================================

@dataclass
class User:
    """Automatically generates __init__, __repr__, __eq__."""
    name: str
    email: str
    age: int
    active: bool = True  # Default value

    def __post_init__(self):
        """Runs after __init__ — use for validation."""
        if self.age < 0:
            raise ValueError("Age cannot be negative")

# Auto-generated:
# - __init__(self, name, email, age, active=True)
# - __repr__() → "User(name='Reza', email='reza@x.com', age=30, active=True)"
# - __eq__() — compares all fields

user = User("Reza", "reza@example.com", 30)
print(user)


# --- Frozen dataclass (immutable) ---
@dataclass(frozen=True)
class Point:
    x: float
    y: float

    @property
    def distance_from_origin(self) -> float:
        return (self.x**2 + self.y**2) ** 0.5

p = Point(3.0, 4.0)
# p.x = 5.0  ← FrozenInstanceError! Can't modify.


# --- Dataclass with complex defaults ---
@dataclass
class Config:
    host: str = "localhost"
    port: int = 8080
    tags: list[str] = field(default_factory=list)  # Mutable default!
    # NEVER do: tags: list[str] = []  ← Same bug as mutable default args

    # Class variable (not an instance field)
    MAX_CONNECTIONS: ClassVar[int] = 100

    # Exclude from __repr__ and __eq__
    _cache: dict = field(default_factory=dict, repr=False, compare=False)


# --- Dataclass ordering ---
@dataclass(order=True)
class Priority:
    """Generates __lt__, __le__, __gt__, __ge__."""
    level: int
    name: str = field(compare=False)  # Don't use name for comparison

tasks = [Priority(3, "Low"), Priority(1, "Critical"), Priority(2, "Medium")]
print(sorted(tasks))  # Sorted by level


# =============================================================================
# 5. NAMED TUPLES — Lightweight Immutable Data
# =============================================================================

# Modern way (class syntax)
class Coordinate(NamedTuple):
    latitude: float
    longitude: float
    altitude: float = 0.0

loc = Coordinate(40.7128, -74.0060)
print(f"Lat: {loc.latitude}, Lon: {loc.longitude}")
# Can unpack: lat, lon, alt = loc


# =============================================================================
# 6. ENUMS — Named Constants
# =============================================================================
class Status(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DELETED = "deleted"

class Priority(Enum):
    LOW = auto()      # 1
    MEDIUM = auto()   # 2
    HIGH = auto()     # 3
    CRITICAL = auto() # 4

# Usage
status = Status.ACTIVE
print(status.value)   # "active"
print(status.name)    # "ACTIVE"

# Enum comparison
if status == Status.ACTIVE:
    print("User is active")

# Iterate over enum
for s in Status:
    print(f"{s.name}: {s.value}")


# =============================================================================
# 7. STRUCTURAL PATTERN MATCHING (Python 3.10+)
# =============================================================================
# Like switch-case on steroids — can match structure, not just values.

@dataclass
class HttpResponse:
    status: int
    body: str

def handle_response(response: HttpResponse) -> str:
    match response:
        case HttpResponse(status=200, body=body):
            return f"Success: {body}"
        case HttpResponse(status=404):
            return "Not found"
        case HttpResponse(status=status) if 500 <= status < 600:
            return f"Server error: {status}"
        case _:
            return "Unknown response"

# Match with dict patterns
def process_event(event: dict) -> str:
    match event:
        case {"type": "click", "x": x, "y": y}:
            return f"Click at ({x}, {y})"
        case {"type": "keypress", "key": key}:
            return f"Key pressed: {key}"
        case {"type": str() as event_type}:
            return f"Unknown event: {event_type}"
        case _:
            return "Invalid event"


# =============================================================================
# 8. FINAL — Prevent Override/Reassignment
# =============================================================================
MAX_RETRIES: Final = 3  # Cannot be reassigned (mypy enforces this)

# from typing import final
# @final
# class Base:
#     """Cannot be subclassed."""
#     pass


# =============================================================================
# 9. PYDANTIC — Runtime Validation (Used by FastAPI)
# =============================================================================
# Pydantic validates data at RUNTIME, not just static analysis.
# This is a preview — covered more in FastAPI file.
"""
from pydantic import BaseModel, field_validator, EmailStr

class UserCreate(BaseModel):
    name: str
    email: EmailStr
    age: int

    @field_validator("age")
    @classmethod
    def validate_age(cls, v):
        if v < 0 or v > 150:
            raise ValueError("Invalid age")
        return v

# Pydantic automatically:
# - Validates types (raises ValidationError if wrong)
# - Converts compatible types (e.g., "30" → 30)
# - Generates JSON schema
user = UserCreate(name="Reza", email="reza@example.com", age=30)
print(user.model_dump())       # {"name": "Reza", "email": "reza@example.com", "age": 30}
print(user.model_dump_json())  # JSON string
"""


# =============================================================================
# 10. BEST PRACTICES SUMMARY
# =============================================================================
"""
✓ Always annotate function signatures in production code
✓ Use `str | None` instead of `Optional[str]` (Python 3.10+)
✓ Use `list[str]` instead of `List[str]` (Python 3.9+)
✓ Use Protocol for duck typing instead of ABC when possible
✓ Use @dataclass for data containers instead of plain classes
✓ Use Enum for fixed sets of constants
✓ Use Literal for restricting string/int values
✓ Use Pydantic for external data validation (API inputs, configs)
✓ Run mypy in CI/CD pipeline
✓ Use `from __future__ import annotations` for forward references
"""


if __name__ == "__main__":
    print("=" * 60)
    print("FILE 03: Type Hints & Modern Python")
    print("=" * 60)

    print("\n--- Type-safe generic function ---")
    print(f"First of [1,2,3]: {first_element([1, 2, 3])}")
    print(f"First of []: {first_element([])}")

    print("\n--- Protocols ---")
    render(Circle())
    render(Square())

    print("\n--- Dataclasses ---")
    print(User("Reza", "reza@example.com", 30))
    print(f"Point distance: {Point(3.0, 4.0).distance_from_origin}")

    print("\n--- Pattern matching ---")
    print(handle_response(HttpResponse(200, "OK")))
    print(handle_response(HttpResponse(404, "")))
    print(handle_response(HttpResponse(503, "")))

    print("\n--- Enums ---")
    print(f"Status.ACTIVE = {Status.ACTIVE.value}")

    print("\n✓ File 03 complete. Move to 04_oop_and_solid.py")
