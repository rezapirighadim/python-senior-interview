# 04 — OOP & SOLID Principles

## OOP Quick Reference

```python
class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name, "Woof")
        self.breed = breed

    def speak(self):  # polymorphism
        return f"{self.name} the {self.breed} says {self.sound}!"
```

- `@classmethod` — factory methods, receives `cls`
- `@staticmethod` — utility functions, no implicit arg
- `@property` — controlled attribute access

## S — Single Responsibility

Each class should have **one reason to change**.

```python
# BAD: UserManager does create + email + report
# GOOD: UserRepository, EmailService, ReportGenerator
```

## O — Open/Closed

Open for **extension**, closed for **modification**.

```python
class DiscountStrategy(Protocol):
    def calculate(self, amount: float) -> float: ...

class VIPDiscount:
    def calculate(self, amount: float) -> float:
        return amount * 0.20

# New discount? Add new class. No existing code changes.
```

## L — Liskov Substitution

Subtypes must be **substitutable** for their base types.

```python
# BAD: Square overrides Rectangle's width setter with side effects
# GOOD: Both implement Shape protocol with area() method
```

## I — Interface Segregation

Don't force classes to implement methods they **don't need**.

```python
# BAD: Robot forced to implement eat() and sleep()
# GOOD: Separate Workable, Eatable, Sleepable protocols
```

## D — Dependency Inversion

Depend on **abstractions**, not implementations.

```python
class Database(Protocol):
    def save(self, data: dict) -> None: ...

class OrderProcessor:
    def __init__(self, db: Database):  # inject dependency!
        self.db = db
```

## Composition Over Inheritance

```python
@dataclass
class Car:
    model: str
    engine: Engine    # has-a (composition)
    gps: GPS | None = None
```

## Abstract Base Classes

```python
from abc import ABC, abstractmethod

class PaymentProcessor(ABC):
    @abstractmethod
    def process_payment(self, amount: float) -> bool: ...

    def validate_amount(self, amount: float) -> bool:  # concrete
        return amount > 0
```

## When to Use What

| Tool | When |
|------|------|
| Protocol | Duck typing with type safety, third-party code |
| ABC | Enforce contracts, shared implementation |
| @dataclass | Data containers |
| Composition | "has-a" relationships, flexibility |
| Inheritance | Clear "is-a", shallow hierarchy (2-3 levels) |
