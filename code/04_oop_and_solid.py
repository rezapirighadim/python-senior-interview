"""
=============================================================================
FILE 04: OOP & SOLID PRINCIPLES
=============================================================================
Object-Oriented Programming done right. SOLID principles with Python
examples that you'll actually use in production.
=============================================================================
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol


# =============================================================================
# 1. QUICK OOP REFRESHER
# =============================================================================

class Animal:
    """Base class demonstrating core OOP concepts."""

    # Class variable — shared across ALL instances
    kingdom: str = "Animalia"

    def __init__(self, name: str, sound: str):
        # Instance variables — unique to each instance
        self.name = name
        self.sound = sound
        self._energy = 100  # Convention: _ prefix = "private" (not enforced)

    def speak(self) -> str:
        """Instance method — operates on instance data."""
        return f"{self.name} says {self.sound}!"

    @classmethod
    def from_dict(cls, data: dict) -> "Animal":
        """Class method — factory pattern. Receives class as first arg."""
        return cls(data["name"], data["sound"])

    @staticmethod
    def is_valid_name(name: str) -> bool:
        """Static method — utility function. No access to class or instance."""
        return len(name) > 0 and name.isalpha()

    @property
    def energy(self) -> int:
        return self._energy

    @energy.setter
    def energy(self, value: int):
        self._energy = max(0, min(100, value))  # Clamp to 0-100


class Dog(Animal):
    """Inheritance example."""

    def __init__(self, name: str, breed: str):
        super().__init__(name, "Woof")  # Always call super().__init__
        self.breed = breed

    def speak(self) -> str:
        """Polymorphism — same method name, different behavior."""
        return f"{self.name} the {self.breed} says {self.sound}!"


# =============================================================================
# 2. S — SINGLE RESPONSIBILITY PRINCIPLE (SRP)
# =============================================================================
# A class should have only ONE reason to change.

# BAD — This class does too many things
class BadUserManager:
    def create_user(self, name: str, email: str):
        # Creates user in database
        pass

    def send_welcome_email(self, email: str):
        # Sends email — WHY is this in UserManager?
        pass

    def generate_report(self):
        # Generates report — this is a completely different responsibility!
        pass


# GOOD — Each class has ONE responsibility
class UserRepository:
    """Handles user data persistence."""

    def create(self, name: str, email: str) -> dict:
        user = {"name": name, "email": email}
        # Save to database
        return user

    def find_by_email(self, email: str) -> dict | None:
        # Query database
        return None


class EmailService:
    """Handles sending emails."""

    def send_welcome(self, email: str) -> None:
        print(f"Sending welcome email to {email}")

    def send_notification(self, email: str, message: str) -> None:
        print(f"Sending notification to {email}: {message}")


class UserReportGenerator:
    """Handles user report generation."""

    def __init__(self, user_repo: UserRepository):
        self.user_repo = user_repo

    def generate(self) -> str:
        return "User Report: ..."


# =============================================================================
# 3. O — OPEN/CLOSED PRINCIPLE (OCP)
# =============================================================================
# Open for EXTENSION, closed for MODIFICATION.
# Add new behavior without changing existing code.

# BAD — Adding new discount type requires modifying this function
def bad_calculate_discount(order_type: str, amount: float) -> float:
    if order_type == "regular":
        return amount * 0.05
    elif order_type == "premium":
        return amount * 0.10
    elif order_type == "vip":
        return amount * 0.20
    # Need to modify this function for every new type!
    return 0


# GOOD — Use abstraction. New types just add new classes.
class DiscountStrategy(Protocol):
    """Protocol defines the interface — no inheritance needed."""
    def calculate(self, amount: float) -> float: ...


class RegularDiscount:
    def calculate(self, amount: float) -> float:
        return amount * 0.05


class PremiumDiscount:
    def calculate(self, amount: float) -> float:
        return amount * 0.10


class VIPDiscount:
    def calculate(self, amount: float) -> float:
        return amount * 0.20


# New discount? Just add a new class. No existing code changes!
class EmployeeDiscount:
    def calculate(self, amount: float) -> float:
        return amount * 0.30


def calculate_discount(strategy: DiscountStrategy, amount: float) -> float:
    """Works with ANY object that has calculate(amount) method."""
    return strategy.calculate(amount)


# =============================================================================
# 4. L — LISKOV SUBSTITUTION PRINCIPLE (LSP)
# =============================================================================
# Subtypes must be substitutable for their base types.
# If it looks like a duck but needs batteries, you violated LSP.

# BAD — Square violates LSP for Rectangle
class Rectangle:
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height


class BadSquare(Rectangle):
    """Violates LSP — changing width also changes height."""
    def __init__(self, side: float):
        super().__init__(side, side)

    @Rectangle.width.setter  # type: ignore
    def width(self, value):
        self._width = value
        self._height = value  # Unexpected side effect!


# GOOD — Use a common abstraction
class Shape(Protocol):
    def area(self) -> float: ...


@dataclass(frozen=True)
class GoodRectangle:
    width: float
    height: float

    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True)
class GoodSquare:
    side: float

    def area(self) -> float:
        return self.side ** 2


def print_area(shape: Shape) -> None:
    """Works with any Shape — LSP satisfied."""
    print(f"Area: {shape.area()}")


# =============================================================================
# 5. I — INTERFACE SEGREGATION PRINCIPLE (ISP)
# =============================================================================
# Don't force classes to implement methods they don't need.

# BAD — One fat interface
class BadWorker(ABC):
    @abstractmethod
    def work(self): ...

    @abstractmethod
    def eat(self): ...

    @abstractmethod
    def sleep(self): ...

# A robot can work but can't eat or sleep!
# class Robot(BadWorker):
#     def eat(self): pass   # Forced to implement useless method
#     def sleep(self): pass  # Same here


# GOOD — Small, focused interfaces (protocols)
class Workable(Protocol):
    def work(self) -> None: ...

class Eatable(Protocol):
    def eat(self) -> None: ...

class Sleepable(Protocol):
    def sleep(self) -> None: ...


class Human:
    def work(self) -> None:
        print("Human working")

    def eat(self) -> None:
        print("Human eating")

    def sleep(self) -> None:
        print("Human sleeping")


class Robot:
    def work(self) -> None:
        print("Robot working")
    # No eat() or sleep() — and that's fine!


def assign_work(worker: Workable) -> None:
    worker.work()  # Works with both Human and Robot


# =============================================================================
# 6. D — DEPENDENCY INVERSION PRINCIPLE (DIP)
# =============================================================================
# High-level modules should not depend on low-level modules.
# Both should depend on abstractions.

# BAD — High-level OrderProcessor depends on low-level MySQLDatabase
class MySQLDatabase:
    def save(self, data: dict) -> None:
        print(f"Saving to MySQL: {data}")

class BadOrderProcessor:
    def __init__(self):
        self.db = MySQLDatabase()  # Tight coupling! Can't use PostgreSQL.

    def process(self, order: dict) -> None:
        self.db.save(order)


# GOOD — Depend on abstraction, not implementation
class Database(Protocol):
    """Abstraction — any database that can save."""
    def save(self, data: dict) -> None: ...

class PostgresDatabase:
    def save(self, data: dict) -> None:
        print(f"Saving to PostgreSQL: {data}")

class MongoDatabase:
    def save(self, data: dict) -> None:
        print(f"Saving to MongoDB: {data}")

class InMemoryDatabase:
    """Perfect for testing!"""
    def __init__(self):
        self.data: list[dict] = []

    def save(self, data: dict) -> None:
        self.data.append(data)

class OrderProcessor:
    def __init__(self, db: Database):  # Inject dependency!
        self.db = db

    def process(self, order: dict) -> None:
        # Validate, transform, etc.
        self.db.save(order)

# Usage — swap databases without changing OrderProcessor:
processor = OrderProcessor(PostgresDatabase())
processor.process({"item": "laptop", "price": 999})

# For testing:
test_db = InMemoryDatabase()
test_processor = OrderProcessor(test_db)
test_processor.process({"item": "test"})
assert len(test_db.data) == 1


# =============================================================================
# 7. COMPOSITION OVER INHERITANCE
# =============================================================================
# "Has-a" is usually better than "is-a".
# Deep inheritance hierarchies are fragile and hard to understand.

# BAD — Deep inheritance
# class Animal → class Mammal → class Pet → class Dog → class GoldenRetriever

# GOOD — Composition
@dataclass
class Engine:
    horsepower: int
    fuel_type: str = "gasoline"

    def start(self) -> str:
        return f"Engine ({self.horsepower}HP) started"

@dataclass
class GPS:
    def navigate(self, destination: str) -> str:
        return f"Navigating to {destination}"

@dataclass
class Car:
    """Car HAS an engine and GPS — composition, not inheritance."""
    model: str
    engine: Engine
    gps: GPS | None = None

    def start(self) -> str:
        return f"{self.model}: {self.engine.start()}"

    def navigate(self, destination: str) -> str:
        if self.gps is None:
            return "No GPS installed"
        return self.gps.navigate(destination)

car = Car(
    model="Tesla Model 3",
    engine=Engine(horsepower=283, fuel_type="electric"),
    gps=GPS(),
)
print(car.start())
print(car.navigate("Home"))


# =============================================================================
# 8. ABSTRACT BASE CLASSES (ABC) — When You NEED Inheritance
# =============================================================================
class PaymentProcessor(ABC):
    """Abstract base class — cannot be instantiated directly."""

    @abstractmethod
    def process_payment(self, amount: float) -> bool:
        """Subclasses MUST implement this."""
        ...

    @abstractmethod
    def refund(self, transaction_id: str) -> bool:
        ...

    def validate_amount(self, amount: float) -> bool:
        """Concrete method — shared by all subclasses."""
        return amount > 0


class StripeProcessor(PaymentProcessor):
    def process_payment(self, amount: float) -> bool:
        if not self.validate_amount(amount):
            return False
        print(f"Processing ${amount} via Stripe")
        return True

    def refund(self, transaction_id: str) -> bool:
        print(f"Refunding {transaction_id} via Stripe")
        return True


class PayPalProcessor(PaymentProcessor):
    def process_payment(self, amount: float) -> bool:
        if not self.validate_amount(amount):
            return False
        print(f"Processing ${amount} via PayPal")
        return True

    def refund(self, transaction_id: str) -> bool:
        print(f"Refunding {transaction_id} via PayPal")
        return True


# =============================================================================
# 9. MIXINS — Reusable Behavior Without Tight Coupling
# =============================================================================
class JsonSerializableMixin:
    """Adds JSON serialization to any dataclass."""

    def to_json(self) -> str:
        import json
        if hasattr(self, "__dataclass_fields__"):
            return json.dumps(dataclasses.asdict(self))
        return json.dumps(self.__dict__)


class LoggableMixin:
    """Adds logging capability."""

    def log(self, message: str) -> None:
        print(f"[{self.__class__.__name__}] {message}")


@dataclass
class Order(JsonSerializableMixin, LoggableMixin):
    """Uses mixins for JSON serialization and logging."""
    order_id: str
    amount: float
    items: list[str]


# =============================================================================
# WHEN TO USE WHAT?
# =============================================================================
"""
Use Protocol when:
  → You want duck typing with type safety
  → You don't control the classes (third-party code)
  → You want structural subtyping

Use ABC when:
  → You want to enforce a contract on subclasses
  → You have shared implementation (concrete methods)
  → You want isinstance() checks

Use @dataclass when:
  → Your class is primarily a data container
  → You want auto-generated __init__, __repr__, __eq__
  → You want immutability (frozen=True)

Use Composition when:
  → You can describe the relationship as "has-a"
  → You want flexibility to swap components
  → You want to avoid deep inheritance hierarchies

Use Inheritance when:
  → There's a clear "is-a" relationship
  → You need shared implementation across subtypes
  → The hierarchy is shallow (max 2-3 levels)
"""


if __name__ == "__main__":
    print("=" * 60)
    print("FILE 04: OOP & SOLID Principles")
    print("=" * 60)

    print("\n--- OOP Basics ---")
    dog = Dog("Rex", "German Shepherd")
    print(dog.speak())

    print("\n--- Open/Closed Principle ---")
    print(f"Regular: {calculate_discount(RegularDiscount(), 100)}")
    print(f"VIP: {calculate_discount(VIPDiscount(), 100)}")
    print(f"Employee: {calculate_discount(EmployeeDiscount(), 100)}")

    print("\n--- Liskov Substitution ---")
    print_area(GoodRectangle(5, 3))
    print_area(GoodSquare(4))

    print("\n--- Interface Segregation ---")
    assign_work(Human())
    assign_work(Robot())

    print("\n--- Dependency Inversion ---")
    OrderProcessor(MongoDatabase()).process({"item": "keyboard"})

    print("\n--- Composition ---")
    print(car.start())

    print("\n--- Mixins ---")
    order = Order("ORD-001", 99.99, ["item1", "item2"])
    print(order.to_json())
    order.log("Order created")

    print("\n✓ File 04 complete. Move to 05_design_patterns.py")
