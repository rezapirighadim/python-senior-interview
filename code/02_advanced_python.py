"""
=============================================================================
FILE 02: ADVANCED PYTHON — What Makes You a Senior Dev
=============================================================================
Decorators, generators, context managers, comprehensions, closures,
metaclasses. This is what separates junior from senior.
=============================================================================
"""
import functools
import time
from contextlib import contextmanager
from typing import Any, Callable


# =============================================================================
# 1. CLOSURES — Functions That Remember Their Environment
# =============================================================================
# A closure is a function that captures variables from its enclosing scope.

def make_multiplier(factor: int) -> Callable[[int], int]:
    """Factory function — returns a function that multiplies by `factor`."""
    def multiplier(x: int) -> int:
        return x * factor  # `factor` is captured from enclosing scope
    return multiplier

double = make_multiplier(2)
triple = make_multiplier(3)
print(double(5))   # 10
print(triple(5))   # 15

# WHY it matters: Decorators are built on closures.


# =============================================================================
# 2. DECORATORS — The Most Pythonic Pattern
# =============================================================================
# A decorator is a function that takes a function and returns a modified function.

# --- Basic decorator ---
def timer(func: Callable) -> Callable:
    """Measures execution time of a function."""
    @functools.wraps(func)  # Preserves original function's name & docstring
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

@timer
def slow_function():
    """This is a slow function."""
    time.sleep(0.1)
    return "done"

# @timer is syntactic sugar for: slow_function = timer(slow_function)


# --- Decorator WITH arguments ---
def retry(max_attempts: int = 3, delay: float = 1.0):
    """Retry a function on failure."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    print(f"Attempt {attempt}/{max_attempts} failed: {e}")
                    if attempt < max_attempts:
                        time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.1)
def unstable_api_call():
    """Simulates an unreliable API."""
    import random
    if random.random() < 0.7:
        raise ConnectionError("API timeout")
    return {"status": "ok"}


# --- Class-based decorator ---
class CacheResult:
    """Decorator that caches function results (simple memoization)."""

    def __init__(self, func: Callable):
        functools.update_wrapper(self, func)
        self.func = func
        self.cache: dict = {}

    def __call__(self, *args):
        if args not in self.cache:
            self.cache[args] = self.func(*args)
        return self.cache[args]

    def clear_cache(self):
        self.cache.clear()

@CacheResult
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Built-in alternative: @functools.lru_cache(maxsize=128)
# or @functools.cache (Python 3.9+, unlimited cache)


# --- Stacking decorators ---
def log_call(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}({args}, {kwargs})")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {result}")
        return result
    return wrapper

@log_call
@timer
def add(a: int, b: int) -> int:
    return a + b

# Execution order: log_call wraps timer which wraps add
# So: log_call → timer → add → timer → log_call


# =============================================================================
# 3. GENERATORS — Lazy Evaluation for Memory Efficiency
# =============================================================================
# Generators produce values ONE AT A TIME instead of storing all in memory.

# --- Generator function (uses yield) ---
def countdown(n: int):
    """Yields numbers from n down to 1."""
    while n > 0:
        yield n  # Pauses here, resumes on next()
        n -= 1

for num in countdown(5):
    print(num, end=" ")  # 5 4 3 2 1
print()

# --- Generator expression (like list comprehension but lazy) ---
# List comprehension: [x**2 for x in range(1_000_000)]  ← stores ALL in memory
# Generator expression: (x**2 for x in range(1_000_000))  ← generates on demand

sum_of_squares = sum(x**2 for x in range(1_000_000))  # Memory efficient!


# --- Real-world generator: Reading chunks from large file ---
def read_in_chunks(file_path: str, chunk_size: int = 1024):
    """Read a file in chunks — essential for processing huge files."""
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk


# --- Generator pipeline (composing generators) ---
def read_lines(text: str):
    for line in text.split("\n"):
        yield line

def filter_comments(lines):
    for line in lines:
        if not line.strip().startswith("#"):
            yield line

def strip_whitespace(lines):
    for line in lines:
        yield line.strip()

# Chain them together — each line flows through the pipeline
text = "  hello  \n# comment\n  world  \n# another comment\n  foo  "
pipeline = strip_whitespace(filter_comments(read_lines(text)))
result = list(pipeline)  # ['hello', 'world', 'foo']


# --- yield from — delegate to sub-generator ---
def flatten(nested_list):
    """Flatten arbitrarily nested lists."""
    for item in nested_list:
        if isinstance(item, list):
            yield from flatten(item)  # Delegate to recursive call
        else:
            yield item

list(flatten([1, [2, [3, 4], 5], [6, 7]]))  # [1, 2, 3, 4, 5, 6, 7]


# --- send() — two-way communication with generators ---
def running_average():
    """Generator that computes running average."""
    total = 0.0
    count = 0
    average = None
    while True:
        value = yield average  # Receive value, yield current average
        total += value
        count += 1
        average = total / count

avg = running_average()
next(avg)           # Prime the generator (advance to first yield)
avg.send(10)        # 10.0
avg.send(20)        # 15.0
avg.send(30)        # 20.0


# =============================================================================
# 4. CONTEXT MANAGERS — Resource Management Done Right
# =============================================================================

# --- Class-based context manager ---
class DatabaseConnection:
    """Simulates a database connection with proper cleanup."""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connected = False

    def __enter__(self):
        print(f"Connecting to {self.connection_string}...")
        self.connected = True
        return self  # This is what `as` variable receives

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Closing connection...")
        self.connected = False
        # Return True to suppress the exception, False to propagate
        return False

# Usage:
# with DatabaseConnection("postgres://localhost/mydb") as db:
#     db.execute("SELECT * FROM users")
# Connection is ALWAYS closed, even if exception occurs


# --- Function-based context manager (simpler!) ---
@contextmanager
def timer_context(label: str):
    """Context manager that times a block of code."""
    start = time.perf_counter()
    try:
        yield  # Code inside `with` block runs here
    finally:
        elapsed = time.perf_counter() - start
        print(f"{label}: {elapsed:.4f}s")

# with timer_context("Processing"):
#     time.sleep(0.1)
# Output: "Processing: 0.1001s"


# --- Practical: Temporary directory ---
@contextmanager
def temporary_change_dir(path: str):
    """Change directory temporarily, restore on exit."""
    import os
    original = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(original)


# =============================================================================
# 5. DUNDER (MAGIC) METHODS — Making Custom Classes Pythonic
# =============================================================================
class Money:
    """Custom class with rich Python protocol support."""

    def __init__(self, amount: float, currency: str = "USD"):
        self.amount = amount
        self.currency = currency

    # String representations
    def __repr__(self) -> str:
        """For developers — unambiguous. Used in debugger/REPL."""
        return f"Money({self.amount}, '{self.currency}')"

    def __str__(self) -> str:
        """For users — readable. Used in print()."""
        return f"${self.amount:.2f} {self.currency}"

    # Comparison
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Money):
            return NotImplemented
        return self.amount == other.amount and self.currency == other.currency

    def __lt__(self, other: "Money") -> bool:
        if self.currency != other.currency:
            raise ValueError("Cannot compare different currencies")
        return self.amount < other.amount

    # Arithmetic
    def __add__(self, other: "Money") -> "Money":
        if self.currency != other.currency:
            raise ValueError("Cannot add different currencies")
        return Money(self.amount + other.amount, self.currency)

    # Make it hashable (required if you define __eq__)
    def __hash__(self) -> int:
        return hash((self.amount, self.currency))

    # Make it work with len()
    def __bool__(self) -> bool:
        return self.amount != 0

    # Make it work with `with` statement
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# =============================================================================
# 6. PROPERTY — Controlled Attribute Access
# =============================================================================
class Temperature:
    """Demonstrates property decorators for validation."""

    def __init__(self, celsius: float):
        self.celsius = celsius  # This calls the setter!

    @property
    def celsius(self) -> float:
        return self._celsius

    @celsius.setter
    def celsius(self, value: float):
        if value < -273.15:
            raise ValueError("Temperature below absolute zero!")
        self._celsius = value

    @property
    def fahrenheit(self) -> float:
        """Computed property — no storage, calculated on access."""
        return self._celsius * 9/5 + 32

    @fahrenheit.setter
    def fahrenheit(self, value: float):
        self.celsius = (value - 32) * 5/9


# =============================================================================
# 7. SLOTS — Memory Optimization
# =============================================================================
class Point:
    """Using __slots__ prevents dynamic attribute creation and saves memory."""
    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

# Without __slots__: each instance has a __dict__ (56+ bytes overhead)
# With __slots__: fixed attributes, ~40% less memory per instance
# Use when you have millions of instances (e.g., data processing)


# =============================================================================
# 8. DESCRIPTORS — Advanced Attribute Protocol
# =============================================================================
class Validated:
    """Descriptor that validates attribute values."""

    def __init__(self, min_value: float = None, max_value: float = None):
        self.min_value = min_value
        self.max_value = max_value

    def __set_name__(self, owner, name):
        self.name = name
        self.private_name = f"_{name}"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.private_name, None)

    def __set__(self, obj, value):
        if self.min_value is not None and value < self.min_value:
            raise ValueError(f"{self.name} must be >= {self.min_value}")
        if self.max_value is not None and value > self.max_value:
            raise ValueError(f"{self.name} must be <= {self.max_value}")
        setattr(obj, self.private_name, value)

class Product:
    price = Validated(min_value=0)
    quantity = Validated(min_value=0, max_value=10_000)

    def __init__(self, name: str, price: float, quantity: int):
        self.name = name
        self.price = price
        self.quantity = quantity


# =============================================================================
# 9. METACLASSES — Classes That Create Classes (Advanced)
# =============================================================================
# 99% of the time you don't need metaclasses. But you should KNOW them.
# Use case: frameworks (Django models, SQLAlchemy, Pydantic)

class SingletonMeta(type):
    """Metaclass that ensures only one instance of a class exists."""
    _instances: dict = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class AppConfig(metaclass=SingletonMeta):
    def __init__(self):
        self.debug = False
        self.database_url = "sqlite:///app.db"

# AppConfig() is AppConfig()  → True (same instance!)


# =============================================================================
# 10. WALRUS + COMPREHENSION + GENERATOR TRICKS
# =============================================================================

# Walrus operator in comprehensions
import math
# Find all numbers whose sqrt is > 5
numbers = range(1, 100)
result = [(n, root) for n in numbers if (root := math.sqrt(n)) > 5]

# Dictionary merge with | operator (Python 3.9+)
defaults = {"theme": "dark", "lang": "en", "font_size": 14}
user_prefs = {"theme": "light", "font_size": 16}
config = defaults | user_prefs  # user_prefs overrides defaults

# Conditional comprehension with walrus
data = ["  hello  ", "", "  world  ", "   ", "  foo  "]
cleaned = [stripped for s in data if (stripped := s.strip())]
# ['hello', 'world', 'foo'] — filters empty strings after stripping


# =============================================================================
# INTERVIEW QUESTIONS YOU MIGHT GET
# =============================================================================
"""
Q: What is the difference between @staticmethod and @classmethod?
A: @staticmethod doesn't receive any implicit first argument.
   @classmethod receives the CLASS as first argument (cls).
   Use @classmethod for factory methods, @staticmethod for utility functions.

Q: What does functools.wraps do?
A: It copies the wrapped function's __name__, __doc__, __module__ etc.
   Without it, decorated functions lose their identity.

Q: When would you use a generator instead of a list?
A: When dealing with large datasets where you don't need all values at once.
   Generators are lazy — they compute values on demand, saving memory.

Q: What is the MRO (Method Resolution Order)?
A: The order Python looks up methods in inheritance. Uses C3 linearization.
   Check with: ClassName.__mro__ or ClassName.mro()

Q: What is __init__ vs __new__?
A: __new__ creates the instance (rarely overridden).
   __init__ initializes the instance (commonly overridden).
   __new__ is called BEFORE __init__.
"""


if __name__ == "__main__":
    print("=" * 60)
    print("FILE 02: Advanced Python")
    print("=" * 60)

    print("\n--- Closures ---")
    print(f"double(5) = {double(5)}")
    print(f"triple(5) = {triple(5)}")

    print("\n--- Decorators ---")
    slow_function()
    print(f"Function name preserved: {slow_function.__name__}")

    print("\n--- Generators ---")
    print(f"Flattened: {list(flatten([1, [2, [3, 4], 5], [6, 7]]))}")

    print("\n--- Dunder methods ---")
    m1 = Money(10.50)
    m2 = Money(20.75)
    print(f"{m1} + {m2} = {m1 + m2}")
    print(f"repr: {repr(m1)}")

    print("\n--- Properties ---")
    t = Temperature(100)
    print(f"{t.celsius}°C = {t.fahrenheit}°F")

    print("\n--- Singleton ---")
    c1 = AppConfig()
    c2 = AppConfig()
    print(f"Same instance? {c1 is c2}")

    print("\n✓ File 02 complete. Move to 03_type_hints_and_modern_python.py")
