# 02 — Advanced Python

## Closures

A closure captures variables from its enclosing scope:

```python
def make_multiplier(factor):
    def multiplier(x):
        return x * factor
    return multiplier

double = make_multiplier(2)
double(5)  # 10
```

## Decorators

A decorator wraps a function to add behavior. Always use `@functools.wraps`.

### Basic decorator

```python
import functools

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.perf_counter()-start:.4f}s")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(0.1)
```

### Decorator with arguments

```python
def retry(max_attempts=3, delay=1.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        raise
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.1)
def unstable_api_call(): ...
```

### Stacking decorators

```python
@log_call      # outer
@timer         # inner
def add(a, b):
    return a + b
# Execution: log_call → timer → add → timer → log_call
```

## Generators

Produce values **one at a time** instead of storing all in memory.

```python
def countdown(n):
    while n > 0:
        yield n
        n -= 1

# Generator expression (lazy)
sum_of_squares = sum(x**2 for x in range(1_000_000))  # memory efficient

# yield from — delegate to sub-generator
def flatten(nested):
    for item in nested:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item
```

### Generator pipeline

```python
def read_lines(text):
    for line in text.split("\n"):
        yield line

def filter_comments(lines):
    for line in lines:
        if not line.startswith("#"):
            yield line

pipeline = filter_comments(read_lines(text))
```

## Context Managers

Guarantee cleanup even on exceptions.

```python
# Class-based
class DatabaseConnection:
    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Function-based (simpler)
from contextlib import contextmanager

@contextmanager
def timer_context(label):
    start = time.perf_counter()
    try:
        yield
    finally:
        print(f"{label}: {time.perf_counter()-start:.4f}s")
```

## Dunder Methods

Make custom classes work with Python's built-in operations:

| Method | Purpose |
|--------|---------|
| `__repr__` | Developer-friendly string (debugger/REPL) |
| `__str__` | User-friendly string (print) |
| `__eq__` | Equality comparison (==) |
| `__lt__` | Less than (<), enables sorting |
| `__add__` | Addition (+) |
| `__hash__` | Make hashable (dict keys, sets) |
| `__bool__` | Truthiness |
| `__len__` | len() support |
| `__getitem__` | Indexing (obj[key]) |
| `__enter__`/`__exit__` | Context manager (with) |

## Properties

```python
class Temperature:
    @property
    def celsius(self):
        return self._celsius

    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Below absolute zero!")
        self._celsius = value

    @property
    def fahrenheit(self):  # computed property
        return self._celsius * 9/5 + 32
```

## `__slots__`

Saves ~40% memory per instance by preventing dynamic attribute creation:

```python
class Point:
    __slots__ = ("x", "y")
    def __init__(self, x, y):
        self.x = x
        self.y = y
```

## Interview Questions

- **`@staticmethod` vs `@classmethod`**: staticmethod gets no implicit arg, classmethod gets the class (`cls`).
- **`functools.wraps`**: Preserves the wrapped function's `__name__`, `__doc__`, etc.
- **`__init__` vs `__new__`**: `__new__` creates the instance, `__init__` initializes it. `__new__` is called first.
- **MRO**: Method Resolution Order — the order Python looks up methods in inheritance chains. Uses C3 linearization. Check with `ClassName.mro()`.
