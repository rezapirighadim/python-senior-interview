"""
=============================================================================
FILE 06: FUNCTIONAL PROGRAMMING IN PYTHON
=============================================================================
Python is multi-paradigm. Knowing FP makes you write cleaner, more
testable code. This is what separates good from great Python code.
=============================================================================
"""
from functools import reduce, partial, lru_cache, wraps
from itertools import (
    chain, islice, groupby, product, permutations,
    combinations, accumulate, starmap, zip_longest
)
from operator import itemgetter, attrgetter, add
from typing import Callable, TypeVar

T = TypeVar("T")


# =============================================================================
# 1. FIRST-CLASS FUNCTIONS — Functions Are Objects
# =============================================================================
def shout(text: str) -> str:
    return text.upper()

def whisper(text: str) -> str:
    return text.lower()

# Functions can be assigned to variables
speak = shout
print(speak("hello"))  # "HELLO"

# Functions can be passed as arguments
def greet(name: str, formatter: Callable[[str], str]) -> str:
    return f"Hi, {formatter(name)}!"

print(greet("reza", shout))    # "Hi, REZA!"
print(greet("reza", whisper))  # "Hi, reza!"

# Functions can be stored in data structures
formatters = {"loud": shout, "quiet": whisper}
print(formatters["loud"]("hello"))


# =============================================================================
# 2. MAP, FILTER, REDUCE — The Classic FP Trio
# =============================================================================

# --- map: apply function to every element ---
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))    # [1, 4, 9, 16, 25]
# Pythonic: [x**2 for x in numbers]  ← prefer this for simple cases

# map with multiple iterables
list(map(pow, [2, 3, 4], [3, 2, 1]))  # [8, 9, 4] → 2³, 3², 4¹

# --- filter: keep elements that match condition ---
evens = list(filter(lambda x: x % 2 == 0, numbers))  # [2, 4]
# Pythonic: [x for x in numbers if x % 2 == 0]

# --- reduce: accumulate into single value ---
total = reduce(add, numbers)           # 15 (1+2+3+4+5)
product_val = reduce(lambda a, b: a * b, numbers)  # 120 (1*2*3*4*5)

# reduce with initial value
total_plus_100 = reduce(add, numbers, 100)  # 115

# When to use reduce vs built-ins:
# sum(numbers)          ← built-in, use for addition
# max(numbers)          ← built-in
# reduce(custom_fn, x)  ← only when you need custom accumulation


# =============================================================================
# 3. FUNCTOOLS — Power Tools for Functions
# =============================================================================

# --- partial: fix some arguments ---
def power(base: int, exponent: int) -> int:
    return base ** exponent

square = partial(power, exponent=2)
cube = partial(power, exponent=3)
print(square(5))  # 25
print(cube(3))    # 27

# Real-world use: configuring functions
import json
pretty_json = partial(json.dumps, indent=2, sort_keys=True)
# Now pretty_json(data) is shorthand for json.dumps(data, indent=2, sort_keys=True)


# --- lru_cache: memoization ---
@lru_cache(maxsize=128)
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(100))  # Instant! Without cache: takes forever
print(fibonacci.cache_info())  # CacheInfo(hits=98, misses=101, ...)
# fibonacci.cache_clear()  # Reset cache


# --- compose functions (not built-in, but common pattern) ---
def compose(*funcs: Callable) -> Callable:
    """Compose functions right to left: compose(f, g)(x) = f(g(x))"""
    def inner(x):
        result = x
        for f in reversed(funcs):
            result = f(result)
        return result
    return inner

def double(x): return x * 2
def increment(x): return x + 1
def stringify(x): return f"Result: {x}"

pipeline = compose(stringify, double, increment)
print(pipeline(5))  # "Result: 12" → increment(5)=6, double(6)=12, stringify(12)


# --- pipe: compose left to right (more readable) ---
def pipe(*funcs: Callable) -> Callable:
    """Pipe functions left to right: pipe(f, g)(x) = g(f(x))"""
    def inner(x):
        result = x
        for f in funcs:
            result = f(result)
        return result
    return inner

pipeline = pipe(increment, double, stringify)
print(pipeline(5))  # Same: "Result: 12"


# =============================================================================
# 4. ITERTOOLS — Efficient Iteration Tools
# =============================================================================
# itertools returns ITERATORS (lazy) — no memory wasted

# --- chain: flatten iterables ---
list(chain([1, 2], [3, 4], [5, 6]))        # [1, 2, 3, 4, 5, 6]
list(chain.from_iterable([[1, 2], [3, 4]])) # [1, 2, 3, 4, 5, 6]

# --- islice: slice an iterator (can't use [:] on iterators) ---
list(islice(range(100), 5, 15, 2))  # [5, 7, 9, 11, 13]

# --- groupby: group consecutive elements (must be sorted first!) ---
data = [
    {"city": "NYC", "name": "Alice"},
    {"city": "NYC", "name": "Bob"},
    {"city": "LA", "name": "Charlie"},
    {"city": "LA", "name": "Diana"},
]
# Data MUST be sorted by the key you're grouping by!
for city, group in groupby(data, key=itemgetter("city")):
    members = [person["name"] for person in group]
    print(f"{city}: {members}")
# NYC: ['Alice', 'Bob']
# LA: ['Charlie', 'Diana']


# --- product: cartesian product ---
list(product("AB", "12"))  # [('A','1'), ('A','2'), ('B','1'), ('B','2')]
# Replaces nested loops:
# for a in "AB":
#     for b in "12":
#         ...


# --- permutations and combinations ---
list(permutations("ABC", 2))   # All ordered pairs: ('A','B'), ('A','C'), ('B','A')...
list(combinations("ABC", 2))   # All unordered pairs: ('A','B'), ('A','C'), ('B','C')


# --- accumulate: running total ---
list(accumulate([1, 2, 3, 4, 5]))        # [1, 3, 6, 10, 15]
list(accumulate([1, 2, 3, 4], max))       # [1, 2, 3, 4] — running max
list(accumulate([3, 1, 4, 1, 5], min))    # [3, 1, 1, 1, 1] — running min


# --- starmap: map with unpacking ---
list(starmap(pow, [(2, 3), (3, 2), (4, 1)]))  # [8, 9, 4]


# --- zip_longest: zip with fill ---
list(zip_longest([1, 2, 3], ["a", "b"], fillvalue="-"))
# [(1, 'a'), (2, 'b'), (3, '-')]


# =============================================================================
# 5. OPERATOR MODULE — Function Versions of Operators
# =============================================================================
# Instead of lambdas, use operator functions for readability

from operator import itemgetter, attrgetter, mul

# Sorting with operator
people = [("Charlie", 30), ("Alice", 25), ("Bob", 35)]
sorted(people, key=itemgetter(1))  # Sort by age
sorted(people, key=itemgetter(0))  # Sort by name

# With dicts
users = [
    {"name": "Charlie", "age": 30},
    {"name": "Alice", "age": 25},
]
sorted(users, key=itemgetter("age"))

# With objects
from dataclasses import dataclass

@dataclass
class Employee:
    name: str
    salary: float

employees = [Employee("Alice", 90000), Employee("Bob", 75000)]
sorted(employees, key=attrgetter("salary"))


# =============================================================================
# 6. HIGHER-ORDER PATTERNS — Real-World Examples
# =============================================================================

# --- Retry with backoff (decorator as higher-order function) ---
def retry_with_backoff(max_retries: int = 3, backoff_factor: float = 2.0):
    """Decorator: retry function with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    wait = backoff_factor ** attempt
                    print(f"Retry {attempt + 1}/{max_retries} in {wait}s: {e}")
                    time.sleep(wait)
        return wrapper
    return decorator


# --- Pipeline pattern for data processing ---
def pipeline_processor(*steps: Callable) -> Callable:
    """Create a data processing pipeline."""
    def process(data):
        result = data
        for step in steps:
            result = step(result)
        return result
    return process

# Example: text cleaning pipeline
clean_text = pipeline_processor(
    str.strip,
    str.lower,
    lambda s: " ".join(s.split()),  # Normalize whitespace
    lambda s: s.replace("  ", " "),
)
print(clean_text("  Hello   World!  "))  # "hello world!"


# --- Validation chain ---
def validate(*validators: Callable[[Any], bool]) -> Callable[[Any], bool]:
    """Combine multiple validators."""
    def combined(value: Any) -> bool:
        return all(v(value) for v in validators)
    return combined

is_positive = lambda x: x > 0
is_even = lambda x: x % 2 == 0
is_small = lambda x: x < 100

is_valid_number = validate(is_positive, is_even, is_small)
print(is_valid_number(42))   # True
print(is_valid_number(-2))   # False
print(is_valid_number(101))  # False


# =============================================================================
# 7. IMMUTABILITY PATTERNS
# =============================================================================

# Using tuple instead of list for immutable sequences
DIRECTIONS = ("north", "south", "east", "west")

# Using frozenset for immutable sets
ALLOWED_ROLES = frozenset({"admin", "editor", "viewer"})

# Using types.MappingProxyType for read-only dict view
from types import MappingProxyType

_config = {"debug": False, "port": 8080}
CONFIG = MappingProxyType(_config)  # Read-only view
# CONFIG["debug"] = True  ← TypeError!


# =============================================================================
# 8. WHEN TO USE FP vs OOP in Python
# =============================================================================
"""
USE FUNCTIONAL STYLE when:
  → Data transformation pipelines (map, filter, reduce)
  → Stateless computations
  → You need composability (combine small functions into bigger ones)
  → Processing collections
  → Writing utility/helper functions

USE OOP when:
  → You have stateful entities (User, Order, Connection)
  → You need polymorphism (different types, same interface)
  → You're building frameworks/libraries
  → Domain modeling (DDD)

PYTHON BEST PRACTICE:
  → Mix both! Use FP for data transforms, OOP for domain models.
  → Prefer list comprehensions over map/filter for simple cases
  → Use functools when you need partial application or memoization
  → Use itertools for complex iteration (it's C-optimized and fast)
"""


if __name__ == "__main__":
    print("=" * 60)
    print("FILE 06: Functional Programming")
    print("=" * 60)

    print("\n--- First-class functions ---")
    print(greet("reza", shout))

    print("\n--- Partial application ---")
    print(f"square(5) = {square(5)}")
    print(f"cube(3) = {cube(3)}")

    print("\n--- Function composition ---")
    print(pipe(increment, double, stringify)(5))

    print("\n--- itertools.groupby ---")
    for city, group in groupby(data, key=itemgetter("city")):
        print(f"  {city}: {[p['name'] for p in group]}")

    print("\n--- accumulate ---")
    print(f"  Running sum: {list(accumulate([1, 2, 3, 4, 5]))}")

    print("\n--- Pipeline pattern ---")
    print(f"  Clean: '{clean_text('  Hello   World!  ')}'")

    print("\n✓ File 06 complete. Move to 07_async_programming.py")
