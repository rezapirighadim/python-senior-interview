# 01 — Python Fundamentals

## Data Types

Python has **immutable** types (`int`, `float`, `str`, `tuple`, `frozenset`, `bytes`) and **mutable** types (`list`, `dict`, `set`, `bytearray`).

```python
name: str = "Reza"
age: int = 30
salary: float = 150_000.50   # underscores for readability
is_active: bool = True
nothing: None = None          # always use `is None`, not `== None`
```

## Strings

Always use **f-strings** (not `.format()` or `%`):

```python
greeting = f"My name is {name} and I'm {age} years old"
```

**Strings are immutable.** Building strings in a loop? Use a list and `join`:

```python
parts = []
for i in range(5):
    parts.append(str(i))
result = "".join(parts)  # O(n) instead of O(n²)
```

Key methods: `upper()`, `lower()`, `strip()`, `split()`, `replace()`, `startswith()`, `find()`, `join()`

## Collections

### List — ordered, mutable, duplicates allowed

```python
fruits = ["apple", "banana", "cherry"]
fruits.append("date")       # O(1)
fruits.insert(0, "avocado") # O(n)
fruits.pop()                # O(1) remove last
fruits.sort()               # O(n log n) in-place
```

**List comprehension** (must-know):

```python
squares = [x**2 for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]
flat = [num for row in matrix for num in row]  # flatten
```

### Tuple — ordered, immutable, duplicates allowed

```python
point = (3, 4)
x, y = point  # unpacking
```

Use tuples when data shouldn't change. Faster and less memory than lists.

### Set — unordered, mutable, no duplicates

```python
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}
a & b   # intersection: {3, 4}
a | b   # union: {1, 2, 3, 4, 5, 6}
a - b   # difference: {1, 2}
```

**Lookups are O(1)** — use sets for membership testing instead of lists.

### Dict — key-value, ordered (3.7+), mutable

```python
person = {"name": "Reza", "age": 30}
person.get("phone", "N/A")  # safe access with default

# dict comprehension
squares = {x: x**2 for x in range(5)}

# merge (3.9+)
merged = d1 | d2
```

### defaultdict and Counter

```python
from collections import defaultdict, Counter

word_count = defaultdict(int)
for word in "hello world hello".split():
    word_count[word] += 1

counts = Counter("abracadabra")  # Counter({'a': 5, 'b': 2, ...})
counts.most_common(2)            # [('a', 5), ('b', 2)]
```

## Control Flow

```python
# ternary
status = "adult" if age >= 18 else "minor"

# walrus operator (3.8+)
if (n := len(data)) > 5:
    print(f"List has {n} elements")

# match statement (3.10+)
match command.split():
    case ["quit"]:
        return "Quitting..."
    case ["hello", name]:
        return f"Hello, {name}!"
    case _:
        return "Unknown"
```

## Functions

```python
# *args and **kwargs
def flexible(*args, **kwargs):
    print(args)    # tuple
    print(kwargs)  # dict

# keyword-only (after *)
def create_user(name: str, *, email: str, role: str = "user"): ...

# positional-only (before /) — 3.8+
def power(base, exp, /): ...
```

## Common Gotchas

### Mutable default arguments

```python
# WRONG
def append_to(item, target=[]):  # shared across calls!
    target.append(item)
    return target

# CORRECT
def append_to(item, target=None):
    if target is None:
        target = []
    target.append(item)
    return target
```

### `is` vs `==`

```python
a = [1, 2, 3]
b = [1, 2, 3]
a == b   # True (same value)
a is b   # False (different objects)
```

### Shallow vs Deep Copy

```python
import copy
original = [[1, 2], [3, 4]]
shallow = original.copy()        # shares inner objects
deep = copy.deepcopy(original)   # fully independent
```

## Essential Built-ins

```python
# enumerate — don't use range(len(...))
for i, name in enumerate(names, start=1):
    print(f"{i}. {name}")

# zip — iterate multiple sequences
for name, age in zip(names, ages):
    print(f"{name} is {age}")

# sorted with key
sorted(people, key=lambda p: p["age"])
```
