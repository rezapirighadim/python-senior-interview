# 06 — Functional Programming

## First-Class Functions

Functions are objects — assign, pass, store them:

```python
formatters = {"loud": str.upper, "quiet": str.lower}
formatters["loud"]("hello")  # "HELLO"
```

## Map, Filter, Reduce

```python
squared = list(map(lambda x: x**2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))
total = reduce(add, numbers)
```

Prefer list comprehensions for simple cases.

## functools

### partial — fix some arguments

```python
from functools import partial
square = partial(pow, exp=2)
pretty_json = partial(json.dumps, indent=2, sort_keys=True)
```

### lru_cache — memoization

```python
@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2: return n
    return fibonacci(n-1) + fibonacci(n-2)
```

## Function Composition

```python
def pipe(*funcs):
    def inner(x):
        for f in funcs:
            x = f(x)
        return x
    return inner

clean = pipe(str.strip, str.lower, lambda s: " ".join(s.split()))
```

## itertools Highlights

| Function | What It Does |
|----------|-------------|
| `chain(a, b)` | Flatten iterables |
| `islice(iter, 5, 15)` | Slice an iterator |
| `groupby(data, key)` | Group consecutive elements |
| `product("AB", "12")` | Cartesian product |
| `combinations("ABC", 2)` | Unordered pairs |
| `permutations("ABC", 2)` | Ordered pairs |
| `accumulate([1,2,3])` | Running total: [1,3,6] |
| `starmap(pow, [(2,3)])` | Map with unpacking |

## operator Module

```python
from operator import itemgetter, attrgetter
sorted(users, key=itemgetter("age"))        # sort dicts
sorted(employees, key=attrgetter("salary")) # sort objects
```

## Immutability

```python
DIRECTIONS = ("north", "south", "east", "west")       # tuple
ALLOWED = frozenset({"admin", "editor"})               # frozenset
CONFIG = MappingProxyType({"debug": False})             # read-only dict
```

## When to Use FP vs OOP

- **FP**: data transforms, stateless computation, pipelines, utilities
- **OOP**: stateful entities, polymorphism, domain modeling
- **Python**: mix both — FP for data, OOP for domain
