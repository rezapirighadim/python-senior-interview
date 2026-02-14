# 90-Minute Python Practice Sheet

Type each section by hand. Don't copy-paste. Writing builds muscle memory.

---

## PART 1: Core Python (15 min)

### Lists, Dicts, Sets

```python
# List comprehension
squares = [x**2 for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]
flat = [n for row in [[1,2],[3,4],[5,6]] for n in row]

# Dict comprehension
scores = {name: 0 for name in ["alice", "bob", "charlie"]}
squares_d = {x: x**2 for x in range(5)}

# Set operations
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}
print(a & b)  # {3, 4} intersection
print(a | b)  # {1,2,3,4,5,6} union
print(a - b)  # {1, 2} difference

# Counter
from collections import Counter, defaultdict
counts = Counter("abracadabra")
print(counts.most_common(2))  # [('a', 5), ('b', 2)]

# defaultdict
grouped = defaultdict(list)
for name in ["alice", "bob", "anna", "brian"]:
    grouped[name[0]].append(name)
print(dict(grouped))
```

### String Operations

```python
s = "  Hello, World!  "
s.strip()
s.lower()
s.split(", ")
"_".join(["a", "b", "c"])
f"Name: {'Reza'}, Age: {30}"

# Build strings efficiently
parts = []
for i in range(5):
    parts.append(str(i))
result = "".join(parts)
```

### Unpacking & Walrus

```python
first, *middle, last = [1, 2, 3, 4, 5]

a, b = 1, 2
a, b = b, a  # swap

data = [1, 2, 3, 4, 5, 6, 7, 8]
if (n := len(data)) > 5:
    print(f"List has {n} elements")
```

---

## PART 2: Functions & Decorators (15 min)

### Args, Kwargs, Lambda

```python
def flexible(*args, **kwargs):
    print(f"args={args}, kwargs={kwargs}")

flexible(1, 2, name="reza")

# Keyword-only (after *)
def create_user(name, *, email, role="user"):
    return {"name": name, "email": email, "role": role}

# Lambda
sorted_names = sorted(["Charlie", "Alice", "Bob"], key=lambda s: s.lower())
```

### Decorator (MUST memorize this pattern)

```python
import functools
import time

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.perf_counter()-start:.4f}s")
        return result
    return wrapper

@timer
def slow():
    time.sleep(0.1)
    return "done"
```

### Decorator with arguments

```python
def retry(max_attempts=3):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        raise
                    print(f"Retry {attempt}: {e}")
        return wrapper
    return decorator

@retry(max_attempts=3)
def unstable():
    import random
    if random.random() < 0.5:
        raise ValueError("fail")
    return "ok"
```

### Generator

```python
def countdown(n):
    while n > 0:
        yield n
        n -= 1

for x in countdown(5):
    print(x, end=" ")

# Generator expression
total = sum(x**2 for x in range(1000000))

# Flatten nested
def flatten(nested):
    for item in nested:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item

print(list(flatten([1, [2, [3, 4], 5]])))
```

### Context Manager

```python
from contextlib import contextmanager

@contextmanager
def timer_ctx(label):
    start = time.perf_counter()
    try:
        yield
    finally:
        print(f"{label}: {time.perf_counter()-start:.4f}s")

with timer_ctx("test"):
    time.sleep(0.1)
```

---

## PART 3: OOP & Patterns (15 min)

### Dataclass

```python
from dataclasses import dataclass, field

@dataclass
class User:
    name: str
    email: str
    age: int
    active: bool = True
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.age < 0:
            raise ValueError("Age cannot be negative")

@dataclass(frozen=True)
class Point:
    x: float
    y: float
```

### Protocol (duck typing with type safety)

```python
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> str: ...

class Circle:
    def draw(self) -> str:
        return "circle"

class Square:
    def draw(self) -> str:
        return "square"

def render(shape: Drawable):
    print(shape.draw())

render(Circle())
render(Square())
```

### Strategy Pattern

```python
class RegularDiscount:
    def calculate(self, amount):
        return amount * 0.05

class VIPDiscount:
    def calculate(self, amount):
        return amount * 0.20

def apply_discount(strategy, amount):
    return strategy.calculate(amount)

print(apply_discount(VIPDiscount(), 100))  # 20.0
```

### Factory Pattern

```python
class NotificationFactory:
    _registry = {}

    @classmethod
    def register(cls, name, klass):
        cls._registry[name] = klass

    @classmethod
    def create(cls, name, **kwargs):
        return cls._registry[name](**kwargs)

class EmailNotif:
    def __init__(self, to):
        self.to = to
    def send(self, msg):
        return f"Email to {self.to}: {msg}"

class SMSNotif:
    def __init__(self, phone):
        self.phone = phone
    def send(self, msg):
        return f"SMS to {self.phone}: {msg}"

NotificationFactory.register("email", EmailNotif)
NotificationFactory.register("sms", SMSNotif)
n = NotificationFactory.create("email", to="reza@x.com")
print(n.send("Hello"))
```

### Observer Pattern

```python
class EventEmitter:
    def __init__(self):
        self._listeners = {}

    def on(self, event, callback):
        self._listeners.setdefault(event, []).append(callback)

    def emit(self, event, **data):
        for cb in self._listeners.get(event, []):
            cb(**data)

bus = EventEmitter()
bus.on("order", lambda order_id, **kw: print(f"Order {order_id} placed"))
bus.emit("order", order_id="ORD-1")
```

---

## PART 4: Async (10 min)

```python
import asyncio

async def fetch(url):
    print(f"Fetching {url}...")
    await asyncio.sleep(1)
    return {"url": url, "status": 200}

async def main():
    # Sequential — 3 seconds
    r1 = await fetch("/a")
    r2 = await fetch("/b")
    r3 = await fetch("/c")

    # Concurrent — 1 second!
    results = await asyncio.gather(
        fetch("/a"),
        fetch("/b"),
        fetch("/c"),
    )
    print(f"Got {len(results)} results")

    # With semaphore (limit concurrency)
    sem = asyncio.Semaphore(2)
    async def limited(url):
        async with sem:
            return await fetch(url)

    urls = [f"/item/{i}" for i in range(10)]
    results = await asyncio.gather(*[limited(u) for u in urls])

asyncio.run(main())
```

---

## PART 5: LeetCode Patterns — Type These (20 min)

### Two Sum (Hash Map)

```python
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

print(two_sum([2, 7, 11, 15], 9))  # [0, 1]
```

### Valid Parentheses (Stack)

```python
def is_valid(s):
    stack = []
    pairs = {")": "(", "}": "{", "]": "["}
    for c in s:
        if c in "({[":
            stack.append(c)
        elif not stack or stack[-1] != pairs[c]:
            return False
        else:
            stack.pop()
    return len(stack) == 0

print(is_valid("({[]})"))  # True
print(is_valid("(]"))      # False
```

### Longest Substring Without Repeating (Sliding Window)

```python
def longest_unique(s):
    seen = set()
    left = 0
    max_len = 0
    for right in range(len(s)):
        while s[right] in seen:
            seen.remove(s[left])
            left += 1
        seen.add(s[right])
        max_len = max(max_len, right - left + 1)
    return max_len

print(longest_unique("abcabcbb"))  # 3
```

### Binary Search

```python
def binary_search(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

print(binary_search([1, 3, 5, 7, 9], 7))  # 3
```

### Reverse Linked List

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse(head):
    prev = None
    curr = head
    while curr:
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt
    return prev
```

### Max Depth of Binary Tree (Recursion)

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def max_depth(root):
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))
```

### Merge Intervals

```python
def merge(intervals):
    intervals.sort()
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged

print(merge([[1,3],[2,6],[8,10],[15,18]]))  # [[1,6],[8,10],[15,18]]
```

### Coin Change (DP)

```python
def coin_change(coins, amount):
    dp = [float("inf")] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i and dp[i - coin] != float("inf"):
                dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[amount] if dp[amount] != float("inf") else -1

print(coin_change([1, 5, 10, 25], 36))  # 3
```

---

## PART 6: Quick-Fire Concepts to Write Down (15 min)

Write these from memory on paper:

### GIL
- Only one thread runs Python bytecode at a time
- I/O threads benefit (GIL released during I/O)
- CPU parallelism → use multiprocessing

### Mutable Default Argument Bug
```python
# BAD
def add(item, lst=[]):
    lst.append(item)
    return lst
# GOOD
def add(item, lst=None):
    if lst is None:
        lst = []
    lst.append(item)
    return lst
```

### is vs ==
```python
a = [1, 2]
b = [1, 2]
a == b   # True (value)
a is b   # False (identity)
# Use `is` only for None
```

### SOLID (one line each)
- S: one class, one reason to change
- O: extend with new classes, don't modify existing
- L: subtypes can replace base types
- I: small focused interfaces, not fat ones
- D: depend on abstractions, not implementations

### When to use what
- asyncio → I/O-bound, many connections
- threading → blocking I/O libraries
- multiprocessing → CPU-bound
- Celery → distributed background tasks

### Pattern recognition
- sorted array + pair → Two Pointers
- subarray/substring → Sliding Window
- min/max with condition → Binary Search on Answer
- count ways / optimize → DP
- generate all X → Backtracking
- overlapping ranges → Sort + Merge
- dependencies → Topological Sort
- next greater → Monotonic Stack
