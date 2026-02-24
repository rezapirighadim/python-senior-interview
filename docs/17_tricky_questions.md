# 17 -- Tricky Python Questions

40+ questions that trip up even senior developers. Each shows a code snippet, the surprising output, and an explanation of why.

---

## Category 1: Mutable Default Arguments

### Q1: The Classic Mutable Default Argument Trap

```python
def append_to(element, target=[]):
    target.append(element)
    return target

print(append_to(1))  # [1]
print(append_to(2))  # [1, 2]  <-- surprise!
print(append_to(3))  # [1, 2, 3]
```

**Output:** `[1]`, `[1, 2]`, `[1, 2, 3]`

**Why:** Default arguments are evaluated **once** at function definition time, not each call. The list `[]` is created once and shared across all calls. Each call mutates the same object.

**Fix:**
```python
def append_to(element, target=None):
    if target is None:
        target = []
    target.append(element)
    return target
```

### Q2: Mutable Default with Dict

```python
def add_student(name, roster={}):
    roster[name] = True
    return roster

print(add_student("Alice"))  # {'Alice': True}
print(add_student("Bob"))    # {'Alice': True, 'Bob': True}
```

**Output:** Both print `{'Alice': True, 'Bob': True}`

**Why:** Same principle as lists. The dict `{}` is created once at function definition time. Every call shares and mutates the same dict. Both return values reference the same dict object.

### Q3: Inspecting the Default Object

```python
def buggy(items=[]):
    items.append("x")
    return items

buggy()
buggy()
print(buggy.__defaults__)  # (['x', 'x'],)
```

**Output:** `(['x', 'x'],)`

**Why:** `func.__defaults__` stores the actual default argument objects. Since the list is mutated on every call, you can see the accumulated state. This is also how you can detect this bug: check if `__defaults__` contains mutable objects.

### Q4: Immutable Defaults Are Fine

```python
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

print(greet("Alice"))            # Hello, Alice!
print(greet("Bob"))              # Hello, Bob!
print(greet("Charlie", "Hey"))   # Hey, Charlie!
print(greet("Dave"))             # Hello, Dave!
```

**Why:** Strings (and ints, tuples, frozensets) are immutable. Even though the default is evaluated once, you cannot mutate an immutable object. The mutable default trap **only** applies to mutable types: list, dict, set, custom objects.

---

## Category 2: Scope & Closures

### Q5: Late Binding Closures in a Loop

```python
funcs = []
for i in range(4):
    funcs.append(lambda: i)

print([f() for f in funcs])  # [3, 3, 3, 3]
```

**Output:** `[3, 3, 3, 3]` -- NOT `[0, 1, 2, 3]`

**Why:** Closures capture **variables**, not **values**. All lambdas reference the same variable `i`. By the time they are called, the loop has finished and `i` is 3.

**Fix:** Use a default argument to capture the value at creation time:
```python
funcs = [lambda i=i: i for i in range(4)]  # [0, 1, 2, 3]
```

### Q6: LEGB Rule

```python
x = "global"

def outer():
    x = "enclosing"
    def inner():
        x = "local"
        print(x)      # local
    inner()
    print(x)           # enclosing

outer()
print(x)               # global
```

**Output:** `local`, `enclosing`, `global`

**Why:** Python follows LEGB -- Local, Enclosing, Global, Built-in. Each `x = ...` creates a new local variable in that scope. Assignment creates a local variable; it does not modify the outer `x`.

### Q7: nonlocal vs global

```python
count = 0

def outer():
    count = 10
    def inner():
        nonlocal count
        count += 1
    inner()
    print("outer count:", count)   # 11

outer()
print("global count:", count)      # 0
```

**Output:** `outer count: 11`, `global count: 0`

**Why:** `nonlocal` binds to the nearest enclosing scope (outer's count), not the global. If you wanted to modify the global, use `global count`.

### Q8: UnboundLocalError Surprise

```python
x = 10

def foo():
    print(x)   # UnboundLocalError!
    x = 20

foo()
```

**Output:** `UnboundLocalError`

**Why:** Python determines variable scope at **compile time**, not runtime. Because `x = 20` exists anywhere in the function, `x` is classified as local for the entire function. When `print(x)` executes, the local `x` has not been assigned yet.

### Q9: Closure Over Loop Variable -- The Fix

```python
# Method 1: Default argument capture
funcs1 = [lambda i=i: i for i in range(4)]

# Method 2: Factory function
def make_func(val):
    return lambda: val
funcs2 = [make_func(i) for i in range(4)]

print([f() for f in funcs1])  # [0, 1, 2, 3]
print([f() for f in funcs2])  # [0, 1, 2, 3]
```

**Why:** Both methods capture the **value** at iteration time instead of the variable. The factory approach is generally considered cleaner.

---

## Category 3: Identity vs Equality

### Q10: Integer Caching (-5 to 256)

```python
a = 256
b = 256
print(a is b)   # True

c = 257
d = 257
print(c is d)   # False (typically)
```

**Why:** CPython caches integers from -5 to 256 as singleton objects. `256 is 256` is True (same object). But 257 creates a new int object each time. **Never** rely on `is` for integer comparison. Always use `==`.

### Q11: String Interning

```python
a = "hello"
b = "hello"
print(a is b)          # True

c = "hello world"      # with space
d = "hello world"
print(c is d)          # May be False at runtime

e = "hello_world"      # identifier-like
f = "hello_world"
print(e is f)          # True
```

**Why:** Python interns strings that look like identifiers (letters, digits, underscores). Strings with spaces may not be interned at runtime. **Never** use `is` for string comparison. Always use `==`.

### Q12: None Comparison -- is vs ==

```python
class Sneaky:
    def __eq__(self, other):
        return True

s = Sneaky()
print(s == None)    # True -- __eq__ was overridden!
print(s is None)    # False -- identity check
print(None is None) # True -- singleton
```

**Why:** `==` calls `__eq__` which can be overridden. `is` checks identity. PEP 8 says: "Comparisons to singletons like None should always be done with `is` or `is not`, never the equality operators."

### Q13: is vs == with Lists and Tuples

```python
a = [1, 2, 3]
b = [1, 2, 3]
print(a == b)   # True (same values)
print(a is b)   # False (different objects)
```

**Why:** `==` compares values element by element. `is` checks if they are the same object in memory. Two independently created containers are equal but not identical.

---

## Category 4: Mutability Tricks

### Q14: Tuple with Mutable Elements

```python
t = (1, 2, [3, 4])
t[2].append(5)
print(t)  # (1, 2, [3, 4, 5])
```

**Why:** Tuples are immutable, but they store **references**. The tuple cannot change which objects it references, but the objects themselves can be mutable. The list inside the tuple can still be modified.

### Q15: Tuple += with Mutable Element

```python
t = (1, 2, [3, 4])
try:
    t[2] += [5, 6]
except TypeError as e:
    print(f"Error: {e}")
print(t)  # (1, 2, [3, 4, 5, 6])  -- modified despite error!
```

**Output:** BOTH an error AND the list is modified.

**Why:** `t[2] += [5, 6]` does two things:
1. Calls `t[2].__iadd__([5, 6])` -- this **succeeds** (mutates the list)
2. Assigns the result back: `t[2] = result` -- this **fails** (tuple is immutable)

The list is mutated in step 1 before the TypeError in step 2. One of Python's most notorious gotchas.

### Q16: Shallow vs Deep Copy

```python
import copy
original = [[1, 2], [3, 4]]
shallow = copy.copy(original)
deep = copy.deepcopy(original)

original[0].append(99)

print(original)  # [[1, 2, 99], [3, 4]]
print(shallow)   # [[1, 2, 99], [3, 4]]  <-- also changed!
print(deep)      # [[1, 2], [3, 4]]       <-- independent
```

**Why:** Shallow copy creates a new outer list but the inner lists are still references to the same objects. Deep copy recursively copies everything.

### Q17: Dict Key Requirements

```python
d = {[1, 2]: "value"}         # TypeError: unhashable type: 'list'
d = {(1, 2): "value"}         # OK: tuples are hashable
d = {(1, [2, 3]): "value"}   # TypeError: tuple contains unhashable list
```

**Why:** Dict keys must be hashable. Lists are mutable and therefore unhashable. Tuples are hashable only if **all** elements are also hashable.

### Q18: Frozenset vs Set

```python
s = {1, 2, 3}
fs = frozenset([1, 2, 3])

print(s == fs)     # True (equal values)
d = {fs: "ok"}    # Works -- frozenset is hashable
d = {s: "ok"}     # TypeError -- set is unhashable
```

**Why:** `frozenset` is the immutable version of `set`. Immutable means hashable, which means it can be used as a dict key or set element.

---

## Category 5: Iterator/Generator Gotchas

### Q19: Exhausted Iterator

```python
nums = [1, 2, 3, 4, 5]
evens = filter(lambda x: x % 2 == 0, nums)

print(list(evens))  # [2, 4]
print(list(evens))  # [] -- empty!
```

**Why:** `filter()` returns an iterator. Iterators can only be consumed **once**. After the first `list(evens)`, the iterator is exhausted.

**Fix:** Convert to list first: `evens = list(filter(...))`

### Q20: Generator Expression Sees Mutations

```python
data = [1, 2, 3]
gen = (x for x in data)
data.append(4)
print(list(gen))  # [1, 2, 3, 4]
```

**Why:** Generator expressions are **lazy**. The generator does not iterate over `data` until consumed. By the time `list(gen)` runs, `data` already has 4 appended.

### Q21: send() to a Generator

```python
def accumulator():
    total = 0
    while True:
        value = yield total
        if value is None:
            break
        total += value

gen = accumulator()
print(next(gen))       # 0 (prime the generator)
print(gen.send(10))    # 10
print(gen.send(20))    # 30
print(gen.send(5))     # 35
```

**Why:** `send(value)` resumes the generator and makes `value` the result of the `yield` expression. You must call `next()` first to "prime" the generator.

### Q22: yield from -- Delegating to Sub-generators

```python
def flatten(nested):
    for item in nested:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item

print(list(flatten([1, [2, 3, [4, 5]], 6])))  # [1, 2, 3, 4, 5, 6]
```

**Why:** `yield from` delegates to another iterator/generator. It yields each item as if this generator yielded it directly. Also properly handles `send()` and `throw()` for bidirectional communication.

---

## Category 6: Class & Inheritance Traps

### Q23: Class Variable vs Instance Variable

```python
class Dog:
    tricks = []  # class variable -- shared!

    def __init__(self, name):
        self.name = name

    def add_trick(self, trick):
        self.tricks.append(trick)

fido = Dog("Fido")
buddy = Dog("Buddy")
fido.add_trick("roll over")
buddy.add_trick("shake")
print(fido.tricks)   # ['roll over', 'shake']
print(buddy.tricks)  # ['roll over', 'shake']  <-- same list!
```

**Why:** `tricks = []` is a class variable -- shared by all instances. Mutating it through any instance affects all.

**Fix:** Initialize in `__init__`: `self.tricks = []`

### Q24: MRO with Diamond Inheritance

```python
class A:
    def method(self): return "A"

class B(A):
    def method(self): return "B"

class C(A):
    def method(self): return "C"

class D(B, C):
    pass

print(D().method())  # "B"
print([c.__name__ for c in D.__mro__])  # ['D', 'B', 'C', 'A', 'object']
```

**Why:** Python uses C3 Linearization. It is **not** depth-first. B comes before C because B is listed first in `class D(B, C)`. Each class appears exactly once.

### Q25: super() with Multiple Inheritance

```python
class Base:
    def __init__(self): print("Base")

class Left(Base):
    def __init__(self): print("Left"); super().__init__()

class Right(Base):
    def __init__(self): print("Right"); super().__init__()

class Child(Left, Right):
    def __init__(self): print("Child"); super().__init__()

Child()
# Output: Child, Left, Right, Base -- each called exactly ONCE
```

**Why:** `super()` follows the MRO, not the direct parent. `super().__init__()` in Left calls `Right.__init__()` (next in MRO), not `Base.__init__()`. This is cooperative multiple inheritance.

### Q26: __slots__ Memory Optimization

```python
class WithSlots:
    __slots__ = ('x', 'y')
    def __init__(self, x, y):
        self.x = x; self.y = y

b = WithSlots(1, 2)
b.z = 3  # AttributeError!
```

**Why:** `__slots__` uses a fixed set of attributes instead of a dynamic `__dict__`. Benefits: ~40-50% less memory, slightly faster access, prevents accidental attributes. Trade-off: no dynamic attributes.

### Q27: __init_subclass__ Hook

```python
class Plugin:
    _registry = {}
    def __init_subclass__(cls, plugin_name=None, **kwargs):
        super().__init_subclass__(**kwargs)
        Plugin._registry[plugin_name or cls.__name__.lower()] = cls

class JSONPlugin(Plugin, plugin_name="json"): pass
class XMLPlugin(Plugin, plugin_name="xml"): pass

print(Plugin._registry)  # {'json': JSONPlugin, 'xml': XMLPlugin}
```

**Why:** `__init_subclass__` is called automatically when a class is subclassed. No metaclass needed. This is the modern way to implement plugin registration. Added in Python 3.6.

---

## Category 7: Decorator Puzzles

### Q28: Decorator Execution Order (Stacking)

```python
def deco_a(func):
    print("A applied")
    def wrapper(*a, **k):
        print("A before"); result = func(*a, **k); print("A after"); return result
    return wrapper

def deco_b(func):
    print("B applied")
    def wrapper(*a, **k):
        print("B before"); result = func(*a, **k); print("B after"); return result
    return wrapper

@deco_a
@deco_b
def hello(): print("Hello!")

# Definition time: "B applied", "A applied"
hello()
# Call time: "A before", "B before", "Hello!", "B after", "A after"
```

**Why:** `@A @B def f` means `f = A(B(f))`. Decorators are **applied bottom-up** but **executed top-down**. Like wrapping a gift: wrap with B first then A; unwrap A first then B.

### Q29: functools.wraps Importance

```python
def bad_deco(func):
    def wrapper(*a, **k): return func(*a, **k)
    return wrapper

def good_deco(func):
    @functools.wraps(func)
    def wrapper(*a, **k): return func(*a, **k)
    return wrapper

@bad_deco
def my_func(): """My docstring."""

print(my_func.__name__)  # "wrapper" -- lost!
print(my_func.__doc__)   # None -- lost!
```

**Why:** Without `@functools.wraps`, the decorated function loses its name, docstring, and metadata. This breaks `help()`, debugging, and serialization. **Always** use `@functools.wraps(func)`.

### Q30: Class as a Decorator

```python
class CountCalls:
    def __init__(self, func):
        self.func = func
        self.count = 0
    def __call__(self, *args, **kwargs):
        self.count += 1
        return self.func(*args, **kwargs)

@CountCalls
def say_hello(name): return f"Hello, {name}!"

say_hello("Alice"); say_hello("Bob")
print(say_hello.count)  # 2
```

**Why:** A class with `__call__` can act as a decorator. The class instance replaces the function. Powerful because class decorators can maintain **state** across calls.

### Q31: Decorator with Arguments

```python
def repeat(n):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return [func(*args, **kwargs) for _ in range(n)]
        return wrapper
    return decorator

@repeat(3)
def greet(name): return f"Hi {name}"

print(greet("World"))  # ['Hi World', 'Hi World', 'Hi World']
```

**Why:** `@repeat(3)` first calls `repeat(3)`, which returns `decorator`. Then decorator is applied. Three-level nesting: `repeat(n) -> decorator(func) -> wrapper(*args)`.

---

## Category 8: String & Number Surprises

### Q32: Floating Point Arithmetic

```python
print(0.1 + 0.2 == 0.3)   # False!
print(0.1 + 0.2)           # 0.30000000000000004
```

**Why:** Floating point numbers cannot represent 0.1 exactly in binary (IEEE 754).

**Fix:** Use `decimal.Decimal` for financial calculations or `math.isclose()` for comparisons.

### Q33: round() -- Banker's Rounding

```python
print(round(0.5))  # 0
print(round(1.5))  # 2
print(round(2.5))  # 2
print(round(3.5))  # 4
```

**Output:** `0, 2, 2, 4` -- NOT `1, 2, 3, 4`

**Why:** Python uses "banker's rounding" (round half to even). When exactly halfway, it rounds to the nearest **even** number. This reduces cumulative rounding bias.

### Q34: bool is a Subclass of int

```python
print(isinstance(True, int))        # True
print(True + True + True)           # 3
print(True * 10)                    # 10
print({True: "yes", 1: "one", 1.0: "float"})  # {True: 'float'}
```

**Why:** `bool` is a subclass of `int`. `True == 1`, `False == 0`. In the dict, `True`, `1`, and `1.0` are all equal with the same hash, so they are the same key. Key stays as `True` (first inserted) but value is overwritten to `'float'` (last set).

### Q35: String Multiplication

```python
print("ha" * 3)     # "hahaha"
print("-" * 0)       # "" (empty)
print("ab" * -1)     # "" (empty)
```

**Why:** String `* n` repeats n times. `* 0` or negative gives an empty string.

---

## Category 9: Async Gotchas

### Q36: Forgetting to await

```python
async def fetch_data():
    return {"status": "ok"}

async def main():
    result = fetch_data()   # Missing await!
    print(type(result))     # <class 'coroutine'>
    print(result)           # <coroutine object ...>

    result2 = await fetch_data()  # Correct
    print(result2)          # {'status': 'ok'}
```

**Why:** Calling an async function without `await` returns a coroutine object. The function body does not execute at all. No error, just a silent coroutine. Very common source of bugs.

### Q37: asyncio.gather Exception Handling

```python
async def task_ok(): return "ok"
async def task_fail(): raise ValueError("boom")

# Without return_exceptions -- raises first exception
await asyncio.gather(task_ok(), task_fail())  # ValueError!

# With return_exceptions=True -- exceptions become values
results = await asyncio.gather(task_ok(), task_fail(), return_exceptions=True)
# results = ["ok", ValueError("boom")]
```

**Why:** By default, `gather()` raises the first exception. With `return_exceptions=True`, exceptions are returned as values. Use this for partial failure handling in production.

### Q38: Blocking Call in Async Code

```python
async def bad(): time.sleep(1)      # BLOCKS entire event loop!
async def good(): await asyncio.sleep(1)  # Non-blocking
async def better():                  # Wrap blocking code
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, time.sleep, 1)
```

**Rule:** Never use blocking I/O in async functions. Use async libraries (aiohttp, asyncpg) or `run_in_executor()` for unavoidable blocking calls.

---

## Category 10: Pythonic Traps

### Q39: Chained Comparisons

```python
print(1 < 2 < 3)         # True (means 1<2 AND 2<3)
print(1 < 2 > 0)         # True (means 1<2 AND 2>0)
print(1 == 1 in [1])     # True (means 1==1 AND 1 in [1])
```

**Why:** Python chains comparisons. `in` and `is` are comparison operators and participate in chaining. `False == False in [False]` is `True`.

### Q40: else on for/while/try

```python
for i in range(5):
    if i == 3: break
else:
    print("no break")   # Skipped -- loop broke

for i in range(5):
    pass
else:
    print("no break")   # Printed -- loop completed

try:
    result = 10 / 2
except ZeroDivisionError:
    print("error")
else:
    print("success")    # Printed -- no exception
```

**Why:**
- `for/else`: else runs only if the loop completes **without** a `break`. Think "for/nobreak".
- `try/else`: else runs only if **no** exception was raised.

### Q41: Walrus Operator Edge Cases

```python
if (n := len("hello")) > 3:
    print(f"Long: {n}")    # Long: 5

results = [y for x in range(10) if (y := x**2) > 20]
print(results)  # [25, 36, 49, 64, 81]
```

**Why:** `:=` assigns and returns a value in one expression. In comprehensions, compute once and use for both filter and result. Parentheses are required in most contexts.

### Q42: Unpacking Tricks

```python
first, *middle, last = [1, 2, 3, 4, 5]
# first=1, middle=[2,3,4], last=5

x, y = 1, 2
x, y = y, x  # Swap! Works because right side is evaluated first

(a, b), (c, d) = [1, 2], [3, 4]  # Nested unpacking
```

**Why:** `*middle` captures everything between first and last. Swap works because Python evaluates the right side fully (creates a tuple) before any assignment.

### Q43: Positional-Only and Keyword-Only Parameters

```python
def func(pos_only, /, normal, *, kw_only):
    pass

func(1, 2, kw_only=3)              # OK
func(1, normal=2, kw_only=3)       # OK
func(pos_only=1, normal=2, kw_only=3)  # TypeError!
func(1, 2, 3)                          # TypeError!
```

**Why:**
- `/` marks everything before it as positional-only
- `*` marks everything after it as keyword-only
- Real-world: `len(obj, /)` -- you cannot do `len(obj=mylist)`

### Q44: Dictionary Merge (3.9+)

```python
d1 = {"a": 1, "b": 2}
d2 = {"b": 3, "c": 4}
print(d1 | d2)   # {'a': 1, 'b': 3, 'c': 4} -- d2 wins
d1 |= d2         # Update in place
```

### Q45: Multiple Assignment Gotcha

```python
a = b = []
a.append(1)
print(a, b)  # [1] [1] -- same object!

x = []
y = []
x.append(1)
print(x, y)  # [1] [] -- independent
```

**Why:** `a = b = []` creates one list with two names pointing to it. They are aliases for the same object.

---

## Bonus Questions

### Q46: all() and any() with Empty Iterables

```python
print(all([]))           # True (vacuous truth)
print(any([]))           # False
print(all([0, 1, 2]))   # False (0 is falsy)
print(any([0, 0, 0]))   # False
```

**Why:** `all([])` is True by vacuous truth -- no elements to be False. `any([])` is False -- no elements to be True.

### Q47: Exception Chaining

```python
try:
    try:
        1 / 0
    except ZeroDivisionError:
        raise ValueError("bad value")
except ValueError as e:
    print(e.__context__)  # division by zero
```

**Why:** When raising inside an except block, Python sets `__context__` to the original exception. Use `raise X from Y` for explicit chaining (sets `__cause__`).

### Q48: 'is not' vs 'not ... is'

```python
x = [1, 2, 3]
y = [1, 2, 3]
print(x is not y)   # True
print(not x is y)   # True -- same result
```

**Why:** `is not` is a single operator. `not x is y` is parsed as `not (x is y)`. Both produce the same bytecode. PEP 8 recommends `is not` for readability.

---

## Key Takeaways

| Category | Rule |
|----------|------|
| Mutable defaults | Use `None` as default, create mutable objects inside the function |
| Closures | Capture values, not variables (use default args or factories) |
| Identity | Use `is` only for `None`/`True`/`False`; use `==` for values |
| Mutability | Tuples hold references; shallow copy shares nested objects |
| Iterators | Can only be consumed once; generators are lazy |
| Classes | Class variables are shared; `super()` follows MRO, not parent |
| Decorators | Applied bottom-up, executed top-down; always use `@wraps` |
| Numbers | Float math is imprecise; `round()` uses banker's rounding |
| Async | Always await coroutines; never block the event loop |
| Pythonic | Chained comparisons include `in`/`is`; for/else means for/nobreak |
