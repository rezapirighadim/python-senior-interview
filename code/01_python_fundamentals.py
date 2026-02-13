"""
=============================================================================
FILE 01: PYTHON FUNDAMENTALS — The Foundation
=============================================================================
Everything you need to remember about Python basics.
Run this file: python 01_python_fundamentals.py
=============================================================================
"""

# =============================================================================
# 1. VARIABLES & DATA TYPES
# =============================================================================
# Python is dynamically typed — variables don't need type declarations.
# But senior devs should KNOW the types and their behavior.

# Immutable types: int, float, str, tuple, frozenset, bytes
# Mutable types:   list, dict, set, bytearray

name: str = "Reza"          # str — immutable sequence of characters
age: int = 30               # int — arbitrary precision (no overflow!)
salary: float = 150_000.50  # float — use underscores for readability
is_active: bool = True      # bool — subclass of int (True == 1, False == 0)
nothing: None = None         # NoneType — singleton, use `is None` not `== None`


# =============================================================================
# 2. STRINGS — Interview Favorite
# =============================================================================
s = "Hello, Python"

# f-strings (Python 3.6+) — ALWAYS use these, not .format() or %
greeting = f"My name is {name} and I'm {age} years old"

# Common string methods you MUST know:
s.upper()          # "HELLO, PYTHON"
s.lower()          # "hello, python"
s.strip()          # Remove whitespace from both ends
s.split(", ")      # ["Hello", "Python"]
s.replace("H", "J")  # "Jello, Python"
s.startswith("He")    # True
s.find("Py")          # 7 (index) or -1 if not found
"".join(["a", "b"])   # "ab"

# IMPORTANT: Strings are IMMUTABLE — every operation creates a new string
# For building strings in a loop, use a list and join:
parts = []
for i in range(5):
    parts.append(str(i))
result = "".join(parts)  # "01234"  — O(n) instead of O(n²)


# =============================================================================
# 3. LISTS, TUPLES, SETS, DICTS
# =============================================================================

# --- LIST: ordered, mutable, allows duplicates ---
fruits = ["apple", "banana", "cherry"]
fruits.append("date")       # Add to end — O(1)
fruits.insert(0, "avocado") # Insert at index — O(n)
fruits.pop()                # Remove last — O(1)
fruits.pop(0)               # Remove first — O(n)
fruits.sort()               # In-place sort — O(n log n)
sorted_fruits = sorted(fruits)  # Returns new sorted list

# List comprehension — MUST KNOW for interviews
squares = [x**2 for x in range(10)]  # [0, 1, 4, 9, 16, ...]
evens = [x for x in range(20) if x % 2 == 0]

# Nested comprehension
matrix = [[1, 2], [3, 4], [5, 6]]
flat = [num for row in matrix for num in row]  # [1, 2, 3, 4, 5, 6]


# --- TUPLE: ordered, IMMUTABLE, allows duplicates ---
point = (3, 4)
x, y = point  # Tuple unpacking — very Pythonic

# Use tuples when data shouldn't change (coordinates, DB records, dict keys)
# Tuples are faster than lists and use less memory


# --- SET: unordered, mutable, NO duplicates ---
unique = {1, 2, 3, 3, 3}  # {1, 2, 3}
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}
a & b   # Intersection: {3, 4}
a | b   # Union: {1, 2, 3, 4, 5, 6}
a - b   # Difference: {1, 2}
a ^ b   # Symmetric difference: {1, 2, 5, 6}

# Set lookups are O(1) — use for membership testing instead of lists!
big_list = list(range(1_000_000))
big_set = set(big_list)
# `999_999 in big_set` is O(1), `999_999 in big_list` is O(n)


# --- DICT: key-value pairs, ordered (Python 3.7+), mutable ---
person = {"name": "Reza", "age": 30, "city": "Toronto"}
person["email"] = "reza@example.com"  # Add/update
person.get("phone", "N/A")           # Safe access with default
person.pop("city")                    # Remove and return

# Dict comprehension
squares_dict = {x: x**2 for x in range(5)}  # {0:0, 1:1, 2:4, 3:9, 4:16}

# Merge dicts (Python 3.9+)
d1 = {"a": 1}
d2 = {"b": 2}
merged = d1 | d2  # {"a": 1, "b": 2}

# defaultdict — avoids KeyError
from collections import defaultdict
word_count = defaultdict(int)
for word in "hello world hello".split():
    word_count[word] += 1  # No need to check if key exists

# Counter — frequency counting
from collections import Counter
counts = Counter("abracadabra")  # Counter({'a': 5, 'b': 2, 'r': 2, ...})
counts.most_common(2)            # [('a', 5), ('b', 2)]


# =============================================================================
# 4. CONTROL FLOW
# =============================================================================

# Ternary expression
status = "adult" if age >= 18 else "minor"

# Walrus operator := (Python 3.8+) — assign inside expressions
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
if (n := len(data)) > 5:
    print(f"List has {n} elements")

# Match statement (Python 3.10+) — structural pattern matching
def handle_command(command: str) -> str:
    match command.split():
        case ["quit"]:
            return "Quitting..."
        case ["hello", name]:
            return f"Hello, {name}!"
        case ["add", *numbers]:
            return str(sum(int(n) for n in numbers))
        case _:
            return "Unknown command"

# For-else — the `else` runs if loop completes WITHOUT `break`
def find_prime(numbers: list[int]) -> int | None:
    for n in numbers:
        if n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1)):
            return n
    return None


# =============================================================================
# 5. FUNCTIONS
# =============================================================================

# *args and **kwargs
def flexible(*args, **kwargs):
    """Accept any number of positional and keyword arguments."""
    print(f"args: {args}")      # tuple
    print(f"kwargs: {kwargs}")  # dict

flexible(1, 2, 3, name="Reza", age=30)

# Keyword-only arguments (after *)
def create_user(name: str, *, email: str, role: str = "user"):
    """email and role MUST be passed as keyword arguments."""
    return {"name": name, "email": email, "role": role}

# Positional-only arguments (before /) — Python 3.8+
def power(base, exp, /):
    """base and exp can ONLY be passed positionally."""
    return base ** exp

# Lambda — anonymous function (keep them simple!)
double = lambda x: x * 2
sorted_names = sorted(["Charlie", "Alice", "Bob"], key=lambda s: s.lower())


# =============================================================================
# 6. UNPACKING & STARRED EXPRESSIONS
# =============================================================================
first, *middle, last = [1, 2, 3, 4, 5]  # first=1, middle=[2,3,4], last=5

# Swap without temp variable
a, b = 1, 2
a, b = b, a  # a=2, b=1

# Unpack into function call
def add(x, y, z):
    return x + y + z

numbers = [1, 2, 3]
add(*numbers)  # Same as add(1, 2, 3)


# =============================================================================
# 7. EXCEPTION HANDLING
# =============================================================================
def divide(a: float, b: float) -> float:
    """Demonstrates proper exception handling."""
    try:
        result = a / b
    except ZeroDivisionError:
        print("Cannot divide by zero!")
        raise  # Re-raise — don't silently swallow exceptions!
    except TypeError as e:
        print(f"Type error: {e}")
        raise
    else:
        # Runs ONLY if no exception was raised
        print(f"Result: {result}")
        return result
    finally:
        # ALWAYS runs — cleanup code goes here
        print("Division attempted")

# Custom exceptions — always inherit from Exception, not BaseException
class InsufficientFundsError(Exception):
    def __init__(self, balance: float, amount: float):
        self.balance = balance
        self.amount = amount
        super().__init__(
            f"Cannot withdraw ${amount:.2f}. Balance: ${balance:.2f}"
        )


# =============================================================================
# 8. FILE I/O — Always Use Context Managers
# =============================================================================
# GOOD — file is auto-closed even if exception occurs
# with open("data.txt", "r") as f:
#     content = f.read()

# NEVER do this:
# f = open("data.txt")
# content = f.read()
# f.close()  # What if an exception happens before this line?

# Reading large files — DON'T read entire file into memory
# with open("huge_file.txt") as f:
#     for line in f:  # Lazy iteration — one line at a time
#         process(line)


# =============================================================================
# 9. IMPORTANT GOTCHAS — Interview Trick Questions
# =============================================================================

# --- Mutable default arguments ---
# WRONG:
def append_to(item, target=[]):
    target.append(item)
    return target
# append_to(1) → [1]
# append_to(2) → [1, 2]  ← BUG! Default list is shared across calls!

# CORRECT:
def append_to_fixed(item, target=None):
    if target is None:
        target = []
    target.append(item)
    return target


# --- `is` vs `==` ---
a = [1, 2, 3]
b = [1, 2, 3]
print(a == b)   # True — same VALUE
print(a is b)   # False — different OBJECTS
# Use `is` only for None, True, False: `if x is None`


# --- Integer caching ---
a = 256
b = 256
print(a is b)   # True — Python caches integers -5 to 256
a = 257
b = 257
print(a is b)   # May be False — outside cache range


# --- Shallow vs Deep Copy ---
import copy
original = [[1, 2], [3, 4]]
shallow = original.copy()       # or list(original)
deep = copy.deepcopy(original)

original[0][0] = 99
print(shallow[0][0])  # 99 — shallow copy shares inner objects!
print(deep[0][0])     # 1  — deep copy is independent


# =============================================================================
# 10. ENUMERATE, ZIP, MAP, FILTER
# =============================================================================

# enumerate — get index and value (don't use range(len(...)))
names = ["Alice", "Bob", "Charlie"]
for i, name in enumerate(names, start=1):
    print(f"{i}. {name}")

# zip — iterate multiple sequences together
ages = [25, 30, 35]
for name, age in zip(names, ages):
    print(f"{name} is {age}")

# zip with strict=True (Python 3.10+) — raises if lengths differ
# for name, age in zip(names, ages, strict=True): ...

# map and filter (prefer list comprehensions in most cases)
doubled = list(map(lambda x: x * 2, [1, 2, 3]))
adults = list(filter(lambda x: x >= 18, [10, 20, 15, 30]))


# =============================================================================
# PRACTICE EXERCISE
# =============================================================================
def practice():
    """
    Exercise: Write a function that takes a list of strings and returns
    a dictionary where keys are the first letters and values are lists
    of words starting with that letter, sorted alphabetically.

    Example: ["apple", "banana", "avocado", "blueberry", "cherry"]
    Returns: {"a": ["apple", "avocado"], "b": ["banana", "blueberry"], "c": ["cherry"]}
    """
    words = ["apple", "banana", "avocado", "blueberry", "cherry", "apricot"]

    # Solution using defaultdict
    grouped = defaultdict(list)
    for word in words:
        grouped[word[0]].append(word)

    # Sort each group
    result = {k: sorted(v) for k, v in sorted(grouped.items())}
    print(result)
    # {'a': ['apple', 'apricot', 'avocado'], 'b': ['banana', 'blueberry'], 'c': ['cherry']}


if __name__ == "__main__":
    print("=" * 60)
    print("FILE 01: Python Fundamentals")
    print("=" * 60)

    print("\n--- String formatting ---")
    print(greeting)

    print("\n--- Match statement ---")
    print(handle_command("hello World"))
    print(handle_command("add 1 2 3"))

    print("\n--- Unpacking ---")
    first, *middle, last = [1, 2, 3, 4, 5]
    print(f"first={first}, middle={middle}, last={last}")

    print("\n--- Practice exercise ---")
    practice()

    print("\n✓ File 01 complete. Move to 02_advanced_python.py")
