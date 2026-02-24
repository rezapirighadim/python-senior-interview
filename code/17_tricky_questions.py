"""
=============================================================================
FILE 17: TRICKY PYTHON QUESTIONS — What Trips Up Even Senior Developers
=============================================================================
40+ questions that expose subtle Python behaviors. Each question shows a
code snippet, asks "What's the output?", reveals the answer, and explains
WHY it behaves that way. Run this file to see all questions in action.
=============================================================================
"""
import sys
import copy
import asyncio
import functools
from textwrap import dedent


question_counter = 0


def question(title: str):
    """Decorator that formats each question with numbering and section headers."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper():
            global question_counter
            question_counter += 1
            print(f"\n{'='*70}")
            print(f"  Q{question_counter}: {title}")
            print(f"{'='*70}")
            func()
        return wrapper
    return decorator


# =============================================================================
# CATEGORY 1: MUTABLE DEFAULT ARGUMENTS
# =============================================================================
print("\n" + "#"*70)
print("#  CATEGORY 1: MUTABLE DEFAULT ARGUMENTS")
print("#"*70)


@question("The Classic Mutable Default Argument Trap")
def q_mutable_default_basic():
    print(dedent("""
        Code:
            def append_to(element, target=[]):
                target.append(element)
                return target

            print(append_to(1))
            print(append_to(2))
            print(append_to(3))

        What's the output?
    """))

    def append_to(element, target=[]):
        target.append(element)
        return target

    result1 = append_to(1)
    result2 = append_to(2)
    result3 = append_to(3)

    print(f"  Actual output:")
    print(f"    {result1}")       # Same object!
    print(f"    {result2}")       # Same object!
    print(f"    {result3}")       # Same object!
    # Note: all three variables point to the SAME list

    print(dedent("""
        Surprise: All three return the SAME list: [1, 2, 3]

        WHY: Default arguments are evaluated ONCE at function definition time,
        not each call. The list `[]` is created once and shared across all
        calls. Each call mutates the same object.

        FIX:
            def append_to(element, target=None):
                if target is None:
                    target = []
                target.append(element)
                return target
    """))


@question("Mutable Default with Dict")
def q_mutable_default_dict():
    print(dedent("""
        Code:
            def add_student(name, roster={}):
                roster[name] = True
                return roster

            print(add_student("Alice"))
            print(add_student("Bob"))

        What's the output?
    """))

    def add_student(name, roster={}):
        roster[name] = True
        return roster

    r1 = add_student("Alice")
    r2 = add_student("Bob")

    print(f"  Actual output:")
    print(f"    {r1}")
    print(f"    {r2}")

    print(dedent("""
        Surprise: Both print {'Alice': True, 'Bob': True}

        WHY: Same principle as lists. The dict `{}` is created once at
        function definition time. Every call shares and mutates the same dict.
        Both r1 and r2 reference the same dict object.

        FIX: Use `roster=None` and create a new dict inside the function.
    """))


@question("Inspecting the Default Object")
def q_mutable_default_inspect():
    print(dedent("""
        Code:
            def buggy(items=[]):
                items.append("x")
                return items

            buggy()
            buggy()
            print(buggy.__defaults__)

        What's the output?
    """))

    def buggy(items=[]):
        items.append("x")
        return items

    buggy()
    buggy()
    result = buggy.__defaults__

    print(f"  Actual output:")
    print(f"    {result}")

    print(dedent("""
        Output: (['x', 'x'],)

        WHY: `func.__defaults__` stores the actual default argument objects.
        Since the list is mutated on every call, you can see the accumulated
        state by inspecting __defaults__. This is also how you can detect
        this bug: check if __defaults__ contains mutable objects.
    """))


@question("Immutable Defaults Are Fine")
def q_mutable_default_immutable():
    print(dedent("""
        Code:
            def greet(name, greeting="Hello"):
                return f"{greeting}, {name}!"

            print(greet("Alice"))
            print(greet("Bob"))
            print(greet("Charlie", "Hey"))
            print(greet("Dave"))

        What's the output?
    """))

    def greet(name, greeting="Hello"):
        return f"{greeting}, {name}!"

    print(f"  Actual output:")
    print(f"    {greet('Alice')}")
    print(f"    {greet('Bob')}")
    print(f"    {greet('Charlie', 'Hey')}")
    print(f"    {greet('Dave')}")

    print(dedent("""
        Output: Hello Alice!, Hello Bob!, Hey Charlie!, Hello Dave!

        WHY: Strings (and ints, tuples, frozensets) are immutable. Even
        though the default is evaluated once, you can't mutate an immutable
        object, so this is perfectly safe. The mutable default trap ONLY
        applies to mutable types: list, dict, set, custom objects.
    """))


q_mutable_default_basic()
q_mutable_default_dict()
q_mutable_default_inspect()
q_mutable_default_immutable()


# =============================================================================
# CATEGORY 2: SCOPE & CLOSURES
# =============================================================================
print("\n" + "#"*70)
print("#  CATEGORY 2: SCOPE & CLOSURES")
print("#"*70)


@question("Late Binding Closures in a Loop")
def q_closure_late_binding():
    print(dedent("""
        Code:
            funcs = []
            for i in range(4):
                funcs.append(lambda: i)

            print([f() for f in funcs])

        What's the output?
    """))

    funcs = []
    for i in range(4):
        funcs.append(lambda: i)

    result = [f() for f in funcs]
    print(f"  Actual output: {result}")

    print(dedent("""
        Surprise: [3, 3, 3, 3] -- NOT [0, 1, 2, 3]

        WHY: Closures capture VARIABLES, not VALUES. All lambdas reference
        the same variable `i`. By the time they're called, the loop has
        finished and `i` is 3.

        FIX: Use a default argument to capture the value at creation time:
            funcs.append(lambda i=i: i)

        Or use functools.partial:
            from functools import partial
            funcs.append(partial(lambda x: x, i))
    """))


@question("LEGB Rule — Local, Enclosing, Global, Built-in")
def q_scope_legb():
    print(dedent("""
        Code:
            x = "global"

            def outer():
                x = "enclosing"
                def inner():
                    x = "local"
                    print(x)
                inner()
                print(x)

            outer()
            print(x)

        What's the output?
    """))

    x = "global"

    def outer():
        x = "enclosing"
        def inner():
            x = "local"
            print(f"    {x}")
        inner()
        print(f"    {x}")

    outer()
    print(f"    {x}")

    print(dedent("""
        Output: local, enclosing, global

        WHY: Python follows LEGB — Local, Enclosing, Global, Built-in.
        Each `x = ...` creates a NEW local variable in that scope. They
        don't modify the outer `x`. Assignment creates a local variable.
    """))


@question("nonlocal vs global")
def q_scope_nonlocal():
    print(dedent("""
        Code:
            count = 0

            def outer():
                count = 10
                def inner():
                    nonlocal count
                    count += 1
                inner()
                print("outer count:", count)

            outer()
            print("global count:", count)

        What's the output?
    """))

    count = 0

    def outer():
        count = 10
        def inner():
            nonlocal count
            count += 1
        inner()
        print(f"    outer count: {count}")

    outer()
    print(f"    global count: {count}")

    print(dedent("""
        Output: outer count: 11, global count: 0

        WHY: `nonlocal` binds to the nearest enclosing scope (outer's count),
        not the global. So inner() modifies outer's count (10 -> 11), but the
        global count stays 0. If you wanted to modify the global, use `global count`.
    """))


@question("UnboundLocalError Surprise")
def q_scope_unbound():
    print(dedent("""
        Code:
            x = 10

            def foo():
                print(x)
                x = 20

            foo()

        What's the output?
    """))

    print(f"  Actual output:")
    try:
        x = 10

        def foo():
            print(x)   # noqa
            x = 20      # noqa

        foo()
    except UnboundLocalError as e:
        print(f"    UnboundLocalError: {e}")

    print(dedent("""
        Surprise: UnboundLocalError, NOT 10!

        WHY: Python determines variable scope at COMPILE TIME, not runtime.
        Because `x = 20` exists anywhere in the function, `x` is classified
        as LOCAL for the entire function. When `print(x)` executes, the
        local `x` hasn't been assigned yet -> UnboundLocalError.

        FIX: Use `global x` at the top of the function, or restructure
        your code to avoid shadowing.
    """))


@question("Closure Over Loop Variable — The Fix")
def q_closure_loop_fix():
    print(dedent("""
        Code:
            # Method 1: Default argument capture
            funcs1 = [lambda i=i: i for i in range(4)]

            # Method 2: Using a factory function
            def make_func(val):
                return lambda: val
            funcs2 = [make_func(i) for i in range(4)]

            print([f() for f in funcs1])
            print([f() for f in funcs2])

        What's the output?
    """))

    funcs1 = [lambda i=i: i for i in range(4)]

    def make_func(val):
        return lambda: val
    funcs2 = [make_func(i) for i in range(4)]

    print(f"  Actual output:")
    print(f"    {[f() for f in funcs1]}")
    print(f"    {[f() for f in funcs2]}")

    print(dedent("""
        Output: [0, 1, 2, 3] and [0, 1, 2, 3]

        WHY: Both methods capture the VALUE at iteration time instead of
        the variable. The default argument trick (`i=i`) evaluates `i`
        at lambda creation time. The factory function creates a new scope
        for each value. The factory approach is generally considered cleaner.
    """))


q_closure_late_binding()
q_scope_legb()
q_scope_nonlocal()
q_scope_unbound()
q_closure_loop_fix()


# =============================================================================
# CATEGORY 3: IDENTITY vs EQUALITY
# =============================================================================
print("\n" + "#"*70)
print("#  CATEGORY 3: IDENTITY vs EQUALITY")
print("#"*70)


@question("Integer Caching (-5 to 256)")
def q_identity_int_cache():
    print(dedent("""
        Code:
            a = 256
            b = 256
            print(a is b)

            c = 257
            d = 257
            print(c is d)

        What's the output?
    """))

    a = 256
    b = 256
    print(f"  Actual output:")
    print(f"    a is b: {a is b}")

    # Note: In CPython, small integers are cached.
    # For 257, behavior may vary by context (script vs REPL vs optimizer).
    # In modern CPython with the peephole optimizer, constants in the same
    # compilation unit may be folded, so we force separate creation:
    c = int("257")
    d = int("257")
    print(f"    c is d (int('257')): {c is d}")

    print(dedent("""
        Output: True, then False (typically)

        WHY: CPython caches integers from -5 to 256 as singleton objects.
        So `256 is 256` is True (same object). But 257 creates a new int
        object each time, so `is` is False.

        CAVEAT: The CPython compiler may intern constants in the same
        code block, making `257 is 257` True in a script. NEVER rely
        on this. Always use `==` for value comparison.
    """))


@question("String Interning")
def q_identity_string_intern():
    print(dedent("""
        Code:
            a = "hello"
            b = "hello"
            print(a is b)

            c = "hello world"
            d = "hello world"
            print(c is d)

            e = "hello_world"
            f = "hello_world"
            print(e is f)

        What's the output?
    """))

    a = "hello"
    b = "hello"
    print(f"  Actual output:")
    print(f"    'hello' is 'hello': {a is b}")

    # Force runtime creation to avoid compiler constant folding
    c = " ".join(["hello", "world"])
    d = " ".join(["hello", "world"])
    print(f"    runtime 'hello world' is 'hello world': {c is d}")

    e = "hello_world"
    f = "hello_world"
    print(f"    'hello_world' is 'hello_world': {e is f}")

    print(dedent("""
        Output: True, False (runtime), True

        WHY: Python interns strings that look like identifiers (letters,
        digits, underscores). "hello" and "hello_world" are interned.
        "hello world" (with a space) may not be interned at runtime.
        Compile-time constants may still be folded by the optimizer.

        RULE: NEVER use `is` for string comparison. Always use `==`.
    """))


@question("None Comparison — is vs ==")
def q_identity_none():
    print(dedent("""
        Code:
            class Sneaky:
                def __eq__(self, other):
                    return True

            s = Sneaky()
            print(s == None)
            print(s is None)
            print(None is None)

        What's the output?
    """))

    class Sneaky:
        def __eq__(self, other):
            return True

    s = Sneaky()
    print(f"  Actual output:")
    print(f"    s == None: {s == None}")    # noqa: E711
    print(f"    s is None: {s is None}")
    print(f"    None is None: {None is None}")

    print(dedent("""
        Output: True, False, True

        WHY: `==` calls __eq__ which can be overridden to return anything.
        `is` checks identity — is it literally the None singleton?
        None is a singleton in Python, so `is None` is always correct.

        RULE: PEP 8 says: "Comparisons to singletons like None should
        always be done with `is` or `is not`, never the equality operators."
    """))


@question("is vs == with Lists and Tuples")
def q_identity_containers():
    print(dedent("""
        Code:
            a = [1, 2, 3]
            b = [1, 2, 3]
            print(a == b)
            print(a is b)

            c = (1, 2, 3)
            d = (1, 2, 3)
            print(c == d)
            print(c is d)

        What's the output?
    """))

    a = [1, 2, 3]
    b = [1, 2, 3]
    print(f"  Actual output:")
    print(f"    list == list: {a == b}")
    print(f"    list is list: {a is b}")

    # Force tuple creation at runtime
    c = tuple([1, 2, 3])
    d = tuple([1, 2, 3])
    print(f"    tuple == tuple: {c == d}")
    print(f"    tuple is tuple: {c is d}")

    print(dedent("""
        Output: True, False, True, False (typically)

        WHY: `==` compares values (element by element). `is` checks if they
        are the same object in memory. Two independently created lists or
        tuples have equal values but are different objects.

        NOTE: CPython may cache small tuples, but never rely on this.
    """))


q_identity_int_cache()
q_identity_string_intern()
q_identity_none()
q_identity_containers()


# =============================================================================
# CATEGORY 4: MUTABILITY TRICKS
# =============================================================================
print("\n" + "#"*70)
print("#  CATEGORY 4: MUTABILITY TRICKS")
print("#"*70)


@question("Tuple with Mutable Elements")
def q_mutability_tuple_list():
    print(dedent("""
        Code:
            t = (1, 2, [3, 4])
            t[2].append(5)
            print(t)

        What's the output?
    """))

    t = (1, 2, [3, 4])
    t[2].append(5)
    print(f"  Actual output: {t}")

    print(dedent("""
        Output: (1, 2, [3, 4, 5])

        WHY: Tuples are immutable, but they store REFERENCES. The tuple
        can't change which objects it references, but the objects themselves
        can be mutable. The list inside the tuple can still be modified.
        You just can't do t[2] = something_else.
    """))


@question("Tuple += with Mutable Element")
def q_mutability_tuple_iadd():
    print(dedent("""
        Code:
            t = (1, 2, [3, 4])
            try:
                t[2] += [5, 6]
            except TypeError as e:
                print(f"Error: {e}")
            print(t)

        What's the output?
    """))

    t = (1, 2, [3, 4])
    try:
        t[2] += [5, 6]
    except TypeError as e:
        print(f"  Error: {e}")
    print(f"  Actual output: {t}")

    print(dedent("""
        Surprise: BOTH an error AND the list is modified!
        Output: Error: ..., then (1, 2, [3, 4, 5, 6])

        WHY: `t[2] += [5, 6]` does TWO things:
        1. Calls t[2].__iadd__([5, 6]) — this SUCCEEDS (mutates the list)
        2. Assigns the result back: t[2] = result — this FAILS (tuple immutable)

        The list is mutated in step 1 before the TypeError in step 2.
        This is one of Python's most notorious gotchas.
    """))


@question("Shallow vs Deep Copy")
def q_mutability_copy():
    print(dedent("""
        Code:
            import copy
            original = [[1, 2], [3, 4]]

            shallow = copy.copy(original)
            deep = copy.deepcopy(original)

            original[0].append(99)

            print("original:", original)
            print("shallow:", shallow)
            print("deep:", deep)

        What's the output?
    """))

    original = [[1, 2], [3, 4]]
    shallow = copy.copy(original)
    deep = copy.deepcopy(original)

    original[0].append(99)

    print(f"  Actual output:")
    print(f"    original: {original}")
    print(f"    shallow:  {shallow}")
    print(f"    deep:     {deep}")

    print(dedent("""
        Output:
            original: [[1, 2, 99], [3, 4]]
            shallow:  [[1, 2, 99], [3, 4]]  <- also changed!
            deep:     [[1, 2], [3, 4]]       <- independent

        WHY: Shallow copy creates a new outer list but the inner lists
        are still references to the same objects. Deep copy recursively
        copies everything, creating fully independent objects.
    """))


@question("Dict Key Requirements")
def q_mutability_dict_keys():
    print(dedent("""
        Code:
            # Can a list be a dict key?
            try:
                d = {[1, 2]: "value"}
            except TypeError as e:
                print(f"List as key: {e}")

            # Can a tuple be a dict key?
            d = {(1, 2): "value"}
            print(f"Tuple as key: {d}")

            # Can a tuple containing a list be a dict key?
            try:
                d = {(1, [2, 3]): "value"}
            except TypeError as e:
                print(f"Tuple with list: {e}")

        What's the output?
    """))

    try:
        d = {[1, 2]: "value"}
    except TypeError as e:
        print(f"    List as key: {e}")

    d = {(1, 2): "value"}
    print(f"    Tuple as key: {d}")

    try:
        d = {(1, [2, 3]): "value"}
    except TypeError as e:
        print(f"    Tuple with list: {e}")

    print(dedent("""
        Output:
            List as key: unhashable type: 'list'
            Tuple as key: {(1, 2): 'value'}
            Tuple with list: unhashable type: 'list'

        WHY: Dict keys must be hashable. Lists are mutable -> unhashable.
        Tuples are immutable -> hashable, BUT only if all elements are
        also hashable. A tuple containing a list is unhashable because
        the list inside it is unhashable.
    """))


@question("Frozenset vs Set")
def q_mutability_frozenset():
    print(dedent("""
        Code:
            s = {1, 2, 3}
            fs = frozenset([1, 2, 3])

            print(type(s), type(fs))
            print(s == fs)

            # Can we use them as dict keys?
            d = {}
            try:
                d[s] = "set"
            except TypeError as e:
                print(f"Set as key: {e}")
            d[fs] = "frozenset"
            print(d)

        What's the output?
    """))

    s = {1, 2, 3}
    fs = frozenset([1, 2, 3])

    print(f"    types: {type(s).__name__}, {type(fs).__name__}")
    print(f"    s == fs: {s == fs}")

    d = {}
    try:
        d[s] = "set"
    except TypeError as e:
        print(f"    Set as key: {e}")
    d[fs] = "frozenset"
    print(f"    dict: {d}")

    print(dedent("""
        Output:
            set and frozenset are equal (==) but not the same type
            set is unhashable (can't be dict key)
            frozenset is hashable (can be dict key)

        WHY: frozenset is the immutable version of set. Immutable ->
        hashable -> can be used as dict key or set element.
    """))


q_mutability_tuple_list()
q_mutability_tuple_iadd()
q_mutability_copy()
q_mutability_dict_keys()
q_mutability_frozenset()


# =============================================================================
# CATEGORY 5: ITERATOR / GENERATOR GOTCHAS
# =============================================================================
print("\n" + "#"*70)
print("#  CATEGORY 5: ITERATOR / GENERATOR GOTCHAS")
print("#"*70)


@question("Exhausted Iterator")
def q_iter_exhausted():
    print(dedent("""
        Code:
            nums = [1, 2, 3, 4, 5]
            evens = filter(lambda x: x % 2 == 0, nums)

            print(list(evens))
            print(list(evens))

        What's the output?
    """))

    nums = [1, 2, 3, 4, 5]
    evens = filter(lambda x: x % 2 == 0, nums)

    print(f"  Actual output:")
    print(f"    First:  {list(evens)}")
    print(f"    Second: {list(evens)}")

    print(dedent("""
        Surprise: [2, 4] then [] (empty!)

        WHY: filter() returns an iterator, not a list. Iterators can only
        be consumed ONCE. After the first list(evens), the iterator is
        exhausted. The second call gets nothing.

        FIX: Convert to list first: evens = list(filter(...))
        Or recreate the iterator when needed.
    """))


@question("Generator Expression vs List Comprehension as Argument")
def q_iter_genexp_arg():
    print(dedent("""
        Code:
            # List comprehension — creates entire list in memory
            sum_list = sum([x**2 for x in range(1000000)])

            # Generator expression — lazy, memory efficient
            sum_gen = sum(x**2 for x in range(1000000))

            print(sum_list == sum_gen)

            # But watch out for this:
            data = [1, 2, 3]
            gen = (x for x in data)
            data.append(4)
            print(list(gen))

        What's the output?
    """))

    data = [1, 2, 3]
    gen = (x for x in data)
    data.append(4)
    result = list(gen)

    print(f"  Actual output:")
    print(f"    sum_list == sum_gen: True")
    print(f"    list(gen) after append: {result}")

    print(dedent("""
        Output: True, then [1, 2, 3, 4]

        WHY: Generator expressions are LAZY. The generator doesn't iterate
        over `data` until you consume it. By the time list(gen) runs,
        data already has 4 appended. The generator sees the modified list.

        This is a subtle source of bugs. If you need a snapshot of the data,
        use a list comprehension instead.
    """))


@question("send() to a Generator")
def q_iter_send():
    print(dedent("""
        Code:
            def accumulator():
                total = 0
                while True:
                    value = yield total
                    if value is None:
                        break
                    total += value

            gen = accumulator()
            print(next(gen))       # Prime the generator
            print(gen.send(10))
            print(gen.send(20))
            print(gen.send(5))

        What's the output?
    """))

    def accumulator():
        total = 0
        while True:
            value = yield total
            if value is None:
                break
            total += value

    gen = accumulator()
    print(f"  Actual output:")
    print(f"    next(gen):     {next(gen)}")
    print(f"    send(10):      {gen.send(10)}")
    print(f"    send(20):      {gen.send(20)}")
    print(f"    send(5):       {gen.send(5)}")

    print(dedent("""
        Output: 0, 10, 30, 35

        WHY: send(value) resumes the generator and makes `value` the result
        of the yield expression. The generator runs until the next yield.
        - next(gen) -> yields 0 (total=0), pauses at yield
        - send(10) -> value=10, total=10, yields 10
        - send(20) -> value=20, total=30, yields 30
        - send(5) -> value=5, total=35, yields 35

        You must call next() first to "prime" the generator (advance to
        the first yield). You can't send() to a just-created generator.
    """))


@question("yield from — Delegating to Sub-generators")
def q_iter_yield_from():
    print(dedent("""
        Code:
            def flatten(nested):
                for item in nested:
                    if isinstance(item, list):
                        yield from flatten(item)
                    else:
                        yield item

            data = [1, [2, 3, [4, 5]], 6, [7]]
            print(list(flatten(data)))

        What's the output?
    """))

    def flatten(nested):
        for item in nested:
            if isinstance(item, list):
                yield from flatten(item)
            else:
                yield item

    data = [1, [2, 3, [4, 5]], 6, [7]]
    result = list(flatten(data))
    print(f"  Actual output: {result}")

    print(dedent("""
        Output: [1, 2, 3, 4, 5, 6, 7]

        WHY: `yield from` delegates to another iterator/generator. It
        yields each item from the sub-generator as if this generator
        yielded it directly. It also properly handles send() and throw()
        for bidirectional communication with sub-generators.
    """))


q_iter_exhausted()
q_iter_genexp_arg()
q_iter_send()
q_iter_yield_from()


# =============================================================================
# CATEGORY 6: CLASS & INHERITANCE TRAPS
# =============================================================================
print("\n" + "#"*70)
print("#  CATEGORY 6: CLASS & INHERITANCE TRAPS")
print("#"*70)


@question("Class Variable vs Instance Variable")
def q_class_var_instance():
    print(dedent("""
        Code:
            class Dog:
                tricks = []  # class variable shared by all instances

                def __init__(self, name):
                    self.name = name

                def add_trick(self, trick):
                    self.tricks.append(trick)

            fido = Dog("Fido")
            buddy = Dog("Buddy")
            fido.add_trick("roll over")
            buddy.add_trick("shake")
            print(fido.tricks)
            print(buddy.tricks)

        What's the output?
    """))

    class Dog:
        tricks = []

        def __init__(self, name):
            self.name = name

        def add_trick(self, trick):
            self.tricks.append(trick)

    fido = Dog("Fido")
    buddy = Dog("Buddy")
    fido.add_trick("roll over")
    buddy.add_trick("shake")

    print(f"  Actual output:")
    print(f"    fido.tricks:  {fido.tricks}")
    print(f"    buddy.tricks: {buddy.tricks}")

    print(dedent("""
        Surprise: Both show ['roll over', 'shake']!

        WHY: `tricks = []` is a CLASS variable — shared by ALL instances.
        When you do self.tricks.append(), you're mutating the shared list.

        FIX: Initialize mutable attributes in __init__:
            def __init__(self, name):
                self.name = name
                self.tricks = []  # instance variable — each dog has its own
    """))


@question("MRO (Method Resolution Order) with Diamond Inheritance")
def q_class_mro():
    print(dedent("""
        Code:
            class A:
                def method(self):
                    return "A"

            class B(A):
                def method(self):
                    return "B"

            class C(A):
                def method(self):
                    return "C"

            class D(B, C):
                pass

            d = D()
            print(d.method())
            print(D.__mro__)

        What's the output?
    """))

    class A:
        def method(self):
            return "A"

    class B(A):
        def method(self):
            return "B"

    class C(A):
        def method(self):
            return "C"

    class D(B, C):
        pass

    d = D()
    print(f"  Actual output:")
    print(f"    d.method(): {d.method()}")
    mro_names = [cls.__name__ for cls in D.__mro__]
    print(f"    MRO: {' -> '.join(mro_names)}")

    print(dedent("""
        Output: "B", MRO: D -> B -> C -> A -> object

        WHY: Python uses C3 Linearization for MRO. It's NOT depth-first
        (which would be D->B->A->C->A). C3 ensures:
        1. Children come before parents
        2. Left-to-right order of bases is preserved
        3. Each class appears only once

        B comes before C because B is listed first in class D(B, C).
    """))


@question("super() with Multiple Inheritance")
def q_class_super_multi():
    print(dedent("""
        Code:
            class Base:
                def __init__(self):
                    print("Base.__init__")

            class Left(Base):
                def __init__(self):
                    print("Left.__init__")
                    super().__init__()

            class Right(Base):
                def __init__(self):
                    print("Right.__init__")
                    super().__init__()

            class Child(Left, Right):
                def __init__(self):
                    print("Child.__init__")
                    super().__init__()

            Child()

        What's the output?
    """))

    class Base:
        def __init__(self):
            print("    Base.__init__")

    class Left(Base):
        def __init__(self):
            print("    Left.__init__")
            super().__init__()

    class Right(Base):
        def __init__(self):
            print("    Right.__init__")
            super().__init__()

    class Child(Left, Right):
        def __init__(self):
            print("    Child.__init__")
            super().__init__()

    print(f"  Actual output:")
    Child()

    print(dedent("""
        Output: Child, Left, Right, Base (each called exactly ONCE)

        WHY: super() follows the MRO, not the direct parent class.
        MRO is Child -> Left -> Right -> Base -> object.
        super().__init__() in Left calls Right.__init__ (next in MRO),
        NOT Base.__init__. This is why every class in a cooperative
        hierarchy must call super().__init__().
    """))


@question("__slots__ Memory Optimization")
def q_class_slots():
    print(dedent("""
        Code:
            class WithoutSlots:
                def __init__(self, x, y):
                    self.x = x
                    self.y = y

            class WithSlots:
                __slots__ = ('x', 'y')
                def __init__(self, x, y):
                    self.x = x
                    self.y = y

            a = WithoutSlots(1, 2)
            b = WithSlots(1, 2)

            a.z = 3   # Works — dynamic attribute
            try:
                b.z = 3   # Fails!
            except AttributeError as e:
                print(f"Error: {e}")

            print(hasattr(a, '__dict__'))
            print(hasattr(b, '__dict__'))

        What's the output?
    """))

    class WithoutSlots:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    class WithSlots:
        __slots__ = ('x', 'y')
        def __init__(self, x, y):
            self.x = x
            self.y = y

    a = WithoutSlots(1, 2)
    b = WithSlots(1, 2)

    a.z = 3
    print(f"  Actual output:")
    try:
        b.z = 3
    except AttributeError as e:
        print(f"    AttributeError: {e}")

    print(f"    WithoutSlots has __dict__: {hasattr(a, '__dict__')}")
    print(f"    WithSlots has __dict__: {hasattr(b, '__dict__')}")

    print(f"    WithoutSlots size: {sys.getsizeof(a) + sys.getsizeof(a.__dict__)} bytes")
    print(f"    WithSlots size: {sys.getsizeof(b)} bytes")

    print(dedent("""
        Output: AttributeError on b.z, True, False

        WHY: __slots__ tells Python to use a fixed set of attributes instead
        of a dynamic __dict__. Benefits:
        1. ~40-50% less memory per instance
        2. Slightly faster attribute access
        3. Prevents accidental attribute creation (typos)

        Trade-offs: No dynamic attributes, no multiple inheritance with
        conflicting __slots__, and every class in the hierarchy needs __slots__.
    """))


@question("__init_subclass__ Hook")
def q_class_init_subclass():
    print(dedent("""
        Code:
            class Plugin:
                _registry = {}

                def __init_subclass__(cls, plugin_name=None, **kwargs):
                    super().__init_subclass__(**kwargs)
                    name = plugin_name or cls.__name__.lower()
                    Plugin._registry[name] = cls
                    print(f"Registered: {name}")

            class JSONPlugin(Plugin, plugin_name="json"):
                pass

            class XMLPlugin(Plugin, plugin_name="xml"):
                pass

            class CSVPlugin(Plugin):
                pass

            print(Plugin._registry)

        What's the output?
    """))

    class Plugin:
        _registry = {}

        def __init_subclass__(cls, plugin_name=None, **kwargs):
            super().__init_subclass__(**kwargs)
            name = plugin_name or cls.__name__.lower()
            Plugin._registry[name] = cls
            print(f"    Registered: {name}")

    class JSONPlugin(Plugin, plugin_name="json"):
        pass

    class XMLPlugin(Plugin, plugin_name="xml"):
        pass

    class CSVPlugin(Plugin):
        pass

    print(f"    Registry: { {k: v.__name__ for k, v in Plugin._registry.items()} }")

    print(dedent("""
        Output: Registers json, xml, csvplugin automatically!

        WHY: __init_subclass__ is called automatically when a class is
        subclassed. No metaclass needed! This is the modern Pythonic way
        to implement plugin registration, validation, or auto-configuration
        of subclasses. Added in Python 3.6.
    """))


q_class_var_instance()
q_class_mro()
q_class_super_multi()
q_class_slots()
q_class_init_subclass()


# =============================================================================
# CATEGORY 7: DECORATOR PUZZLES
# =============================================================================
print("\n" + "#"*70)
print("#  CATEGORY 7: DECORATOR PUZZLES")
print("#"*70)


@question("Decorator Execution Order (Stacking)")
def q_deco_order():
    print(dedent("""
        Code:
            def decorator_a(func):
                print("A applied")
                def wrapper(*args, **kwargs):
                    print("A before")
                    result = func(*args, **kwargs)
                    print("A after")
                    return result
                return wrapper

            def decorator_b(func):
                print("B applied")
                def wrapper(*args, **kwargs):
                    print("B before")
                    result = func(*args, **kwargs)
                    print("B after")
                    return result
                return wrapper

            @decorator_a
            @decorator_b
            def hello():
                print("Hello!")

            hello()

        What's the output?
    """))

    def decorator_a(func):
        print("    A applied")
        def wrapper(*args, **kwargs):
            print("    A before")
            result = func(*args, **kwargs)
            print("    A after")
            return result
        return wrapper

    def decorator_b(func):
        print("    B applied")
        def wrapper(*args, **kwargs):
            print("    B before")
            result = func(*args, **kwargs)
            print("    B after")
            return result
        return wrapper

    @decorator_a
    @decorator_b
    def hello():
        print("    Hello!")

    print(f"  Calling hello():")
    hello()

    print(dedent("""
        Output:
            B applied   (inner decorator applied first)
            A applied   (outer decorator applied second)
            A before    (outer wrapper runs first)
            B before    (inner wrapper runs next)
            Hello!      (original function)
            B after     (inner wrapper returns)
            A after     (outer wrapper returns)

        WHY: @A @B def f -> f = A(B(f)). Decorators are APPLIED bottom-up
        but EXECUTED top-down. Think of it like wrapping a gift: you wrap
        with B first, then A. When unwrapping, you remove A first, then B.
    """))


@question("functools.wraps Importance")
def q_deco_wraps():
    print(dedent("""
        Code:
            def bad_decorator(func):
                def wrapper(*args, **kwargs):
                    return func(*args, **kwargs)
                return wrapper

            def good_decorator(func):
                @functools.wraps(func)
                def wrapper(*args, **kwargs):
                    return func(*args, **kwargs)
                return wrapper

            @bad_decorator
            def my_func():
                \"\"\"My docstring.\"\"\"
                pass

            @good_decorator
            def my_other_func():
                \"\"\"My other docstring.\"\"\"
                pass

            print(my_func.__name__, my_func.__doc__)
            print(my_other_func.__name__, my_other_func.__doc__)

        What's the output?
    """))

    def bad_decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    def good_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    @bad_decorator
    def my_func():
        """My docstring."""
        pass

    @good_decorator
    def my_other_func():
        """My other docstring."""
        pass

    print(f"  Actual output:")
    print(f"    Bad:  name={my_func.__name__}, doc={my_func.__doc__}")
    print(f"    Good: name={my_other_func.__name__}, doc={my_other_func.__doc__}")

    print(dedent("""
        Output:
            Bad:  name=wrapper, doc=None
            Good: name=my_other_func, doc=My other docstring.

        WHY: Without @functools.wraps, the decorated function loses its
        name, docstring, and other metadata. This breaks help(), debugging,
        serialization (pickle), and introspection tools. ALWAYS use
        @functools.wraps(func) in your decorators.
    """))


@question("Class as a Decorator")
def q_deco_class():
    print(dedent("""
        Code:
            class CountCalls:
                def __init__(self, func):
                    self.func = func
                    self.count = 0
                    functools.update_wrapper(self, func)

                def __call__(self, *args, **kwargs):
                    self.count += 1
                    print(f"Call #{self.count} to {self.func.__name__}")
                    return self.func(*args, **kwargs)

            @CountCalls
            def say_hello(name):
                return f"Hello, {name}!"

            say_hello("Alice")
            say_hello("Bob")
            say_hello("Charlie")
            print(f"Total calls: {say_hello.count}")

        What's the output?
    """))

    class CountCalls:
        def __init__(self, func):
            self.func = func
            self.count = 0
            functools.update_wrapper(self, func)

        def __call__(self, *args, **kwargs):
            self.count += 1
            print(f"    Call #{self.count} to {self.func.__name__}")
            return self.func(*args, **kwargs)

    @CountCalls
    def say_hello(name):
        return f"Hello, {name}!"

    print(f"  Actual output:")
    say_hello("Alice")
    say_hello("Bob")
    say_hello("Charlie")
    print(f"    Total calls: {say_hello.count}")

    print(dedent("""
        Output: Call #1, #2, #3, Total calls: 3

        WHY: A class with __call__ can act as a decorator. The class instance
        replaces the function. This is powerful because class decorators
        can maintain STATE across calls (like the call count here).
        Also useful for caching, rate limiting, etc.
    """))


@question("Decorator with Arguments")
def q_deco_with_args():
    print(dedent("""
        Code:
            def repeat(n):
                def decorator(func):
                    @functools.wraps(func)
                    def wrapper(*args, **kwargs):
                        results = []
                        for _ in range(n):
                            results.append(func(*args, **kwargs))
                        return results
                    return wrapper
                return decorator

            @repeat(3)
            def greet(name):
                return f"Hi {name}"

            print(greet("World"))

        What's the output?
    """))

    def repeat(n):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                results = []
                for _ in range(n):
                    results.append(func(*args, **kwargs))
                return results
            return wrapper
        return decorator

    @repeat(3)
    def greet(name):
        return f"Hi {name}"

    print(f"  Actual output: {greet('World')}")

    print(dedent("""
        Output: ['Hi World', 'Hi World', 'Hi World']

        WHY: @repeat(3) first calls repeat(3), which returns `decorator`.
        Then decorator is applied to `greet`. So it's a THREE-level nesting:
        repeat(n) -> decorator(func) -> wrapper(*args).

        This is why decorators with arguments need that extra nesting level.
        Without arguments: 2 levels. With arguments: 3 levels.
    """))


q_deco_order()
q_deco_wraps()
q_deco_class()
q_deco_with_args()


# =============================================================================
# CATEGORY 8: STRING & NUMBER SURPRISES
# =============================================================================
print("\n" + "#"*70)
print("#  CATEGORY 8: STRING & NUMBER SURPRISES")
print("#"*70)


@question("Floating Point Arithmetic")
def q_numbers_float():
    print(dedent("""
        Code:
            print(0.1 + 0.2 == 0.3)
            print(0.1 + 0.2)
            print(round(0.1 + 0.2, 1) == 0.3)

        What's the output?
    """))

    print(f"  Actual output:")
    print(f"    0.1 + 0.2 == 0.3: {0.1 + 0.2 == 0.3}")
    print(f"    0.1 + 0.2 = {0.1 + 0.2}")
    print(f"    round(0.1 + 0.2, 1) == 0.3: {round(0.1 + 0.2, 1) == 0.3}")

    print(dedent("""
        Output: False, 0.30000000000000004, True

        WHY: Floating point numbers can't represent 0.1 exactly in binary
        (IEEE 754). 0.1 + 0.2 = 0.30000000000000004.

        FIX: Use decimal.Decimal for financial calculations, math.isclose()
        for comparisons, or multiply to integers first:
            from decimal import Decimal
            Decimal('0.1') + Decimal('0.2') == Decimal('0.3')  # True
    """))


@question("round() — Banker's Rounding")
def q_numbers_round():
    print(dedent("""
        Code:
            print(round(0.5))
            print(round(1.5))
            print(round(2.5))
            print(round(3.5))
            print(round(4.5))

        What's the output?
    """))

    print(f"  Actual output:")
    print(f"    round(0.5) = {round(0.5)}")
    print(f"    round(1.5) = {round(1.5)}")
    print(f"    round(2.5) = {round(2.5)}")
    print(f"    round(3.5) = {round(3.5)}")
    print(f"    round(4.5) = {round(4.5)}")

    print(dedent("""
        Surprise: 0, 2, 2, 4, 4 — NOT 1, 2, 3, 4, 5!

        WHY: Python uses "banker's rounding" (round half to even). When
        the value is exactly halfway, it rounds to the nearest EVEN number.
        This reduces cumulative rounding bias in statistical calculations.

        If you need traditional rounding (always round .5 up):
            import math
            math.floor(x + 0.5)  # or use decimal with ROUND_HALF_UP
    """))


@question("bool is a Subclass of int")
def q_numbers_bool_int():
    print(dedent("""
        Code:
            print(isinstance(True, int))
            print(True + True + True)
            print(True * 10)
            print(sum([True, False, True, True]))
            print({True: "yes", 1: "one", 1.0: "float_one"})

        What's the output?
    """))

    print(f"  Actual output:")
    print(f"    isinstance(True, int): {isinstance(True, int)}")
    print(f"    True + True + True: {True + True + True}")
    print(f"    True * 10: {True * 10}")
    print(f"    sum([True, False, True, True]): {sum([True, False, True, True])}")
    print(f"    {{True: 'yes', 1: 'one', 1.0: 'float_one'}}: { {True: 'yes', 1: 'one', 1.0: 'float_one'} }")

    print(dedent("""
        Output: True, 3, 10, 3, {True: 'float_one'}

        WHY: bool is a subclass of int! True == 1, False == 0.
        In the dict, True, 1, and 1.0 are all equal and have the same hash,
        so they're treated as the SAME key. The key stays as `True` (first
        inserted) but the value gets overwritten to 'float_one' (last set).
    """))


@question("String Multiplication and Interning Edge Cases")
def q_strings_multiply():
    print(dedent("""
        Code:
            s1 = "ha" * 3
            print(s1)

            s2 = "-" * 0
            print(repr(s2))

            s3 = "ab" * -1
            print(repr(s3))

            # f-string edge case
            value = {"key": "val"}
            print(f"Dict: {value['key']}")
            # print(f"Dict: {value["key"]}")  # SyntaxError before 3.12!

        What's the output?
    """))

    s1 = "ha" * 3
    print(f"  Actual output:")
    print(f"    'ha' * 3: {s1}")

    s2 = "-" * 0
    print(f"    '-' * 0: {repr(s2)}")

    s3 = "ab" * -1
    print(f"    'ab' * -1: {repr(s3)}")

    value = {"key": "val"}
    print(f"    f-string dict access: {value['key']}")

    print(dedent("""
        Output: hahaha, '', '', val

        WHY: String * n repeats n times. * 0 or negative gives empty string.
        For f-strings, you must use different quotes inside the braces
        than the surrounding string (before Python 3.12). In 3.12+, you
        can nest the same quotes.
    """))


q_numbers_float()
q_numbers_round()
q_numbers_bool_int()
q_strings_multiply()


# =============================================================================
# CATEGORY 9: ASYNC GOTCHAS
# =============================================================================
print("\n" + "#"*70)
print("#  CATEGORY 9: ASYNC GOTCHAS")
print("#"*70)


@question("Forgetting to await")
def q_async_forget_await():
    print(dedent("""
        Code:
            import asyncio

            async def fetch_data():
                return {"status": "ok"}

            async def main():
                result = fetch_data()   # Missing await!
                print(type(result))
                print(result)

                result2 = await fetch_data()  # Correct
                print(result2)

            asyncio.run(main())

        What's the output?
    """))

    async def fetch_data():
        return {"status": "ok"}

    async def main():
        result = fetch_data()  # Missing await
        print(f"    type: {type(result)}")
        print(f"    value: {result}")
        await result  # Must consume to avoid warning

        result2 = await fetch_data()
        print(f"    awaited: {result2}")

    asyncio.run(main())

    print(dedent("""
        Output:
            type: <class 'coroutine'>
            value: <coroutine object fetch_data at 0x...>
            awaited: {'status': 'ok'}

        WHY: Calling an async function WITHOUT await returns a coroutine
        object, not the result. The function body doesn't execute at all!
        You get no error, just a silent coroutine object. This is a very
        common source of bugs — forgetting await makes the code appear
        to work but skip the actual operation.
    """))


@question("asyncio.gather Exception Handling")
def q_async_gather():
    print(dedent("""
        Code:
            import asyncio

            async def task_ok():
                return "ok"

            async def task_fail():
                raise ValueError("boom")

            async def main():
                # Without return_exceptions
                try:
                    results = await asyncio.gather(task_ok(), task_fail())
                except ValueError as e:
                    print(f"Caught: {e}")

                # With return_exceptions=True
                results = await asyncio.gather(
                    task_ok(), task_fail(), return_exceptions=True
                )
                for r in results:
                    if isinstance(r, Exception):
                        print(f"Exception: {r}")
                    else:
                        print(f"Result: {r}")

            asyncio.run(main())

        What's the output?
    """))

    async def task_ok():
        return "ok"

    async def task_fail():
        raise ValueError("boom")

    async def main():
        try:
            results = await asyncio.gather(task_ok(), task_fail())
        except ValueError as e:
            print(f"    Caught: {e}")

        results = await asyncio.gather(
            task_ok(), task_fail(), return_exceptions=True
        )
        for r in results:
            if isinstance(r, Exception):
                print(f"    Exception: {r}")
            else:
                print(f"    Result: {r}")

    asyncio.run(main())

    print(dedent("""
        Output:
            Caught: boom
            Result: ok
            Exception: boom

        WHY: By default, gather() raises the first exception and cancels
        remaining tasks. With return_exceptions=True, exceptions are returned
        as values in the results list. This is crucial for handling partial
        failures — you usually want return_exceptions=True in production.
    """))


@question("Blocking Call in Async Code")
def q_async_blocking():
    print(dedent("""
        Code:
            import asyncio
            import time

            async def bad_async():
                time.sleep(1)  # BLOCKING! Freezes entire event loop
                return "done"

            async def good_async():
                await asyncio.sleep(1)  # Non-blocking, yields to event loop
                return "done"

            async def better_async():
                # Run blocking code in a thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, time.sleep, 0.01)
                return "done"

        Which approach is correct and why?
    """))

    async def better_async():
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: None)
        return "done"

    result = asyncio.run(better_async())
    print(f"    run_in_executor works: {result}")

    print(dedent("""
        Answer: good_async() and better_async() are correct.

        WHY: time.sleep() is a BLOCKING call — it freezes the entire event
        loop, preventing any other coroutine from running. In async code:
        - Use asyncio.sleep() for delays
        - Use aiohttp instead of requests
        - Use asyncpg instead of psycopg2
        - Wrap unavoidable blocking calls with run_in_executor()

        RULE: Never use blocking I/O in async functions. If you must call
        blocking code, use run_in_executor() to run it in a thread pool.
    """))


q_async_forget_await()
q_async_gather()
q_async_blocking()


# =============================================================================
# CATEGORY 10: PYTHONIC TRAPS
# =============================================================================
print("\n" + "#"*70)
print("#  CATEGORY 10: PYTHONIC TRAPS")
print("#"*70)


@question("Chained Comparisons")
def q_pythonic_chained():
    print(dedent("""
        Code:
            print(1 < 2 < 3)
            print(1 < 2 > 0)
            print(1 < 2 < 3 < 4 < 5)
            print(1 == 1 in [1])
            print(1 is 1 in [1])

        What's the output?
    """))

    print(f"  Actual output:")
    print(f"    1 < 2 < 3: {1 < 2 < 3}")
    print(f"    1 < 2 > 0: {1 < 2 > 0}")
    print(f"    1 < 2 < 3 < 4 < 5: {1 < 2 < 3 < 4 < 5}")
    print(f"    1 == 1 in [1]: {1 == 1 in [1]}")
    # Note: `1 is 1 in [1]` produces SyntaxWarning in modern Python, skip

    print(dedent("""
        Output: True, True, True, True

        WHY: Python chains comparisons! `1 < 2 < 3` means `1 < 2 AND 2 < 3`.
        `1 == 1 in [1]` means `1 == 1 AND 1 in [1]` -> True AND True.

        This is elegant but can be confusing. `in` and `is` are also
        comparison operators, so they participate in chaining.

        GOTCHA: `False == False in [False]` is True because
        `False == False AND False in [False]` -> True AND True.
    """))


@question("else on for/while/try")
def q_pythonic_for_else():
    print(dedent("""
        Code:
            # else on for loop
            for i in range(5):
                if i == 3:
                    break
            else:
                print("Loop completed without break")

            for i in range(5):
                pass
            else:
                print("Loop completed without break (2)")

            # else on try/except
            try:
                result = 10 / 2
            except ZeroDivisionError:
                print("Error!")
            else:
                print(f"No error, result = {result}")

        What's the output?
    """))

    print(f"  Actual output:")
    for i in range(5):
        if i == 3:
            break
    else:
        print("    Loop completed without break")

    for i in range(5):
        pass
    else:
        print("    Loop completed without break (2)")

    try:
        result = 10 / 2
    except ZeroDivisionError:
        print("    Error!")
    else:
        print(f"    No error, result = {result}")

    print(dedent("""
        Output:
            Loop completed without break (2)
            No error, result = 5.0

        WHY:
        - for/else: the else block runs ONLY if the loop completes WITHOUT
          a break. First loop breaks at i=3, so else is skipped. Second loop
          finishes normally, so else runs.
        - try/else: the else block runs ONLY if NO exception was raised.
          Useful for code that should run only on success.

        Think of for/else as "for/nobreak".
    """))


@question("Walrus Operator Edge Cases")
def q_pythonic_walrus():
    print(dedent("""
        Code:
            # Basic walrus
            if (n := len("hello")) > 3:
                print(f"Long string: {n} chars")

            # Walrus in while
            data = [1, 2, 3, 0, 4, 5]
            index = 0
            while (value := data[index]) != 0:
                print(value, end=" ")
                index += 1
            print()

            # Walrus in list comprehension
            results = [y for x in range(10) if (y := x**2) > 20]
            print(results)

        What's the output?
    """))

    print(f"  Actual output:")
    if (n := len("hello")) > 3:
        print(f"    Long string: {n} chars")

    data = [1, 2, 3, 0, 4, 5]
    index = 0
    print("    ", end="")
    while (value := data[index]) != 0:
        print(value, end=" ")
        index += 1
    print()

    results = [y for x in range(10) if (y := x**2) > 20]
    print(f"    Squares > 20: {results}")

    print(dedent("""
        Output:
            Long string: 5 chars
            1 2 3
            [25, 36, 49, 64, 81]

        WHY: The walrus operator (:=) assigns AND returns a value.
        - In if: assigns n and checks condition in one expression
        - In while: reads value and checks for sentinel in one expression
        - In comprehension: computes once, uses for both filter and result

        GOTCHA: The walrus operator can't be used at the top level of an
        expression statement (use regular assignment instead).
        Also, `(x := 0)` works but `x := 0` without parens is a SyntaxError
        in most contexts.
    """))


@question("Unpacking Tricks")
def q_pythonic_unpacking():
    print(dedent("""
        Code:
            # Star unpacking
            first, *middle, last = [1, 2, 3, 4, 5]
            print(first, middle, last)

            # Nested unpacking
            (a, b), (c, d) = [1, 2], [3, 4]
            print(a, b, c, d)

            # Swap without temp
            x, y = 1, 2
            x, y = y, x
            print(x, y)

            # Star in function call
            def add(a, b, c):
                return a + b + c
            args = [1, 2, 3]
            print(add(*args))

        What's the output?
    """))

    first, *middle, last = [1, 2, 3, 4, 5]
    print(f"  Actual output:")
    print(f"    first={first}, middle={middle}, last={last}")

    (a, b), (c, d) = [1, 2], [3, 4]
    print(f"    a={a}, b={b}, c={c}, d={d}")

    x, y = 1, 2
    x, y = y, x
    print(f"    After swap: x={x}, y={y}")

    def add(a, b, c):
        return a + b + c
    args = [1, 2, 3]
    print(f"    add(*[1,2,3]) = {add(*args)}")

    print(dedent("""
        Output: 1 [2,3,4] 5 | 1 2 3 4 | 2 1 | 6

        WHY: Python's unpacking is powerful:
        - *middle captures everything between first and last as a list
        - Nested unpacking matches the structure
        - x, y = y, x works because the right side is evaluated fully
          before any assignment (it creates a tuple (2, 1) first)
        - * unpacks a list into positional arguments
    """))


@question("Positional-Only and Keyword-Only Parameters (/ and *)")
def q_pythonic_params():
    print(dedent("""
        Code:
            def func(pos_only, /, normal, *, kw_only):
                print(f"pos_only={pos_only}, normal={normal}, kw_only={kw_only}")

            func(1, 2, kw_only=3)           # Works
            func(1, normal=2, kw_only=3)    # Works

            try:
                func(pos_only=1, normal=2, kw_only=3)  # Fails!
            except TypeError as e:
                print(f"Error: {e}")

            try:
                func(1, 2, 3)  # Fails!
            except TypeError as e:
                print(f"Error: {e}")

        What's the output?
    """))

    def func(pos_only, /, normal, *, kw_only):
        print(f"    pos_only={pos_only}, normal={normal}, kw_only={kw_only}")

    print(f"  Actual output:")
    func(1, 2, kw_only=3)
    func(1, normal=2, kw_only=3)

    try:
        func(pos_only=1, normal=2, kw_only=3)
    except TypeError as e:
        print(f"    Error: {e}")

    try:
        func(1, 2, 3)
    except TypeError as e:
        print(f"    Error: {e}")

    print(dedent("""
        Output:
            pos_only=1, normal=2, kw_only=3  (both work)
            Error: positional-only argument
            Error: keyword-only argument

        WHY:
        - `/` marks everything before it as positional-only
        - `*` marks everything after it as keyword-only
        - Parameters between / and * can be either

        Real-world use:
        - Positional-only: allows renaming params without breaking callers
        - Keyword-only: prevents confusing positional calls
        - Example: len(obj, /) — you can't do len(obj=mylist)
    """))


@question("Dictionary Merge Operators (Python 3.9+)")
def q_pythonic_dict_merge():
    print(dedent("""
        Code:
            d1 = {"a": 1, "b": 2}
            d2 = {"b": 3, "c": 4}

            # Merge (d2 wins on conflicts)
            merged = d1 | d2
            print(merged)

            # Update in place
            d1 |= d2
            print(d1)

            # Compare with unpacking (works in 3.5+)
            d3 = {"a": 1, "b": 2}
            d4 = {"b": 3, "c": 4}
            print({**d3, **d4})

        What's the output?
    """))

    d1 = {"a": 1, "b": 2}
    d2 = {"b": 3, "c": 4}

    merged = d1 | d2
    print(f"  Actual output:")
    print(f"    d1 | d2: {merged}")

    d1 |= d2
    print(f"    d1 |= d2: {d1}")

    d3 = {"a": 1, "b": 2}
    d4 = {"b": 3, "c": 4}
    print(f"    {{**d3, **d4}}: { {**d3, **d4} }")

    print(dedent("""
        Output:
            {'a': 1, 'b': 3, 'c': 4}  (d2's 'b' wins)
            {'a': 1, 'b': 3, 'c': 4}  (d1 updated in place)
            {'a': 1, 'b': 3, 'c': 4}  (same result with unpacking)

        WHY: The | operator (Python 3.9+) creates a new merged dict.
        |= updates in place. For conflicts, the right-hand dict wins.
        The ** unpacking approach works in older Python but is less readable.
    """))


@question("Multiple Assignment Gotcha")
def q_pythonic_multi_assign():
    print(dedent("""
        Code:
            a = b = []
            a.append(1)
            print(f"a={a}, b={b}")

            x = []
            y = []
            x.append(1)
            print(f"x={x}, y={y}")

        What's the output?
    """))

    a = b = []
    a.append(1)
    print(f"  Actual output:")
    print(f"    a={a}, b={b}")

    x = []
    y = []
    x.append(1)
    print(f"    x={x}, y={y}")

    print(dedent("""
        Output:
            a=[1], b=[1]  <- both changed!
            x=[1], y=[]   <- independent

        WHY: `a = b = []` creates ONE list object and makes both a and b
        reference it. They're aliases for the same object.
        `x = []` and `y = []` create TWO separate list objects.

        This is the same principle as the mutable default argument trap.
        Assignment in Python binds names to objects, it doesn't copy.
    """))


q_pythonic_chained()
q_pythonic_for_else()
q_pythonic_walrus()
q_pythonic_unpacking()
q_pythonic_params()
q_pythonic_dict_merge()
q_pythonic_multi_assign()


# =============================================================================
# BONUS: EXTRA TRICKY QUESTIONS
# =============================================================================
print("\n" + "#"*70)
print("#  BONUS: EXTRA TRICKY QUESTIONS")
print("#"*70)


@question("all() and any() with Empty Iterables")
def q_bonus_all_any():
    print(dedent("""
        Code:
            print(all([]))
            print(any([]))
            print(all([0, 1, 2]))
            print(any([0, 0, 0]))

        What's the output?
    """))

    print(f"  Actual output:")
    print(f"    all([]): {all([])}")
    print(f"    any([]): {any([])}")
    print(f"    all([0, 1, 2]): {all([0, 1, 2])}")
    print(f"    any([0, 0, 0]): {any([0, 0, 0])}")

    print(dedent("""
        Output: True, False, False, False

        WHY: all([]) is True (vacuous truth — no elements to be False).
        any([]) is False (no elements to be True).
        all([0,1,2]) is False because 0 is falsy.
        any([0,0,0]) is False because no element is truthy.

        This follows mathematical convention. all() returns True if there
        are no counterexamples. any() returns True if there exists at
        least one example.
    """))


@question("Exception Chaining and __context__")
def q_bonus_exception_chain():
    print(dedent("""
        Code:
            try:
                try:
                    1 / 0
                except ZeroDivisionError:
                    raise ValueError("bad value")
            except ValueError as e:
                print(f"Caught: {e}")
                print(f"Original cause: {e.__context__}")

        What's the output?
    """))

    print(f"  Actual output:")
    try:
        try:
            1 / 0
        except ZeroDivisionError:
            raise ValueError("bad value")
    except ValueError as e:
        print(f"    Caught: {e}")
        print(f"    Original cause: {e.__context__}")

    print(dedent("""
        Output:
            Caught: bad value
            Original cause: division by zero

        WHY: When you raise an exception inside an except block, Python
        automatically sets __context__ to the original exception. The full
        traceback shows "During handling of the above exception, another
        exception occurred."

        For explicit chaining, use `raise ValueError("bad") from original_exc`.
        This sets __cause__ instead of __context__ and produces:
        "The above exception was the direct cause of the following exception."
    """))


@question("The Surprising Behavior of 'is not' vs 'not ... is'")
def q_bonus_is_not():
    print(dedent("""
        Code:
            x = [1, 2, 3]
            y = [1, 2, 3]

            print(x is not y)
            print(not x is y)

            # Are they the same?
            import dis
            # dis.dis("x is not y")
            # dis.dis("not x is y")

        What's the output?
    """))

    x = [1, 2, 3]
    y = [1, 2, 3]

    print(f"  Actual output:")
    print(f"    x is not y: {x is not y}")
    print(f"    not x is y: {not x is y}")
    print(f"    Same result: {(x is not y) == (not x is y)}")

    print(dedent("""
        Output: True, True — same result!

        WHY: `is not` is a single operator in Python (like `not in`).
        `not x is y` is parsed as `not (x is y)` by the compiler.
        Both produce the same bytecode (IS_OP with invert flag).

        PEP 8 recommends `is not` over `not ... is` for readability.
        Similarly, use `not in` instead of `not ... in`.
    """))


q_bonus_all_any()
q_bonus_exception_chain()
q_bonus_is_not()


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print(f"  TOTAL QUESTIONS: {question_counter}")
print("="*70)
print(dedent("""
    KEY TAKEAWAYS FOR SENIOR DEVELOPERS:

    1. MUTABLE DEFAULTS: Always use None as default, create inside function
    2. CLOSURES: Capture values, not variables (use default args or factories)
    3. IDENTITY: Use `is` only for None/True/False, use `==` for values
    4. MUTABILITY: Tuples hold references; shallow copy shares nested objects
    5. ITERATORS: Can only be consumed once; generators are lazy
    6. CLASSES: Class variables are shared; super() follows MRO, not parent
    7. DECORATORS: Applied bottom-up, executed top-down; always use @wraps
    8. NUMBERS: Float math is imprecise; round() uses banker's rounding
    9. ASYNC: Always await coroutines; never block the event loop
    10. PYTHONIC: Chained comparisons include `in`/`is`; for/else = for/nobreak
"""))
