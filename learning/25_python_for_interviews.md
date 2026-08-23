# Python for Data Science Interviews — Comprehensive Preparation Guide

**Candidate:** Rahul Sharma | **Level:** Advanced Python | **Experience:** 4+ years  
**Education:** MS Data Science, University of Maryland (4.0 GPA)  
**Focus:** AI Engineer / Data Scientist — Python, ML Engineering, Backend Development, Data Pipelines  
**Core Skills:** Python 3.10+, NumPy, Pandas, Scikit-learn, FastAPI, AsyncIO, OOP, Functional Programming  
**Document Scope:** Complete Python mastery guide — fundamentals through advanced patterns, data science libraries, coding challenges, and interview strategy. **If coding is your weak round, start at §0, then §7.0–§7.3, then §10–§12. Do not binge §1–§6 the night before a live coding loop.**

**Companion drills:** `projects/FusionSpan_Python_Coding_Questions.md` (30+ extra problems) · `projects/practice_coding.py` (scratch pad)

---

## Table of Contents

0. [**Start here — how the coding round actually works**](#0-start-here--how-the-coding-round-actually-works)
1. [Python Fundamentals](#1-python-fundamentals)
2. [Data Structures](#2-data-structures)
3. [Object-Oriented Programming](#3-object-oriented-programming)
4. [Functional Programming](#4-functional-programming)
5. [Advanced Concepts](#5-advanced-concepts)
6. [Data Science Python](#6-data-science-python)
7. [Common Coding Patterns](#7-common-coding-patterns)
   - 7.0 [The 60-second pattern test](#70-the-60-second-pattern-test)
   - 7.1 [Two Pointers](#71-two-pointers)
   - 7.2 [Sliding Window](#72-sliding-window)
   - 7.3 [Hash Map Frequency Counting](#73-hash-map-frequency-counting)
   - 7.4 [Matrix Traversal](#74-matrix-traversal)
   - 7.5 [BFS / DFS Basics](#75-bfs--dfs-basics)
   - 7.6 [Prefix sums](#76-prefix-sums)
   - 7.7 [Binary search (including on the answer)](#77-binary-search-including-on-the-answer)
   - 7.8 [Heaps and Top-K](#78-heaps-and-top-k)
   - 7.9 [Intervals and merge](#79-intervals-and-merge)
   - 7.10 [Stack / monotonic](#710-stack--monotonic)
8. [Python Best Practices](#8-python-best-practices)
9. [Common Interview Coding Questions (10+)](#9-common-interview-coding-questions)
10. [**What each role actually codes**](#10-what-each-role-actually-codes)
11. [**The 6-step brainstorm (say it out loud)**](#11-the-6-step-brainstorm-say-it-out-loud)
12. [**Pocket complexity + 14-day drill**](#12-pocket-complexity--14-day-drill)
13. [Key Takeaways](#13-key-takeaways)

---

# 0. Start here — how the coding round actually works

This is the section to re-read the morning of a coding loop. Sections 1–6 are language depth. **The live round is pattern recognition under talking.** You already know Python. The failure mode is freezing, then jumping into code before the problem is pinned.

## 0.1 What they are testing (not LeetCode rank)

| Role you are targeting | Typical coding ask | What a pass looks like | What a fail looks like |
|---|---|---|---|
| **AI Engineer / GenAI** | Parse messy JSON, chunk text, merge intervals, LRU/rate-limit, BFS on a small graph, implement a 20-line retriever helper (RRF, unique-by-id) | Clear types, edge cases, you mention complexity, you ask about duplicates / empty / streaming | Silent coding; O(n²) on "unique docs"; no tests |
| **Data Scientist** | Two-sum / frequency / group-by in pure Python, sliding window, pandas transform, entity-aware split, a tiny metric (precision@k) | Correct on a hand example; you say train-only fit; you don't leak | `for` over a DataFrame; shuffle then split when IDs repeat |
| **Forward Deployed** | Dirty strings, windowed counts, "make this CSV usable" | You ask about encoding, nulls, late events | Assume clean input |
| **Data Engineer** | Two pointers on sorted logs, hash join sketch, generator over a file, deque window | O(n) streaming mindset | `list(file)` then process |

You will **not** be asked a hard DP contest problem for these roles. You **will** be asked something that looks easy and dies on duplicates, empty input, or off-by-one.

## 0.2 The only four structures you must retrieve under pressure

Write these on scratch paper before they finish the prompt.

| Structure | Python | O(1) superpower | Default problem |
|---|---|---|---|
| **Hash map** | `dict` / `Counter` / `defaultdict(list)` | lookup, count, group, complement | two-sum, anagrams, first duplicate, prefix-sum index |
| **Two pointers** | `left`, `right` or `read`/`write` | O(n) on sorted / in-place | pair sum (sorted), palindrome, compact uniques, merge two lists |
| **Sliding window** | `left` + running state | longest/shortest contiguous | max sum of k, longest unique substring, moving average |
| **Heap** | `heapq` | top-k without full sort | k most frequent, merge k lists, streaming median (two heaps) |

Backup: **stack** (parens, undo, monotonic next-greater), **deque** (BFS, bounded window), **set** (seen / uniqueness).

If the prompt has the word **contiguous**, reach for a window or Kadane/prefix. If it has **pair / two numbers / anagram / frequency**, reach for a hash map. If it says **sorted** and **in-place**, reach for two pointers.

## 0.3 The 6-step loop (say each step out loud)

Full script with phrases is **§11**. The skeleton:

1. **Restate** — one sentence in your words.
2. **Ask** — sorted? duplicates? in-place? n up to? streaming or fits in RAM?
3. **Tiny example** — 5–7 items, including an empty / one-element / all-duplicates case.
4. **Name the pattern** — "hash map of value → index" / "left-right on sorted" / …
5. **Code happy path** — then edges.
6. **Complexity + one test** — time, extra space, run the example.

Until step 4 is spoken, **do not type**. Interviewers grade the narration.

## 0.4 20-minute warm-up (do this the morning of)

1. Two-sum unsorted (hash) + two-sum sorted (pointers) — §7.1, §7.3, Q12
2. Valid parentheses — Q13
3. Sliding window max-sum of k — §7.2
4. Group anagrams — §7.3
5. Merge two sorted lists — Q14

If those five compile from memory, you are ready. Extra drills: FusionSpan file.

---

# 1. Python Fundamentals

---

## 1.1 Data Types

Python is **dynamically typed** (types are checked at runtime, not compile time) and **strongly typed** (no implicit coercion between unrelated types — `"3" + 4` raises `TypeError`, unlike JavaScript).

### Scalar Types

| Type | Description | Example | Immutable? |
|------|-------------|---------|------------|
| `int` | Arbitrary-precision integer | `x = 42` | Yes |
| `float` | IEEE 754 double-precision (64-bit) | `y = 3.14` | Yes |
| `complex` | Complex number | `z = 2 + 3j` | Yes |
| `bool` | Boolean (subclass of `int`) | `flag = True` | Yes |
| `NoneType` | Singleton null value | `x = None` | Yes |
| `str` | Unicode text sequence | `s = "hello"` | Yes |
| `bytes` | Immutable byte sequence | `b = b"\x00\x01"` | Yes |

```python
# bool is a subclass of int
isinstance(True, int)   # True
True + True             # 2
False * 10              # 0

# int has arbitrary precision — no overflow
2 ** 1000               # Works perfectly, returns a huge number

# float precision gotcha
0.1 + 0.2 == 0.3       # False!
0.1 + 0.2              # 0.30000000000000004

# Use decimal for exact arithmetic
from decimal import Decimal
Decimal('0.1') + Decimal('0.2') == Decimal('0.3')  # True

# None is a singleton — always use `is` not `==`
x = None
x is None      # True  (correct)
x == None      # True  (works but bad practice — __eq__ could be overridden)
```

### Type Checking

```python
type(42)              # <class 'int'>
isinstance(42, int)   # True
isinstance(True, int) # True — bool is a subclass of int

# Check multiple types
isinstance(42, (int, float))  # True
```

> **Interview tip:** When asked "What are Python's basic data types?", start with the scalar types, then mention collections. Always note that Python is dynamically typed but strongly typed — this distinction catches many candidates off guard.

---

## 1.2 Collections

### Built-in Collection Types

| Type | Ordered | Mutable | Duplicates | Hashable | Example |
|------|---------|---------|------------|----------|---------|
| `list` | Yes | Yes | Yes | No | `[1, 2, 3]` |
| `tuple` | Yes | No | Yes | Yes* | `(1, 2, 3)` |
| `dict` | Yes (3.7+) | Yes | Keys: No | No | `{"a": 1}` |
| `set` | No | Yes | No | No | `{1, 2, 3}` |
| `frozenset` | No | No | No | Yes | `frozenset({1, 2})` |

*\*Tuples are hashable only if all their elements are hashable.*

```python
# list — workhorse mutable sequence
nums = [3, 1, 4, 1, 5, 9]
nums.append(2)        # [3, 1, 4, 1, 5, 9, 2]
nums.extend([6, 5])   # [3, 1, 4, 1, 5, 9, 2, 6, 5]
nums.insert(0, 0)     # [0, 3, 1, 4, 1, 5, 9, 2, 6, 5]
nums.pop()            # returns 5, list is now [0, 3, 1, 4, 1, 5, 9, 2, 6]
nums.remove(1)        # removes first 1: [0, 3, 4, 1, 5, 9, 2, 6]

# tuple — immutable, used for fixed collections
point = (3, 4)
# point[0] = 5        # TypeError: 'tuple' object does not support item assignment

# dict — key-value pairs (insertion-ordered since Python 3.7)
profile = {"name": "Rahul", "role": "Data Scientist"}
profile["skills"] = ["Python", "ML"]   # add key
profile.get("age", "N/A")             # safe access with default

# set — unique unordered elements
unique = {1, 2, 2, 3, 3, 3}  # {1, 2, 3}

# frozenset — immutable set, can be used as dict key or set element
fs = frozenset([1, 2, 3])
d = {fs: "value"}  # Works because frozenset is hashable
```

### When to Use What

| Use Case | Best Collection |
|----------|----------------|
| Ordered, mutable sequence | `list` |
| Fixed record / dictionary key | `tuple` |
| Fast lookup by key | `dict` |
| Unique elements, set operations | `set` |
| Immutable set (as dict key) | `frozenset` |

---

## 1.3 Mutability: Mutable vs Immutable

This is one of the most commonly tested concepts. Understanding mutability is essential for avoiding bugs and reasoning about Python's pass-by-object-reference semantics.

### Classification

| Immutable | Mutable |
|-----------|---------|
| `int`, `float`, `bool`, `str` | `list`, `dict`, `set` |
| `tuple`, `frozenset`, `bytes` | `bytearray` |

```python
# Immutable: reassignment creates a new object
a = "hello"
b = a
a += " world"
print(b)  # "hello" — b still points to original object
print(id(a) == id(b))  # False — a now points to a new object

# Mutable: modifications affect all references
x = [1, 2, 3]
y = x            # y is an alias, NOT a copy
x.append(4)
print(y)         # [1, 2, 3, 4] — y sees the change!

# The trap: default mutable arguments
def bad_append(item, lst=[]):    # DEFAULT LIST IS SHARED ACROSS CALLS
    lst.append(item)
    return lst

bad_append(1)   # [1]
bad_append(2)   # [1, 2]  — NOT [2]! The default list persists!

# Correct pattern
def good_append(item, lst=None):
    if lst is None:
        lst = []
    lst.append(item)
    return lst
```

### Deep Copy vs Shallow Copy

```python
import copy

original = [[1, 2], [3, 4]]

# Shallow copy — new outer list, but inner lists are still shared
shallow = copy.copy(original)     # or original.copy() or list(original) or original[:]
shallow[0].append(99)
print(original)  # [[1, 2, 99], [3, 4]] — inner list mutated!

# Deep copy — completely independent at all levels
original = [[1, 2], [3, 4]]
deep = copy.deepcopy(original)
deep[0].append(99)
print(original)  # [[1, 2], [3, 4]] — untouched
```

> **Interview tip:** A *very* common question is "What happens when you pass a list to a function?" The answer: Python uses pass-by-object-reference. The function receives a reference to the same object — mutations inside the function are visible outside, but reassignment inside the function creates a new local binding.

---

## 1.4 Comprehensions and Generator Expressions

Comprehensions are Python's most idiomatic and performant way to create collections. They are not syntactic sugar — they are compiled to faster bytecode than equivalent for-loops.

### List Comprehension

```python
# Basic
squares = [x**2 for x in range(10)]
# [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# With filter
evens = [x for x in range(20) if x % 2 == 0]

# Nested (flattening)
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat = [val for row in matrix for val in row]
# [1, 2, 3, 4, 5, 6, 7, 8, 9]
# Read as: for row in matrix → for val in row → val

# With transformation and filter
words = ["Hello", "WORLD", "Python", "DATA"]
lower_long = [w.lower() for w in words if len(w) > 4]
# ['hello', 'world', 'python']

# Conditional expression (ternary) inside comprehension
labels = ["even" if x % 2 == 0 else "odd" for x in range(5)]
# ['even', 'odd', 'even', 'odd', 'even']
```

### Dict Comprehension

```python
# Basic
sq_dict = {x: x**2 for x in range(5)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# Inverting a dictionary
original = {"a": 1, "b": 2, "c": 3}
inverted = {v: k for k, v in original.items()}
# {1: 'a', 2: 'b', 3: 'c'}

# Filtering
scores = {"alice": 85, "bob": 62, "charlie": 91, "diana": 58}
passed = {name: score for name, score in scores.items() if score >= 70}
# {'alice': 85, 'charlie': 91}
```

### Set Comprehension

```python
text = "hello world"
unique_chars = {ch for ch in text if ch != ' '}
# {'d', 'e', 'h', 'l', 'o', 'r', 'w'}
```

### Generator Expression

```python
# Generator expression — lazy evaluation, O(1) memory
gen = (x**2 for x in range(1_000_000))
# No list is created in memory — values are produced one at a time

# Use in function calls (no extra parentheses needed)
total = sum(x**2 for x in range(1_000_000))

# Memory comparison
import sys
list_comp = [x**2 for x in range(1_000_000)]
gen_exp = (x**2 for x in range(1_000_000))
print(sys.getsizeof(list_comp))  # ~8 MB
print(sys.getsizeof(gen_exp))    # ~200 bytes (constant!)
```

> **Interview tip:** Always mention generator expressions when discussing memory efficiency. They are the go-to answer for "How would you process a very large file/dataset in Python?"

---

## 1.5 String Formatting

```python
name = "Rahul"
score = 95.678

# f-strings (Python 3.6+) — fastest and most readable
f"Hello, {name}! Score: {score:.2f}"
# 'Hello, Rahul! Score: 95.68'

# Expression inside f-string
f"{2 ** 10 = }"                     # '2 ** 10 = 1024' (Python 3.8+, debug format)
f"{'hello':>20}"                    # '               hello' (right-align in 20 chars)
f"{1_000_000:,}"                    # '1,000,000'
f"{0.75:.1%}"                       # '75.0%'

# str.format()
"Hello, {}! Score: {:.2f}".format(name, score)

# Named placeholders
"Hello, {name}! Score: {score:.2f}".format(name=name, score=score)

# Old style (avoid in new code)
"Hello, %s! Score: %.2f" % (name, score)
```

### Common Format Specifiers

| Specifier | Meaning | Example | Output |
|-----------|---------|---------|--------|
| `:.2f` | Float, 2 decimals | `f"{3.14159:.2f}"` | `3.14` |
| `:,` | Thousands separator | `f"{1000000:,}"` | `1,000,000` |
| `:.1%` | Percentage | `f"{0.856:.1%}"` | `85.6%` |
| `:>10` | Right-align, width 10 | `f"{'hi':>10}"` | `        hi` |
| `:<10` | Left-align, width 10 | `f"{'hi':<10}"` | `hi        ` |
| `:^10` | Center, width 10 | `f"{'hi':^10}"` | `    hi    ` |
| `:08d` | Zero-padded integer | `f"{42:08d}"` | `00000042` |

---

## 1.6 Unpacking: *args and **kwargs

```python
# *args — collects positional arguments into a tuple
def add(*args):
    return sum(args)

add(1, 2, 3)     # 6
add(10, 20)       # 30

# **kwargs — collects keyword arguments into a dict
def greet(**kwargs):
    for key, val in kwargs.items():
        print(f"{key}: {val}")

greet(name="Rahul", role="Data Scientist")
# name: Rahul
# role: Data Scientist

# Combined — ORDER MATTERS: positional, *args, keyword, **kwargs
def func(a, b, *args, key="default", **kwargs):
    print(f"a={a}, b={b}, args={args}, key={key}, kwargs={kwargs}")

func(1, 2, 3, 4, key="custom", extra=True)
# a=1, b=2, args=(3, 4), key=custom, kwargs={'extra': True}
```

### Unpacking in Assignments and Function Calls

```python
# Iterable unpacking
first, *middle, last = [1, 2, 3, 4, 5]
# first=1, middle=[2, 3, 4], last=5

a, b, *_ = range(10)
# a=0, b=1, _=[2, 3, 4, 5, 6, 7, 8, 9]

# Unpacking in function calls
def multiply(x, y, z):
    return x * y * z

args = [2, 3, 4]
multiply(*args)       # 24

kwargs = {"x": 2, "y": 3, "z": 4}
multiply(**kwargs)    # 24

# Merging dicts (Python 3.9+)
d1 = {"a": 1, "b": 2}
d2 = {"b": 3, "c": 4}
merged = d1 | d2       # {'a': 1, 'b': 3, 'c': 4} — d2 overwrites
merged = {**d1, **d2}  # Same result, works in 3.5+
```

> **Interview tip:** Know the parameter order: `def f(pos, /, normal, *args, kw_only, **kwargs)`. The `/` (Python 3.8+) forces parameters before it to be positional-only.

---

# 2. Data Structures

---

## 2.1 Lists — Deep Dive

### Time Complexity

| Operation | Average | Worst | Notes |
|-----------|---------|-------|-------|
| `append(x)` | O(1)* | O(n) | Amortized O(1) due to dynamic array resizing |
| `pop()` | O(1) | O(1) | Pop from end |
| `pop(i)` | O(n) | O(n) | Shifts elements after index i |
| `insert(i, x)` | O(n) | O(n) | Shifts elements |
| `remove(x)` | O(n) | O(n) | Linear search + shift |
| `x in list` | O(n) | O(n) | Linear scan |
| `list[i]` | O(1) | O(1) | Direct index access |
| `list[i:j]` | O(j-i) | O(j-i) | Creates new list |
| `sort()` | O(n log n) | O(n log n) | Timsort (hybrid merge + insertion) |
| `len(list)` | O(1) | O(1) | Stored attribute |

### Slicing

```python
nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Basic: list[start:stop:step]
nums[2:5]        # [2, 3, 4]       — indices 2, 3, 4 (stop is exclusive)
nums[:3]         # [0, 1, 2]       — from start
nums[7:]         # [7, 8, 9]       — to end
nums[::2]        # [0, 2, 4, 6, 8] — every 2nd element
nums[::-1]       # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0] — reversed

# Negative indices
nums[-3:]        # [7, 8, 9]       — last 3 elements
nums[:-3]        # [0, 1, 2, 3, 4, 5, 6] — all except last 3
nums[-5:-2]      # [5, 6, 7]

# Slice assignment (mutable!)
nums[2:5] = [20, 30]  # [0, 1, 20, 30, 5, 6, 7, 8, 9] — length can change!
```

### Sorting

```python
# In-place sort (returns None)
nums = [3, 1, 4, 1, 5, 9]
nums.sort()                    # [1, 1, 3, 4, 5, 9]
nums.sort(reverse=True)       # [9, 5, 4, 3, 1, 1]

# sorted() — returns new list (original unchanged)
original = [3, 1, 4, 1, 5]
new_sorted = sorted(original)  # [1, 1, 3, 4, 5]

# Custom key
words = ["banana", "cherry", "apple", "date"]
sorted(words, key=len)                    # ['date', 'apple', 'banana', 'cherry']
sorted(words, key=lambda w: w[-1])        # sort by last character

# Stable sort — equal elements maintain original order
# Timsort (Python's sort) is stable — critical for multi-key sorting
students = [("Alice", 85), ("Bob", 92), ("Charlie", 85)]
sorted(students, key=lambda s: s[1])
# [('Alice', 85), ('Charlie', 85), ('Bob', 92)] — Alice before Charlie (stable)

# Multi-key sort with tuple key
data = [("Alice", 85), ("Bob", 92), ("Charlie", 85), ("Diana", 92)]
sorted(data, key=lambda s: (-s[1], s[0]))
# [('Bob', 92), ('Diana', 92), ('Alice', 85), ('Charlie', 85)]
```

---

## 2.2 Dictionaries — Deep Dive

Dictionaries are Python's hash map implementation. Under the hood, they use open addressing with a compact hash table (since Python 3.6, they maintain insertion order; this became guaranteed in 3.7).

### Time Complexity

| Operation | Average | Worst | Notes |
|-----------|---------|-------|-------|
| `d[key]` | O(1) | O(n) | Hash collision worst case |
| `d[key] = val` | O(1)* | O(n) | Amortized O(1) |
| `key in d` | O(1) | O(n) | Hash lookup |
| `del d[key]` | O(1) | O(n) | |
| `d.keys()` | O(1) | O(1) | Returns a view (not a copy) |
| `len(d)` | O(1) | O(1) | |
| Iteration | O(n) | O(n) | |

### Essential Operations

```python
d = {"a": 1, "b": 2, "c": 3}

# Safe access
d.get("x", 0)            # 0 (default if key missing)
d.setdefault("x", 10)    # Sets d["x"]=10 if missing, returns value

# Iteration
for key in d:               pass   # keys
for val in d.values():      pass   # values
for key, val in d.items():  pass   # both

# Remove
d.pop("a")         # returns 1, removes key "a"
d.pop("z", None)   # returns None if key doesn't exist (no KeyError)

# Merge (Python 3.9+)
d1 = {"a": 1}
d2 = {"b": 2}
d1 | d2            # {'a': 1, 'b': 2}
d1 |= d2           # in-place merge
```

### collections Module: Specialized Dicts

```python
from collections import defaultdict, OrderedDict, Counter, ChainMap

# defaultdict — auto-initializes missing keys
word_count = defaultdict(int)    # default value: 0
for word in "the cat sat on the mat".split():
    word_count[word] += 1
# defaultdict(<class 'int'>, {'the': 2, 'cat': 1, 'sat': 1, 'on': 1, 'mat': 1})

# Group items
groups = defaultdict(list)
data = [("fruit", "apple"), ("fruit", "banana"), ("veg", "carrot")]
for category, item in data:
    groups[category].append(item)
# {'fruit': ['apple', 'banana'], 'veg': ['carrot']}

# Counter — frequency counting powerhouse
text = "abracadabra"
counter = Counter(text)
# Counter({'a': 5, 'b': 2, 'r': 2, 'c': 1, 'd': 1})

counter.most_common(2)     # [('a', 5), ('b', 2)]
counter["a"]               # 5
counter["z"]               # 0 (no KeyError!)

# Counter arithmetic
c1 = Counter(a=3, b=1)
c2 = Counter(a=1, b=2)
c1 + c2     # Counter({'a': 4, 'b': 3})
c1 - c2     # Counter({'a': 2})  — drops zero/negative counts

# OrderedDict — preserves insertion order (pre-3.7) + move_to_end / equality
from collections import OrderedDict
od = OrderedDict()
od["first"] = 1
od["second"] = 2
od.move_to_end("first")   # move to end
# OrderedDict matters for == comparison: order matters for OrderedDict, not for regular dict

# ChainMap — searches multiple dicts in order
defaults = {"color": "blue", "size": "medium"}
user_prefs = {"color": "red"}
config = ChainMap(user_prefs, defaults)
config["color"]   # "red"  (found in user_prefs first)
config["size"]    # "medium" (falls through to defaults)
```

---

## 2.3 Sets

Sets are hash-based collections with O(1) average lookup. They are the answer to any problem involving uniqueness or membership testing.

```python
a = {1, 2, 3, 4, 5}
b = {4, 5, 6, 7, 8}

# Set operations
a | b              # Union:        {1, 2, 3, 4, 5, 6, 7, 8}
a & b              # Intersection: {4, 5}
a - b              # Difference:   {1, 2, 3}
a ^ b              # Symmetric difference: {1, 2, 3, 6, 7, 8}
a <= b             # Subset check: False
{1, 2} <= a        # True — {1,2} is a subset of a

# In-place operations
a |= b             # a.update(b)
a &= b             # a.intersection_update(b)
a -= b             # a.difference_update(b)

# Practical use cases
# 1. Deduplication
unique = list(set([1, 2, 2, 3, 3, 3]))  # [1, 2, 3]

# 2. Fast membership testing
valid_ids = {1001, 1002, 1003, 1004}
if user_id in valid_ids:    # O(1) vs O(n) for list
    process(user_id)

# 3. Finding common/different elements
set1 = set(list1)
set2 = set(list2)
common = set1 & set2
only_in_first = set1 - set2
```

| Operation | Average | Method Form |
|-----------|---------|-------------|
| `x in s` | O(1) | — |
| `s.add(x)` | O(1) | — |
| `s.remove(x)` | O(1) | KeyError if missing |
| `s.discard(x)` | O(1) | No error if missing |
| `s \| t` | O(len(s) + len(t)) | `s.union(t)` |
| `s & t` | O(min(len(s), len(t))) | `s.intersection(t)` |
| `s - t` | O(len(s)) | `s.difference(t)` |

---

## 2.4 Stacks, Queues, and Heaps

```python
from collections import deque
import heapq

# ─── STACK (LIFO) ─── Use list or deque
stack = []
stack.append(1)     # push
stack.append(2)
stack.append(3)
stack.pop()         # 3 — LIFO

# ─── QUEUE (FIFO) ─── Use deque (NOT list — list.pop(0) is O(n))
queue = deque()
queue.append(1)     # enqueue (right)
queue.append(2)
queue.append(3)
queue.popleft()     # 1 — FIFO, O(1)

# deque also supports maxlen for bounded buffers
recent = deque(maxlen=3)
for i in range(5):
    recent.append(i)
# recent = deque([2, 3, 4], maxlen=3) — auto-evicts oldest

# ─── HEAP (Priority Queue) ─── heapq is a MIN-heap
nums = [5, 3, 8, 1, 9, 2]
heapq.heapify(nums)          # in-place, O(n)
heapq.heappush(nums, 0)      # push
smallest = heapq.heappop(nums)  # 0 — pops smallest

# Top-k elements
top_3 = heapq.nlargest(3, nums)    # [9, 8, 5]
bottom_3 = heapq.nsmallest(3, nums) # [1, 2, 3]

# Max-heap trick: negate values
max_heap = []
heapq.heappush(max_heap, -5)
heapq.heappush(max_heap, -1)
heapq.heappush(max_heap, -9)
largest = -heapq.heappop(max_heap)  # 9
```

| Structure | Push | Pop | Peek | Use Case |
|-----------|------|-----|------|----------|
| Stack (`list`) | O(1)* | O(1) | O(1) `[-1]` | Undo, DFS, expression parsing |
| Queue (`deque`) | O(1) | O(1) | O(1) `[0]` | BFS, task scheduling |
| Min-Heap (`heapq`) | O(log n) | O(log n) | O(1) `[0]` | Priority queue, top-k |

---

## 2.5 Named Tuples and Dataclasses

```python
# ─── NamedTuple ─── immutable, lightweight
from collections import namedtuple

Point = namedtuple("Point", ["x", "y"])
p = Point(3, 4)
p.x           # 3
p.y           # 4
# p.x = 5    # AttributeError — immutable

# Typed version (Python 3.6+)
from typing import NamedTuple

class Point(NamedTuple):
    x: float
    y: float
    label: str = "origin"

p = Point(3.0, 4.0)
p.label     # "origin"

# ─── Dataclass ─── mutable by default, auto-generates __init__, __repr__, __eq__
from dataclasses import dataclass, field

@dataclass
class Employee:
    name: str
    role: str
    salary: float
    skills: list = field(default_factory=list)  # mutable default done right

e = Employee("Rahul", "Data Scientist", 150_000)
e.skills.append("Python")
print(e)  # Employee(name='Rahul', role='Data Scientist', salary=150000, skills=['Python'])

# Frozen dataclass — immutable, hashable
@dataclass(frozen=True)
class Config:
    model: str
    lr: float
    epochs: int

c = Config("ResNet", 0.001, 50)
# c.lr = 0.01  # FrozenInstanceError

# Can use as dict key or set element because it's hashable
configs = {c: "experiment_1"}

# dataclass with ordering
@dataclass(order=True)
class Student:
    gpa: float
    name: str = field(compare=False)  # excluded from comparisons

students = [Student(3.5, "Alice"), Student(3.9, "Bob"), Student(3.5, "Charlie")]
sorted(students)  # sorted by gpa only
```

### When to Use What

| Feature | `namedtuple` | `dataclass` | `dict` |
|---------|-------------|-------------|--------|
| Immutable | Yes | Optional (`frozen=True`) | No |
| Type hints | Optional | Yes | No |
| Default values | Yes | Yes | N/A |
| Methods | Limited | Full class | N/A |
| Memory | Smallest | Medium | Largest |
| Hashable | Yes | If frozen | No |
| Use case | Simple records, function returns | Domain models, configs | Flexible data |

---

# 3. Object-Oriented Programming

---

## 3.1 Classes, `__init__`, and `self`

```python
class DataPipeline:
    """A simple data processing pipeline."""
    
    # Class variable — shared across all instances
    version = "1.0"
    _instance_count = 0
    
    def __init__(self, name: str, steps: list[str] | None = None):
        # Instance variables — unique to each instance
        self.name = name
        self.steps = steps or []
        self._created_at = datetime.now()    # "private" by convention
        DataPipeline._instance_count += 1
    
    def add_step(self, step: str) -> None:
        self.steps.append(step)
    
    def run(self) -> dict:
        results = {}
        for step in self.steps:
            results[step] = f"Completed: {step}"
        return results
    
    def __repr__(self) -> str:
        return f"DataPipeline(name='{self.name}', steps={self.steps})"
    
    def __len__(self) -> int:
        return len(self.steps)

pipeline = DataPipeline("ETL", ["extract", "transform"])
pipeline.add_step("load")
len(pipeline)  # 3
```

> **Key concept:** `self` is the instance reference. Python passes it automatically — it is not a keyword, just a strong convention. Every instance method receives `self` as its first argument.

---

## 3.2 Inheritance, Multiple Inheritance, and MRO

```python
class BaseModel:
    def __init__(self, name: str):
        self.name = name
    
    def predict(self, X):
        raise NotImplementedError("Subclasses must implement predict()")
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"

class LinearModel(BaseModel):
    def __init__(self, name: str, regularization: float = 0.01):
        super().__init__(name)   # Call parent __init__
        self.regularization = regularization
    
    def predict(self, X):
        return [sum(row) for row in X]  # placeholder

class TreeModel(BaseModel):
    def __init__(self, name: str, max_depth: int = 5):
        super().__init__(name)
        self.max_depth = max_depth
    
    def predict(self, X):
        return [0] * len(X)  # placeholder

# Multiple inheritance
class EnsembleModel(LinearModel, TreeModel):
    def __init__(self, name: str):
        super().__init__(name)
    
    def predict(self, X):
        linear_pred = LinearModel.predict(self, X)
        tree_pred = TreeModel.predict(self, X)
        return [(l + t) / 2 for l, t in zip(linear_pred, tree_pred)]
```

### Method Resolution Order (MRO)

MRO determines the order Python searches for methods in a class hierarchy. Python uses the **C3 Linearization** algorithm.

```python
print(EnsembleModel.__mro__)
# (<class 'EnsembleModel'>, <class 'LinearModel'>, <class 'TreeModel'>, 
#  <class 'BaseModel'>, <class 'object'>)

# super() follows MRO, not just the direct parent
# This is why super() works correctly even with diamond inheritance
```

### The Diamond Problem

```
       BaseModel
       /       \
  LinearModel  TreeModel
       \       /
    EnsembleModel
```

Python's C3 linearization ensures each class appears exactly once in the MRO and `super()` calls chain correctly through all parents.

---

## 3.3 Encapsulation and @property

```python
class Temperature:
    def __init__(self, celsius: float):
        self._celsius = celsius  # "protected" by convention
    
    @property
    def celsius(self) -> float:
        """Getter — accessed like an attribute, not a method."""
        return self._celsius
    
    @celsius.setter
    def celsius(self, value: float) -> None:
        """Setter — with validation."""
        if value < -273.15:
            raise ValueError("Temperature below absolute zero!")
        self._celsius = value
    
    @property
    def fahrenheit(self) -> float:
        """Computed property — derived from celsius."""
        return self._celsius * 9/5 + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value: float) -> None:
        self._celsius = (value - 32) * 5/9

t = Temperature(100)
t.celsius        # 100      — uses getter
t.fahrenheit     # 212.0    — computed property
t.celsius = 0    # uses setter with validation
t.fahrenheit     # 32.0
# t.celsius = -300  # ValueError
```

### Naming Conventions for Access Control

| Convention | Meaning | Enforcement |
|-----------|---------|-------------|
| `name` | Public | None |
| `_name` | Protected (internal use) | Convention only |
| `__name` | Private (name-mangled) | Compiler rewrites to `_ClassName__name` |
| `__name__` | Dunder / magic method | Reserved for Python |

```python
class MyClass:
    def __init__(self):
        self.public = 1
        self._protected = 2
        self.__private = 3

obj = MyClass()
obj.public       # 1
obj._protected   # 2  — accessible but "please don't"
# obj.__private  # AttributeError
obj._MyClass__private  # 3  — name mangling, accessible but ugly
```

---

## 3.4 Polymorphism and Dunder Methods

Dunder (double underscore) methods let your objects work with Python's built-in operations and protocols.

```python
class Vector:
    def __init__(self, *components):
        self._components = tuple(components)
    
    # ─── String representations ───
    def __repr__(self) -> str:
        """Unambiguous representation (for developers)."""
        return f"Vector({', '.join(map(str, self._components))})"
    
    def __str__(self) -> str:
        """User-friendly representation."""
        return f"({', '.join(map(str, self._components))})"
    
    # ─── Container protocol ───
    def __len__(self) -> int:
        return len(self._components)
    
    def __getitem__(self, index):
        return self._components[index]
    
    def __iter__(self):
        return iter(self._components)
    
    def __contains__(self, item):
        return item in self._components
    
    # ─── Arithmetic ───
    def __add__(self, other: "Vector") -> "Vector":
        if len(self) != len(other):
            raise ValueError("Vectors must have same dimension")
        return Vector(*(a + b for a, b in zip(self, other)))
    
    def __mul__(self, scalar: float) -> "Vector":
        return Vector(*(x * scalar for x in self))
    
    def __rmul__(self, scalar: float) -> "Vector":
        return self * scalar  # supports 3 * v
    
    # ─── Comparison ───
    def __eq__(self, other) -> bool:
        return self._components == other._components
    
    def __hash__(self) -> int:
        return hash(self._components)
    
    # ─── Magnitude ───
    def __abs__(self) -> float:
        return sum(x**2 for x in self) ** 0.5
    
    def __bool__(self) -> bool:
        return abs(self) > 0

v1 = Vector(1, 2, 3)
v2 = Vector(4, 5, 6)
v1 + v2         # Vector(5, 7, 9)
v1 * 3          # Vector(3, 6, 9)
3 * v1          # Vector(3, 6, 9) — thanks to __rmul__
len(v1)         # 3
v1[0]           # 1
abs(v1)         # 3.7416...
2 in v1         # True
```

### Key Dunder Methods Reference

| Category | Methods | Triggered By |
|----------|---------|--------------|
| Construction | `__init__`, `__new__`, `__del__` | `obj = Class()`, garbage collection |
| String | `__repr__`, `__str__`, `__format__` | `repr()`, `str()`, `f"{obj}"` |
| Container | `__len__`, `__getitem__`, `__setitem__`, `__contains__`, `__iter__` | `len()`, `obj[i]`, `in` |
| Arithmetic | `__add__`, `__sub__`, `__mul__`, `__truediv__` | `+`, `-`, `*`, `/` |
| Comparison | `__eq__`, `__lt__`, `__le__`, `__gt__`, `__ge__` | `==`, `<`, `<=`, `>`, `>=` |
| Callable | `__call__` | `obj()` |
| Context | `__enter__`, `__exit__` | `with obj:` |
| Attribute | `__getattr__`, `__setattr__`, `__delattr__` | `obj.x`, `obj.x = v` |
| Hash | `__hash__` | `hash(obj)`, dict keys, set elements |

---

## 3.5 Abstract Classes and Protocols

```python
from abc import ABC, abstractmethod

class Transformer(ABC):
    """Abstract base class for data transformers."""
    
    @abstractmethod
    def fit(self, X):
        """Learn parameters from data."""
        ...
    
    @abstractmethod
    def transform(self, X):
        """Transform data using learned parameters."""
        ...
    
    def fit_transform(self, X):
        """Convenience method (concrete — uses abstract methods)."""
        self.fit(X)
        return self.transform(X)

# Cannot instantiate ABC directly
# t = Transformer()  # TypeError: Can't instantiate abstract class

class StandardScaler(Transformer):
    def fit(self, X):
        self.mean_ = sum(X) / len(X)
        self.std_ = (sum((x - self.mean_)**2 for x in X) / len(X)) ** 0.5
        return self
    
    def transform(self, X):
        return [(x - self.mean_) / self.std_ for x in X]

scaler = StandardScaler()
scaler.fit_transform([1, 2, 3, 4, 5])  # Works!
```

### Protocols (Structural Subtyping) — Python 3.8+

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Predictable(Protocol):
    """Any class with a predict method satisfies this protocol."""
    def predict(self, X: list) -> list: ...

class MyModel:
    def predict(self, X: list) -> list:
        return [0] * len(X)

# No inheritance needed — structural ("duck") typing
isinstance(MyModel(), Predictable)  # True at runtime
```

---

## 3.6 Decorators: @staticmethod, @classmethod, Custom

```python
class MathUtils:
    precision = 2
    
    @staticmethod
    def is_prime(n: int) -> bool:
        """No access to class or instance — pure utility."""
        if n < 2: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0: return False
        return True
    
    @classmethod
    def set_precision(cls, p: int) -> None:
        """Access to the class (cls), not a specific instance."""
        cls.precision = p
    
    @classmethod
    def from_string(cls, s: str) -> "MathUtils":
        """Alternative constructor pattern."""
        return cls()

MathUtils.is_prime(17)       # True — no instance needed
MathUtils.set_precision(4)   # Modifies class variable
```

### Custom Decorators

```python
import time
from functools import wraps

def timer(func):
    """Measure execution time of a function."""
    @wraps(func)  # preserves func.__name__, __doc__, etc.
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

@timer
def slow_sum(n):
    """Sum numbers from 0 to n."""
    return sum(range(n))

slow_sum(10_000_000)
# slow_sum took 0.2134s

# Decorator with arguments — needs an extra wrapper layer
def retry(max_attempts: int = 3, delay: float = 1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        raise
                    print(f"Attempt {attempt} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.5)
def fetch_data(url: str):
    """Fetch data with automatic retry."""
    ...
```

---

# 4. Functional Programming

---

## 4.1 map, filter, reduce

```python
from functools import reduce

# map — apply function to every element
nums = [1, 2, 3, 4, 5]
doubled = list(map(lambda x: x * 2, nums))    # [2, 4, 6, 8, 10]
# Preferred: [x * 2 for x in nums]

# filter — keep elements where function returns True
evens = list(filter(lambda x: x % 2 == 0, nums))  # [2, 4]
# Preferred: [x for x in nums if x % 2 == 0]

# reduce — accumulate values (left-to-right)
total = reduce(lambda acc, x: acc + x, nums, 0)  # 15
product = reduce(lambda acc, x: acc * x, nums, 1) # 120

# Practical reduce: flatten nested list
nested = [[1, 2], [3, 4], [5, 6]]
flat = reduce(lambda acc, lst: acc + lst, nested, [])
# [1, 2, 3, 4, 5, 6]
# Better: [x for sublist in nested for x in sublist]

# map with multiple iterables
list(map(lambda x, y: x + y, [1, 2, 3], [10, 20, 30]))
# [11, 22, 33]
```

> **Interview tip:** In practice, list comprehensions are preferred over `map`/`filter` for readability. Know both styles — interviewers sometimes specifically ask about the functional approach.

---

## 4.2 Lambda Functions

```python
# Lambda is an anonymous, single-expression function
square = lambda x: x ** 2
square(5)  # 25

# Common use cases: sort keys, callbacks
pairs = [(1, "one"), (3, "three"), (2, "two")]
sorted(pairs, key=lambda p: p[1])  # sort by second element

# Immediately invoked
(lambda x, y: x + y)(3, 4)  # 7

# Lambda limitations: single expression only, no statements
# NO:  lambda x: if x > 0: return x  (syntax error)
# YES: lambda x: x if x > 0 else 0   (ternary expression)
```

---

## 4.3 functools

```python
from functools import lru_cache, partial, wraps, reduce, cached_property

# ─── lru_cache: memoization ───
@lru_cache(maxsize=128)
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

fibonacci(100)  # Instant — without cache this would be impossibly slow
fibonacci.cache_info()  # CacheInfo(hits=98, misses=101, maxsize=128, currsize=101)
fibonacci.cache_clear()

# Python 3.9+: @cache is lru_cache with unlimited maxsize
from functools import cache

@cache
def expensive_computation(x):
    ...

# ─── partial: fix some arguments ───
from functools import partial

def power(base, exponent):
    return base ** exponent

square = partial(power, exponent=2)
cube = partial(power, exponent=3)

square(5)  # 25
cube(3)    # 27

# Practical: partial with pandas
import pandas as pd
round_2 = partial(round, ndigits=2)
# df["col"] = df["col"].apply(round_2)

# ─── wraps: preserve function metadata in decorators ───
# (See custom decorator example in section 3.6)
# Without @wraps, the decorated function loses its __name__, __doc__, etc.

# ─── cached_property (Python 3.8+) ───
class DataLoader:
    @cached_property
    def data(self):
        """Computed once, then cached as an instance attribute."""
        print("Loading data...")
        return pd.read_csv("large_file.csv")
```

---

## 4.4 itertools

```python
import itertools

# ─── chain: flatten iterables ───
list(itertools.chain([1, 2], [3, 4], [5, 6]))
# [1, 2, 3, 4, 5, 6]

list(itertools.chain.from_iterable([[1, 2], [3, 4], [5, 6]]))
# [1, 2, 3, 4, 5, 6]

# ─── combinations and permutations ───
list(itertools.combinations("ABCD", 2))
# [('A','B'), ('A','C'), ('A','D'), ('B','C'), ('B','D'), ('C','D')]

list(itertools.permutations("ABC", 2))
# [('A','B'), ('A','C'), ('B','A'), ('B','C'), ('C','A'), ('C','B')]

list(itertools.combinations_with_replacement("AB", 2))
# [('A','A'), ('A','B'), ('B','B')]

# ─── product: Cartesian product ───
list(itertools.product("AB", [1, 2]))
# [('A',1), ('A',2), ('B',1), ('B',2)]

# Grid search style
params = {"lr": [0.01, 0.001], "batch_size": [32, 64]}
for combo in itertools.product(*params.values()):
    config = dict(zip(params.keys(), combo))
    print(config)
# {'lr': 0.01, 'batch_size': 32}
# {'lr': 0.01, 'batch_size': 64}
# {'lr': 0.001, 'batch_size': 32}
# {'lr': 0.001, 'batch_size': 64}

# ─── groupby: group consecutive elements ───
from itertools import groupby

data = [("fruit", "apple"), ("fruit", "banana"), ("veg", "carrot"), ("veg", "pea")]
# IMPORTANT: data must be sorted by key first!
for key, group in groupby(data, key=lambda x: x[0]):
    print(key, list(group))
# fruit [('fruit', 'apple'), ('fruit', 'banana')]
# veg   [('fruit', 'carrot'), ('fruit', 'pea')]

# ─── islice: slice any iterable (including generators) ───
from itertools import islice

gen = (x**2 for x in range(1000))
first_5 = list(islice(gen, 5))  # [0, 1, 4, 9, 16]

# ─── accumulate: running accumulation ───
from itertools import accumulate
import operator

list(accumulate([1, 2, 3, 4, 5]))                         # [1, 3, 6, 10, 15]
list(accumulate([1, 2, 3, 4, 5], operator.mul))            # [1, 2, 6, 24, 120]
list(accumulate([3, 1, 4, 1, 5], max))                     # [3, 3, 4, 4, 5] (running max)

# ─── count, cycle, repeat: infinite iterators ───
counter = itertools.count(start=10, step=2)
[next(counter) for _ in range(5)]  # [10, 12, 14, 16, 18]

cycler = itertools.cycle("ABC")
[next(cycler) for _ in range(7)]  # ['A', 'B', 'C', 'A', 'B', 'C', 'A']

list(itertools.repeat("x", 3))    # ['x', 'x', 'x']

# ─── zip_longest: zip with fill value ───
list(itertools.zip_longest([1, 2, 3], [10, 20], fillvalue=0))
# [(1, 10), (2, 20), (3, 0)]
```

---

# 5. Advanced Concepts

---

## 5.1 Generators and `yield`

Generators are functions that produce a sequence of values lazily — one value at a time, suspending state between calls. They are fundamental to memory-efficient data processing.

```python
# Generator function — contains yield
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# Usage
fib = fibonacci()
[next(fib) for _ in range(10)]
# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

# Generator for large file processing — O(1) memory
def read_large_csv(filepath: str):
    """Process a file line by line without loading it all into memory."""
    with open(filepath) as f:
        header = next(f).strip().split(",")
        for line in f:
            values = line.strip().split(",")
            yield dict(zip(header, values))

# Chaining generators (pipeline pattern)
def parse_records(lines):
    for line in lines:
        yield line.strip().split(",")

def filter_valid(records):
    for record in records:
        if len(record) == 3 and record[0].strip():
            yield record

def to_dict(records, keys):
    for record in records:
        yield dict(zip(keys, record))

# Pipeline: nothing executes until iteration
lines = open("data.csv")
pipeline = to_dict(filter_valid(parse_records(lines)), ["name", "age", "city"])
for row in pipeline:  # Only now does processing happen, one row at a time
    process(row)
```

### `yield from` — Delegating to Sub-generators

```python
def flatten(nested):
    """Recursively flatten any nested iterable."""
    for item in nested:
        if isinstance(item, (list, tuple)):
            yield from flatten(item)   # delegate to recursive call
        else:
            yield item

list(flatten([1, [2, [3, 4], 5], [6, 7]]))
# [1, 2, 3, 4, 5, 6, 7]
```

### `send()` — Two-Way Communication

```python
def running_average():
    """Generator that accepts values and yields running average."""
    total = 0
    count = 0
    average = None
    while True:
        value = yield average
        if value is not None:
            total += value
            count += 1
            average = total / count

avg = running_average()
next(avg)           # Prime the generator (returns None)
avg.send(10)        # 10.0
avg.send(20)        # 15.0
avg.send(30)        # 20.0
```

### Memory Comparison

| Approach | Memory | Use Case |
|----------|--------|----------|
| `[x**2 for x in range(10**7)]` | ~80 MB | Need random access, reuse |
| `(x**2 for x in range(10**7))` | ~120 bytes | Single pass, streaming |

> **Interview tip:** When asked "How would you process a 100GB file in Python?", the answer is generators. Read line by line (`for line in file`), process with generator pipelines, never load the whole file.

---

## 5.2 Context Managers

Context managers ensure proper resource cleanup — file handles, database connections, locks, temporary state.

```python
# ─── Using with statement ───
with open("data.txt", "r") as f:
    content = f.read()
# File is guaranteed to be closed here, even if an exception occurred

# ─── Class-based context manager ───
class Timer:
    """Context manager to time a block of code."""
    
    def __enter__(self):
        self.start = time.perf_counter()
        return self  # the value bound by `as`
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start
        print(f"Elapsed: {self.elapsed:.4f}s")
        return False  # don't suppress exceptions

with Timer() as t:
    sum(range(10_000_000))
# Elapsed: 0.1234s

# ─── Generator-based context manager (simpler) ───
from contextlib import contextmanager

@contextmanager
def database_connection(url: str):
    """Manage database connection lifecycle."""
    conn = connect(url)      # setup
    try:
        yield conn           # hand control to the with block
    except Exception:
        conn.rollback()      # cleanup on error
        raise
    finally:
        conn.close()         # always cleanup

with database_connection("postgres://...") as conn:
    conn.execute("SELECT ...")

# ─── Suppress specific exceptions ───
from contextlib import suppress

with suppress(FileNotFoundError):
    os.remove("temp_file.txt")  # no error if file doesn't exist

# ─── Temporary directory ───
import tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    # tmpdir is a string path to a temp directory
    # automatically deleted when with block exits
    pass

# ─── Multiple context managers ───
with open("in.txt") as fin, open("out.txt", "w") as fout:
    for line in fin:
        fout.write(line.upper())
```

---

## 5.3 Decorators — How They Work

A decorator is a function that takes a function and returns a modified function. The `@decorator` syntax is syntactic sugar.

```python
# These are equivalent:
@timer
def my_func():
    pass

# Same as:
def my_func():
    pass
my_func = timer(my_func)
```

### Chaining Decorators

```python
def bold(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return f"<b>{func(*args, **kwargs)}</b>"
    return wrapper

def italic(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return f"<i>{func(*args, **kwargs)}</i>"
    return wrapper

@bold
@italic
def greet(name):
    return f"Hello, {name}"

greet("Rahul")  # "<b><i>Hello, Rahul</i></b>"

# Execution order: bold(italic(greet)) — italic applied first, bold wraps the result
```

### Decorator with Arguments

```python
def validate_types(**type_hints):
    """Decorator that validates argument types at runtime."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            for param_name, expected_type in type_hints.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    if not isinstance(value, expected_type):
                        raise TypeError(
                            f"{param_name} must be {expected_type.__name__}, "
                            f"got {type(value).__name__}"
                        )
            return func(*args, **kwargs)
        return wrapper
    return decorator

@validate_types(x=int, y=int)
def add(x, y):
    return x + y

add(1, 2)      # 3
# add("1", 2)  # TypeError: x must be int, got str
```

### Class-Based Decorator

```python
class CountCalls:
    """Decorator that counts how many times a function is called."""
    
    def __init__(self, func):
        self.func = func
        self.count = 0
        wraps(func)(self)  # preserve metadata
    
    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"{self.func.__name__} called {self.count} times")
        return self.func(*args, **kwargs)

@CountCalls
def process():
    pass

process()  # process called 1 times
process()  # process called 2 times
process.count  # 2
```

---

## 5.4 Type Hints

```python
from typing import (
    Optional, Union, List, Dict, Tuple, Set,
    Any, Callable, Iterator, Generator,
    TypeVar, Generic, Protocol, Literal
)

# Basic type hints
def greet(name: str) -> str:
    return f"Hello, {name}"

# Optional — value can be None
def find_user(user_id: int) -> Optional[dict]:
    """Returns None if user not found."""
    ...

# Union — multiple possible types
def process(data: Union[str, bytes]) -> str:
    ...

# Python 3.10+ pipe syntax
def process(data: str | bytes | None) -> str:
    ...

# Collection types (Python 3.9+: use built-in types directly)
def analyze(
    items: list[str],               # was List[str]
    counts: dict[str, int],         # was Dict[str, int]
    pair: tuple[float, float],      # fixed-length tuple
    records: tuple[str, ...],       # variable-length tuple
    unique: set[int],               # was Set[int]
) -> None:
    ...

# Callable
def apply_func(
    data: list[int],
    transform: Callable[[int], float]  # takes int, returns float
) -> list[float]:
    return [transform(x) for x in data]

# TypeVar for generics
T = TypeVar("T")

def first(items: list[T]) -> T:
    return items[0]

# Literal — restrict to specific values
def set_mode(mode: Literal["train", "eval", "test"]) -> None:
    ...

# TypeAlias (Python 3.10+)
type Vector = list[float]
type Matrix = list[Vector]

# Generic class
class Stack(Generic[T]):
    def __init__(self) -> None:
        self._items: list[T] = []
    
    def push(self, item: T) -> None:
        self._items.append(item)
    
    def pop(self) -> T:
        return self._items.pop()

stack: Stack[int] = Stack()
stack.push(42)
```

> **Interview tip:** Type hints are not enforced at runtime — they are for documentation, IDE support, and static analysis tools (`mypy`, `pyright`). In data science, they massively improve code quality in production systems.

---

## 5.5 Async/Await

```python
import asyncio
import aiohttp

# Coroutine — defined with async def
async def fetch_url(session: aiohttp.ClientSession, url: str) -> str:
    async with session.get(url) as response:
        return await response.text()

# Running multiple requests concurrently
async def fetch_all(urls: list[str]) -> list[str]:
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)  # run concurrently
        return results

# Entry point
urls = ["https://api.example.com/1", "https://api.example.com/2"]
results = asyncio.run(fetch_all(urls))

# Async generator
async def stream_data(url: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            async for chunk in response.content.iter_chunks():
                yield chunk

# Async context manager
class AsyncDBConnection:
    async def __aenter__(self):
        self.conn = await connect_to_db()
        return self.conn
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.conn.close()

# asyncio.gather vs asyncio.wait
# gather: run all, return results in order, fails if any task fails
# wait:   run all, return (done, pending) sets, more control over completion

# Semaphore for rate limiting
async def rate_limited_fetch(urls, max_concurrent=5):
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def bounded_fetch(session, url):
        async with semaphore:
            return await fetch_url(session, url)
    
    async with aiohttp.ClientSession() as session:
        tasks = [bounded_fetch(session, url) for url in urls]
        return await asyncio.gather(*tasks)
```

### Sync vs Async Decision

| Scenario | Use |
|----------|-----|
| CPU-bound work (math, ML training) | `multiprocessing` or sync |
| I/O-bound, few operations | Sync is fine |
| I/O-bound, many operations | `asyncio` |
| Web API backend | `asyncio` (FastAPI) |
| Data pipeline with API calls | `asyncio` |

---

## 5.6 GIL (Global Interpreter Lock)

The GIL is a mutex that protects access to Python objects, preventing multiple native threads from executing Python bytecodes simultaneously in CPython.

### What It Means

```
┌─────────────────────────────────────────────┐
│              CPython Process                 │
│                                              │
│   Thread 1  ──┐     ┌──  Thread 2           │
│               │ GIL │                        │
│   Thread 3  ──┘     └──  Thread 4           │
│                                              │
│   Only ONE thread executes Python            │
│   bytecode at a time                         │
└─────────────────────────────────────────────┘
```

### Key Facts

| Aspect | Detail |
|--------|--------|
| What | Mutex in CPython that allows only one thread to execute Python bytecode at a time |
| Why it exists | Simplifies memory management (reference counting), makes C extensions easier |
| Impact | CPU-bound multithreading doesn't achieve true parallelism |
| Doesn't affect | I/O-bound operations (GIL is released during I/O), multiprocessing, C extensions (NumPy) |
| Workarounds | `multiprocessing`, C extensions, `asyncio`, `concurrent.futures` |
| Other implementations | Jython, PyPy (has GIL), IronPython — no GIL. CPython 3.13+ has experimental `--disable-gil` |

```python
import threading
import multiprocessing
import time

def cpu_task(n):
    """CPU-bound work."""
    total = 0
    for i in range(n):
        total += i * i
    return total

# Threading — NO speedup for CPU-bound (GIL)
start = time.time()
threads = [threading.Thread(target=cpu_task, args=(10**7,)) for _ in range(4)]
for t in threads: t.start()
for t in threads: t.join()
print(f"Threading: {time.time() - start:.2f}s")  # ~4s (sequential!)

# Multiprocessing — real parallelism
start = time.time()
with multiprocessing.Pool(4) as pool:
    results = pool.map(cpu_task, [10**7] * 4)
print(f"Multiprocessing: {time.time() - start:.2f}s")  # ~1s (parallel!)
```

---

## 5.7 Threading vs Multiprocessing

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

# ─── ThreadPoolExecutor: I/O-bound tasks ───
def download(url: str) -> str:
    """Simulate network I/O."""
    time.sleep(1)  # GIL is released during sleep/I/O
    return f"Downloaded {url}"

urls = [f"https://api.example.com/{i}" for i in range(10)]

with ThreadPoolExecutor(max_workers=5) as executor:
    results = list(executor.map(download, urls))
# ~2s instead of 10s (5 concurrent threads)

# ─── ProcessPoolExecutor: CPU-bound tasks ───
def heavy_computation(n: int) -> int:
    """CPU-bound work."""
    return sum(i * i for i in range(n))

with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(heavy_computation, [10**7] * 4))
# ~1s instead of 4s (4 parallel processes)

# ─── submit + as_completed: handle results as they finish ───
from concurrent.futures import as_completed

with ThreadPoolExecutor(max_workers=5) as executor:
    future_to_url = {executor.submit(download, url): url for url in urls}
    
    for future in as_completed(future_to_url):
        url = future_to_url[future]
        try:
            result = future.result()
            print(f"{url}: {result}")
        except Exception as e:
            print(f"{url} failed: {e}")
```

### Decision Matrix

| Criteria | Threading | Multiprocessing | AsyncIO |
|----------|-----------|-----------------|---------|
| Best for | I/O-bound | CPU-bound | High-concurrency I/O |
| Parallelism | Concurrent, not parallel (GIL) | True parallel | Concurrent, single thread |
| Memory | Shared (lightweight) | Separate (heavier) | Shared (lightest) |
| Overhead | Low | High (process creation) | Minimal |
| Data sharing | Easy (shared memory) | Needs IPC (Queue, Pipe) | Easy (single thread) |
| GIL impact | Limited by GIL | Bypasses GIL | Not affected |
| Complexity | Medium | Medium | Higher (async everywhere) |

---

# 6. Data Science Python

---

## 6.1 NumPy

NumPy is the foundation of scientific computing in Python. Its arrays are stored in contiguous memory blocks, enabling vectorized operations that are 10-100x faster than Python loops.

```python
import numpy as np

# ─── Array creation ───
a = np.array([1, 2, 3, 4, 5])                    # from list
z = np.zeros((3, 4))                               # 3x4 zeros
o = np.ones((2, 3))                                # 2x3 ones
r = np.arange(0, 10, 0.5)                          # like range but for floats
l = np.linspace(0, 1, 5)                           # [0, 0.25, 0.5, 0.75, 1.0]
e = np.eye(3)                                       # 3x3 identity matrix
rand = np.random.randn(3, 3)                        # 3x3 standard normal
rand_int = np.random.randint(0, 100, size=(3, 4))  # 3x4 random integers

# ─── Array properties ───
a = np.array([[1, 2, 3], [4, 5, 6]])
a.shape    # (2, 3)
a.ndim     # 2
a.dtype    # int64
a.size     # 6
a.nbytes   # 48 (6 elements × 8 bytes)
```

### Broadcasting

Broadcasting allows NumPy to perform operations on arrays of different shapes without explicitly copying data.

```python
# Rule: dimensions are compared right-to-left
# Compatible if: (1) equal, or (2) one of them is 1

a = np.array([[1, 2, 3],
              [4, 5, 6]])     # shape (2, 3)

b = np.array([10, 20, 30])    # shape (3,) → broadcasts to (2, 3)
a + b
# array([[11, 22, 33],
#        [14, 25, 36]])

c = np.array([[10], [20]])    # shape (2, 1) → broadcasts to (2, 3)
a + c
# array([[11, 12, 13],
#        [24, 25, 26]])

# Broadcasting diagram:
#    a: 2 × 3
#    b:     3  →  1 × 3  →  2 × 3
#    c: 2 × 1            →  2 × 3
```

### Vectorization — Why It Matters

```python
import time

n = 10_000_000
a = np.random.randn(n)
b = np.random.randn(n)

# Python loop: ~5s
start = time.time()
result = [a[i] + b[i] for i in range(n)]
print(f"Loop: {time.time() - start:.2f}s")

# NumPy vectorized: ~0.02s (250x faster!)
start = time.time()
result = a + b
print(f"NumPy: {time.time() - start:.4f}s")
```

### Common Operations

```python
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Aggregations (with axis)
a.sum()          # 45      — total
a.sum(axis=0)    # [12, 15, 18] — column sums
a.sum(axis=1)    # [6, 15, 24]  — row sums
a.mean(axis=0)   # [4., 5., 6.]
a.std(axis=1)    # per-row std deviation
a.min(), a.max(), a.argmin(), a.argmax()

# Reshaping
a.reshape(1, 9)     # (1, 9)
a.reshape(-1)       # flatten to 1D — (9,)
a.T                 # transpose — (3, 3)

# Indexing
a[0, :]         # first row: [1, 2, 3]
a[:, 0]         # first column: [1, 4, 7]
a[a > 5]        # boolean indexing: [6, 7, 8, 9]
a[np.where(a > 5)]  # same result

# Fancy indexing
rows = [0, 2]
cols = [1, 2]
a[rows, cols]   # [2, 9] — elements at (0,1) and (2,2)

# Stacking
np.vstack([a, a])       # vertical stack: (6, 3)
np.hstack([a, a])       # horizontal stack: (3, 6)
np.concatenate([a, a], axis=0)  # same as vstack

# Linear algebra
np.dot(a, a)            # matrix multiply
a @ a                   # same — Python 3.5+ operator
np.linalg.inv(a)        # inverse (if invertible)
np.linalg.eig(a)        # eigenvalues and eigenvectors
np.linalg.norm(a)       # Frobenius norm
```

---

## 6.2 Pandas

Pandas is the data manipulation workhorse. A DataFrame is a 2D labeled data structure — think of it as a Python dict of NumPy arrays with an aligned index.

### Core Operations

```python
import pandas as pd
import numpy as np

# ─── Creation ───
df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
    "age": [25, 30, 35, 28, 32],
    "salary": [70000, 85000, 92000, 78000, 88000],
    "department": ["Engineering", "Marketing", "Engineering", "Marketing", "Engineering"]
})

# ─── Selection ───
df["name"]                    # Series
df[["name", "salary"]]        # DataFrame (multiple columns)
df.loc[0:2, "name":"salary"]  # label-based (inclusive on both ends)
df.iloc[0:2, 0:3]             # integer-based (exclusive end)

# ─── Filtering ───
df[df["age"] > 28]
df[df["department"] == "Engineering"]
df.query("age > 28 and department == 'Engineering'")  # string query syntax

# ─── Adding/modifying columns ───
df["bonus"] = df["salary"] * 0.10
df["senior"] = df["age"].apply(lambda x: x >= 30)
df.assign(tax=lambda d: d["salary"] * 0.22)  # returns new df (functional style)

# ─── Sorting ───
df.sort_values("salary", ascending=False)
df.sort_values(["department", "salary"], ascending=[True, False])

# ─── Missing data ───
df.isna().sum()               # count NAs per column
df.dropna(subset=["salary"])  # drop rows with NA in salary
df.fillna({"age": df["age"].median()})  # fill with median
df["salary"].interpolate()    # linear interpolation
```

### GroupBy

```python
# GroupBy is split-apply-combine
grouped = df.groupby("department")

# Aggregation
grouped["salary"].mean()
grouped["salary"].agg(["mean", "median", "std", "count"])

# Multiple aggregations on different columns
df.groupby("department").agg(
    avg_salary=("salary", "mean"),
    max_age=("age", "max"),
    headcount=("name", "count")
)

# Transform — returns same-shaped output (broadcast back)
df["salary_zscore"] = df.groupby("department")["salary"].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Filter — keep groups that satisfy a condition
df.groupby("department").filter(lambda g: g["salary"].mean() > 80000)

# Apply — most flexible
def top_earner(group):
    return group.nlargest(1, "salary")

df.groupby("department").apply(top_earner)
```

### Merge / Join

```python
employees = pd.DataFrame({
    "emp_id": [1, 2, 3, 4],
    "name": ["Alice", "Bob", "Charlie", "Diana"],
    "dept_id": [10, 20, 10, 30]
})

departments = pd.DataFrame({
    "dept_id": [10, 20, 40],
    "dept_name": ["Engineering", "Marketing", "Sales"]
})

# Inner join (default) — only matching keys
pd.merge(employees, departments, on="dept_id", how="inner")
# emp_id 4 (dept_id=30) dropped — no match in departments
# dept_id 40 dropped — no match in employees

# Left join — keep all from left
pd.merge(employees, departments, on="dept_id", how="left")
# Diana (dept_id=30) included with NaN for dept_name

# Right, outer
pd.merge(employees, departments, on="dept_id", how="right")
pd.merge(employees, departments, on="dept_id", how="outer")

# Join on different column names
pd.merge(employees, departments, left_on="dept_id", right_on="dept_id")

# concat — stack DataFrames
pd.concat([df1, df2], axis=0)  # vertical (row-wise)
pd.concat([df1, df2], axis=1)  # horizontal (column-wise)
```

### Window Functions

```python
# Rolling window
df["salary_rolling_avg"] = df["salary"].rolling(window=3).mean()
df["salary_expanding_avg"] = df["salary"].expanding().mean()

# Rank
df["salary_rank"] = df.groupby("department")["salary"].rank(ascending=False)

# Shift (lag/lead)
df["prev_salary"] = df["salary"].shift(1)   # lag
df["next_salary"] = df["salary"].shift(-1)  # lead
df["salary_change"] = df["salary"].pct_change()

# Cumulative
df["cumulative_salary"] = df.groupby("department")["salary"].cumsum()
```

### Method Chaining

```python
# Clean, readable data transformation pipeline
result = (
    df
    .query("age >= 25")
    .assign(
        bonus=lambda d: d["salary"] * 0.10,
        total_comp=lambda d: d["salary"] + d["salary"] * 0.10
    )
    .groupby("department")
    .agg(
        avg_comp=("total_comp", "mean"),
        headcount=("name", "count")
    )
    .sort_values("avg_comp", ascending=False)
    .reset_index()
)

# Using pipe() for custom functions in chains
def log_shape(df, label=""):
    print(f"{label} Shape: {df.shape}")
    return df

result = (
    df
    .pipe(log_shape, "Initial")
    .dropna()
    .pipe(log_shape, "After dropna")
    .query("salary > 75000")
    .pipe(log_shape, "After filter")
)
```

### Pandas Performance Tips

| Tip | Why |
|-----|-----|
| Use vectorized ops over `apply` | `apply` is a loop in disguise |
| Use `category` dtype for low-cardinality strings | 10-50x memory reduction |
| Use `read_csv(usecols=[...])` | Don't load unneeded columns |
| Use `query()` for complex filters | Often faster than boolean indexing |
| Use `assign()` over `df["col"] = ...` | Enables method chaining |
| Use `.values` or `.to_numpy()` | When you need raw arrays for ML |

---

## 6.3 Matplotlib / Seaborn

```python
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Matplotlib basics ───
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Line plot
axes[0].plot(x, y, label="Model A", linewidth=2)
axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()

# Scatter plot
axes[1].scatter(x, y, c=colors, cmap="viridis", alpha=0.7)

# Bar chart
axes[1].bar(categories, values)

plt.tight_layout()
plt.savefig("figure.png", dpi=300, bbox_inches="tight")

# ─── Seaborn ───
# Distribution
sns.histplot(df["salary"], kde=True, bins=30)
sns.kdeplot(df["salary"], fill=True)
sns.boxplot(x="department", y="salary", data=df)
sns.violinplot(x="department", y="salary", data=df)

# Relationships
sns.scatterplot(x="age", y="salary", hue="department", data=df)
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
sns.pairplot(df, hue="department")

# Categorical
sns.countplot(x="department", data=df)
sns.barplot(x="department", y="salary", data=df, estimator=np.mean)
```

---

## 6.4 Scikit-learn: Pipeline and Transformers

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

# ─── ColumnTransformer: different processing for different columns ───
numeric_features = ["age", "salary", "experience"]
categorical_features = ["department", "education"]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ─── Full pipeline: preprocessing + model ───
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# ─── Cross-validation ───
scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
print(f"Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

# ─── GridSearchCV ───
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [5, 10, None],
    "preprocessor__num__imputer__strategy": ["mean", "median"]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.3f}")

# ─── Fit and predict ───
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
probabilities = pipeline.predict_proba(X_test)
```

> **Interview tip:** Always mention sklearn Pipelines when discussing ML workflows. They prevent data leakage (fitting scalers on training data only), make code reproducible, and are production-ready.

---

# 7. Common Coding Patterns

---

## 7.0 The 60-second pattern test

Read the prompt. Pick **one** row. If two rows both fit, say both and pick the cheaper one (usually O(n) hash over O(n log n) sort).

| If you hear… | Pattern | First line you write |
|---|---|---|
| "two numbers sum to target", array **unsorted** | Hash map `value → index` | `seen = {}` |
| same, array **sorted**, or "in-place, O(1) extra memory" | Two pointers from ends | `left, right = 0, n-1` |
| "remove duplicates in-place" / "compact" | Read/write pointers | `write = 1` |
| "contiguous subarray / substring of length k" | Fixed window | running `total`, add right, drop left |
| "longest / shortest substring that…" | Variable window | `left = 0`, grow `right`, shrink when invalid |
| "anagram / frequency / top k frequent" | `Counter` (+ heap if k) | `Counter(nums).most_common(k)` |
| "group by some key" | `defaultdict(list)` | `key = tuple(sorted(word))` |
| "subarray sum equals k" (not necessarily contiguous-length-k) | Prefix sum + map | `prefix = 0; seen = {0: 1}` |
| "kth largest / merge k sorted" | Heap | `heapq.nlargest` / `heappush` |
| "overlapping meetings / merge ranges" | Sort intervals | `intervals.sort(key=lambda x: x[0])` |
| "balanced brackets / undo / next greater" | Stack | `stack = []` |
| "shortest path, unweighted graph / word ladder" | BFS + `deque` | `q = deque([start])` |
| "all combinations / parse nested" | DFS / recursion | helper with backtrack |
| "middle of linked list / cycle" | Fast/slow pointers | `slow = fast = head` |

**Hash map vs two pointers for two-sum — the question they love:**

- Unsorted, need **indices**, extra memory OK → **hash**, O(n) time / O(n) space.
- Sorted, need **values or indices**, extra memory **not** OK → **two pointers**, O(n) time / O(1) space.
- If you sort first to use two pointers, sorting is O(n log n) and **you lose original indices** unless you store `(value, i)` pairs.

---

## 7.1 Two Pointers

Used when working with sorted arrays or when searching for pairs/subarrays.

```python
def two_sum_sorted(nums: list[int], target: int) -> tuple[int, int]:
    """Find two numbers that add up to target in a SORTED array."""
    left, right = 0, len(nums) - 1
    
    while left < right:
        current_sum = nums[left] + nums[right]
        if current_sum == target:
            return (left, right)
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return (-1, -1)

# Time: O(n), Space: O(1)

def remove_duplicates(nums: list[int]) -> int:
    """Remove duplicates in-place from sorted array. Return new length."""
    if not nums:
        return 0
    
    write = 1
    for read in range(1, len(nums)):
        if nums[read] != nums[read - 1]:
            nums[write] = nums[read]
            write += 1
    
    return write

# Time: O(n), Space: O(1)
```

### How to see "two pointers" (say this)

There are **four families**. Draw two index arrows on the array; if they only move one way and never need a hash, you are in this pattern.

| Family | Arrows | When | Example in this file |
|---|---|---|---|
| **Opposite ends** | `left=0`, `right=n-1` | Sorted array, palindrome, container-with-water | `two_sum_sorted` |
| **Read / write** | `read` scans, `write` places keepers | In-place compact, remove element | `remove_duplicates` |
| **Same direction (slow/fast on array)** | both start left, `fast` explores, `slow` is boundary | Some windows, partition | sliding window is the cousin |
| **Fast / slow (linked list)** | `fast = fast.next.next` | Cycle, middle node | below |

**Invariant to speak:** *"left is the smallest unused candidate, right is the largest; the sum tells me which side is impossible."*

```python
def is_palindrome_two_pointers(s: str) -> bool:
    """Opposite-ends: skip non-alnum, compare inward."""
    left, right = 0, len(s) - 1
    while left < right:
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        if s[left].lower() != s[right].lower():
            return False
        left += 1
        right -= 1
    return True


def move_zeroes(nums: list[int]) -> None:
    """Read/write: keep non-zeros in order, fill rest with 0. In-place."""
    write = 0
    for read in range(len(nums)):
        if nums[read] != 0:
            nums[write], nums[read] = nums[read], nums[write]
            write += 1


def has_cycle(head) -> bool:
    """Fast/slow on a linked list. If they meet, there is a cycle."""
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True
    return False


def merge_two_sorted(a: list[int], b: list[int]) -> list[int]:
    """Two pointers on TWO arrays — same idea as Q14."""
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i]); i += 1
        else:
            out.append(b[j]); j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out
```

**Common bugs (they watch for these):**
- Off-by-one: `while left <= right` vs `left < right` — for pairs you usually want `left < right` so you don't use the same index twice.
- Forgetting to **move a pointer on every iteration** → infinite loop.
- Using two pointers on an **unsorted** two-sum and getting a wrong pair.

---

## 7.2 Sliding Window

Used for subarray/substring problems with a contiguous range.

```python
def max_sum_subarray(nums: list[int], k: int) -> int:
    """Maximum sum of any subarray of length k."""
    window_sum = sum(nums[:k])
    max_sum = window_sum
    
    for i in range(k, len(nums)):
        window_sum += nums[i] - nums[i - k]  # slide: add right, remove left
        max_sum = max(max_sum, window_sum)
    
    return max_sum

# Time: O(n), Space: O(1)

def longest_unique_substring(s: str) -> int:
    """Length of longest substring without repeating characters."""
    seen = {}
    left = 0
    max_len = 0
    
    for right, char in enumerate(s):
        if char in seen and seen[char] >= left:
            left = seen[char] + 1
        seen[char] = right
        max_len = max(max_len, right - left + 1)
    
    return max_len

# Time: O(n), Space: O(min(n, alphabet_size))
```

**Fixed vs variable window:**
- **Fixed k:** add `nums[right]`, subtract `nums[right-k]`. One loop. (moving average, max sum of k)
- **Variable:** `right` always grows; `left` grows only when the window **breaks the constraint** (too many uniques, sum too big, etc.). The invariant is "window `[left, right]` is always valid."

```python
def min_subarray_len(target: int, nums: list[int]) -> int:
    """Shortest contiguous subarray with sum >= target. Variable window."""
    left = total = 0
    best = float("inf")
    for right, x in enumerate(nums):
        total += x
        while total >= target:
            best = min(best, right - left + 1)
            total -= nums[left]
            left += 1
    return 0 if best == float("inf") else best
```

Tie to work: Jet2 event detection rolling windows and AI Cargo rolling temp std are the *same idea* — a contiguous time window with a running statistic.

---

## 7.3 Hash Map Frequency Counting

```python
from collections import Counter

def top_k_frequent(nums: list[int], k: int) -> list[int]:
    """Return the k most frequent elements."""
    return [x for x, _ in Counter(nums).most_common(k)]

def group_anagrams(words: list[str]) -> list[list[str]]:
    """Group words that are anagrams of each other."""
    from collections import defaultdict
    groups = defaultdict(list)
    
    for word in words:
        key = tuple(sorted(word))  # anagrams have the same sorted form
        groups[key].append(word)
    
    return list(groups.values())

# Example: group_anagrams(["eat","tea","tan","ate","nat","bat"])
# [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]

def two_sum(nums: list[int], target: int) -> list[int]:
    """Find indices of two numbers that add up to target (unsorted)."""
    seen = {}  # value → index
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

# Time: O(n), Space: O(n)
```

### How to see "hash map" (say this)

A hash map is **O(1) memory of the past**. Every time the problem says "have I seen this before?" / "how many times?" / "group equals" / "the complement exists", you store the past instead of scanning it again (which would be O(n²)).

| Job | Python | Key you store |
|---|---|---|
| Complement (two-sum) | `dict[value] = index` | the number itself |
| Frequency | `Counter` or `defaultdict(int)` | the number / char |
| Group | `defaultdict(list)` | a **canonical** key (`tuple(sorted(word))`, or 26-count tuple) |
| First index / last index | `dict[value] = i` | update only on first seen if you need earliest |
| Uniqueness | `set` | same as a map with dummy values |
| Prefix-sum → count of subarrays | `dict[prefix] = count` | running sum |

**Two-sum walk-through (the one you must not fumble):**

```
nums = [2, 7, 11, 15], target = 9
i=0, num=2, need=7, seen={}        → store 2→0
i=1, num=7, need=2, 2 in seen      → return [0, 1]
```

Store **after** the lookup so you do not use the same index twice.

```python
from collections import defaultdict, Counter

def first_duplicate(nums: list[int]) -> int | None:
    """First value that appears a second time (scan order)."""
    seen: set[int] = set()
    for x in nums:
        if x in seen:
            return x
        seen.add(x)
    return None


def longest_consecutive(nums: list[int]) -> int:
    """Longest run of consecutive values, unsorted. Set, not sort."""
    s = set(nums)
    best = 0
    for x in s:
        if x - 1 in s:      # only start a run at the beginning
            continue
        length = 1
        while x + length in s:
            length += 1
        best = max(best, length)
    return best


def subarray_sum_equals_k(nums: list[int], k: int) -> int:
    """Count contiguous subarrays whose sum is k. Prefix + hash."""
    count = 0
    prefix = 0
    seen = defaultdict(int)
    seen[0] = 1
    for x in nums:
        prefix += x
        count += seen[prefix - k]   # how many earlier prefixes make this window = k
        seen[prefix] += 1
    return count
```

**Canonical keys (anagrams and "same bag of chars"):**
- `tuple(sorted(word))` — simple, O(L log L) per word
- 26-length count tuple — O(L), better if they ask

**Bugs:**
- Using a list as a dict key → `TypeError` (unhashable). Use `tuple`.
- `if complement in seen` then `seen[num] = i` — if you store first, `[3, 3]` target 6 can still work if you lookup before store; **lookup then store** is the safe order.
- `Counter` equality is the right anagram check; sorting both strings is the slower backup.

**Tie to work:** Amazon Pay feature selection and ATCS DQ lookups are the same mental model — key → value, don't scan the whole table.

---

## 7.4 Matrix Traversal

```python
def spiral_order(matrix: list[list[int]]) -> list[int]:
    """Return elements in spiral order."""
    result = []
    if not matrix:
        return result
    
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    
    while top <= bottom and left <= right:
        for col in range(left, right + 1):       # → right
            result.append(matrix[top][col])
        top += 1
        
        for row in range(top, bottom + 1):        # ↓ down
            result.append(matrix[row][right])
        right -= 1
        
        if top <= bottom:
            for col in range(right, left - 1, -1): # ← left
                result.append(matrix[bottom][col])
            bottom -= 1
        
        if left <= right:
            for row in range(bottom, top - 1, -1):  # ↑ up
                result.append(matrix[row][left])
            left += 1
    
    return result

def transpose(matrix: list[list[int]]) -> list[list[int]]:
    """Transpose a matrix."""
    return [list(row) for row in zip(*matrix)]
    # or: [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
```

---

## 7.5 BFS / DFS Basics

```python
from collections import deque

# ─── BFS: Level-order traversal / shortest path ───
def bfs(graph: dict[str, list[str]], start: str) -> list[str]:
    """Breadth-first search — explores level by level."""
    visited = set()
    queue = deque([start])
    visited.add(start)
    order = []
    
    while queue:
        node = queue.popleft()
        order.append(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return order

# ─── DFS: Go deep before going wide ───
def dfs_iterative(graph: dict[str, list[str]], start: str) -> list[str]:
    """Depth-first search using explicit stack."""
    visited = set()
    stack = [start]
    order = []
    
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                stack.append(neighbor)
    
    return order

def dfs_recursive(graph, node, visited=None):
    """Depth-first search — recursive."""
    if visited is None:
        visited = set()
    
    visited.add(node)
    print(node, end=" ")
    
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited)

# Example graph
graph = {
    "A": ["B", "C"],
    "B": ["A", "D", "E"],
    "C": ["A", "F"],
    "D": ["B"],
    "E": ["B", "F"],
    "F": ["C", "E"]
}

bfs(graph, "A")  # ['A', 'B', 'C', 'D', 'E', 'F']
```

| Algorithm | Data Structure | Time | Space | Use Case |
|-----------|---------------|------|-------|----------|
| BFS | Queue (`deque`) | O(V + E) | O(V) | Shortest path, level-order |
| DFS | Stack / recursion | O(V + E) | O(V) | Topological sort, cycle detection, path finding |

**BFS vs DFS one-liner:** BFS for *shortest* unweighted path (level by level). DFS for *explore all / detect cycle / topo order*. Recursion DFS needs to mention stack overflow on deep graphs — prefer iterative in interviews if n can be large.

---

## 7.6 Prefix sums

Turn range-sum queries from O(n) into O(1) after an O(n) build. `prefix[i] = nums[0] + … + nums[i-1]` (length n+1, `prefix[0] = 0`).

```python
def range_sum(nums: list[int], left: int, right: int) -> int:
    """Sum of nums[left:right+1], 0-based inclusive."""
    prefix = [0]
    for x in nums:
        prefix.append(prefix[-1] + x)
    return prefix[right + 1] - prefix[left]


def max_subarray_kadane(nums: list[int]) -> int:
    """Best contiguous sum (Kadane). Not prefix — running max of (extend vs restart)."""
    best = cur = nums[0]
    for x in nums[1:]:
        cur = max(x, cur + x)
        best = max(best, cur)
    return best
```

Subarray-sum-equals-k is prefix **plus hash** (§7.3). Kadane is the "best sum" cousin. Do not mix them.

---

## 7.7 Binary search (including on the answer)

On a **sorted** array: standard `bisect`. In DS/ML interviews they also ask **binary search on a monotonic predicate** — "smallest k such that a window of size k works."

```python
import bisect

def search_insert(nums: list[int], target: int) -> int:
    """Index of target, or where it would be inserted. Sorted nums."""
    return bisect.bisect_left(nums, target)


def min_capacity_to_ship(weights: list[int], days: int) -> int:
    """Classic 'search the answer': smallest capacity that ships in `days`."""
    def feasible(cap: int) -> bool:
        need, load = 1, 0
        for w in weights:
            if load + w > cap:
                need += 1
                load = 0
            load += w
        return need <= days

    lo, hi = max(weights), sum(weights)
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo
```

**Template:** `while lo < hi: mid = (lo+hi)//2; if feasible(mid): hi = mid; else: lo = mid+1` → `lo` is the smallest feasible. Draw a T/F array on the number line if you blank.

---

## 7.8 Heaps and Top-K

```python
import heapq
from collections import Counter

def top_k_frequent_heap(nums: list[int], k: int) -> list[int]:
    """When they ban Counter.most_common and want you to show a heap."""
    freq = Counter(nums)
    return [x for x, _ in heapq.nlargest(k, freq.items(), key=lambda kv: kv[1])]


def kth_largest(nums: list[int], k: int) -> int:
    """Min-heap of size k — root is the kth largest."""
    h: list[int] = []
    for x in nums:
        if len(h) < k:
            heapq.heappush(h, x)
        elif x > h[0]:
            heapq.heapreplace(h, x)
    return h[0]
```

`heapq` is a **min-heap**. Max-heap = push `-x`. nlargest/nsmallest are fine to use if you name the complexity: O(n log k).

---

## 7.9 Intervals and merge

Sort by start. Sweep. This is the "merge overlapping meetings / time windows" problem — also how you merge retrieval score lists conceptually.

```python
def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    out = [intervals[0][:]]
    for start, end in intervals[1:]:
        if start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out
```

---

## 7.10 Stack / monotonic

You already have valid parens in Q13. Next-greater is the other stack classic:

```python
def next_greater(nums: list[int]) -> list[int]:
    """For each value, next strictly greater to the right, else -1."""
    n = len(nums)
    ans = [-1] * n
    stack: list[int] = []  # indices of increasing candidates
    for i, x in enumerate(nums):
        while stack and nums[stack[-1]] < x:
            ans[stack.pop()] = x
        stack.append(i)
    return ans
```

---

# 8. Python Best Practices

---

## 8.1 PEP 8 Style Guide

| Rule | Example |
|------|---------|
| snake_case for functions/variables | `calculate_score()`, `user_name` |
| PascalCase for classes | `DataPipeline`, `ModelTrainer` |
| UPPER_CASE for constants | `MAX_RETRIES = 3` |
| 4-space indentation | (not tabs) |
| Max line length: 79 (code), 72 (docstrings) | Use `\` or parens for continuation |
| Two blank lines before top-level definitions | Functions, classes |
| One blank line inside classes | Between methods |
| Imports at top, grouped | stdlib → third-party → local |

```python
# Import order
import os                          # 1. Standard library
import sys
from pathlib import Path

import numpy as np                 # 2. Third-party
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from myproject.utils import clean  # 3. Local
from myproject.models import Model
```

---

## 8.2 Virtual Environments

```bash
# venv (built-in)
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
.venv\Scripts\activate             # Windows
pip install -r requirements.txt
deactivate

# conda
conda create -n myenv python=3.11
conda activate myenv
conda install numpy pandas scikit-learn
conda deactivate

# uv (modern, fast — Rust-based)
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

---

## 8.3 Dependency Management

```ini
# requirements.txt — pinned versions for reproducibility
numpy==1.26.4
pandas==2.2.0
scikit-learn==1.4.0
fastapi==0.109.0
uvicorn==0.27.0
```

```toml
# pyproject.toml — modern standard (PEP 621)
[project]
name = "my-project"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.26",
    "pandas>=2.2",
    "scikit-learn>=1.4",
]

[project.optional-dependencies]
dev = ["pytest", "ruff", "mypy"]
```

---

## 8.4 Logging vs Print

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)

logger = logging.getLogger(__name__)

# Use logging levels appropriately
logger.debug("Detailed diagnostic info")
logger.info("Training started with lr=0.001")
logger.warning("GPU memory usage above 90%")
logger.error("Failed to load model checkpoint")
logger.critical("Database connection lost")

# Lazy formatting (no string construction if level is filtered)
logger.info("Processing batch %d of %d", batch_num, total_batches)

# Never use print() in production code — use logging instead
# print cannot be: filtered by level, routed to files, structured, disabled
```

---

## 8.5 Error Handling

```python
# ─── Basic try/except/else/finally ───
def safe_divide(a: float, b: float) -> float | None:
    try:
        result = a / b
    except ZeroDivisionError:
        logger.error(f"Division by zero: {a}/{b}")
        return None
    except TypeError as e:
        logger.error(f"Type error: {e}")
        raise  # re-raise after logging
    else:
        # Runs only if no exception occurred
        logger.info(f"Result: {result}")
        return result
    finally:
        # Always runs (cleanup)
        logger.debug("Division operation completed")

# ─── Specific exceptions over broad ones ───
# BAD
try:
    result = process_data(raw)
except Exception:  # catches everything — hides bugs
    pass

# GOOD
try:
    result = process_data(raw)
except (ValueError, KeyError) as e:
    logger.error(f"Data processing failed: {e}")
    raise

# ─── Custom exceptions ───
class ModelNotTrainedError(Exception):
    """Raised when prediction is attempted before training."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        super().__init__(f"Model '{model_name}' must be trained before prediction")

class DataValidationError(Exception):
    """Raised when input data fails validation."""
    def __init__(self, field: str, reason: str):
        self.field = field
        self.reason = reason
        super().__init__(f"Validation failed for '{field}': {reason}")

# Usage
def predict(model, X):
    if not model.is_trained:
        raise ModelNotTrainedError(model.name)
    return model.predict(X)

# ─── EAFP vs LBYL ───
# LBYL (Look Before You Leap) — C-style
if key in dictionary:
    value = dictionary[key]

# EAFP (Easier to Ask Forgiveness than Permission) — Pythonic
try:
    value = dictionary[key]
except KeyError:
    value = default

# Even more Pythonic:
value = dictionary.get(key, default)
```

---

## 8.6 Testing: pytest

```python
# test_math_utils.py
import pytest
from math_utils import factorial, fibonacci, is_palindrome

# ─── Basic tests ───
def test_factorial():
    assert factorial(0) == 1
    assert factorial(5) == 120
    assert factorial(10) == 3628800

def test_factorial_negative():
    with pytest.raises(ValueError, match="must be non-negative"):
        factorial(-1)

# ─── Parametrized tests ───
@pytest.mark.parametrize("input_val, expected", [
    ("racecar", True),
    ("hello", False),
    ("", True),
    ("a", True),
    ("Aba", True),  # case-insensitive
])
def test_is_palindrome(input_val, expected):
    assert is_palindrome(input_val) == expected

# ─── Fixtures ───
@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    import pandas as pd
    return pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "score": [85, 92, 78]
    })

def test_average_score(sample_dataframe):
    avg = sample_dataframe["score"].mean()
    assert avg == pytest.approx(85.0, rel=1e-2)

# ─── Fixtures with cleanup ───
@pytest.fixture
def temp_file(tmp_path):
    """Create and clean up a temporary file."""
    filepath = tmp_path / "test_data.csv"
    filepath.write_text("name,score\nAlice,85\nBob,92\n")
    yield filepath
    # Cleanup happens automatically with tmp_path

# ─── Mocking ───
from unittest.mock import patch, MagicMock

def test_api_call():
    with patch("requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"result": "success"}
        )
        response = fetch_data("https://api.example.com")
        assert response["result"] == "success"
        mock_get.assert_called_once()
```

```bash
# Running tests
pytest                         # all tests
pytest test_math_utils.py      # specific file
pytest -v                      # verbose
pytest -x                      # stop on first failure
pytest -k "palindrome"         # filter by name
pytest --cov=src               # with coverage
```

---

# 9. Common Interview Coding Questions

---

## Q1: "Explain list vs tuple"

| Feature | `list` | `tuple` |
|---------|--------|---------|
| Mutable | Yes | No |
| Syntax | `[1, 2, 3]` | `(1, 2, 3)` |
| Hashable | No | Yes (if elements are) |
| Performance | Slightly slower | Slightly faster |
| Memory | More (over-allocates for growth) | Less |
| Use case | Dynamic collections | Fixed records, dict keys, function returns |

```python
import sys

lst = [1, 2, 3, 4, 5]
tup = (1, 2, 3, 4, 5)

sys.getsizeof(lst)  # 120 bytes (extra space for growth)
sys.getsizeof(tup)  # 80 bytes (exact fit)

# Tuples can be dict keys; lists cannot
d = {(1, 2): "point"}    # Works
# d = {[1, 2]: "point"}  # TypeError: unhashable type: 'list'
```

> **Answer framework:** Start with mutability difference, then mention hashability (dict keys, set elements), performance, and memory. Give a practical example of when you'd choose each.

---

## Q2: "What is the GIL?"

> **Concise answer:** The GIL (Global Interpreter Lock) is a mutex in CPython that allows only one thread to execute Python bytecode at a time. It exists to simplify memory management (reference counting is not thread-safe). It means CPU-bound multithreading doesn't achieve true parallelism — use `multiprocessing` or `concurrent.futures.ProcessPoolExecutor` instead. The GIL is released during I/O operations, so threading still helps for I/O-bound tasks. NumPy also releases the GIL during C-level computations.

*(See Section 5.6 for the full deep dive.)*

---

## Q3: "How does a decorator work?"

> **Concise answer:** A decorator is a function that takes a function as input and returns a new function (usually a wrapper). The `@decorator` syntax is syntactic sugar — `@dec def f()` is equivalent to `f = dec(f)`. Decorators are used for cross-cutting concerns like logging, timing, caching, authentication, and retry logic. Always use `@functools.wraps` to preserve the original function's metadata.

```python
from functools import wraps

def log_calls(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {result}")
        return result
    return wrapper

@log_calls
def add(a, b):
    return a + b

add(3, 4)
# Calling add with args=(3, 4), kwargs={}
# add returned 7
```

---

## Q4: "Generator vs list comprehension?"

| Aspect | List Comprehension | Generator Expression |
|--------|-------------------|---------------------|
| Syntax | `[x for x in ...]` | `(x for x in ...)` |
| Returns | `list` | `generator` object |
| Memory | O(n) — stores all elements | O(1) — one at a time |
| Reusable | Yes (iterate multiple times) | No (single pass, exhausted) |
| Speed | Slightly faster for small n | Faster for large n (no allocation) |

```python
# Use list comprehension when you need random access or multiple iterations
squares = [x**2 for x in range(100)]
print(squares[50])     # works
print(sum(squares))    # works again

# Use generator for large datasets or single-pass operations
total = sum(x**2 for x in range(10**8))  # O(1) memory
# list version would need ~800 MB of memory!
```

---

## Q5: "Deep copy vs shallow copy"

```python
import copy

original = {"name": "Rahul", "scores": [90, 85, 92]}

# Shallow copy — new dict, but inner list is shared
shallow = copy.copy(original)
shallow["scores"].append(88)
print(original["scores"])  # [90, 85, 92, 88] — MUTATED!
shallow["name"] = "New"
print(original["name"])    # "Rahul" — strings are immutable, safe

# Deep copy — completely independent
original = {"name": "Rahul", "scores": [90, 85, 92]}
deep = copy.deepcopy(original)
deep["scores"].append(88)
print(original["scores"])  # [90, 85, 92] — UNCHANGED
```

**Rule of thumb:** If your object contains only immutable elements, shallow copy is fine. If it contains nested mutable objects (lists, dicts, sets), use deep copy for true independence.

---

## Q6: "How does Python manage memory?"

> **Answer structure:**

1. **Reference counting** — Every object has a count of references pointing to it. When count reaches 0, memory is freed immediately.

2. **Garbage collector** — Handles circular references (A → B → A) that reference counting can't detect. Uses generational collection (gen 0, 1, 2).

3. **Memory allocator (pymalloc)** — Small object allocator for objects ≤ 512 bytes. Uses memory pools to reduce overhead of frequent malloc/free calls.

4. **Interning** — Python caches small integers (-5 to 256) and short strings for reuse.

```python
import sys

a = [1, 2, 3]
sys.getrefcount(a)  # 2 (a + the temporary reference from getrefcount)

b = a
sys.getrefcount(a)  # 3

del b
sys.getrefcount(a)  # 2

# Integer interning
x = 256
y = 256
x is y  # True — same object (cached)

x = 257
y = 257
x is y  # False (may vary — not cached outside -5 to 256 in standard REPL)

# Garbage collector for circular references
import gc
gc.collect()           # force garbage collection
gc.get_stats()         # collection statistics per generation
gc.get_threshold()     # (700, 10, 10) — default thresholds
```

---

## Q7: "Explain *args and **kwargs"

> **Concise answer:** `*args` collects extra positional arguments into a tuple; `**kwargs` collects extra keyword arguments into a dict. Together, they let a function accept any number of arguments. They're essential for writing wrappers (decorators), forwarding arguments, and creating flexible APIs.

```python
def flexible(*args, **kwargs):
    print(f"args: {args}")        # tuple
    print(f"kwargs: {kwargs}")    # dict

flexible(1, 2, 3, name="Rahul", role="DS")
# args: (1, 2, 3)
# kwargs: {'name': 'Rahul', 'role': 'DS'}

# Key use: forwarding arguments
def wrapper(func, *args, **kwargs):
    print(f"Calling {func.__name__}")
    return func(*args, **kwargs)
```

*(See Section 1.6 for the complete deep dive.)*

---

## Q8: Code Challenge — Anagram Check

```python
def is_anagram(s1: str, s2: str) -> bool:
    """Check if two strings are anagrams."""
    # Method 1: Sorting — O(n log n) time, O(n) space
    # return sorted(s1.lower()) == sorted(s2.lower())
    
    # Method 2: Counter — O(n) time, O(1) space (bounded alphabet)
    from collections import Counter
    return Counter(s1.lower()) == Counter(s2.lower())

assert is_anagram("listen", "silent") == True
assert is_anagram("hello", "world") == False
assert is_anagram("Astronomer", "Moon starer") == False  # spaces differ
```

---

## Q9: Code Challenge — Moving Average (Sliding Window)

```python
from collections import deque

class MovingAverage:
    """Calculate moving average over the last k values."""
    
    def __init__(self, k: int):
        self.k = k
        self.window = deque(maxlen=k)
        self.total = 0.0
    
    def next(self, val: float) -> float:
        if len(self.window) == self.k:
            self.total -= self.window[0]  # remove oldest (auto-evicted by maxlen)
        self.window.append(val)
        self.total += val
        return self.total / len(self.window)

ma = MovingAverage(3)
ma.next(1)    # 1.0
ma.next(10)   # 5.5
ma.next(3)    # 4.667
ma.next(5)    # 6.0  — window is [10, 3, 5], 1 was evicted
```

---

## Q10: Code Challenge — Flatten Nested List

```python
def flatten(nested) -> list:
    """Flatten an arbitrarily nested list."""
    result = []
    for item in nested:
        if isinstance(item, (list, tuple)):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

assert flatten([1, [2, [3, 4], 5], [6, 7]]) == [1, 2, 3, 4, 5, 6, 7]
assert flatten([1, [2], [[3]], [[[4]]]]) == [1, 2, 3, 4]

# Generator version — memory efficient
def flatten_gen(nested):
    for item in nested:
        if isinstance(item, (list, tuple)):
            yield from flatten_gen(item)
        else:
            yield item

list(flatten_gen([1, [2, [3, 4], 5], [6, 7]]))
# [1, 2, 3, 4, 5, 6, 7]
```

---

## Q11: Code Challenge — LRU Cache Implementation

```python
from collections import OrderedDict

class LRUCache:
    """Least Recently Used cache with O(1) get and put."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)  # mark as recently used
        return self.cache[key]
    
    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # evict oldest (front)

# Usage
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
cache.get(1)       # 1 (marks key 1 as recently used)
cache.put(3, 3)    # evicts key 2 (least recently used)
cache.get(2)       # -1 (not found)
cache.get(3)       # 3
```

### Double-Linked List + HashMap (From Scratch)

```python
class Node:
    def __init__(self, key=0, val=0):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None

class LRUCacheManual:
    """LRU Cache using doubly linked list + hash map."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key → Node
        
        # Dummy head and tail for easy edge-case handling
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _remove(self, node: Node) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def _add_to_front(self, node: Node) -> None:
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node
    
    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._remove(node)
        self._add_to_front(node)
        return node.val
    
    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self._remove(self.cache[key])
        
        node = Node(key, value)
        self.cache[key] = node
        self._add_to_front(node)
        
        if len(self.cache) > self.capacity:
            lru = self.tail.prev
            self._remove(lru)
            del self.cache[lru.key]
```

| Operation | Time Complexity |
|-----------|----------------|
| `get()` | O(1) |
| `put()` | O(1) |
| Space | O(capacity) |

---

## Q12: Code Challenge — Two Sum (Hash Map)

```python
def two_sum(nums: list[int], target: int) -> list[int]:
    """Return indices of two numbers that sum to target."""
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

assert two_sum([2, 7, 11, 15], 9) == [0, 1]
assert two_sum([3, 2, 4], 6) == [1, 2]
```

---

## Q13: Code Challenge — Valid Parentheses (Stack)

```python
def is_valid_parens(s: str) -> bool:
    """Check if brackets are balanced."""
    stack = []
    pairs = {")": "(", "]": "[", "}": "{"}
    
    for char in s:
        if char in "({[":
            stack.append(char)
        elif char in ")}]":
            if not stack or stack[-1] != pairs[char]:
                return False
            stack.pop()
    
    return len(stack) == 0

assert is_valid_parens("()[]{}") == True
assert is_valid_parens("([)]") == False
assert is_valid_parens("{[()]}") == True
```

---

## Q14: Code Challenge — Merge Two Sorted Lists

```python
def merge_sorted(a: list[int], b: list[int]) -> list[int]:
    """Merge two sorted lists into one sorted list."""
    result = []
    i = j = 0
    
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    
    result.extend(a[i:])
    result.extend(b[j:])
    return result

# Time: O(n + m), Space: O(n + m)
assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]
```

---

## Bonus Q15: "What's new in recent Python versions?"

| Version | Key Feature |
|---------|-------------|
| **3.8** | Walrus operator `:=`, positional-only params `/` |
| **3.9** | Dict merge `\|`, `list[int]` (no `List`), `str.removeprefix()` |
| **3.10** | Structural pattern matching (`match/case`), `X \| Y` union types |
| **3.11** | 10-60% faster (specializing adaptive interpreter), exception groups |
| **3.12** | Per-interpreter GIL (subinterpreters), `type` statement for type aliases |
| **3.13** | Experimental free-threaded mode (`--disable-gil`), JIT compiler (experimental) |

```python
# Walrus operator (3.8) — assign in expression
if (n := len(data)) > 10:
    print(f"Processing {n} items")

# While reading lines
while (line := f.readline()):
    process(line)

# Pattern matching (3.10)
def process_command(command):
    match command.split():
        case ["quit"]:
            return "Goodbye"
        case ["greet", name]:
            return f"Hello, {name}!"
        case ["add", *numbers]:
            return sum(int(n) for n in numbers)
        case _:
            return "Unknown command"
```

---

# 10. What each role actually codes

Do **not** grind 200 random LeetCode hards. Grind the column for the role you booked this week.

## 10.1 AI Engineer / GenAI (your primary target)

They are checking: can you write correct Python around retrieval, JSON, windows, and caches — the same muscles as Bilbo / AI Cargo.

| Ask (they may disguise the story) | Pattern | In this doc | Project tie-in |
|---|---|---|---|
| Dedup retrieved chunks by `doc_id`, keep highest score | hash map `id → best` | §7.3 | P07 RRF / rerank |
| Reciprocal rank fusion of two ranked lists | hash map `id → score`, then sort | Q16 below | P07 hybrid retrieval |
| Chunk text with overlap | sliding index, not a window-sum | Q17 | P07 / P15 RAG |
| Streaming tokens / moving metric | deque + running sum | Q9 | P04 / P15 windows |
| Tool-call graph: shortest path to a tool | BFS | §7.5 | P15 LangGraph is the *product*; BFS is the *round* |
| LRU prompt cache | OrderedDict | Q11 | doc `50` prefix cache is the systems analogue |
| Merge overlapping citation spans | intervals | §7.9 | P07 citations |
| Rate-limit an LLM client | deque of timestamps | Q18 | doc `49` budgets |
| Parse nested KYC-like JSON | recursion / generators | Q10 flatten | P06 JSON flatten |

**You will almost never be asked** to implement attention from scratch in a 45-minute *coding* round. That is a *whiteboard* question from `04`. Coding is helpers around the model.

## 10.2 Data Scientist

| Ask | Pattern | Tie-in |
|---|---|---|
| Two-sum / frequency / top-k | hash / heap | P06 chi²/MI is conceptual; the code round is Counter |
| Group-by in pure Python | `defaultdict(list)` | don't use pandas unless they allow it |
| Entity-aware split (no leakage) | hash `id → fold` | P15 shipment split, P04 walk-forward |
| Precision / recall from labels | four counts | P01 / P03 metrics |
| WOE-style "count good/bad per bin" | `defaultdict` of pairs | P01, doc `51` — they want the dict, not sklearn |
| Time-based window feature | sliding / prefix | P04 |

**Pandas is fair game** if they say "use pandas." If they say "pure Python," a `.apply(axis=1)` loop is a fail. Vectorize or use dicts.

## 10.3 Forward Deployed / Data Engineer

Dirty strings, generators over files, two-pointer merge of sorted logs, `deque` for late events. FusionSpan file is the extra problem bank.

## 10.4 Worked problems for YOUR stack (code these from memory)

### Q16: Reciprocal Rank Fusion (AI Engineer)

```python
from collections import defaultdict

def rrf(rank_lists: list[list[str]], k: int = 60) -> list[str]:
    """
    rank_lists: each is doc_ids best-first (rank 1 = index 0).
    score(d) = sum 1/(k + rank). Return doc ids by score desc.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rank_lists:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return [d for d, _ in sorted(scores.items(), key=lambda kv: -kv[1])]

assert rrf([["a", "b"], ["b", "a"]])[0] in {"a", "b"}
```

> Spoken: *"Same combiner I used on Bilbo for BM25 + dense — here it's a 15-line dict."*

### Q17: Overlapping chunks (RAG)

```python
def chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    """Word-ish char chunks. Last chunk may be short. overlap < size."""
    if size <= 0 or overlap >= size:
        raise ValueError("need size > overlap >= 0")
    step = size - overlap
    out = []
    i = 0
    n = len(text)
    while i < n:
        out.append(text[i : i + size])
        i += step
        if i >= n:
            break
    return out
```

### Q18: Sliding-window rate limiter

```python
from collections import deque
import time

class RateLimiter:
    """Allow at most `limit` events in any `window_seconds` span."""

    def __init__(self, limit: int, window_seconds: float):
        self.limit = limit
        self.window = window_seconds
        self.hits: deque[float] = deque()

    def allow(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        while self.hits and now - self.hits[0] > self.window:
            self.hits.popleft()
        if len(self.hits) >= self.limit:
            return False
        self.hits.append(now)
        return True
```

### Q19: Entity-aware split (Data Scientist — no leakage)

```python
from collections import defaultdict
import random

def group_split(ids: list[str], labels: list[int], test_frac: float = 0.2, seed: int = 0):
    """All rows with the same id stay in train OR test, never both."""
    rng = random.Random(seed)
    by_id: dict[str, list[int]] = defaultdict(list)
    for i, eid in enumerate(ids):
        by_id[eid].append(i)
    unique = list(by_id)
    rng.shuffle(unique)
    cut = int((1 - test_frac) * len(unique))
    train_ids, test_ids = set(unique[:cut]), set(unique[cut:])
    train = [i for eid in train_ids for i in by_id[eid]]
    test = [i for eid in test_ids for i in by_id[eid]]
    return train, test
```

### Q20: Precision@k from ranked IDs

```python
def precision_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    hit = sum(1 for doc in ranked[:k] if doc in relevant)
    return hit / k
```

---

# 11. The 6-step brainstorm (say it out loud)

Memorize the **bold** questions. This is the round, not the code.

**Step 1 — Restate (15 s)**  
*"So I need to return the indices of two numbers that add to target, and I can assume there is at most one valid pair?"*

**Step 2 — Ask constraints (30 s)** — pick the ones that change the algorithm:

| Ask | If they say yes… | Then use |
|---|---|---|
| Is the array sorted? | Yes | two pointers, O(1) extra |
| Can I use extra memory? | No | two pointers / in-place / sort |
| Duplicates? Multiple answers? | Yes | clarify: any pair vs all pairs |
| In-place? | Yes | read/write pointers |
| n? Does it fit in RAM? | Streaming / huge | generator, one pass |
| Linked list vs array? | List | fast/slow, not index math |

**Step 3 — Example (45 s)**  
Write `[2, 7, 11, 15], 9 → [0, 1]`. Then `[]`, `[1]`, `[3, 3]` target 6. If your plan dies on `[3, 3]`, fix the plan before coding.

**Step 4 — Pattern name (10 s)**  
*"Unsorted plus indices → hash map of value to index, one pass."*  
If you cannot name it, brute force first: *"O(n²) nested loops is correct; I will then replace the inner scan with a dict."* Brute-then-optimize is a **pass**. Silent brute is a fail.

**Step 5 — Code (the rest)**  
Types on the signature. Happy path. Then edges. Do not optimize names while the logic is wrong.

**Step 6 — Close (20 s)**  
*"Time O(n), extra space O(n) for the map. Empty list returns []. I stored after lookup so I don't use the same index twice."*

### If you get stuck

1. Say *"I'm going to write the O(n²) version so we have a correct baseline."*
2. Point at the inner loop: *"this scan is 'have I seen complement?' — that's a dict."*
3. If still stuck, ask: *"May I sort? If yes I can two-pointer but I lose indices unless I store pairs."*

### If they allow Copilot / ChatGPT in the round

Narrate **before** the model types. *"I want a dict from value to index, lookup complement, store after."* Then debug their output: empty input, duplicates, complexity. They are grading whether you **steer and verify**, not whether you memorize.

### What you can ask them (this looks senior)

- Expected n and value ranges  
- Duplicate policy  
- Mutate input or return new  
- Integer vs float  
- Follow-up they will ask anyway: *"How would this change if the stream never ends?"* → one-pass hash / deque window, not `list(stream)`

---

# 12. Pocket complexity + 14-day drill

## 12.1 Retrieve in two seconds

| Structure | Access | Search | Insert / delete | Notes |
|---|---|---|---|---|
| `list` | O(1) | O(n) | end O(1) amort., middle O(n) | `pop(0)` is O(n) — use `deque` |
| `deque` | ends O(1) | O(n) | ends O(1) | BFS, sliding window |
| `dict` / `set` | — | avg O(1) | avg O(1) | worst O(n) if pathological hashes |
| `heapq` | peek O(1) | — | O(log n) | min-heap |
| sorted list + `bisect` | O(1) | O(log n) | O(n) | insert still shifts |

Sorting is O(n log n). Two nested loops O(n²). If n is 10⁵, n² is already too slow — say that.

## 12.2 14-day plan (30–40 min/day, not 4 hours of reading §1)

| Day | From-memory (no notes) | Then check |
|---|---|---|
| 1 | two-sum hash + two-sum sorted | §7.1, §7.3 |
| 2 | anagram Counter + group anagrams | §7.3 |
| 3 | valid parens + merge two sorted | Q13, Q14 |
| 4 | max sum of k + longest unique substring | §7.2 |
| 5 | remove dups in-place + move zeroes | §7.1 |
| 6 | prefix sum + subarray sum = k | §7.6, §7.3 |
| 7 | **mock 45 min:** two-sum + flatten nested + talk §11** | Q10, Q12 |
| 8 | LRU OrderedDict | Q11 |
| 9 | BFS + RRF | §7.5, Q16 |
| 10 | merge intervals + chunk_text | §7.9, Q17 |
| 11 | rate limiter + precision@k | Q18, Q20 |
| 12 | entity split + Kadane | Q19, §7.6 |
| 13 | heap top-k + binary search template | §7.7, §7.8 |
| 14 | **mock:** one AI-eng helper + one DS split, out loud | §10 |

After day 7, if still freezing, record yourself on step 1–4 only (no code) until the script is automatic.

## 12.3 Night-before card (screenshot this)

```
unsorted pair/sum/index     → dict value→i, lookup then store
sorted pair / in-place      → left/right or read/write
contiguous constraint       → window (fixed k add/drop, else shrink left)
frequency / anagram / group → Counter / defaultdict / canonical key
subarray sum = k            → prefix + dict
top k                       → heap / Counter.most_common
overlap ranges              → sort by start, sweep
brackets / next greater     → stack
shortest unweighted         → BFS deque
then: empty, dupes, [3,3], complexity out loud
```

---

# 13. Key Takeaways

---

### For Rahul's Interviews

1. **Python is strongly and dynamically typed.** Know the difference — it's often the first question.

2. **Mutability is the root of many bugs.** Understand which types are mutable, the mutable default argument trap, and shallow vs deep copy.

3. **Comprehensions > loops.** Use list/dict/set comprehensions for clean, Pythonic code. Use generator expressions for memory efficiency.

4. **Know your data structures and their time complexity.** `list` for sequences, `dict`/`set` for O(1) lookup, `deque` for queues, `heapq` for priority queues.

5. **OOP is about encapsulation and polymorphism**, not just inheritance. Know dunder methods — they make objects work naturally with Python's built-in operations.

6. **Decorators are functions that wrap functions.** Understand the mechanics (closure + `@wraps`), chaining, and decorators with arguments.

7. **Generators are your answer to "How do you handle large data in Python?"** Lazy evaluation, O(1) memory, pipeline pattern.

8. **The GIL is a CPython implementation detail.** Threading helps for I/O, multiprocessing for CPU-bound. `concurrent.futures` provides a clean API for both.

9. **Type hints are for documentation and tooling**, not runtime enforcement. They massively improve code quality in production systems.

10. **In data science interviews, show you know the ecosystem.** NumPy broadcasting, Pandas method chaining, sklearn Pipelines — these demonstrate practical experience beyond textbook knowledge.

11. **Always mention edge cases.** Empty inputs, None values, duplicate keys, single-element lists. This is what separates senior candidates.

12. **Testing is not optional.** Know `pytest`, parametrized tests, fixtures, and mocking. Production code requires tests.

13. **The live coding round is a talking pattern-match, not a language exam.** Restate → ask sorted/duplicates/memory → tiny example → name hash vs two-pointers vs window → code → complexity. Until the pattern is named, do not type. Script is §11.

14. **Four structures cover almost every DS/AI-eng screen:** `dict`/`Counter`, two pointers, sliding window, `heapq`. Stack, deque, and BFS are the backups. §0.2 and §7.0.

15. **Unsorted two-sum is a hash map; sorted two-sum is two pointers.** Mixing them is the most common fail. Lookup complement **before** you store the current value.

16. **Code what the job codes.** AI Engineer: RRF, chunking, LRU, rate limit, interval merge, entity-safe split. Data Scientist: group-by dicts, windows, precision@k, no-leak splits. Not 200 LeetCode hards. §10.

17. **Brute force out loud is a pass; silent heroics are a fail.** O(n²) then "the inner scan is a dict" is how you recover when stuck.

---

*Document prepared for Rahul Sharma — Python for AI Engineer / Data Science Interviews. Language core last updated February 2026; coding-round path (§0, §7.0–§7.10, §10–§12) added August 2026.*
