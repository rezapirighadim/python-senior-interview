# 12 — Data Structures & Algorithms

## Complexity Cheat Sheet

### Python Built-in Operations

| Structure | Operation | Complexity |
|-----------|-----------|-----------|
| **list** | Access by index | O(1) |
| | Append | O(1) amortized |
| | Insert/Delete at i | O(n) |
| | Search | O(n) |
| | Sort | O(n log n) |
| **dict** | Get/Set/Delete | O(1) avg |
| **set** | Add/Remove/Lookup | O(1) |
| **deque** | Append/Pop (both ends) | O(1) |
| **heapq** | Push/Pop | O(log n) |
| | Peek min | O(1) |

## Hash Map Problems

### Two Sum

```python
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        if target - num in seen:
            return [seen[target - num], i]
        seen[num] = i
```

## Linked List

```python
def reverse(head):
    prev, curr = None, head
    while curr:
        curr.next, prev, curr = prev, curr, curr.next
    return prev

def has_cycle(head):  # Floyd's fast/slow
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
        if slow is fast: return True
    return False
```

## Stack

### Valid Parentheses

```python
def is_valid(s):
    stack, pairs = [], {")": "(", "}": "{", "]": "["}
    for c in s:
        if c in "({[": stack.append(c)
        elif not stack or stack.pop() != pairs[c]: return False
    return not stack
```

## Trees

```python
def max_depth(root):
    if not root: return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))

def level_order(root):  # BFS with deque
    queue, result = deque([root]), []
    while queue:
        level = [queue.popleft() for _ in range(len(queue))]
        result.append([n.val for n in level])
        for n in level:
            if n.left: queue.append(n.left)
            if n.right: queue.append(n.right)
    return result
```

## Graphs

```python
def bfs(graph, start):
    visited, queue = {start}, deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

def dfs(graph, node, visited=None):
    if visited is None: visited = set()
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
```

## Heap / Priority Queue

```python
import heapq
# Python's heapq is a MIN-heap
heapq.heappush(heap, item)
smallest = heapq.heappop(heap)
top_k = heapq.nlargest(k, items)
```

## Binary Search

```python
def binary_search(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target: return mid
        elif nums[mid] < target: lo = mid + 1
        else: hi = mid - 1
    return -1
```

## Trie

```python
class Trie:
    def __init__(self):
        self.root = {}

    def insert(self, word):
        node = self.root
        for c in word:
            node = node.setdefault(c, {})
        node["#"] = True

    def search(self, word):
        node = self.root
        for c in word:
            if c not in node: return False
            node = node[c]
        return "#" in node
```
