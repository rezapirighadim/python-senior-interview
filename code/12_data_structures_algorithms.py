"""
=============================================================================
FILE 12: DATA STRUCTURES & ALGORITHMS — Python Implementation
=============================================================================
You MUST know the time complexity of common operations. Interviewers WILL
ask "what's the time complexity?" for every solution you write.

BIG-O CHEAT SHEET:
  O(1)       → Constant    → dict lookup, list append
  O(log n)   → Logarithmic → binary search
  O(n)       → Linear      → list traversal
  O(n log n) → Linearithmic → sorting (timsort)
  O(n²)      → Quadratic   → nested loops
  O(2ⁿ)      → Exponential → recursive subsets
  O(n!)      → Factorial   → permutations
=============================================================================
"""
from collections import defaultdict, deque
import heapq


# =============================================================================
# 1. PYTHON BUILT-IN DATA STRUCTURE COMPLEXITY
# =============================================================================
"""
LIST (dynamic array):
  Access by index:  O(1)
  Append:           O(1) amortized
  Insert at i:      O(n)
  Delete at i:      O(n)
  Search:           O(n)
  Sort:             O(n log n)

DICT (hash table):
  Get/Set/Delete:   O(1) average, O(n) worst case
  Search by key:    O(1)
  Search by value:  O(n)

SET (hash table):
  Add/Remove:       O(1)
  Membership test:  O(1)   ← This is why sets are great for lookups!
  Union:            O(m+n)
  Intersection:     O(min(m,n))

DEQUE (double-ended queue):
  Append left/right: O(1)
  Pop left/right:    O(1)
  Access by index:   O(n)  ← Worse than list for random access!

HEAPQ (binary heap / priority queue):
  Push:              O(log n)
  Pop min:           O(log n)
  Peek min:          O(1)
"""


# =============================================================================
# 2. HASH MAP — The Most Used Data Structure
# =============================================================================

# --- Two Sum (classic interview problem) ---
def two_sum(nums: list[int], target: int) -> list[int]:
    """
    Find two indices whose values add up to target.
    Time: O(n), Space: O(n)

    THINKING PROCESS:
    1. Brute force: check every pair → O(n²)
    2. Better: for each number, I need `target - number`. Where is it?
    3. Store seen numbers in a hash map → O(1) lookup!
    """
    seen = {}  # value → index
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

assert two_sum([2, 7, 11, 15], 9) == [0, 1]
assert two_sum([3, 2, 4], 6) == [1, 2]


# --- Group Anagrams ---
def group_anagrams(strs: list[str]) -> list[list[str]]:
    """
    Group strings that are anagrams of each other.
    Time: O(n * k log k) where k is max string length
    Space: O(n * k)

    KEY INSIGHT: Anagrams have the same sorted characters.
    "eat" → "aet", "tea" → "aet", "ate" → "aet"
    """
    groups = defaultdict(list)
    for s in strs:
        key = "".join(sorted(s))  # or tuple(sorted(s))
        groups[key].append(s)
    return list(groups.values())


# =============================================================================
# 3. LINKED LIST
# =============================================================================

class ListNode:
    def __init__(self, val: int = 0, next: "ListNode | None" = None):
        self.val = val
        self.next = next


def reverse_linked_list(head: ListNode | None) -> ListNode | None:
    """
    Reverse a linked list in-place.
    Time: O(n), Space: O(1)

    THINKING: Use three pointers: prev, curr, next
    At each step: point curr.next to prev, then advance all pointers.
    """
    prev = None
    curr = head
    while curr:
        next_node = curr.next  # Save next
        curr.next = prev       # Reverse link
        prev = curr            # Advance prev
        curr = next_node       # Advance curr
    return prev


def has_cycle(head: ListNode | None) -> bool:
    """
    Detect cycle in linked list using Floyd's algorithm (fast/slow pointers).
    Time: O(n), Space: O(1)

    THINKING: If there's a cycle, a fast pointer (2 steps) will
    eventually meet a slow pointer (1 step) — like running on a track.
    """
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True
    return False


# =============================================================================
# 4. STACK & QUEUE
# =============================================================================

# --- Valid Parentheses (classic stack problem) ---
def is_valid_parentheses(s: str) -> bool:
    """
    Check if parentheses are balanced.
    Time: O(n), Space: O(n)

    THINKING: Opening brackets push onto stack.
    Closing brackets must match the most recent opening → LIFO → stack!
    """
    stack = []
    pairs = {")": "(", "}": "{", "]": "["}
    for char in s:
        if char in "({[":
            stack.append(char)
        elif char in ")}]":
            if not stack or stack[-1] != pairs[char]:
                return False
            stack.pop()
    return len(stack) == 0

assert is_valid_parentheses("(){}[]") is True
assert is_valid_parentheses("(]") is False
assert is_valid_parentheses("([)]") is False
assert is_valid_parentheses("{[]}") is True


# --- Min Stack ---
class MinStack:
    """
    Stack that supports O(1) get_min.
    TRICK: Store (value, current_min) pairs.
    """
    def __init__(self):
        self.stack: list[tuple[int, int]] = []

    def push(self, val: int) -> None:
        current_min = min(val, self.stack[-1][1] if self.stack else val)
        self.stack.append((val, current_min))

    def pop(self) -> int:
        return self.stack.pop()[0]

    def top(self) -> int:
        return self.stack[-1][0]

    def get_min(self) -> int:
        return self.stack[-1][1]


# =============================================================================
# 5. TREES
# =============================================================================

class TreeNode:
    def __init__(self, val: int = 0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# --- Tree traversals ---
def inorder(root: TreeNode | None) -> list[int]:
    """Left → Root → Right (sorted order for BST)"""
    if not root:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)

def preorder(root: TreeNode | None) -> list[int]:
    """Root → Left → Right"""
    if not root:
        return []
    return [root.val] + preorder(root.left) + preorder(root.right)

def level_order(root: TreeNode | None) -> list[list[int]]:
    """BFS — level by level. Uses QUEUE (deque)."""
    if not root:
        return []
    result = []
    queue = deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    return result


# --- Max depth of binary tree ---
def max_depth(root: TreeNode | None) -> int:
    """
    Time: O(n), Space: O(h) where h is height

    THINKING: Depth of tree = 1 + max(depth of left, depth of right)
    This is the essence of tree recursion.
    """
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))


# --- Validate BST ---
def is_valid_bst(root: TreeNode | None) -> bool:
    """
    THINKING: Each node has a valid range. Root: (-inf, inf).
    Left child of 5: (-inf, 5). Right child of 5: (5, inf).
    """
    def validate(node, low=float("-inf"), high=float("inf")):
        if not node:
            return True
        if node.val <= low or node.val >= high:
            return False
        return (
            validate(node.left, low, node.val)
            and validate(node.right, node.val, high)
        )
    return validate(root)


# =============================================================================
# 6. GRAPHS
# =============================================================================

# Graph representation: adjacency list
graph = {
    "A": ["B", "C"],
    "B": ["A", "D", "E"],
    "C": ["A", "F"],
    "D": ["B"],
    "E": ["B", "F"],
    "F": ["C", "E"],
}


def bfs(graph: dict, start: str) -> list[str]:
    """
    Breadth-First Search — explores level by level.
    Use for: shortest path (unweighted), level-order traversal.
    Time: O(V + E), Space: O(V)
    """
    visited = set()
    queue = deque([start])
    order = []
    visited.add(start)

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return order


def dfs(graph: dict, start: str) -> list[str]:
    """
    Depth-First Search — explores as deep as possible first.
    Use for: cycle detection, topological sort, path finding.
    """
    visited = set()
    order = []

    def explore(node):
        visited.add(node)
        order.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                explore(neighbor)

    explore(start)
    return order


# --- Number of Islands (classic BFS/DFS problem) ---
def num_islands(grid: list[list[str]]) -> int:
    """
    Count connected components of "1"s in a grid.
    Time: O(m*n), Space: O(m*n)

    THINKING: For each unvisited "1", start BFS/DFS to mark all
    connected "1"s as visited. Each BFS/DFS = one island.
    """
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    count = 0

    def sink_island(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != "1":
            return
        grid[r][c] = "0"  # Mark as visited
        sink_island(r + 1, c)
        sink_island(r - 1, c)
        sink_island(r, c + 1)
        sink_island(r, c - 1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1":
                count += 1
                sink_island(r, c)

    return count


# =============================================================================
# 7. HEAP / PRIORITY QUEUE
# =============================================================================

# Python's heapq is a MIN-heap
def top_k_frequent(nums: list[int], k: int) -> list[int]:
    """
    Find k most frequent elements.
    Time: O(n log k), Space: O(n)

    THINKING: Count frequencies, then use heap to get top k.
    """
    from collections import Counter
    count = Counter(nums)
    # nlargest uses a heap internally
    return [item for item, _ in count.most_common(k)]

assert top_k_frequent([1, 1, 1, 2, 2, 3], 2) == [1, 2]


# --- Merge K Sorted Lists ---
def merge_k_sorted(lists: list[list[int]]) -> list[int]:
    """
    Merge k sorted lists into one sorted list.
    Time: O(N log k), Space: O(k)

    THINKING: Use min-heap to always pick the smallest available element.
    """
    heap = []
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))  # (value, list_index, elem_index)

    result = []
    while heap:
        val, list_i, elem_i = heapq.heappop(heap)
        result.append(val)
        if elem_i + 1 < len(lists[list_i]):
            next_val = lists[list_i][elem_i + 1]
            heapq.heappush(heap, (next_val, list_i, elem_i + 1))

    return result


# =============================================================================
# 8. SORTING — Know These
# =============================================================================

"""
Python's sort: Timsort — O(n log n), stable, adaptive

Built-in sorting:
  sorted(items)                    # Returns new list
  items.sort()                     # In-place
  sorted(items, key=lambda x: x.age)  # Custom key
  sorted(items, reverse=True)      # Descending

When interviewers ask about sorting algorithms:
  Quick Sort: O(n log n) avg, O(n²) worst — not stable
  Merge Sort: O(n log n) always — stable, extra space O(n)
  Heap Sort:  O(n log n) always — not stable, O(1) extra space
  Counting Sort: O(n+k) — only for integers in range [0, k]
"""

# Quick sort implementation (for interview)
def quicksort(arr: list[int]) -> list[int]:
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)


# =============================================================================
# 9. BINARY SEARCH — O(log n)
# =============================================================================

def binary_search(nums: list[int], target: int) -> int:
    """
    Classic binary search. Returns index or -1.
    MUST KNOW: This is the foundation of many interview problems.
    """
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2  # Avoid overflow
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1


# --- Binary search variant: find first/last occurrence ---
def find_first(nums: list[int], target: int) -> int:
    """Find the FIRST occurrence of target."""
    left, right = 0, len(nums) - 1
    result = -1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            result = mid
            right = mid - 1  # Keep searching left
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return result


# =============================================================================
# 10. TRIE — Prefix Tree
# =============================================================================

class TrieNode:
    def __init__(self):
        self.children: dict[str, TrieNode] = {}
        self.is_word: bool = False

class Trie:
    """
    Prefix tree — efficient for word search, autocomplete.
    Insert/Search: O(m) where m is word length.
    """
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True

    def search(self, word: str) -> bool:
        node = self._find_node(word)
        return node is not None and node.is_word

    def starts_with(self, prefix: str) -> bool:
        return self._find_node(prefix) is not None

    def _find_node(self, prefix: str) -> TrieNode | None:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node


if __name__ == "__main__":
    print("=" * 60)
    print("FILE 12: Data Structures & Algorithms")
    print("=" * 60)

    print("\n--- Two Sum ---")
    print(f"  two_sum([2,7,11,15], 9) = {two_sum([2, 7, 11, 15], 9)}")

    print("\n--- Valid Parentheses ---")
    print(f"  '(){{}}[]' → {is_valid_parentheses('(){}[]')}")
    print(f"  '(]'     → {is_valid_parentheses('(]')}")

    print("\n--- Binary Tree ---")
    root = TreeNode(3, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7)))
    print(f"  Level order: {level_order(root)}")
    print(f"  Max depth: {max_depth(root)}")

    print("\n--- Graph BFS ---")
    print(f"  BFS from A: {bfs(graph, 'A')}")
    print(f"  DFS from A: {dfs(graph, 'A')}")

    print("\n--- Heap: Top K Frequent ---")
    print(f"  top_k_frequent([1,1,1,2,2,3], 2) = {top_k_frequent([1,1,1,2,2,3], 2)}")

    print("\n--- Binary Search ---")
    nums = [1, 3, 5, 7, 9, 11, 13]
    print(f"  search({nums}, 7) = index {binary_search(nums, 7)}")

    print("\n--- Trie ---")
    trie = Trie()
    for word in ["apple", "app", "apricot", "banana"]:
        trie.insert(word)
    print(f"  search('app') = {trie.search('app')}")
    print(f"  starts_with('app') = {trie.starts_with('app')}")
    print(f"  search('application') = {trie.search('application')}")

    print("\n✓ File 12 complete. Move to 13_leetcode_patterns.py")
