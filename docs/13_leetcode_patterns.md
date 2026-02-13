# 13 — LeetCode Patterns

## How to Approach Any Problem

1. **Understand** — restate it, ask clarifying questions
2. **Examples** — work through examples, find edge cases
3. **Pattern** — which pattern does this match?
4. **Brute force** — start simple, then optimize
5. **Code** — clean, meaningful names
6. **Test** — examples + edge cases
7. **Complexity** — state time & space

## Pattern 1: Two Pointers

**When:** Sorted array, finding pairs, palindrome

```python
def three_sum(nums):
    nums.sort()
    result = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i-1]: continue
        lo, hi = i + 1, len(nums) - 1
        while lo < hi:
            s = nums[i] + nums[lo] + nums[hi]
            if s == 0:
                result.append([nums[i], nums[lo], nums[hi]])
                lo += 1; hi -= 1
                while lo < hi and nums[lo] == nums[lo-1]: lo += 1
            elif s < 0: lo += 1
            else: hi -= 1
    return result
```

## Pattern 2: Sliding Window

**When:** Subarray/substring, contiguous sequence

```python
def longest_unique_substring(s):
    seen, left, max_len = set(), 0, 0
    for right in range(len(s)):
        while s[right] in seen:
            seen.remove(s[left]); left += 1
        seen.add(s[right])
        max_len = max(max_len, right - left + 1)
    return max_len
```

## Pattern 3: Binary Search on Answer

**When:** "Find minimum/maximum satisfying condition"

```python
def koko_bananas(piles, h):
    lo, hi = 1, max(piles)
    while lo < hi:
        mid = (lo + hi) // 2
        if sum(ceil(p/mid) for p in piles) <= h:
            hi = mid
        else:
            lo = mid + 1
    return lo
```

## Pattern 4: Dynamic Programming

**When:** "Count ways", "find optimal", overlapping subproblems

```python
def coin_change(coins, amount):
    dp = [float("inf")] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i-coin] + 1)
    return dp[amount] if dp[amount] != float("inf") else -1
```

## Pattern 5: Backtracking

**When:** "Generate all", combinations, permutations

```python
def subsets(nums):
    result = []
    def bt(start, current):
        result.append(current[:])
        for i in range(start, len(nums)):
            current.append(nums[i])
            bt(i + 1, current)
            current.pop()
    bt(0, [])
    return result
```

## Pattern 6: Merge Intervals

```python
def merge(intervals):
    intervals.sort()
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged
```

## Pattern 7: Topological Sort

```python
def course_order(n, prereqs):
    graph, indegree = defaultdict(list), [0]*n
    for c, p in prereqs:
        graph[p].append(c); indegree[c] += 1
    queue = deque(i for i in range(n) if indegree[i] == 0)
    order = []
    while queue:
        c = queue.popleft(); order.append(c)
        for nxt in graph[c]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0: queue.append(nxt)
    return order if len(order) == n else []
```

## Pattern 8: Monotonic Stack

```python
def daily_temperatures(temps):
    result, stack = [0]*len(temps), []
    for i, t in enumerate(temps):
        while stack and t > temps[stack[-1]]:
            j = stack.pop()
            result[j] = i - j
        stack.append(i)
    return result
```

## Pattern Recognition

| See This | Think This |
|----------|-----------|
| Sorted array + find pair | Two Pointers |
| Subarray of size k | Sliding Window |
| Min/max with condition | Binary Search on Answer |
| Count ways / optimize | DP |
| Generate all X | Backtracking |
| Overlapping ranges | Merge Intervals |
| Dependencies | Topological Sort |
| Next greater element | Monotonic Stack |
| Shortest path | BFS |
| Connected components | DFS / Union-Find |
| k-th largest | Heap |
