"""
=============================================================================
FILE 13: LEETCODE PATTERNS — How to THINK About Problems
=============================================================================
87% of FAANG interview questions use 10-12 core patterns. Learn the
PATTERN, not individual problems. When you see a new problem, ask:
"Which pattern does this remind me of?"

HOW TO APPROACH ANY PROBLEM:
  1. Understand the problem (restate it, ask clarifying questions)
  2. Think about examples (edge cases!)
  3. Identify the PATTERN
  4. Start with brute force, then optimize
  5. Code it cleanly
  6. Test with examples + edge cases
  7. Analyze time & space complexity
=============================================================================
"""


# =============================================================================
# PATTERN 1: TWO POINTERS
# =============================================================================
# WHEN: Sorted array, finding pairs, palindrome, removing duplicates
# KEY IDEA: Use two pointers moving toward each other or in same direction

def two_sum_sorted(numbers: list[int], target: int) -> list[int]:
    """
    Two Sum II — Input array is sorted.
    Time: O(n), Space: O(1)

    THINKING: Since sorted, if sum too big → move right pointer left.
    If sum too small → move left pointer right.
    """
    left, right = 0, len(numbers) - 1
    while left < right:
        current_sum = numbers[left] + numbers[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return []


def is_palindrome(s: str) -> bool:
    """
    Check if string is palindrome (ignoring non-alphanumeric).
    Time: O(n), Space: O(1)
    """
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


def three_sum(nums: list[int]) -> list[list[int]]:
    """
    Find all unique triplets that sum to zero.
    Time: O(n²), Space: O(1) excluding output

    THINKING: Sort array. Fix one number, then use two pointers for the rest.
    This reduces O(n³) brute force to O(n²).
    """
    nums.sort()
    result = []
    for i in range(len(nums) - 2):
        # Skip duplicates for first number
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        left, right = i + 1, len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total == 0:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:
                    left += 1  # Skip duplicates
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1
    return result


# =============================================================================
# PATTERN 2: SLIDING WINDOW
# =============================================================================
# WHEN: Subarray/substring problems, contiguous sequence, max/min in window
# KEY IDEA: Maintain a window [left, right], expand right, shrink left

def max_subarray_sum(nums: list[int], k: int) -> int:
    """
    Maximum sum of subarray of size k.
    Time: O(n), Space: O(1)

    THINKING: Instead of recalculating sum each time,
    slide the window: add new element, remove old element.
    """
    window_sum = sum(nums[:k])
    max_sum = window_sum
    for i in range(k, len(nums)):
        window_sum += nums[i] - nums[i - k]  # Slide: add new, remove old
        max_sum = max(max_sum, window_sum)
    return max_sum


def length_of_longest_substring(s: str) -> int:
    """
    Longest substring without repeating characters.
    Time: O(n), Space: O(min(n, 26))

    THINKING: Expand window right. If we see a duplicate, shrink from left
    until no duplicate. Track characters with a set.
    """
    char_set = set()
    left = 0
    max_length = 0
    for right in range(len(s)):
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        char_set.add(s[right])
        max_length = max(max_length, right - left + 1)
    return max_length


def min_window_substring(s: str, t: str) -> str:
    """
    Minimum window substring containing all characters of t.
    Time: O(n), Space: O(n)

    THINKING: Expand right until window contains all chars of t.
    Then shrink left to minimize. Track with frequency counter.
    """
    from collections import Counter

    if not s or not t:
        return ""

    need = Counter(t)
    have = {}
    formed = 0
    required = len(need)
    left = 0
    min_len = float("inf")
    min_start = 0

    for right in range(len(s)):
        char = s[right]
        have[char] = have.get(char, 0) + 1

        if char in need and have[char] == need[char]:
            formed += 1

        while formed == required:
            # Update minimum
            if right - left + 1 < min_len:
                min_len = right - left + 1
                min_start = left

            # Shrink from left
            left_char = s[left]
            have[left_char] -= 1
            if left_char in need and have[left_char] < need[left_char]:
                formed -= 1
            left += 1

    return "" if min_len == float("inf") else s[min_start:min_start + min_len]


# =============================================================================
# PATTERN 3: BINARY SEARCH (on answer space)
# =============================================================================
# WHEN: "Find minimum/maximum value that satisfies condition"
# KEY IDEA: Binary search isn't just for sorted arrays!

def search_rotated(nums: list[int], target: int) -> int:
    """
    Search in a rotated sorted array. [4,5,6,7,0,1,2], target=0 → 4
    Time: O(log n)

    THINKING: At each mid, one half is ALWAYS sorted. Check if target
    is in the sorted half. If yes, search there. If no, search other half.
    """
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid

        # Left half is sorted
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Right half is sorted
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1


def koko_eating_bananas(piles: list[int], h: int) -> int:
    """
    Koko eats bananas at speed k. Find minimum k to finish in h hours.
    Time: O(n * log(max_pile))

    THINKING: Binary search on the ANSWER (speed k).
    For each candidate speed, check if she can finish in time.
    """
    import math

    def can_finish(speed: int) -> bool:
        hours = sum(math.ceil(pile / speed) for pile in piles)
        return hours <= h

    left, right = 1, max(piles)
    while left < right:
        mid = (left + right) // 2
        if can_finish(mid):
            right = mid  # Try slower speed
        else:
            left = mid + 1  # Need faster speed
    return left


# =============================================================================
# PATTERN 4: DYNAMIC PROGRAMMING
# =============================================================================
# WHEN: "Count number of ways", "Find optimal", overlapping subproblems
# KEY IDEA: Break into subproblems, store results, build up solution

def climb_stairs(n: int) -> int:
    """
    Number of ways to climb n stairs (1 or 2 steps at a time).
    Time: O(n), Space: O(1)

    THINKING: dp[i] = dp[i-1] + dp[i-2] (Fibonacci!)
    At step i, you could have come from step i-1 or i-2.
    """
    if n <= 2:
        return n
    prev2, prev1 = 1, 2
    for _ in range(3, n + 1):
        curr = prev1 + prev2
        prev2, prev1 = prev1, curr
    return prev1


def longest_common_subsequence(text1: str, text2: str) -> int:
    """
    LCS — classic 2D DP.
    Time: O(m*n), Space: O(m*n)

    THINKING: If chars match, LCS extends by 1. If not, take max of
    skipping from either string.

    dp[i][j] = LCS of text1[:i] and text2[:j]
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def coin_change(coins: list[int], amount: int) -> int:
    """
    Minimum coins to make amount. Return -1 if impossible.
    Time: O(amount * len(coins)), Space: O(amount)

    THINKING: dp[i] = min coins to make amount i.
    For each amount, try all coins: dp[i] = min(dp[i], dp[i-coin] + 1)
    """
    dp = [float("inf")] * (amount + 1)
    dp[0] = 0

    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i and dp[i - coin] != float("inf"):
                dp[i] = min(dp[i], dp[i - coin] + 1)

    return dp[amount] if dp[amount] != float("inf") else -1


# =============================================================================
# PATTERN 5: BACKTRACKING
# =============================================================================
# WHEN: "Generate all", "find all combinations/permutations"
# KEY IDEA: Try a choice, recurse, UNDO the choice (backtrack)

def subsets(nums: list[int]) -> list[list[int]]:
    """
    Generate all subsets.
    Time: O(n * 2^n)

    THINKING: For each element, choose to INCLUDE or EXCLUDE it.
    """
    result = []

    def backtrack(start: int, current: list[int]):
        result.append(current[:])  # Copy current subset
        for i in range(start, len(nums)):
            current.append(nums[i])     # Choose
            backtrack(i + 1, current)    # Explore
            current.pop()               # Un-choose (backtrack!)

    backtrack(0, [])
    return result


def permutations(nums: list[int]) -> list[list[int]]:
    """
    Generate all permutations.
    Time: O(n * n!)
    """
    result = []

    def backtrack(current: list[int], remaining: set[int]):
        if not remaining:
            result.append(current[:])
            return
        for num in list(remaining):
            current.append(num)
            remaining.remove(num)
            backtrack(current, remaining)
            current.pop()
            remaining.add(num)

    backtrack([], set(nums))
    return result


# =============================================================================
# PATTERN 6: MERGE INTERVALS
# =============================================================================
# WHEN: Overlapping ranges, scheduling, time conflicts

def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    """
    Merge overlapping intervals.
    Time: O(n log n), Space: O(n)

    THINKING: Sort by start. If current overlaps with last merged,
    extend the last merged interval. Otherwise, add new interval.
    """
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]

    for start, end in intervals[1:]:
        if start <= merged[-1][1]:  # Overlaps
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return merged


# =============================================================================
# PATTERN 7: TOPOLOGICAL SORT
# =============================================================================
# WHEN: Dependencies, course scheduling, build ordering

def course_schedule(num_courses: int, prerequisites: list[list[int]]) -> list[int]:
    """
    Find a valid order to take courses (Kahn's algorithm — BFS).
    Time: O(V + E)

    THINKING: Start with courses that have NO prerequisites.
    After "taking" them, reduce dependency count of dependent courses.
    Repeat. If all courses taken, order is valid.
    """
    from collections import deque

    graph = {i: [] for i in range(num_courses)}
    in_degree = [0] * num_courses

    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1

    # Start with courses that have no prerequisites
    queue = deque([i for i in range(num_courses) if in_degree[i] == 0])
    order = []

    while queue:
        course = queue.popleft()
        order.append(course)
        for next_course in graph[course]:
            in_degree[next_course] -= 1
            if in_degree[next_course] == 0:
                queue.append(next_course)

    return order if len(order) == num_courses else []  # Empty = cycle!


# =============================================================================
# PATTERN 8: MONOTONIC STACK
# =============================================================================
# WHEN: "Next greater/smaller element", temperature problems

def daily_temperatures(temperatures: list[int]) -> list[int]:
    """
    For each day, find how many days until a warmer day.
    Time: O(n), Space: O(n)

    THINKING: Maintain a stack of indices with decreasing temperatures.
    When we find a warmer day, pop from stack and record the difference.
    """
    n = len(temperatures)
    result = [0] * n
    stack = []  # Stores indices

    for i in range(n):
        while stack and temperatures[i] > temperatures[stack[-1]]:
            prev_idx = stack.pop()
            result[prev_idx] = i - prev_idx
        stack.append(i)

    return result


# =============================================================================
# PATTERN RECOGNITION CHEAT SHEET
# =============================================================================
"""
"sorted array + find pair"           → Two Pointers
"subarray/substring of size k"       → Sliding Window
"find min/max satisfying condition"  → Binary Search on Answer
"count ways / optimize"             → Dynamic Programming
"generate all combinations"          → Backtracking
"overlapping intervals"              → Merge Intervals → Sort + Merge
"dependencies / ordering"            → Topological Sort
"next greater element"               → Monotonic Stack
"shortest path (unweighted)"         → BFS
"connected components"               → DFS / Union-Find
"k-th largest/smallest"              → Heap
"prefix lookup / autocomplete"       → Trie
"find duplicate / cycle"             → Fast & Slow Pointers
"""


if __name__ == "__main__":
    print("=" * 60)
    print("FILE 13: LeetCode Patterns")
    print("=" * 60)

    print("\n--- Two Pointers: Three Sum ---")
    print(f"  [-1,0,1,2,-1,-4] → {three_sum([-1, 0, 1, 2, -1, -4])}")

    print("\n--- Sliding Window: Longest Substring ---")
    print(f"  'abcabcbb' → {length_of_longest_substring('abcabcbb')}")

    print("\n--- Binary Search: Rotated Array ---")
    print(f"  [4,5,6,7,0,1,2] find 0 → idx {search_rotated([4,5,6,7,0,1,2], 0)}")

    print("\n--- DP: Coin Change ---")
    print(f"  coins=[1,5,10,25] amount=36 → {coin_change([1,5,10,25], 36)} coins")

    print("\n--- DP: LCS ---")
    print(f"  'abcde' vs 'ace' → {longest_common_subsequence('abcde', 'ace')}")

    print("\n--- Backtracking: Subsets ---")
    print(f"  [1,2,3] → {subsets([1, 2, 3])}")

    print("\n--- Merge Intervals ---")
    intervals = [[1,3],[2,6],[8,10],[15,18]]
    print(f"  {intervals} → {merge_intervals(intervals)}")

    print("\n--- Topological Sort: Course Schedule ---")
    print(f"  4 courses, [[1,0],[2,0],[3,1],[3,2]] → {course_schedule(4, [[1,0],[2,0],[3,1],[3,2]])}")

    print("\n--- Monotonic Stack: Daily Temperatures ---")
    temps = [73, 74, 75, 71, 69, 72, 76, 73]
    print(f"  {temps} → {daily_temperatures(temps)}")

    print("\n✓ File 13 complete. Move to 14_python_for_ai.py")
