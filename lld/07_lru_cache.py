"""
=============================================================================
LLD 07: LRU CACHE
=============================================================================
Design a Least Recently Used cache with O(1) get and put.

KEY CONCEPTS:
  - Doubly linked list for O(1) removal and insertion
  - Hash map for O(1) lookup
  - This is a real interview classic (LeetCode #146)
=============================================================================
"""


class Node:
    """Doubly linked list node."""
    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key: int = 0, value: int = 0):
        self.key = key
        self.value = value
        self.prev: Node | None = None
        self.next: Node | None = None


class LRUCache:
    """
    Least Recently Used Cache.
    - get(key): O(1)
    - put(key, value): O(1)
    - Evicts least recently used item when capacity is full.

    HOW IT WORKS:
    - Hash map: key -> Node (O(1) lookup)
    - Doubly linked list: ordered by recency
      HEAD <-> [most recent] <-> ... <-> [least recent] <-> TAIL
    - On access: move node to front
    - On evict: remove node from back
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: dict[int, Node] = {}

        # Dummy head and tail (avoid edge cases)
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node) -> None:
        """Remove node from linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_front(self, node: Node) -> None:
        """Add node right after head (most recent position)."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def _move_to_front(self, node: Node) -> None:
        """Mark as most recently used."""
        self._remove(node)
        self._add_to_front(node)

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._move_to_front(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            self._move_to_front(node)
            return

        # Add new node
        node = Node(key, value)
        self.cache[key] = node
        self._add_to_front(node)

        # Evict if over capacity
        if len(self.cache) > self.capacity:
            lru = self.tail.prev  # least recently used
            self._remove(lru)
            del self.cache[lru.key]

    def __repr__(self) -> str:
        items = []
        node = self.head.next
        while node != self.tail:
            items.append(f"{node.key}:{node.value}")
            node = node.next
        return f"LRUCache([{', '.join(items)}])"


# --- Simpler version using OrderedDict (also valid in interviews) ---
from collections import OrderedDict


class LRUCacheSimple:
    """
    Same behavior, using Python's OrderedDict.
    More Pythonic but interviewer may ask you to implement from scratch.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # remove oldest


# --- Demo ---
if __name__ == "__main__":
    print("=== LRU Cache (Linked List implementation) ===")
    cache = LRUCache(3)

    cache.put(1, 10)
    cache.put(2, 20)
    cache.put(3, 30)
    print(f"After adding 1,2,3: {cache}")

    print(f"get(1) = {cache.get(1)}")  # 10, moves 1 to front
    print(f"After get(1): {cache}")

    cache.put(4, 40)  # evicts key 2 (least recent)
    print(f"After put(4): {cache}")
    print(f"get(2) = {cache.get(2)}")  # -1, was evicted

    cache.put(5, 50)  # evicts key 3
    print(f"After put(5): {cache}")
    print(f"get(3) = {cache.get(3)}")  # -1, was evicted
    print(f"get(4) = {cache.get(4)}")  # 40

    print(f"\n=== LRU Cache (OrderedDict implementation) ===")
    cache2 = LRUCacheSimple(2)
    cache2.put(1, 1)
    cache2.put(2, 2)
    print(f"get(1) = {cache2.get(1)}")  # 1
    cache2.put(3, 3)                     # evicts 2
    print(f"get(2) = {cache2.get(2)}")  # -1
