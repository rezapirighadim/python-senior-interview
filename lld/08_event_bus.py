"""
=============================================================================
LLD 08: IN-MEMORY EVENT BUS / MESSAGE BROKER
=============================================================================
Design an event bus that supports pub/sub, topic filtering,
and async handlers.

KEY CONCEPTS:
  - Observer pattern
  - Wildcard topic matching
  - Async support
  - Middleware/interceptors
=============================================================================
"""
import asyncio
import fnmatch
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable
from collections import defaultdict


@dataclass
class Event:
    topic: str
    data: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: str = ""

    def __post_init__(self):
        if not self.event_id:
            import uuid
            self.event_id = str(uuid.uuid4())[:8]


# Type aliases
SyncHandler = Callable[[Event], None]
AsyncHandler = Callable[[Event], Any]  # coroutine


class EventBus:
    """
    In-memory event bus with topic-based pub/sub.
    Supports exact topics and wildcard patterns.

    Examples:
      bus.subscribe("order.created", handler)     # exact match
      bus.subscribe("order.*", handler)            # wildcard
      bus.publish("order.created", {"id": "123"})  # triggers both
    """

    def __init__(self):
        self._handlers: dict[str, list[SyncHandler]] = defaultdict(list)
        self._async_handlers: dict[str, list[AsyncHandler]] = defaultdict(list)
        self._middlewares: list[Callable[[Event], Event | None]] = []
        self._history: list[Event] = []

    # --- Subscribe ---
    def subscribe(self, topic: str, handler: SyncHandler) -> Callable:
        """Subscribe a sync handler to a topic (supports wildcards)."""
        self._handlers[topic].append(handler)
        return lambda: self._handlers[topic].remove(handler)  # unsubscribe fn

    def subscribe_async(self, topic: str, handler: AsyncHandler) -> Callable:
        """Subscribe an async handler."""
        self._async_handlers[topic].append(handler)
        return lambda: self._async_handlers[topic].remove(handler)

    # --- Middleware ---
    def use(self, middleware: Callable[[Event], Event | None]) -> None:
        """Add middleware. Return None from middleware to block the event."""
        self._middlewares.append(middleware)

    # --- Publish ---
    def publish(self, topic: str, data: dict[str, Any] | None = None) -> Event:
        """Publish an event synchronously."""
        event = Event(topic=topic, data=data or {})

        # Run through middleware
        for mw in self._middlewares:
            event = mw(event)
            if event is None:
                return Event(topic=topic, data=data or {})  # blocked

        self._history.append(event)

        # Find matching handlers (exact + wildcard)
        for pattern, handlers in self._handlers.items():
            if fnmatch.fnmatch(topic, pattern):
                for handler in handlers:
                    try:
                        handler(event)
                    except Exception as e:
                        print(f"  [ERROR] Handler failed: {e}")

        return event

    async def publish_async(self, topic: str, data: dict[str, Any] | None = None) -> Event:
        """Publish an event and run async handlers."""
        event = self.publish(topic, data)  # run sync handlers first

        for pattern, handlers in self._async_handlers.items():
            if fnmatch.fnmatch(topic, pattern):
                await asyncio.gather(
                    *[h(event) for h in handlers],
                    return_exceptions=True,
                )

        return event

    # --- Utility ---
    def get_history(self, topic: str | None = None) -> list[Event]:
        if topic:
            return [e for e in self._history if fnmatch.fnmatch(e.topic, topic)]
        return self._history

    def clear(self) -> None:
        self._handlers.clear()
        self._async_handlers.clear()
        self._history.clear()


# --- Demo ---
if __name__ == "__main__":
    bus = EventBus()

    # Add logging middleware
    def log_middleware(event: Event) -> Event:
        print(f"  [MW] Event: {event.topic} | {event.data}")
        return event

    bus.use(log_middleware)

    # Subscribe to exact topics
    bus.subscribe("order.created", lambda e: print(f"  [OrderService] New order: {e.data}"))
    bus.subscribe("order.created", lambda e: print(f"  [EmailService] Sending confirmation"))
    bus.subscribe("order.cancelled", lambda e: print(f"  [RefundService] Processing refund"))

    # Subscribe with wildcard
    bus.subscribe("order.*", lambda e: print(f"  [Analytics] Tracking: {e.topic}"))
    bus.subscribe("*", lambda e: print(f"  [Audit] Logged: {e.topic}"))

    # Publish events
    print("--- Creating order ---")
    bus.publish("order.created", {"order_id": "ORD-001", "amount": 99.99})

    print("\n--- Cancelling order ---")
    bus.publish("order.cancelled", {"order_id": "ORD-001", "reason": "changed mind"})

    print("\n--- User signup ---")
    bus.publish("user.signup", {"user_id": "U-001", "email": "reza@example.com"})

    # Unsubscribe example
    print("\n--- Unsubscribe and publish again ---")
    unsub = bus.subscribe("test.topic", lambda e: print("  Should appear once"))
    bus.publish("test.topic", {})
    unsub()  # unsubscribe
    bus.publish("test.topic", {})  # handler won't fire

    # History
    print(f"\n--- History ---")
    print(f"Total events: {len(bus.get_history())}")
    print(f"Order events: {len(bus.get_history('order.*'))}")
