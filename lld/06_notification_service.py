"""
=============================================================================
LLD 06: NOTIFICATION SERVICE
=============================================================================
Design a notification system that sends via email, SMS, push.
Supports user preferences and priority levels.

KEY CONCEPTS:
  - Strategy pattern for delivery channels
  - Observer pattern for event-driven dispatch
  - Template method for formatting
=============================================================================
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Protocol


class Channel(Enum):
    EMAIL = auto()
    SMS = auto()
    PUSH = auto()


class Priority(Enum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    URGENT = auto()


@dataclass
class Notification:
    notification_id: str
    recipient_id: str
    channel: Channel
    subject: str
    body: str
    priority: Priority = Priority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    sent: bool = False
    error: str | None = None


@dataclass
class UserPreferences:
    user_id: str
    name: str
    email: str | None = None
    phone: str | None = None
    device_token: str | None = None
    enabled_channels: set[Channel] = field(
        default_factory=lambda: {Channel.EMAIL}
    )
    quiet_hours: tuple[int, int] | None = None  # (start_hour, end_hour)

    def is_quiet_time(self) -> bool:
        if not self.quiet_hours:
            return False
        now_hour = datetime.now().hour
        start, end = self.quiet_hours
        if start <= end:
            return start <= now_hour < end
        return now_hour >= start or now_hour < end  # wraps midnight


# --- Delivery handlers (Strategy) ---
class DeliveryHandler(Protocol):
    def send(self, notification: Notification, prefs: UserPreferences) -> bool: ...


class EmailHandler:
    def send(self, notification: Notification, prefs: UserPreferences) -> bool:
        if not prefs.email:
            return False
        print(f"  [EMAIL] To: {prefs.email} | Subject: {notification.subject}")
        return True


class SMSHandler:
    def send(self, notification: Notification, prefs: UserPreferences) -> bool:
        if not prefs.phone:
            return False
        # SMS: truncate to 160 chars
        body = notification.body[:157] + "..." if len(notification.body) > 160 else notification.body
        print(f"  [SMS] To: {prefs.phone} | {body}")
        return True


class PushHandler:
    def send(self, notification: Notification, prefs: UserPreferences) -> bool:
        if not prefs.device_token:
            return False
        print(f"  [PUSH] Device: {prefs.device_token[:8]}... | {notification.subject}")
        return True


# --- Notification Service ---
class NotificationService:
    def __init__(self):
        self._handlers: dict[Channel, DeliveryHandler] = {
            Channel.EMAIL: EmailHandler(),
            Channel.SMS: SMSHandler(),
            Channel.PUSH: PushHandler(),
        }
        self._users: dict[str, UserPreferences] = {}
        self._history: list[Notification] = []
        self._counter = 0

    def register_user(self, prefs: UserPreferences) -> None:
        self._users[prefs.user_id] = prefs

    def send(
        self,
        recipient_id: str,
        subject: str,
        body: str,
        priority: Priority = Priority.MEDIUM,
        channels: set[Channel] | None = None,
    ) -> list[Notification]:
        prefs = self._users.get(recipient_id)
        if not prefs:
            raise ValueError(f"User {recipient_id} not found")

        # Respect quiet hours (except URGENT)
        if prefs.is_quiet_time() and priority != Priority.URGENT:
            print(f"  Skipping {prefs.name} — quiet hours")
            return []

        # Use requested channels or user's enabled channels
        target_channels = channels or prefs.enabled_channels
        # Only send to channels the user has enabled
        target_channels = target_channels & prefs.enabled_channels

        results = []
        for channel in target_channels:
            self._counter += 1
            notification = Notification(
                notification_id=f"N-{self._counter:04d}",
                recipient_id=recipient_id,
                channel=channel,
                subject=subject,
                body=body,
                priority=priority,
            )

            handler = self._handlers.get(channel)
            if handler:
                try:
                    notification.sent = handler.send(notification, prefs)
                except Exception as e:
                    notification.error = str(e)

            self._history.append(notification)
            results.append(notification)

        return results

    def broadcast(
        self,
        subject: str,
        body: str,
        priority: Priority = Priority.MEDIUM,
    ) -> int:
        """Send to ALL registered users."""
        sent_count = 0
        for user_id in self._users:
            results = self.send(user_id, subject, body, priority)
            sent_count += sum(1 for n in results if n.sent)
        return sent_count

    def get_history(self, user_id: str | None = None) -> list[Notification]:
        if user_id:
            return [n for n in self._history if n.recipient_id == user_id]
        return self._history


# --- Demo ---
if __name__ == "__main__":
    service = NotificationService()

    # Register users with preferences
    service.register_user(UserPreferences(
        user_id="u1",
        name="Reza",
        email="reza@example.com",
        phone="+1234567890",
        device_token="abc123def456",
        enabled_channels={Channel.EMAIL, Channel.PUSH},
    ))
    service.register_user(UserPreferences(
        user_id="u2",
        name="Alice",
        email="alice@example.com",
        enabled_channels={Channel.EMAIL, Channel.SMS},
        phone="+0987654321",
    ))

    print("--- Send to Reza (email + push) ---")
    service.send("u1", "Welcome!", "Hello Reza, welcome aboard!", Priority.HIGH)

    print("\n--- Send to Alice (email + sms) ---")
    service.send("u2", "New Feature", "Check out our new AI tools!", Priority.MEDIUM)

    print("\n--- Broadcast ---")
    count = service.broadcast("Maintenance", "System maintenance at midnight.")
    print(f"  Sent to {count} channels total")

    print(f"\n--- History ---")
    for n in service.get_history():
        status = "SENT" if n.sent else "FAILED"
        print(f"  [{status}] {n.channel.name} -> {n.recipient_id}: {n.subject}")
