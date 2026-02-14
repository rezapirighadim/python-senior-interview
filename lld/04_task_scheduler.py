"""
=============================================================================
LLD 04: TASK SCHEDULER
=============================================================================
Design a task scheduler that runs tasks at specified times,
supports priorities, and handles recurring tasks.

KEY CONCEPTS:
  - Heap (priority queue) for scheduling
  - Enum for task state
  - Callable tasks with error handling
=============================================================================
"""
import heapq
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Callable


class TaskStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()


class Priority(Enum):
    LOW = 3
    MEDIUM = 2
    HIGH = 1       # lower number = higher priority (heap is min-heap)
    CRITICAL = 0


@dataclass(order=True)
class ScheduledTask:
    # Fields used for ordering (heap comparison)
    run_at: datetime = field(compare=True)
    priority: int = field(compare=True)

    # Task details (not used for comparison)
    task_id: str = field(compare=False)
    name: str = field(compare=False)
    action: Callable = field(compare=False)
    status: TaskStatus = field(default=TaskStatus.PENDING, compare=False)
    repeat_seconds: int | None = field(default=None, compare=False)
    error: str | None = field(default=None, compare=False)


class TaskScheduler:
    def __init__(self):
        self._heap: list[ScheduledTask] = []
        self._task_counter = 0
        self._cancelled: set[str] = set()
        self.history: list[ScheduledTask] = []

    def schedule(
        self,
        name: str,
        action: Callable,
        run_at: datetime | None = None,
        delay_seconds: float = 0,
        priority: Priority = Priority.MEDIUM,
        repeat_seconds: int | None = None,
    ) -> str:
        self._task_counter += 1
        task_id = f"task-{self._task_counter:04d}"

        if run_at is None:
            run_at = datetime.now() + timedelta(seconds=delay_seconds)

        task = ScheduledTask(
            run_at=run_at,
            priority=priority.value,
            task_id=task_id,
            name=name,
            action=action,
            repeat_seconds=repeat_seconds,
        )
        heapq.heappush(self._heap, task)
        return task_id

    def cancel(self, task_id: str) -> bool:
        self._cancelled.add(task_id)
        return True

    def run_pending(self) -> list[ScheduledTask]:
        """Run all tasks that are due. Returns list of executed tasks."""
        executed = []
        now = datetime.now()

        while self._heap and self._heap[0].run_at <= now:
            task = heapq.heappop(self._heap)

            # Skip cancelled tasks
            if task.task_id in self._cancelled:
                self._cancelled.discard(task.task_id)
                continue

            # Execute
            task.status = TaskStatus.RUNNING
            try:
                task.action()
                task.status = TaskStatus.COMPLETED
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)

            executed.append(task)
            self.history.append(task)

            # Reschedule recurring tasks
            if task.repeat_seconds and task.status == TaskStatus.COMPLETED:
                self.schedule(
                    name=task.name,
                    action=task.action,
                    delay_seconds=task.repeat_seconds,
                    priority=Priority(task.priority),
                    repeat_seconds=task.repeat_seconds,
                )

        return executed

    def pending_count(self) -> int:
        return len(self._heap)

    def next_task(self) -> ScheduledTask | None:
        return self._heap[0] if self._heap else None


# --- Demo ---
if __name__ == "__main__":
    scheduler = TaskScheduler()

    # Schedule tasks
    scheduler.schedule(
        "Send welcome email",
        lambda: print("  -> Sending welcome email"),
        priority=Priority.HIGH,
    )
    scheduler.schedule(
        "Generate report",
        lambda: print("  -> Generating report"),
        delay_seconds=0.1,
        priority=Priority.LOW,
    )
    scheduler.schedule(
        "Health check",
        lambda: print("  -> Health check OK"),
        repeat_seconds=2,
        priority=Priority.MEDIUM,
    )
    cancel_id = scheduler.schedule(
        "This will be cancelled",
        lambda: print("  -> Should not run"),
        priority=Priority.LOW,
    )

    # Cancel one task
    scheduler.cancel(cancel_id)

    print(f"Pending tasks: {scheduler.pending_count()}")

    # Run immediately due tasks
    time.sleep(0.01)
    executed = scheduler.run_pending()
    print(f"\nRound 1 — executed {len(executed)} tasks:")
    for t in executed:
        print(f"  [{t.status.name}] {t.name}")

    # Wait and run again (recurring tasks)
    time.sleep(0.15)
    executed = scheduler.run_pending()
    print(f"\nRound 2 — executed {len(executed)} tasks:")
    for t in executed:
        print(f"  [{t.status.name}] {t.name}")

    print(f"\nPending: {scheduler.pending_count()}")
    print(f"History: {len(scheduler.history)} tasks executed total")
