"""
=============================================================================
LLD 10: ELEVATOR SYSTEM
=============================================================================
Design an elevator system for a building with multiple elevators.

KEY CONCEPTS:
  - State machine (elevator states)
  - Strategy pattern (elevator selection)
  - Queue-based request handling
=============================================================================
"""
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque


class Direction(Enum):
    UP = auto()
    DOWN = auto()
    IDLE = auto()


class DoorState(Enum):
    OPEN = auto()
    CLOSED = auto()


@dataclass
class Request:
    floor: int
    direction: Direction | None = None  # None for internal requests


class Elevator:
    def __init__(self, elevator_id: str, min_floor: int = 1, max_floor: int = 10):
        self.elevator_id = elevator_id
        self.current_floor = 1
        self.direction = Direction.IDLE
        self.door = DoorState.CLOSED
        self.min_floor = min_floor
        self.max_floor = max_floor
        self.up_stops: set[int] = set()
        self.down_stops: set[int] = set()

    @property
    def is_idle(self) -> bool:
        return self.direction == Direction.IDLE

    @property
    def pending_stops(self) -> int:
        return len(self.up_stops) + len(self.down_stops)

    def add_stop(self, floor: int) -> None:
        if floor < self.min_floor or floor > self.max_floor:
            raise ValueError(f"Floor {floor} out of range")

        if floor > self.current_floor:
            self.up_stops.add(floor)
        elif floor < self.current_floor:
            self.down_stops.add(floor)
        else:
            self._open_doors()  # already at that floor

        # Start moving if idle
        if self.direction == Direction.IDLE:
            if self.up_stops:
                self.direction = Direction.UP
            elif self.down_stops:
                self.direction = Direction.DOWN

    def step(self) -> str | None:
        """Simulate one step of movement. Returns action taken."""
        if self.direction == Direction.IDLE:
            return None

        if self.direction == Direction.UP:
            if self.current_floor in self.up_stops:
                self.up_stops.discard(self.current_floor)
                self._open_doors()
                return f"{self.elevator_id}: Stopped at floor {self.current_floor} (UP)"

            if self.up_stops:
                self.current_floor += 1
                return f"{self.elevator_id}: Moving UP to {self.current_floor}"
            else:
                # No more up stops, switch direction or idle
                if self.down_stops:
                    self.direction = Direction.DOWN
                else:
                    self.direction = Direction.IDLE
                return f"{self.elevator_id}: Switching to {self.direction.name}"

        if self.direction == Direction.DOWN:
            if self.current_floor in self.down_stops:
                self.down_stops.discard(self.current_floor)
                self._open_doors()
                return f"{self.elevator_id}: Stopped at floor {self.current_floor} (DOWN)"

            if self.down_stops:
                self.current_floor -= 1
                return f"{self.elevator_id}: Moving DOWN to {self.current_floor}"
            else:
                if self.up_stops:
                    self.direction = Direction.UP
                else:
                    self.direction = Direction.IDLE
                return f"{self.elevator_id}: Switching to {self.direction.name}"

    def _open_doors(self) -> None:
        self.door = DoorState.OPEN
        # In real system: wait, then close
        self.door = DoorState.CLOSED

    def __repr__(self) -> str:
        return (
            f"Elevator({self.elevator_id}, floor={self.current_floor}, "
            f"dir={self.direction.name}, stops_up={self.up_stops}, "
            f"stops_down={self.down_stops})"
        )


class ElevatorController:
    """Manages multiple elevators and dispatches requests."""

    def __init__(self, num_elevators: int, num_floors: int):
        self.elevators = [
            Elevator(f"E{i+1}", min_floor=1, max_floor=num_floors)
            for i in range(num_elevators)
        ]
        self.num_floors = num_floors

    def request(self, floor: int, direction: Direction | None = None) -> Elevator:
        """Assign the best elevator to handle a floor request."""
        best = self._find_best_elevator(floor, direction)
        best.add_stop(floor)
        return best

    def _find_best_elevator(self, floor: int, direction: Direction | None) -> Elevator:
        """
        Selection strategy — pick the closest suitable elevator:
        1. Idle elevator closest to the floor
        2. Elevator moving toward the floor in the right direction
        3. Elevator with fewest pending stops
        """
        best = None
        best_score = float("inf")

        for elev in self.elevators:
            distance = abs(elev.current_floor - floor)

            if elev.is_idle:
                score = distance
            elif elev.direction == Direction.UP and floor >= elev.current_floor:
                score = distance  # on the way
            elif elev.direction == Direction.DOWN and floor <= elev.current_floor:
                score = distance  # on the way
            else:
                score = distance + 100  # penalty: wrong direction

            # Tie-break: fewer pending stops
            score += elev.pending_stops * 0.1

            if score < best_score:
                best_score = score
                best = elev

        return best

    def press_floor(self, elevator_id: str, floor: int) -> None:
        """Person inside elevator presses a floor button."""
        for elev in self.elevators:
            if elev.elevator_id == elevator_id:
                elev.add_stop(floor)
                return
        raise ValueError(f"Elevator {elevator_id} not found")

    def step_all(self) -> list[str]:
        """Advance all elevators by one step."""
        actions = []
        for elev in self.elevators:
            action = elev.step()
            if action:
                actions.append(action)
        return actions

    def status(self) -> list[str]:
        return [
            f"  {e.elevator_id}: floor={e.current_floor} dir={e.direction.name} "
            f"up={e.up_stops or '{}'} down={e.down_stops or '{}'}"
            for e in self.elevators
        ]


# --- Demo ---
if __name__ == "__main__":
    controller = ElevatorController(num_elevators=3, num_floors=10)

    print("--- Initial status ---")
    for s in controller.status():
        print(s)

    # Someone on floor 5 presses UP
    e = controller.request(5, Direction.UP)
    print(f"\nFloor 5 UP -> assigned to {e.elevator_id}")

    # Someone on floor 8 presses DOWN
    e = controller.request(8, Direction.DOWN)
    print(f"Floor 8 DOWN -> assigned to {e.elevator_id}")

    # Someone on floor 3 presses UP
    e = controller.request(3, Direction.UP)
    print(f"Floor 3 UP -> assigned to {e.elevator_id}")

    # Simulate movement
    print("\n--- Simulation ---")
    for step in range(12):
        actions = controller.step_all()
        if actions:
            for a in actions:
                print(f"  Step {step+1}: {a}")

    print("\n--- Final status ---")
    for s in controller.status():
        print(s)
