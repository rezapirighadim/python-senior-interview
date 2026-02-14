"""
=============================================================================
LLD 01: PARKING LOT SYSTEM
=============================================================================
Design a parking lot that supports multiple vehicle types,
multiple floors, and tracks availability.

KEY CONCEPTS:
  - Enum for fixed categories
  - Strategy pattern for pricing
  - Single Responsibility (separate Ticket from ParkingLot)
=============================================================================
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Protocol


# --- Enums ---
class VehicleType(Enum):
    MOTORCYCLE = auto()
    CAR = auto()
    TRUCK = auto()


class SpotSize(Enum):
    SMALL = auto()    # motorcycle
    MEDIUM = auto()   # car
    LARGE = auto()    # truck


# Maps which vehicle fits which spot
VEHICLE_TO_SPOT: dict[VehicleType, SpotSize] = {
    VehicleType.MOTORCYCLE: SpotSize.SMALL,
    VehicleType.CAR: SpotSize.MEDIUM,
    VehicleType.TRUCK: SpotSize.LARGE,
}


# --- Models ---
@dataclass
class Vehicle:
    license_plate: str
    vehicle_type: VehicleType


@dataclass
class ParkingSpot:
    spot_id: str
    floor: int
    size: SpotSize
    vehicle: Vehicle | None = None

    @property
    def is_available(self) -> bool:
        return self.vehicle is None

    def park(self, vehicle: Vehicle) -> None:
        if not self.is_available:
            raise ValueError(f"Spot {self.spot_id} is occupied")
        self.vehicle = vehicle

    def remove(self) -> Vehicle:
        if self.is_available:
            raise ValueError(f"Spot {self.spot_id} is empty")
        vehicle = self.vehicle
        self.vehicle = None
        return vehicle


@dataclass
class Ticket:
    ticket_id: str
    vehicle: Vehicle
    spot: ParkingSpot
    entry_time: datetime = field(default_factory=datetime.now)
    exit_time: datetime | None = None

    @property
    def duration_hours(self) -> float:
        end = self.exit_time or datetime.now()
        return (end - self.entry_time).total_seconds() / 3600


# --- Pricing (Strategy pattern) ---
class PricingStrategy(Protocol):
    def calculate(self, ticket: Ticket) -> float: ...


class HourlyPricing:
    RATES = {
        SpotSize.SMALL: 2.0,
        SpotSize.MEDIUM: 5.0,
        SpotSize.LARGE: 10.0,
    }

    def calculate(self, ticket: Ticket) -> float:
        hours = max(1, ticket.duration_hours)  # minimum 1 hour
        rate = self.RATES[ticket.spot.size]
        return round(hours * rate, 2)


# --- Parking Lot ---
class ParkingLot:
    def __init__(self, name: str, pricing: PricingStrategy):
        self.name = name
        self.pricing = pricing
        self.spots: list[ParkingSpot] = []
        self.active_tickets: dict[str, Ticket] = {}  # license_plate -> ticket
        self._ticket_counter = 0

    def add_spots(self, floor: int, size: SpotSize, count: int) -> None:
        for i in range(count):
            spot_id = f"F{floor}-{size.name[0]}{len(self.spots) + 1}"
            self.spots.append(ParkingSpot(spot_id, floor, size))

    def find_available_spot(self, vehicle_type: VehicleType) -> ParkingSpot | None:
        needed = VEHICLE_TO_SPOT[vehicle_type]
        for spot in self.spots:
            if spot.is_available and spot.size == needed:
                return spot
        return None

    def park_vehicle(self, vehicle: Vehicle) -> Ticket:
        if vehicle.license_plate in self.active_tickets:
            raise ValueError(f"{vehicle.license_plate} is already parked")

        spot = self.find_available_spot(vehicle.vehicle_type)
        if spot is None:
            raise ValueError(f"No spot available for {vehicle.vehicle_type.name}")

        spot.park(vehicle)
        self._ticket_counter += 1
        ticket = Ticket(
            ticket_id=f"T-{self._ticket_counter:04d}",
            vehicle=vehicle,
            spot=spot,
        )
        self.active_tickets[vehicle.license_plate] = ticket
        return ticket

    def exit_vehicle(self, license_plate: str) -> float:
        if license_plate not in self.active_tickets:
            raise ValueError(f"{license_plate} not found")

        ticket = self.active_tickets.pop(license_plate)
        ticket.exit_time = datetime.now()
        ticket.spot.remove()

        fee = self.pricing.calculate(ticket)
        return fee

    def available_spots_count(self) -> dict[SpotSize, int]:
        counts = {size: 0 for size in SpotSize}
        for spot in self.spots:
            if spot.is_available:
                counts[spot.size] += 1
        return counts


# --- Demo ---
if __name__ == "__main__":
    lot = ParkingLot("Downtown Parking", HourlyPricing())

    # Add spots: 2 floors
    lot.add_spots(floor=1, size=SpotSize.SMALL, count=5)
    lot.add_spots(floor=1, size=SpotSize.MEDIUM, count=10)
    lot.add_spots(floor=1, size=SpotSize.LARGE, count=3)
    lot.add_spots(floor=2, size=SpotSize.MEDIUM, count=10)

    print("Available:", lot.available_spots_count())

    # Park vehicles
    car = Vehicle("ABC-123", VehicleType.CAR)
    ticket = lot.park_vehicle(car)
    print(f"Parked {car.license_plate} at {ticket.spot.spot_id} (ticket: {ticket.ticket_id})")

    moto = Vehicle("MOTO-1", VehicleType.MOTORCYCLE)
    lot.park_vehicle(moto)

    print("Available:", lot.available_spots_count())

    # Exit
    fee = lot.exit_vehicle("ABC-123")
    print(f"Fee for ABC-123: ${fee}")
    print("Available:", lot.available_spots_count())
