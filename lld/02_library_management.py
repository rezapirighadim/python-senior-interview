"""
=============================================================================
LLD 02: LIBRARY MANAGEMENT SYSTEM
=============================================================================
Design a library system where users can search, borrow, and return books.

KEY CONCEPTS:
  - Repository pattern for data access
  - Enum for status tracking
  - Clean separation: Book, Member, Library
=============================================================================
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto


class BookStatus(Enum):
    AVAILABLE = auto()
    BORROWED = auto()
    RESERVED = auto()


@dataclass
class Book:
    isbn: str
    title: str
    author: str
    status: BookStatus = BookStatus.AVAILABLE


@dataclass
class Member:
    member_id: str
    name: str
    email: str
    borrowed_books: list[str] = field(default_factory=list)  # list of ISBNs
    max_books: int = 5


@dataclass
class BorrowRecord:
    member_id: str
    isbn: str
    borrow_date: datetime = field(default_factory=datetime.now)
    due_date: datetime | None = None
    return_date: datetime | None = None

    def __post_init__(self):
        if self.due_date is None:
            self.due_date = self.borrow_date + timedelta(days=14)

    @property
    def is_overdue(self) -> bool:
        if self.return_date:
            return False
        return datetime.now() > self.due_date

    @property
    def overdue_days(self) -> int:
        if not self.is_overdue:
            return 0
        return (datetime.now() - self.due_date).days


class Library:
    LATE_FEE_PER_DAY = 0.50

    def __init__(self, name: str):
        self.name = name
        self.books: dict[str, Book] = {}
        self.members: dict[str, Member] = {}
        self.records: list[BorrowRecord] = []

    # --- Book management ---
    def add_book(self, book: Book) -> None:
        self.books[book.isbn] = book

    def search_by_title(self, query: str) -> list[Book]:
        query_lower = query.lower()
        return [b for b in self.books.values() if query_lower in b.title.lower()]

    def search_by_author(self, query: str) -> list[Book]:
        query_lower = query.lower()
        return [b for b in self.books.values() if query_lower in b.author.lower()]

    # --- Member management ---
    def register_member(self, member: Member) -> None:
        self.members[member.member_id] = member

    # --- Borrow / Return ---
    def borrow_book(self, member_id: str, isbn: str) -> BorrowRecord:
        member = self.members.get(member_id)
        if not member:
            raise ValueError(f"Member {member_id} not found")

        book = self.books.get(isbn)
        if not book:
            raise ValueError(f"Book {isbn} not found")

        if book.status != BookStatus.AVAILABLE:
            raise ValueError(f"'{book.title}' is not available")

        if len(member.borrowed_books) >= member.max_books:
            raise ValueError(f"{member.name} has reached the borrow limit")

        # Execute borrow
        book.status = BookStatus.BORROWED
        member.borrowed_books.append(isbn)

        record = BorrowRecord(member_id=member_id, isbn=isbn)
        self.records.append(record)
        return record

    def return_book(self, member_id: str, isbn: str) -> float:
        member = self.members.get(member_id)
        if not member:
            raise ValueError(f"Member {member_id} not found")

        if isbn not in member.borrowed_books:
            raise ValueError(f"{member.name} didn't borrow {isbn}")

        book = self.books[isbn]

        # Find the active borrow record
        record = None
        for r in reversed(self.records):
            if r.member_id == member_id and r.isbn == isbn and r.return_date is None:
                record = r
                break

        if not record:
            raise ValueError("No active borrow record found")

        # Execute return
        book.status = BookStatus.AVAILABLE
        member.borrowed_books.remove(isbn)
        record.return_date = datetime.now()

        # Calculate late fee
        fee = record.overdue_days * self.LATE_FEE_PER_DAY
        return fee

    def get_member_books(self, member_id: str) -> list[Book]:
        member = self.members.get(member_id)
        if not member:
            return []
        return [self.books[isbn] for isbn in member.borrowed_books]

    def get_available_books(self) -> list[Book]:
        return [b for b in self.books.values() if b.status == BookStatus.AVAILABLE]


# --- Demo ---
if __name__ == "__main__":
    lib = Library("City Library")

    # Add books
    lib.add_book(Book("978-1", "Clean Code", "Robert Martin"))
    lib.add_book(Book("978-2", "Design Patterns", "Gang of Four"))
    lib.add_book(Book("978-3", "Python Cookbook", "David Beazley"))
    lib.add_book(Book("978-4", "Clean Architecture", "Robert Martin"))

    # Register members
    lib.register_member(Member("M001", "Reza", "reza@example.com"))
    lib.register_member(Member("M002", "Alice", "alice@example.com"))

    # Search
    print("Search 'clean':", [b.title for b in lib.search_by_title("clean")])
    print("Search author 'martin':", [b.title for b in lib.search_by_author("martin")])

    # Borrow
    record = lib.borrow_book("M001", "978-1")
    print(f"\nBorrowed: '{lib.books['978-1'].title}' due: {record.due_date.date()}")

    lib.borrow_book("M001", "978-3")

    print(f"Reza's books: {[b.title for b in lib.get_member_books('M001')]}")
    print(f"Available: {[b.title for b in lib.get_available_books()]}")

    # Return
    fee = lib.return_book("M001", "978-1")
    print(f"\nReturned 'Clean Code', late fee: ${fee:.2f}")
    print(f"Available: {[b.title for b in lib.get_available_books()]}")
