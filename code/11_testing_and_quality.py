"""
=============================================================================
FILE 11: TESTING & CODE QUALITY — pytest, Mocking, TDD
=============================================================================
"Code without tests is broken by design." — Senior devs are expected to
write testable code and comprehensive tests. This is NON-NEGOTIABLE.

pip install pytest pytest-asyncio pytest-cov hypothesis
=============================================================================
"""
import json
from dataclasses import dataclass
from typing import Protocol
from unittest.mock import AsyncMock, MagicMock, Mock, patch


# =============================================================================
# 1. PYTEST BASICS
# =============================================================================

# --- Simple test functions ---
def add(a: int, b: int) -> int:
    return a + b

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0

# Run: pytest 11_testing_and_quality.py -v


# --- Testing exceptions ---
def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

def test_divide_by_zero():
    import pytest
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        divide(10, 0)


# =============================================================================
# 2. FIXTURES — Setup & Teardown
# =============================================================================

"""
import pytest

# --- Basic fixture ---
@pytest.fixture
def sample_user():
    return {"name": "Reza", "email": "reza@example.com", "age": 30}

def test_user_has_name(sample_user):
    assert sample_user["name"] == "Reza"


# --- Fixture with cleanup ---
@pytest.fixture
def database():
    db = InMemoryDatabase()
    db.connect()
    yield db          # Test runs here
    db.disconnect()   # Cleanup — always runs!


# --- Fixture scopes ---
@pytest.fixture(scope="session")    # Once per test session
def api_client():
    return create_api_client()

@pytest.fixture(scope="module")     # Once per test module
def config():
    return load_config()

@pytest.fixture(scope="function")   # Once per test function (default)
def fresh_data():
    return {"items": []}


# --- Parametrize — run same test with different inputs ---
@pytest.mark.parametrize("input,expected", [
    ("hello", "HELLO"),
    ("world", "WORLD"),
    ("Python", "PYTHON"),
    ("", ""),
])
def test_uppercase(input, expected):
    assert input.upper() == expected


@pytest.mark.parametrize("a,b,expected", [
    (2, 3, 5),
    (-1, 1, 0),
    (100, 200, 300),
    (0, 0, 0),
])
def test_add_parametrized(a, b, expected):
    assert add(a, b) == expected
"""


# =============================================================================
# 3. MOCKING — Isolate What You're Testing
# =============================================================================

# The code we want to test
class EmailService(Protocol):
    def send(self, to: str, subject: str, body: str) -> bool: ...


class UserService:
    def __init__(self, email_service: EmailService):
        self.email_service = email_service

    def register(self, name: str, email: str) -> dict:
        user = {"id": 1, "name": name, "email": email}
        # Send welcome email
        self.email_service.send(
            to=email,
            subject="Welcome!",
            body=f"Hi {name}, welcome aboard!"
        )
        return user


def test_register_sends_welcome_email():
    """Test that registration sends a welcome email."""
    # Create a mock email service
    mock_email = Mock(spec=EmailService)
    mock_email.send.return_value = True

    # Inject mock
    service = UserService(mock_email)
    user = service.register("Reza", "reza@example.com")

    # Verify the email was sent
    assert user["name"] == "Reza"
    mock_email.send.assert_called_once_with(
        to="reza@example.com",
        subject="Welcome!",
        body="Hi Reza, welcome aboard!"
    )


# --- patch decorator — mock module-level dependencies ---
"""
# In production code: user_service.py
import requests

def fetch_user_from_api(user_id: int) -> dict:
    response = requests.get(f"https://api.example.com/users/{user_id}")
    response.raise_for_status()
    return response.json()

# In test:
@patch("user_service.requests.get")
def test_fetch_user(mock_get):
    mock_response = Mock()
    mock_response.json.return_value = {"id": 1, "name": "Reza"}
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    result = fetch_user_from_api(1)

    assert result["name"] == "Reza"
    mock_get.assert_called_once_with("https://api.example.com/users/1")
"""


# --- MagicMock — mocks with magic methods ---
def test_magic_mock():
    mock_dict = MagicMock()
    mock_dict["key"] = "value"
    mock_dict.__getitem__.return_value = "value"
    assert mock_dict["key"] == "value"

    # Track all calls
    mock_fn = MagicMock()
    mock_fn(1, 2, 3)
    mock_fn("a", b="c")

    assert mock_fn.call_count == 2
    mock_fn.assert_any_call(1, 2, 3)
    mock_fn.assert_any_call("a", b="c")


# --- AsyncMock — for async code ---
"""
import pytest

@pytest.mark.asyncio
async def test_async_service():
    mock_db = AsyncMock()
    mock_db.find_user.return_value = {"id": 1, "name": "Reza"}

    result = await mock_db.find_user(1)
    assert result["name"] == "Reza"
    mock_db.find_user.assert_awaited_once_with(1)
"""


# =============================================================================
# 4. TEST PATTERNS — How to Write Good Tests
# =============================================================================

# --- Arrange-Act-Assert (AAA) pattern ---
def test_user_registration_aaa():
    # ARRANGE — set up test data and dependencies
    mock_email = Mock(spec=EmailService)
    mock_email.send.return_value = True
    service = UserService(mock_email)

    # ACT — perform the action
    user = service.register("Reza", "reza@example.com")

    # ASSERT — verify results
    assert user["name"] == "Reza"
    assert user["email"] == "reza@example.com"
    mock_email.send.assert_called_once()


# --- Testing with dependency injection (easy to test!) ---
class OrderRepository(Protocol):
    def save(self, order: dict) -> str: ...
    def find(self, order_id: str) -> dict | None: ...

class OrderService:
    def __init__(self, repo: OrderRepository, email: EmailService):
        self.repo = repo
        self.email = email

    def create_order(self, items: list[str], user_email: str) -> dict:
        order = {"id": "ord-1", "items": items, "status": "created"}
        self.repo.save(order)
        self.email.send(user_email, "Order Confirmation", f"Order: {order['id']}")
        return order


def test_create_order():
    mock_repo = Mock()
    mock_repo.save.return_value = "ord-1"
    mock_email = Mock()
    mock_email.send.return_value = True

    service = OrderService(mock_repo, mock_email)
    order = service.create_order(["item1", "item2"], "user@example.com")

    assert order["status"] == "created"
    mock_repo.save.assert_called_once()
    mock_email.send.assert_called_once()


# =============================================================================
# 5. PROPERTY-BASED TESTING — Hypothesis
# =============================================================================

"""
from hypothesis import given, strategies as st

# Instead of picking specific test cases, let Hypothesis generate thousands!

@given(st.integers(), st.integers())
def test_add_commutative(a, b):
    '''Addition is commutative: a + b == b + a'''
    assert add(a, b) == add(b, a)

@given(st.integers())
def test_add_identity(a):
    '''Adding 0 doesn't change the number.'''
    assert add(a, 0) == a

@given(st.lists(st.integers()))
def test_sort_is_idempotent(lst):
    '''Sorting twice gives same result as sorting once.'''
    assert sorted(sorted(lst)) == sorted(lst)

@given(st.lists(st.integers()))
def test_sort_preserves_length(lst):
    '''Sorting doesn't change length.'''
    assert len(sorted(lst)) == len(lst)

@given(st.text())
def test_upper_lower_roundtrip(s):
    '''upper then lower doesn't change case-insensitive content.'''
    assert s.upper().lower() == s.lower()
"""


# =============================================================================
# 6. TEST ORGANIZATION & CONFTEST
# =============================================================================

"""
tests/
├── conftest.py         # Shared fixtures (auto-discovered by pytest)
├── unit/
│   ├── test_services.py
│   ├── test_models.py
│   └── test_utils.py
├── integration/
│   ├── test_api.py
│   └── test_database.py
└── e2e/
    └── test_flows.py

# conftest.py — shared fixtures
import pytest

@pytest.fixture
def api_client():
    from httpx import AsyncClient, ASGITransport
    from app.main import app
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")

@pytest.fixture
def mock_db():
    return InMemoryDatabase()

# Run specific tests:
# pytest tests/unit/ -v              # Only unit tests
# pytest tests/ -k "test_create"     # Only tests matching pattern
# pytest tests/ -x                   # Stop on first failure
# pytest tests/ --cov=app --cov-report=html  # Coverage report
"""


# =============================================================================
# 7. CODE QUALITY TOOLS
# =============================================================================

"""
MUST-KNOW tools for senior Python devs:

1. ruff          → Linter + formatter (replaces flake8, black, isort)
   pip install ruff
   ruff check .
   ruff format .

2. mypy          → Static type checker
   pip install mypy
   mypy src/

3. pytest-cov    → Test coverage
   pytest --cov=app --cov-report=html

4. pre-commit    → Run checks before every commit
   pip install pre-commit

# .pre-commit-config.yaml:
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks:
      - id: ruff
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    hooks:
      - id: mypy

# pyproject.toml:
[tool.ruff]
target-version = "py312"
line-length = 88

[tool.ruff.lint]
select = ["E", "W", "F", "I", "N", "UP", "B", "SIM"]

[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = "-v --tb=short"
"""


# =============================================================================
# 8. WRITING TESTABLE CODE — Design Principles
# =============================================================================

"""
RULES FOR TESTABLE CODE:

1. DEPENDENCY INJECTION — Don't hardcode dependencies
   BAD:  def process(): db = MySQL(); db.query(...)
   GOOD: def process(db: Database): db.query(...)

2. SINGLE RESPONSIBILITY — Small, focused functions
   BAD:  def handle_order(data): validate + save + email + log
   GOOD: validate(data), save(order), send_email(order), log(order)

3. PURE FUNCTIONS — Same input → same output, no side effects
   BAD:  def get_price(): return db.query(...)  # Depends on DB state
   GOOD: def calculate_price(base, tax_rate): return base * (1 + tax_rate)

4. AVOID GLOBAL STATE — Makes tests interfere with each other
   BAD:  counter = 0; def increment(): global counter; counter += 1
   GOOD: class Counter: def increment(self): self.count += 1

5. INTERFACE SEGREGATION — Small interfaces are easier to mock
   BAD:  Mock entire database with 50 methods
   GOOD: Mock UserRepository with 3 methods
"""


if __name__ == "__main__":
    print("=" * 60)
    print("FILE 11: Testing & Code Quality")
    print("=" * 60)

    # Run the inline tests
    print("\n--- Running tests ---")
    test_add()
    print("  ✓ test_add passed")

    test_register_sends_welcome_email()
    print("  ✓ test_register_sends_welcome_email passed")

    test_magic_mock()
    print("  ✓ test_magic_mock passed")

    test_user_registration_aaa()
    print("  ✓ test_user_registration_aaa passed")

    test_create_order()
    print("  ✓ test_create_order passed")

    print("\n  All tests passed!")

    print("""
KEY TESTING COMMANDS:
  pytest                          # Run all tests
  pytest -v                       # Verbose output
  pytest -x                       # Stop on first failure
  pytest -k "pattern"             # Run matching tests
  pytest --cov=app                # With coverage
  pytest --cov=app --cov-fail-under=80  # Fail if < 80% coverage
  pytest -n auto                  # Run in parallel (pytest-xdist)
    """)

    print("✓ File 11 complete. Move to 12_data_structures_algorithms.py")
