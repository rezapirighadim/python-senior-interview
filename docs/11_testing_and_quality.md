# 11 — Testing & Code Quality

## pytest Basics

```python
def test_add():
    assert add(2, 3) == 5

def test_exception():
    with pytest.raises(ValueError, match="Cannot divide"):
        divide(10, 0)
```

## Fixtures

```python
@pytest.fixture
def sample_user():
    return {"name": "Reza", "email": "reza@x.com"}

@pytest.fixture
def database():
    db = InMemoryDB()
    yield db        # test runs here
    db.cleanup()    # always runs

# Scopes: function (default), module, session
```

## Parametrize

```python
@pytest.mark.parametrize("input,expected", [
    ("hello", "HELLO"),
    ("", ""),
    ("Python", "PYTHON"),
])
def test_upper(input, expected):
    assert input.upper() == expected
```

## Mocking

```python
from unittest.mock import Mock, patch, AsyncMock

# Mock an injected dependency
mock_email = Mock(spec=EmailService)
mock_email.send.return_value = True
service = UserService(mock_email)
service.register("Reza", "reza@x.com")
mock_email.send.assert_called_once()

# Patch a module-level dependency
@patch("mymodule.requests.get")
def test_fetch(mock_get):
    mock_get.return_value.json.return_value = {"id": 1}
    result = fetch_user(1)
    assert result["id"] == 1
```

## AAA Pattern

```python
def test_registration():
    # ARRANGE
    mock_email = Mock()
    service = UserService(mock_email)

    # ACT
    user = service.register("Reza", "reza@x.com")

    # ASSERT
    assert user["name"] == "Reza"
    mock_email.send.assert_called_once()
```

## Property-Based Testing (Hypothesis)

```python
from hypothesis import given, strategies as st

@given(st.integers(), st.integers())
def test_add_commutative(a, b):
    assert add(a, b) == add(b, a)
```

## Code Quality Tools

| Tool | Purpose |
|------|---------|
| **ruff** | Linter + formatter (replaces flake8, black, isort) |
| **mypy** | Static type checker |
| **pytest-cov** | Test coverage |
| **pre-commit** | Run checks before every commit |

## Writing Testable Code

1. **Dependency injection** — don't hardcode dependencies
2. **Single responsibility** — small, focused functions
3. **Pure functions** — same input, same output
4. **Avoid global state** — prevents test interference
5. **Small interfaces** — easier to mock

## Common Commands

```bash
pytest -v                            # verbose
pytest -x                            # stop on first failure
pytest -k "pattern"                  # match test names
pytest --cov=app --cov-fail-under=80 # coverage with threshold
pytest -n auto                       # parallel (pytest-xdist)
```
