"""Pytest configuration and shared fixtures."""

import os
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

# Set test environment
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-key-for-testing")
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://test:test@localhost:5432/kappa_test"
)
os.environ.setdefault("KAPPA_DEBUG", "true")
os.environ.setdefault("KAPPA_LOG_LEVEL", "DEBUG")


@pytest.fixture
def mock_settings() -> Generator:
    """Provide mock settings for testing."""
    from src.core.config import Settings, clear_settings_cache

    # Clear any cached settings
    clear_settings_cache()

    yield

    # Clear again after test
    clear_settings_cache()


@pytest.fixture
def sample_specification() -> str:
    """Provide a sample project specification."""
    return """Build a REST API for a todo application with the following features:
    - User authentication (register, login, logout)
    - Todo CRUD operations (create, read, update, delete)
    - Todo categories/tags
    - Due dates and reminders
    - PostgreSQL database
    """


@pytest.fixture
def sample_tasks() -> list:
    """Provide sample tasks for testing."""
    from src.decomposition.models import Complexity, Task, TaskCategory

    return [
        Task(
            id="task-1",
            name="Initialize project",
            description="Set up project structure",
            category=TaskCategory.SETUP,
            complexity=Complexity.LOW,
            wave=0,
            dependencies=[],
        ),
        Task(
            id="task-2",
            name="Create User model",
            description="Define User SQLAlchemy model",
            category=TaskCategory.DATA_MODEL,
            complexity=Complexity.MEDIUM,
            wave=1,
            dependencies=["task-1"],
        ),
        Task(
            id="task-3",
            name="Create Todo model",
            description="Define Todo SQLAlchemy model",
            category=TaskCategory.DATA_MODEL,
            complexity=Complexity.MEDIUM,
            wave=1,
            dependencies=["task-1"],
        ),
        Task(
            id="task-4",
            name="Implement auth service",
            description="Create authentication service",
            category=TaskCategory.BUSINESS_LOGIC,
            complexity=Complexity.HIGH,
            wave=2,
            dependencies=["task-2"],
        ),
        Task(
            id="task-5",
            name="Create API endpoints",
            description="Implement REST API endpoints",
            category=TaskCategory.API,
            complexity=Complexity.MEDIUM,
            wave=3,
            dependencies=["task-2", "task-3", "task-4"],
        ),
    ]


@pytest.fixture
def mock_claude_client() -> MagicMock:
    """Provide a mock Claude client."""
    client = MagicMock()
    client.query = AsyncMock(return_value=iter(["Mock response"]))
    client.close = AsyncMock()
    return client


@pytest_asyncio.fixture
async def mock_db_session() -> AsyncGenerator:
    """Provide a mock database session."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.flush = AsyncMock()
    session.add = MagicMock()

    yield session


@pytest.fixture
def sample_task_results() -> list:
    """Provide sample task results for testing."""
    return [
        {
            "task_id": "task-1",
            "session_id": "session-1",
            "success": True,
            "output": "Project initialized",
            "files_modified": ["pyproject.toml", "README.md"],
            "duration_seconds": 10.5,
        },
        {
            "task_id": "task-2",
            "session_id": "session-2",
            "success": True,
            "output": "User model created",
            "files_modified": ["src/models/user.py"],
            "duration_seconds": 25.3,
        },
        {
            "task_id": "task-3",
            "session_id": "session-3",
            "success": True,
            "output": "Todo model created",
            "files_modified": ["src/models/todo.py", "src/models/user.py"],  # Conflict!
            "duration_seconds": 22.1,
        },
    ]


@pytest.fixture
def sample_conflicts() -> list:
    """Provide sample conflicts for testing."""
    return [
        {
            "id": "conflict-1",
            "file_path": "src/models/user.py",
            "session_a_id": "session-2",
            "session_b_id": "session-3",
            "conflict_type": "merge",
            "description": "Both sessions modified src/models/user.py",
        },
    ]


# Markers
def pytest_configure(config: pytest.Config) -> None:
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "e2e: mark test as end-to-end test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
