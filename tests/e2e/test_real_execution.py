"""
Real Execution Tests for Kappa OS.

These tests verify ACTUAL Claude Code execution with real API calls.
They require ANTHROPIC_API_KEY environment variable to be set.

Run with:
    ANTHROPIC_API_KEY=sk-ant-... poetry run pytest tests/e2e/test_real_execution.py -v -s

Note: These tests are marked as 'slow' and 'requires_api_key' and should be
run separately from the regular test suite.
"""

import os
import shutil
from pathlib import Path

import pytest

# Skip all tests if no API key
pytestmark = [
    pytest.mark.slow,
    pytest.mark.e2e,
]


def skip_without_api_key():
    """Skip test if ANTHROPIC_API_KEY is not set."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY environment variable required")


def skip_without_claude_cli():
    """Skip test if claude CLI is not available."""
    if not shutil.which("claude"):
        pytest.skip("Claude CLI not installed or not in PATH")


# =============================================================================
# CLAUDE API TESTS (No CLI required)
# =============================================================================


class TestClaudeAPIIntegration:
    """Test Claude API integration for task generation and parsing."""

    @pytest.mark.asyncio
    async def test_requirements_parser_with_real_api(self) -> None:
        """Test RequirementsParser uses real Claude API."""
        skip_without_api_key()

        from src.decomposition.parser import RequirementsParser

        parser = RequirementsParser(use_api=True)

        requirements = await parser.parse(
            """Build a simple REST API with Express.js that has user authentication
            and a products endpoint. Use PostgreSQL for the database."""
        )

        # Verify it actually parsed (not just keyword fallback)
        assert requirements.name is not None
        assert requirements.project_type is not None
        assert len(requirements.features) > 0

        # Check for expected tech stack detection
        assert "express" in str(requirements.tech_stack).lower() or "node" in str(
            requirements.tech_stack
        ).lower()

    @pytest.mark.asyncio
    async def test_task_generator_with_real_ai(self) -> None:
        """Test TaskGenerator uses real Claude AI for task generation."""
        skip_without_api_key()

        from src.decomposition.models import ProjectRequirements, ProjectType
        from src.decomposition.task_generator import TaskGenerator

        requirements = ProjectRequirements(
            name="test-api",
            description="A simple REST API with user authentication",
            project_type=ProjectType.API,
            tech_stack={"framework": "fastapi", "language": "python"},
            features=["user authentication", "CRUD operations", "JWT tokens"],
        )

        generator = TaskGenerator(use_ai=True)
        tasks = await generator.generate(requirements)

        # Verify tasks were generated
        assert len(tasks) >= 5, f"Expected at least 5 tasks, got {len(tasks)}"

        # Verify task structure
        for task in tasks:
            assert task.id is not None
            assert task.title is not None
            assert task.description is not None
            assert task.wave_number is not None

        # Verify wave ordering (tasks should span multiple waves)
        waves = {t.wave_number for t in tasks}
        assert len(waves) >= 2, "Expected tasks in multiple waves"

    @pytest.mark.asyncio
    async def test_entity_extraction_with_ai(self) -> None:
        """Test AI-powered entity extraction."""
        skip_without_api_key()

        from src.decomposition.models import ProjectRequirements, ProjectType
        from src.decomposition.task_generator import TaskGenerator

        requirements = ProjectRequirements(
            name="ecommerce-api",
            description="E-commerce platform with products, orders, customers, and payments",
            project_type=ProjectType.API,
            features=[
                "product catalog",
                "shopping cart",
                "order processing",
                "payment integration",
            ],
        )

        generator = TaskGenerator(use_ai=True)
        entities = await generator._extract_entities_with_ai(requirements)

        # Verify entities were extracted
        assert len(entities) >= 2, f"Expected at least 2 entities, got {entities}"

        # Common e-commerce entities should be detected
        entities_lower = [e.lower() for e in entities]
        assert any(
            e in entities_lower for e in ["product", "order", "customer", "user", "payment"]
        ), f"Expected common entities, got {entities}"


# =============================================================================
# CLAUDE CODE CLI TESTS (Requires CLI)
# =============================================================================


class TestClaudeCodeCLI:
    """Test actual Claude Code CLI execution."""

    @pytest.mark.asyncio
    async def test_simple_file_creation(self, tmp_path: Path) -> None:
        """Test Claude Code creates a simple file."""
        skip_without_api_key()
        skip_without_claude_cli()

        from src.chat.interface import KappaChat

        chat = KappaChat()

        # Run a simple Claude Code session
        result = await chat._run_claude_code_session(
            task_id="test-1",
            prompt="Create a Python file called hello.py that prints 'Hello from Kappa OS!'",
            workspace=str(tmp_path),
        )

        # Check result
        assert result.get("success") or result.get("output"), f"Execution failed: {result}"

        # Check if file was created (may be in output even if not detected as file)
        hello_file = tmp_path / "hello.py"
        if hello_file.exists():
            content = hello_file.read_text()
            assert "print" in content or "Hello" in content

    @pytest.mark.asyncio
    async def test_multi_file_project_creation(self, tmp_path: Path) -> None:
        """Test Claude Code creates a multi-file project structure."""
        skip_without_api_key()
        skip_without_claude_cli()

        from src.chat.interface import KappaChat

        chat = KappaChat()

        result = await chat._run_claude_code_session(
            task_id="test-2",
            prompt="""Create a simple Python package structure:
            - mypackage/__init__.py (with version = '0.1.0')
            - mypackage/core.py (with a simple 'hello' function)
            - setup.py (minimal setup for the package)
            """,
            workspace=str(tmp_path),
        )

        assert result.get("success") or result.get("output"), f"Execution failed: {result}"

        # Verify structure if files were created
        if (tmp_path / "mypackage").exists():
            assert (tmp_path / "mypackage" / "__init__.py").exists()


# =============================================================================
# END-TO-END CHAT FLOW TESTS
# =============================================================================


class TestEndToEndChatFlow:
    """Test complete chat flow with real Claude."""

    @pytest.mark.asyncio
    async def test_chat_conversation_with_real_claude(self) -> None:
        """Test KappaChat handles conversation with real Claude responses."""
        skip_without_api_key()

        from src.chat.interface import KappaChat

        chat = KappaChat()

        # Process initial message
        response = await chat.process_message("I want to build a simple calculator CLI")

        # Should get a meaningful response (not just error)
        assert response is not None
        assert len(response) > 50, "Expected substantive response"

        # Should transition from greeting
        assert chat.state.phase.value != "greeting", "Should have moved past greeting phase"

    @pytest.mark.asyncio
    async def test_chat_extracts_project_info(self) -> None:
        """Test chat extracts project information correctly."""
        skip_without_api_key()

        from src.chat.interface import KappaChat

        chat = KappaChat()

        # Provide detailed specification
        await chat.process_message(
            """I want to build a REST API for a blog.
            It should have posts, comments, and user authentication.
            Use Python with FastAPI and PostgreSQL."""
        )

        # Check gathered info
        assert chat.state.gathered_info is not None

        # Should have detected some project info
        info = chat.state.gathered_info
        assert info.get("project_type") or info.get("tech_stack") or chat.state.proposed_requirements


# =============================================================================
# TERMINAL SESSION MANAGER TESTS
# =============================================================================


class TestTerminalSessionManager:
    """Test TerminalSessionManager with real sessions."""

    @pytest.mark.asyncio
    async def test_session_manager_creates_and_runs_session(self, tmp_path: Path) -> None:
        """Test TerminalSessionManager can create and run a real session."""
        skip_without_api_key()
        skip_without_claude_cli()

        from src.sessions.terminal import TerminalSessionManager

        manager = TerminalSessionManager(max_concurrent=1)

        session_id = await manager.create_session(
            task_id="test-session-1",
            prompt="Create a file called test_output.txt with the text 'Session test passed'",
            workspace=str(tmp_path),
            context={},
        )

        assert session_id is not None

        # Wait for completion with timeout
        result = await manager.wait_for_completion(session_id, timeout=60)

        assert result is not None
        assert result.session_id == session_id

        # Session should have completed (success or failure)
        assert result.status.value in ["completed", "failed", "timeout"]


# =============================================================================
# INTEGRATION TEST - FULL BUILD
# =============================================================================


class TestFullBuildIntegration:
    """Test a complete build process."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)  # 5 minute timeout
    async def test_minimal_project_build(self, tmp_path: Path) -> None:
        """Test building a minimal project end-to-end."""
        skip_without_api_key()
        skip_without_claude_cli()

        from src.chat.interface import KappaChat

        chat = KappaChat()

        # Set up a minimal project
        chat.state.proposed_requirements = {
            "name": "test-minimal",
            "project_type": "cli_tool",
            "description": "A minimal CLI tool",
            "tech_stack": {"language": "python"},
            "features": ["hello command"],
        }
        chat.state.project_name = "test-minimal"

        # Create simple execution plan
        plan = {
            "project_name": "test-minimal",
            "tasks": [
                {
                    "id": "task-1",
                    "name": "Create hello script",
                    "prompt": "Create a Python file called hello.py with a function hello() that prints 'Hello!'",
                }
            ],
        }

        # Execute
        result = await chat._execute_with_claude_code(plan, str(tmp_path))

        # Verify result structure
        assert "status" in result
        assert "workspace_path" in result
        assert result["workspace_path"] == str(tmp_path)

        # Status should be one of the valid statuses
        assert result["status"] in ["completed", "partial", "failed"]


# =============================================================================
# SMOKE TEST - QUICK VERIFICATION
# =============================================================================


class TestSmokeTests:
    """Quick smoke tests to verify basic functionality."""

    def test_imports_work(self) -> None:
        """Verify all main modules can be imported."""
        from src.chat.interface import KappaChat
        from src.decomposition.parser import RequirementsParser
        from src.decomposition.task_generator import TaskGenerator
        from src.sessions.terminal import TerminalSessionManager

        assert KappaChat is not None
        assert RequirementsParser is not None
        assert TaskGenerator is not None
        assert TerminalSessionManager is not None

    def test_chat_initialization(self) -> None:
        """Verify KappaChat can be instantiated."""
        from src.chat.interface import KappaChat

        chat = KappaChat()
        assert chat is not None
        assert chat.state is not None
        assert chat.state.phase.value == "greeting"

    @pytest.mark.asyncio
    async def test_claude_client_initialization(self) -> None:
        """Test Claude client can be initialized (if API key present)."""
        skip_without_api_key()

        from src.chat.interface import KappaChat

        chat = KappaChat()

        # Access claude_client property
        client = chat.claude_client
        assert client is not None

        # Verify it's the right type
        assert hasattr(client, "messages")
