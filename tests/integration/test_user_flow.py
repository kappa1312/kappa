"""
Integration tests for the complete user flow.

Tests the end-to-end flow from ideation to project execution.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.chat.interface import ConversationPhase, KappaChat

# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def mock_kappa():
    """Create a mock Kappa orchestrator."""
    kappa = MagicMock()
    kappa.execute = AsyncMock(
        return_value={
            "status": "completed",
            "project_id": "test-project-123",
            "workspace_path": "/tmp/test-workspace",
            "final_output": "Project built successfully",
        }
    )
    return kappa


@pytest.fixture
def mock_parser():
    """Create a mock RequirementsParser."""
    from src.decomposition.models import ProjectRequirements, ProjectType

    parser = MagicMock()
    parser.parse = AsyncMock(
        return_value=ProjectRequirements(
            name="test-project",
            description="A test project",
            project_type=ProjectType.WEBSITE,
            tech_stack={"framework": "Next.js 14", "language": "TypeScript"},
            pages=["Home", "About"],
            features=["Contact form"],
        )
    )
    return parser


# =============================================================================
# TEST CHAT TO EXECUTION FLOW
# =============================================================================


class TestChatToExecutionFlow:
    """Tests for complete chat to execution flow."""

    @pytest.mark.asyncio
    async def test_complete_flow_greeting_to_discovery(self) -> None:
        """Test flow from greeting to discovery."""
        chat = KappaChat()

        # Start with greeting
        assert chat.state.phase == ConversationPhase.GREETING

        # Send initial message
        response = await chat.process_message("I want to build a website")

        # Should move to discovery
        assert chat.state.phase == ConversationPhase.DISCOVERY
        assert "question" in response.lower() or "build" in response.lower()

    @pytest.mark.asyncio
    async def test_complete_flow_discovery_to_proposal(self) -> None:
        """Test flow from discovery to proposal."""
        chat = KappaChat()

        # Initial message
        await chat.process_message("I want to build a website called TestSite")

        # Provide details
        await chat.process_message(
            "I need Home, About, Contact pages with Next.js, TypeScript, Tailwind, and a contact form"
        )

        # Should be in proposal, confirmation, or clarification
        assert chat.state.phase in [
            ConversationPhase.PROPOSAL,
            ConversationPhase.CONFIRMATION,
            ConversationPhase.CLARIFICATION,
        ]

    @pytest.mark.asyncio
    async def test_modification_flow(self) -> None:
        """Test modifying requirements before execution."""
        chat = KappaChat()

        # Get to proposal
        await chat.process_message("Build a website with Home page, Next.js, Tailwind")
        await chat.process_message("Call it TestSite with contact form")

        if chat.state.phase in [ConversationPhase.CONFIRMATION, ConversationPhase.PROPOSAL]:
            # Request modification
            response = await chat.process_message("Add a blog section")

            # Should acknowledge change
            assert (
                "blog" in response.lower()
                or "added" in response.lower()
                or "change" in response.lower()
            )

    @pytest.mark.asyncio
    async def test_execution_start(self) -> None:
        """Test starting execution."""
        chat = KappaChat()

        # Mock the Kappa orchestrator to avoid actual execution
        with patch.object(chat, "_execute_project", new_callable=AsyncMock):
            # Get to confirmation
            await chat.process_message("Build a website with Home, Next.js, Tailwind")
            await chat.process_message("Named TestSite with contact form")

            if chat.state.phase in [ConversationPhase.CONFIRMATION, ConversationPhase.PROPOSAL]:
                # Confirm
                response = await chat.process_message("yes build it")

                # Should start execution
                assert (
                    chat.state.phase == ConversationPhase.EXECUTION
                    or "start" in response.lower()
                    or "build" in response.lower()
                )


# =============================================================================
# TEST API TO ORCHESTRATOR
# =============================================================================


class TestAPIToOrchestrator:
    """Tests for API triggering orchestrator execution."""

    @pytest.mark.asyncio
    async def test_chat_creates_kappa_on_demand(self) -> None:
        """Test Kappa is created lazily."""
        chat = KappaChat()

        # Kappa should not be created yet
        assert chat._kappa is None

        # Access kappa property
        kappa = chat.kappa

        # Now it should be created
        assert kappa is not None

    @pytest.mark.asyncio
    async def test_chat_with_workspace(self, tmp_path) -> None:
        """Test chat with custom workspace."""
        workspace = str(tmp_path / "test-workspace")
        chat = KappaChat(workspace=workspace)

        assert chat.workspace == workspace


# =============================================================================
# TEST WEBSOCKET UPDATES
# =============================================================================


class TestWebSocketUpdates:
    """Tests for WebSocket update integration."""

    @pytest.mark.asyncio
    async def test_progress_callback(self) -> None:
        """Test progress update callback is called."""
        chat = KappaChat()
        updates_received = []

        def callback(result):
            updates_received.append(result)

        chat.on_progress_update = callback

        # Simulate build completion
        chat.state.build_result = {"status": "completed"}
        if chat.on_progress_update:
            chat.on_progress_update(chat.state.build_result)

        assert len(updates_received) == 1
        assert updates_received[0]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_phase_change_callback(self) -> None:
        """Test phase change callback is called."""
        chat = KappaChat()
        phases_seen = []

        chat.on_phase_change = lambda p: phases_seen.append(p)

        await chat.process_message("Hello, I want to build a website")

        assert len(phases_seen) > 0
        assert ConversationPhase.DISCOVERY in phases_seen


# =============================================================================
# TEST PROJECT LIFECYCLE
# =============================================================================


class TestProjectLifecycle:
    """Tests for project lifecycle management."""

    @pytest.mark.asyncio
    async def test_new_project_after_completion(self) -> None:
        """Test starting new project after completion."""
        chat = KappaChat()

        # Simulate completed project
        chat.state.phase = ConversationPhase.COMPLETION
        chat.state.project_name = "old-project"
        chat.state.build_complete = True
        chat.state.build_result = {"status": "completed"}

        # Request new project
        response = await chat.process_message("Start a new project")

        # Should reset state
        assert chat.state.phase == ConversationPhase.GREETING
        assert chat.state.project_name is None
        assert "new" in response.lower() or "build" in response.lower()

    @pytest.mark.asyncio
    async def test_reset_clears_state(self) -> None:
        """Test reset clears all state."""
        chat = KappaChat()

        # Set some state
        chat.state.phase = ConversationPhase.EXECUTION
        chat.state.project_name = "test"
        chat.state.gathered_info = {"key": "value"}

        # Reset
        chat.reset()

        # Verify clean state
        assert chat.state.phase == ConversationPhase.GREETING
        assert chat.state.project_name is None
        assert chat.state.gathered_info == {}

    @pytest.mark.asyncio
    async def test_conversation_history_preserved(self) -> None:
        """Test conversation history is preserved."""
        chat = KappaChat()

        await chat.process_message("Hello")
        await chat.process_message("I want a website")

        history = chat.get_conversation_history()

        assert len(history) >= 4  # 2 user + 2 assistant messages
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"


# =============================================================================
# TEST CLI COMMANDS
# =============================================================================


class TestCLICommands:
    """Tests for CLI command functionality."""

    def test_cli_app_has_commands(self) -> None:
        """Test CLI app has expected commands."""
        from src.cli.main import app

        # Get command names from Typer app
        command_names = list(app.registered_commands)
        # Typer commands are stored differently - check the command group
        cmd_list = [c.name or c.callback.__name__ if c.callback else None for c in command_names]

        expected_commands = ["run", "init", "status", "decompose", "health", "chat", "dashboard", "build"]
        for cmd in expected_commands:
            assert cmd in cmd_list, f"Command '{cmd}' not found in {cmd_list}"

    def test_cli_has_version(self) -> None:
        """Test CLI has version callback."""
        from src.cli.main import app

        # Check that version option exists
        assert app.info.options_metavar is not None or len(app.registered_callbacks) > 0


# =============================================================================
# TEST ERROR HANDLING
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in user flow."""

    @pytest.mark.asyncio
    async def test_chat_handles_empty_input(self) -> None:
        """Test chat handles empty input gracefully."""
        chat = KappaChat()

        # Empty string should still work
        response = await chat.process_message("")

        assert response is not None
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_execution_query_during_build(self) -> None:
        """Test querying during build phase."""
        chat = KappaChat()
        chat.state.phase = ConversationPhase.EXECUTION
        chat.state.is_building = True

        response = await chat.process_message("What's happening?")

        assert "progress" in response.lower() or "building" in response.lower()

    @pytest.mark.asyncio
    async def test_build_complete_detection(self) -> None:
        """Test detecting when build is complete."""
        chat = KappaChat()
        chat.state.phase = ConversationPhase.EXECUTION
        chat.state.is_building = False
        chat.state.build_complete = True
        chat.state.build_result = {"status": "completed", "workspace_path": "/tmp/test"}

        response = await chat.process_message("What's the status?")

        # Should transition to completion
        assert chat.state.phase == ConversationPhase.COMPLETION or "complete" in response.lower()
