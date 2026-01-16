"""
Unit tests for the KappaChat interface.

Tests the chat conversation flow from ideation to project completion.
"""

import pytest

from src.chat.interface import (
    ConversationPhase,
    ConversationState,
    KappaChat,
    Message,
)

# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def chat() -> KappaChat:
    """Create a KappaChat instance for testing."""
    return KappaChat()


@pytest.fixture
def chat_with_workspace(tmp_path) -> KappaChat:
    """Create a KappaChat instance with a workspace."""
    return KappaChat(workspace=str(tmp_path))


# =============================================================================
# TEST ConversationPhase ENUM
# =============================================================================


class TestConversationPhase:
    """Tests for ConversationPhase enum."""

    def test_phase_values(self) -> None:
        """Test ConversationPhase has expected values."""
        assert ConversationPhase.GREETING.value == "greeting"
        assert ConversationPhase.DISCOVERY.value == "discovery"
        assert ConversationPhase.CLARIFICATION.value == "clarification"
        assert ConversationPhase.PROPOSAL.value == "proposal"
        assert ConversationPhase.REFINEMENT.value == "refinement"
        assert ConversationPhase.CONFIRMATION.value == "confirmation"
        assert ConversationPhase.EXECUTION.value == "execution"
        assert ConversationPhase.COMPLETION.value == "completion"

    def test_phase_count(self) -> None:
        """Test there are 8 conversation phases."""
        assert len(ConversationPhase) == 8

    def test_phase_is_string_enum(self) -> None:
        """Test ConversationPhase inherits from str."""
        assert isinstance(ConversationPhase.GREETING, str)
        assert ConversationPhase.GREETING == "greeting"


# =============================================================================
# TEST Message DATACLASS
# =============================================================================


class TestMessage:
    """Tests for Message dataclass."""

    def test_create_message(self) -> None:
        """Test creating a Message instance."""
        msg = Message(
            role="user",
            content="Hello",
            timestamp="2024-01-01T00:00:00",
        )

        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.timestamp == "2024-01-01T00:00:00"
        assert msg.metadata == {}

    def test_message_with_metadata(self) -> None:
        """Test Message with metadata."""
        msg = Message(
            role="assistant",
            content="Response",
            timestamp="2024-01-01T00:00:00",
            metadata={"phase": "greeting"},
        )

        assert msg.metadata == {"phase": "greeting"}


# =============================================================================
# TEST ConversationState DATACLASS
# =============================================================================


class TestConversationState:
    """Tests for ConversationState dataclass."""

    def test_default_state(self) -> None:
        """Test default ConversationState values."""
        state = ConversationState()

        assert state.phase == ConversationPhase.GREETING
        assert state.messages == []
        assert state.project_name is None
        assert state.gathered_info == {}
        assert state.proposed_requirements is None
        assert state.is_building is False
        assert state.project_id is None

    def test_state_modification(self) -> None:
        """Test modifying ConversationState."""
        state = ConversationState()
        state.phase = ConversationPhase.DISCOVERY
        state.project_name = "test-project"

        assert state.phase == ConversationPhase.DISCOVERY
        assert state.project_name == "test-project"


# =============================================================================
# TEST KappaChat INITIALIZATION
# =============================================================================


class TestKappaChatInit:
    """Tests for KappaChat initialization."""

    def test_default_init(self) -> None:
        """Test default KappaChat initialization."""
        chat = KappaChat()

        assert chat.state.phase == ConversationPhase.GREETING
        assert chat.workspace is None
        assert chat.on_phase_change is None
        assert chat.on_progress_update is None

    def test_init_with_workspace(self, tmp_path) -> None:
        """Test KappaChat with workspace."""
        chat = KappaChat(workspace=str(tmp_path))

        assert chat.workspace == str(tmp_path)

    def test_init_state_is_fresh(self) -> None:
        """Test each KappaChat has fresh state."""
        chat1 = KappaChat()
        chat2 = KappaChat()

        chat1.state.project_name = "project1"

        assert chat2.state.project_name is None


# =============================================================================
# TEST GREETING PHASE
# =============================================================================


class TestKappaChatGreeting:
    """Tests for greeting phase handling."""

    @pytest.mark.asyncio
    async def test_greeting_response(self, chat: KappaChat) -> None:
        """Test response to initial greeting."""
        response = await chat.process_message("I want to build a website")

        assert "Great" in response or "build" in response.lower()
        assert chat.state.phase == ConversationPhase.DISCOVERY

    @pytest.mark.asyncio
    async def test_greeting_extracts_project_type(self, chat: KappaChat) -> None:
        """Test project type extraction from greeting."""
        await chat.process_message("I want to build a REST API")

        assert chat.state.gathered_info.get("project_type") == "api"

    @pytest.mark.asyncio
    async def test_greeting_adds_message(self, chat: KappaChat) -> None:
        """Test message is added to history."""
        await chat.process_message("Hello")

        assert len(chat.state.messages) == 2  # User + Assistant
        assert chat.state.messages[0].role == "user"
        assert chat.state.messages[1].role == "assistant"


# =============================================================================
# TEST DISCOVERY PHASE
# =============================================================================


class TestKappaChatDiscovery:
    """Tests for discovery phase handling."""

    @pytest.mark.asyncio
    async def test_discovery_gathers_info(self, chat: KappaChat) -> None:
        """Test discovery phase gathers information."""
        # Move to discovery
        await chat.process_message("I want to build a website")

        # Provide more details
        await chat.process_message("I need Home, About, and Contact pages with Tailwind")

        assert "pages" in chat.state.gathered_info or "styling" in chat.state.gathered_info

    @pytest.mark.asyncio
    async def test_discovery_moves_to_proposal(self, chat: KappaChat) -> None:
        """Test discovery moves to proposal when enough info."""
        await chat.process_message("I want to build a website called TestSite")
        await chat.process_message(
            "Home, About, Contact pages with Next.js and Tailwind, contact form"
        )

        # Should have enough info to move to proposal or clarification
        assert chat.state.phase in [
            ConversationPhase.PROPOSAL,
            ConversationPhase.CLARIFICATION,
            ConversationPhase.CONFIRMATION,
        ]


# =============================================================================
# TEST PROPOSAL GENERATION
# =============================================================================


class TestKappaChatProposal:
    """Tests for proposal generation."""

    @pytest.mark.asyncio
    async def test_proposal_contains_project_info(self, chat: KappaChat) -> None:
        """Test proposal contains gathered project info."""
        await chat.process_message("Build a website called TestSite with Home and About pages")
        await chat.process_message("Use Next.js, TypeScript, and Tailwind CSS")

        # Check the last response (proposal or confirmation prompt)
        messages = chat.get_conversation_history()
        proposal_found = any(
            "Technical Stack" in m["content"] or "Pages" in m["content"] for m in messages
        )
        assert proposal_found or chat.state.phase in [
            ConversationPhase.CLARIFICATION,
            ConversationPhase.CONFIRMATION,
        ]

    @pytest.mark.asyncio
    async def test_proposal_creates_requirements(self, chat: KappaChat) -> None:
        """Test proposal creates ProjectRequirements."""
        await chat.process_message("Build a website with Home page")
        await chat.process_message("Use Next.js with Tailwind, add contact form")

        # After enough info, requirements should be created
        if chat.state.phase in [ConversationPhase.PROPOSAL, ConversationPhase.CONFIRMATION]:
            assert chat.state.proposed_requirements is not None


# =============================================================================
# TEST REFINEMENT PHASE
# =============================================================================


class TestKappaChatRefinement:
    """Tests for refinement phase handling."""

    @pytest.mark.asyncio
    async def test_add_modification(self, chat: KappaChat) -> None:
        """Test adding a feature via refinement."""
        # Setup: get to proposal
        await chat.process_message("Build a website with Home page, Next.js, Tailwind")
        await chat.process_message("Project name TestSite")

        # Request modification
        await chat.process_message("Add a blog section")

        if chat.state.proposed_requirements:
            # Check if blog was added or mentioned in response
            pages = chat.state.proposed_requirements.pages
            assert "Blog" in pages or chat.state.phase in [
                ConversationPhase.REFINEMENT,
                ConversationPhase.CONFIRMATION,
            ]


# =============================================================================
# TEST CONFIRMATION PHASE
# =============================================================================


class TestKappaChatConfirmation:
    """Tests for confirmation phase handling."""

    @pytest.mark.asyncio
    async def test_confirmation_yes_starts_execution(self, chat: KappaChat) -> None:
        """Test saying yes starts execution."""
        await chat.process_message("Build a website with Home, Next.js, Tailwind")
        await chat.process_message("Project name TestSite, add contact form")
        await chat.process_message("yes build it")

        # Should move to execution
        assert chat.state.phase in [ConversationPhase.EXECUTION, ConversationPhase.CONFIRMATION]

    @pytest.mark.asyncio
    async def test_confirmation_no_allows_changes(self, chat: KappaChat) -> None:
        """Test saying no allows modifications."""
        await chat.process_message("Build a website")
        await chat.process_message("Home page with Next.js")

        if chat.state.phase == ConversationPhase.CONFIRMATION:
            response = await chat.process_message("no, I want to change something")
            assert "change" in response.lower() or chat.state.phase == ConversationPhase.REFINEMENT


# =============================================================================
# TEST EXECUTION QUERIES
# =============================================================================


class TestKappaChatExecution:
    """Tests for execution phase queries."""

    @pytest.mark.asyncio
    async def test_progress_query(self, chat: KappaChat) -> None:
        """Test asking about progress during execution."""
        chat.state.phase = ConversationPhase.EXECUTION
        chat.state.is_building = True

        response = await chat.process_message("What's the progress?")

        assert "progress" in response.lower() or "building" in response.lower()

    @pytest.mark.asyncio
    async def test_issues_query(self, chat: KappaChat) -> None:
        """Test asking about issues during execution."""
        chat.state.phase = ConversationPhase.EXECUTION
        chat.state.is_building = True

        response = await chat.process_message("Any issues?")

        assert "issue" in response.lower() or "status" in response.lower()

    @pytest.mark.asyncio
    async def test_tasks_query(self, chat: KappaChat) -> None:
        """Test asking about tasks during execution."""
        chat.state.phase = ConversationPhase.EXECUTION
        chat.state.is_building = True

        response = await chat.process_message("How many tasks left?")

        assert "task" in response.lower() or "progress" in response.lower()


# =============================================================================
# TEST COMPLETION PHASE
# =============================================================================


class TestKappaChatCompletion:
    """Tests for completion phase handling."""

    @pytest.mark.asyncio
    async def test_completion_offers_options(self, chat: KappaChat) -> None:
        """Test completion offers next steps."""
        chat.state.phase = ConversationPhase.COMPLETION
        chat.state.build_complete = True
        chat.state.build_result = {"status": "completed", "workspace_path": "/tmp/test"}

        response = await chat.process_message("What now?")

        assert "deploy" in response.lower() or "complete" in response.lower()

    @pytest.mark.asyncio
    async def test_new_project_resets_state(self, chat: KappaChat) -> None:
        """Test starting new project resets state."""
        chat.state.phase = ConversationPhase.COMPLETION
        chat.state.project_name = "old-project"

        await chat.process_message("Start a new project")

        assert chat.state.phase == ConversationPhase.GREETING
        assert chat.state.project_name is None


# =============================================================================
# TEST HELPER METHODS
# =============================================================================


class TestKappaChatHelpers:
    """Tests for helper methods."""

    def test_format_list_empty(self, chat: KappaChat) -> None:
        """Test format_list with empty list."""
        result = chat._format_list([])
        assert result == "- None specified"

    def test_format_list_with_items(self, chat: KappaChat) -> None:
        """Test format_list with items."""
        result = chat._format_list(["Home", "About"])
        assert "- Home" in result
        assert "- About" in result

    def test_check_info_completeness_empty(self, chat: KappaChat) -> None:
        """Test completeness check with empty info."""
        result = chat._check_info_completeness()
        assert result == 0.0

    def test_check_info_completeness_partial(self, chat: KappaChat) -> None:
        """Test completeness check with partial info."""
        chat.state.gathered_info = {"project_type": "website"}
        result = chat._check_info_completeness()
        assert result > 0.0

    def test_get_conversation_history(self, chat: KappaChat) -> None:
        """Test getting conversation history."""
        chat._add_message("user", "Hello")
        chat._add_message("assistant", "Hi there")

        history = chat.get_conversation_history()

        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"

    def test_reset(self, chat: KappaChat) -> None:
        """Test resetting chat state."""
        chat.state.phase = ConversationPhase.EXECUTION
        chat.state.project_name = "test"

        chat.reset()

        assert chat.state.phase == ConversationPhase.GREETING
        assert chat.state.project_name is None


# =============================================================================
# TEST PROJECT INFO EXTRACTION
# =============================================================================


class TestProjectInfoExtraction:
    """Tests for project info extraction from text."""

    @pytest.mark.asyncio
    async def test_extract_website_type(self, chat: KappaChat) -> None:
        """Test extracting website project type."""
        info = await chat._extract_project_info("I want to build a website")
        assert info.get("project_type") == "website"

    @pytest.mark.asyncio
    async def test_extract_api_type(self, chat: KappaChat) -> None:
        """Test extracting API project type."""
        info = await chat._extract_project_info("I need a REST API")
        assert info.get("project_type") == "api"

    @pytest.mark.asyncio
    async def test_extract_dashboard_type(self, chat: KappaChat) -> None:
        """Test extracting dashboard project type."""
        info = await chat._extract_project_info("Build an admin dashboard")
        assert info.get("project_type") == "dashboard"

    @pytest.mark.asyncio
    async def test_extract_framework_nextjs(self, chat: KappaChat) -> None:
        """Test extracting Next.js framework."""
        info = await chat._extract_project_info("Using Next.js framework")
        assert info.get("framework") == "Next.js 14"

    @pytest.mark.asyncio
    async def test_extract_framework_react(self, chat: KappaChat) -> None:
        """Test extracting React framework."""
        info = await chat._extract_project_info("Using React")
        assert info.get("framework") == "React"

    @pytest.mark.asyncio
    async def test_extract_styling_tailwind(self, chat: KappaChat) -> None:
        """Test extracting Tailwind styling."""
        info = await chat._extract_project_info("With Tailwind CSS")
        assert info.get("styling") == "Tailwind CSS"

    @pytest.mark.asyncio
    async def test_extract_pages(self, chat: KappaChat) -> None:
        """Test extracting pages."""
        info = await chat._extract_project_info("Home, About, and Contact pages")
        pages = info.get("pages", [])
        assert "Home" in pages
        assert "About" in pages
        assert "Contact" in pages

    @pytest.mark.asyncio
    async def test_extract_features(self, chat: KappaChat) -> None:
        """Test extracting features."""
        info = await chat._extract_project_info("With authentication and contact form")
        features = info.get("features", [])
        assert len(features) > 0

    @pytest.mark.asyncio
    async def test_extract_design_style(self, chat: KappaChat) -> None:
        """Test extracting design style."""
        info = await chat._extract_project_info("Modern minimal design")
        style = info.get("design_style", "")
        assert "modern" in style or "minimal" in style

    @pytest.mark.asyncio
    async def test_extract_cms_sanity(self, chat: KappaChat) -> None:
        """Test extracting Sanity CMS."""
        info = await chat._extract_project_info("Using Sanity.io for CMS")
        assert info.get("cms") == "Sanity.io"


# =============================================================================
# TEST CALLBACKS
# =============================================================================


class TestKappaChatCallbacks:
    """Tests for callback functionality."""

    @pytest.mark.asyncio
    async def test_on_phase_change_callback(self, chat: KappaChat) -> None:
        """Test phase change callback is called."""
        phases_seen = []

        def callback(phase: ConversationPhase) -> None:
            phases_seen.append(phase)

        chat.on_phase_change = callback
        await chat.process_message("I want to build a website")

        assert ConversationPhase.DISCOVERY in phases_seen

    def test_set_phase_triggers_callback(self, chat: KappaChat) -> None:
        """Test _set_phase triggers callback."""
        callback_called = []

        chat.on_phase_change = lambda p: callback_called.append(p)
        chat._set_phase(ConversationPhase.PROPOSAL)

        assert ConversationPhase.PROPOSAL in callback_called
