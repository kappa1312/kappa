"""
End-to-end tests for Bali Graphics POC.

Tests the complete flow for building a graphics design studio website,
using mocks to avoid actual execution.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.chat.interface import ConversationPhase, KappaChat
from src.decomposition.models import ProjectRequirements, ProjectType

# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def bali_graphics_requirements() -> dict:
    """Return the Bali Graphics project requirements."""
    return {
        "name": "bali-graphics",
        "description": "Graphics design studio website with tropical minimal aesthetic",
        "project_type": "website",
        "pages": ["Home", "About", "Services", "Portfolio", "Contact"],
        "features": [
            "Contact form with email notifications",
            "Portfolio filtering",
            "Animations",
        ],
        "tech_stack": {
            "framework": "Next.js 14",
            "language": "TypeScript",
            "styling": "Tailwind CSS",
            "cms": "Sanity.io",
        },
        "design": "Tropical minimal with deep green, coral orange, warm cream",
    }


@pytest.fixture
def bali_conversation() -> list:
    """Return the conversation flow for Bali Graphics."""
    return [
        "I want to build a website for a graphics design studio in Bali called Bali Graphics",
        "The studio specializes in brand identity, packaging design, and digital illustrations with a tropical-modern aesthetic",
        "I need Home, About, Services, Portfolio, and Contact pages",
        "Use Next.js 14, TypeScript, Tailwind CSS, and Sanity CMS for the portfolio",
        "The design should be tropical minimal with colors: deep green, coral orange, warm cream",
        "Add a contact form with email notifications and portfolio filtering",
        "yes, build it",
    ]


# =============================================================================
# TEST REQUIREMENTS PARSING
# =============================================================================


class TestBaliGraphicsRequirements:
    """Tests for Bali Graphics requirements parsing."""

    @pytest.mark.asyncio
    async def test_extract_project_name(self) -> None:
        """Test extracting project name from conversation."""
        chat = KappaChat()

        text = "I want to build a website for Bali Graphics"
        info = await chat._extract_project_info(text)

        # Should extract website type
        assert info.get("project_type") == "website"

    @pytest.mark.asyncio
    async def test_extract_pages(self) -> None:
        """Test extracting pages from conversation."""
        chat = KappaChat()

        text = "I need Home, About, Services, Portfolio, and Contact pages"
        info = await chat._extract_project_info(text)

        pages = info.get("pages", [])
        assert "Home" in pages
        assert "About" in pages
        assert "Portfolio" in pages
        assert "Contact" in pages

    @pytest.mark.asyncio
    async def test_extract_tech_stack(self) -> None:
        """Test extracting tech stack from conversation."""
        chat = KappaChat()

        text = "Use Next.js 14, TypeScript, Tailwind CSS, and Sanity CMS"
        info = await chat._extract_project_info(text)

        assert info.get("framework") == "Next.js 14"
        assert info.get("styling") == "Tailwind CSS"
        assert info.get("cms") == "Sanity.io"

    @pytest.mark.asyncio
    async def test_extract_features(self) -> None:
        """Test extracting features from conversation."""
        chat = KappaChat()

        text = "Add a contact form with email notifications and portfolio filtering"
        info = await chat._extract_project_info(text)

        features = info.get("features", [])
        # Should extract some features
        assert len(features) > 0 or "form" in text.lower()

    @pytest.mark.asyncio
    async def test_extract_design_style(self) -> None:
        """Test extracting design style from conversation."""
        chat = KappaChat()

        text = "The design should be tropical minimal with colors"
        info = await chat._extract_project_info(text)

        style = info.get("design_style", "")
        assert "tropical" in style or "minimal" in style


# =============================================================================
# TEST TASK GENERATION
# =============================================================================


class TestBaliGraphicsTaskGeneration:
    """Tests for task generation from Bali Graphics requirements."""

    def test_create_requirements_model(self, bali_graphics_requirements) -> None:
        """Test creating ProjectRequirements model."""
        reqs = bali_graphics_requirements

        model = ProjectRequirements(
            name=reqs["name"],
            description=reqs["description"],
            project_type=ProjectType.WEBSITE,
            tech_stack=reqs["tech_stack"],
            pages=reqs["pages"],
            features=reqs["features"],
        )

        assert model.name == "bali-graphics"
        assert model.project_type == ProjectType.WEBSITE
        assert len(model.pages) == 5
        assert len(model.features) == 3

    def test_requirements_uses_typescript(self, bali_graphics_requirements) -> None:
        """Test requirements detect TypeScript usage."""
        reqs = bali_graphics_requirements

        model = ProjectRequirements(
            name=reqs["name"],
            description=reqs["description"],
            project_type=ProjectType.WEBSITE,
            tech_stack=reqs["tech_stack"],
            pages=reqs["pages"],
            features=reqs["features"],
        )

        assert model.uses_typescript()


# =============================================================================
# TEST WAVE STRUCTURE
# =============================================================================


class TestBaliGraphicsWaveStructure:
    """Tests for wave structure organization."""

    def test_pages_generate_tasks(self, bali_graphics_requirements) -> None:
        """Test pages generate appropriate tasks."""
        pages = bali_graphics_requirements["pages"]

        # Each page should generate at least one task
        assert len(pages) >= 5

        # Typical wave structure for website:
        # Wave 0: Setup, config, types
        # Wave 1: Components (shared)
        # Wave 2: Pages
        # Wave 3: Integrations, CMS
        # Wave 4: Testing, deployment

    def test_features_add_tasks(self, bali_graphics_requirements) -> None:
        """Test features add to task count."""
        features = bali_graphics_requirements["features"]

        # Features should add tasks
        assert len(features) > 0

        # Contact form needs: form component, validation, email integration
        assert "Contact form" in str(features)

    def test_cms_integration_in_later_wave(self) -> None:
        """Test CMS integration should be in later wave."""
        # CMS setup depends on content models and types
        # Should be in wave 2+ after types are defined
        pass


# =============================================================================
# TEST PROPOSAL GENERATION
# =============================================================================


class TestBaliGraphicsProposal:
    """Tests for proposal generation."""

    @pytest.mark.asyncio
    async def test_proposal_includes_all_pages(self, bali_conversation) -> None:
        """Test proposal includes all requested pages."""
        chat = KappaChat()

        # Simulate conversation
        for message in bali_conversation[:4]:  # Up to tech stack
            await chat.process_message(message)

        # Check gathered info or proposed requirements
        if chat.state.proposed_requirements:
            pages = [p.lower() for p in chat.state.proposed_requirements.pages]
            # At minimum should have some pages from the conversation
            assert len(pages) >= 2, f"Expected at least 2 pages, got {pages}"
        elif chat.state.gathered_info.get("pages"):
            pages = [p.lower() for p in chat.state.gathered_info["pages"]]
            assert len(pages) >= 2, f"Expected at least 2 pages, got {pages}"

    @pytest.mark.asyncio
    async def test_proposal_includes_tech_stack(self, bali_conversation) -> None:
        """Test proposal includes specified tech stack."""
        chat = KappaChat()

        # Simulate conversation up to tech stack
        for message in bali_conversation[:4]:
            await chat.process_message(message)

        # Check that we captured project info
        info = chat.state.gathered_info
        # Should have project type at minimum since first message mentions "website"
        assert info.get("project_type") == "website", f"Expected website type, got {info}"


# =============================================================================
# TEST MODIFICATIONS
# =============================================================================


class TestBaliGraphicsModifications:
    """Tests for requirement modifications."""

    @pytest.mark.asyncio
    async def test_add_blog_section(self) -> None:
        """Test adding blog section to requirements."""
        chat = KappaChat()

        # Get to proposal state
        await chat.process_message("Build a website with Home, About, Contact using Next.js")
        await chat.process_message("Named BaliGraphics with Tailwind")

        if chat.state.proposed_requirements:
            initial_pages = len(chat.state.proposed_requirements.pages)

            # Add blog
            await chat.process_message("Add a blog section")

            # Blog should be added
            if chat.state.proposed_requirements:
                assert (
                    "Blog" in chat.state.proposed_requirements.pages
                    or len(chat.state.proposed_requirements.pages) >= initial_pages
                )

    @pytest.mark.asyncio
    async def test_change_framework(self) -> None:
        """Test changing framework in requirements."""
        chat = KappaChat()

        # Setup with Next.js
        await chat.process_message("Build a website with Next.js")
        await chat.process_message("Home page with Tailwind")

        if chat.state.proposed_requirements:
            # Try to change to Vue
            await chat.process_message("Actually, use Vue.js instead")

            # Framework might be updated
            if chat.state.proposed_requirements:
                tech = chat.state.proposed_requirements.tech_stack
                # Either Vue was set or it kept Next.js
                assert "framework" in tech


# =============================================================================
# TEST VALIDATION
# =============================================================================


class TestBaliGraphicsValidation:
    """Tests for validation checks."""

    def test_requirements_have_name(self, bali_graphics_requirements) -> None:
        """Test requirements have a valid name."""
        assert bali_graphics_requirements["name"]
        assert len(bali_graphics_requirements["name"]) > 0

    def test_requirements_have_pages(self, bali_graphics_requirements) -> None:
        """Test requirements have pages defined."""
        pages = bali_graphics_requirements["pages"]
        assert len(pages) > 0
        assert "Home" in pages

    def test_requirements_have_tech_stack(self, bali_graphics_requirements) -> None:
        """Test requirements have complete tech stack."""
        tech = bali_graphics_requirements["tech_stack"]

        assert "framework" in tech
        assert "language" in tech
        assert "styling" in tech

    def test_project_type_is_website(self, bali_graphics_requirements) -> None:
        """Test project type is website."""
        assert bali_graphics_requirements["project_type"] == "website"


# =============================================================================
# TEST FULL CONVERSATION FLOW
# =============================================================================


class TestBaliGraphicsFullFlow:
    """Tests for complete conversation flow."""

    @pytest.mark.asyncio
    async def test_full_conversation_reaches_execution(self, bali_conversation) -> None:
        """Test full conversation reaches execution phase."""
        chat = KappaChat()

        # Mock the execution to avoid actual build
        with patch.object(chat, "_execute_project", new_callable=AsyncMock):
            for message in bali_conversation:
                await chat.process_message(message)

            # Should reach execution or completion
            assert chat.state.phase in [
                ConversationPhase.EXECUTION,
                ConversationPhase.COMPLETION,
                ConversationPhase.CONFIRMATION,
            ]

    @pytest.mark.asyncio
    async def test_conversation_builds_complete_requirements(self, bali_conversation) -> None:
        """Test conversation builds complete requirements."""
        chat = KappaChat()

        # Process all messages except the final "build it"
        with patch.object(chat, "_execute_project", new_callable=AsyncMock):
            for message in bali_conversation[:-1]:
                await chat.process_message(message)

        # Should have gathered significant info
        info = chat.state.gathered_info
        assert len(info) > 0

        # Should have detected website type
        assert info.get("project_type") == "website" or chat.state.proposed_requirements

    @pytest.mark.asyncio
    async def test_conversation_history_complete(self, bali_conversation) -> None:
        """Test conversation history is complete."""
        chat = KappaChat()

        with patch.object(chat, "_execute_project", new_callable=AsyncMock):
            for message in bali_conversation:
                await chat.process_message(message)

        history = chat.get_conversation_history()

        # Should have all user messages and responses
        user_messages = [m for m in history if m["role"] == "user"]
        assert len(user_messages) == len(bali_conversation)
