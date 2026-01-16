"""
Interactive Chat Interface for Kappa OS.

Handles the ideation -> development conversation flow, guiding users
from initial project ideas to fully built applications.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from rich.console import Console


class ConversationPhase(str, Enum):
    """Phases of the ideation -> development flow."""

    GREETING = "greeting"
    DISCOVERY = "discovery"  # Understanding what user wants
    CLARIFICATION = "clarification"  # Asking follow-up questions
    PROPOSAL = "proposal"  # Presenting structured proposal
    REFINEMENT = "refinement"  # User adjusts the proposal
    CONFIRMATION = "confirmation"  # User confirms to proceed
    EXECUTION = "execution"  # Building the project
    COMPLETION = "completion"  # Project complete


@dataclass
class Message:
    """Single chat message."""

    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationState:
    """Current state of the conversation."""

    phase: ConversationPhase = ConversationPhase.GREETING
    messages: list[Message] = field(default_factory=list)
    project_name: str | None = None
    gathered_info: dict[str, Any] = field(default_factory=dict)
    proposed_requirements: Any | None = None  # ProjectRequirements
    is_building: bool = False
    project_id: str | None = None
    build_complete: bool = False
    build_result: dict[str, Any] | None = None


class KappaChat:
    """
    Interactive chat interface for Kappa OS.

    Guides users from ideation to completed project through
    a natural conversation flow powered by Claude.

    Usage:
        chat = KappaChat()
        response = await chat.process_message("I want to build a website")

    Attributes:
        state: Current conversation state.
        on_phase_change: Optional callback when phase changes.
        on_progress_update: Optional callback for build progress.
    """

    # System prompt for Claude-powered conversation
    SYSTEM_PROMPT = """You are Kappa OS, an autonomous development assistant that helps users build software projects.

Your role is to:
1. Understand what the user wants to build
2. Ask clarifying questions to gather requirements
3. Create a comprehensive project proposal
4. Refine the proposal based on feedback
5. Execute the build when approved

Current conversation phase: {phase}
Current gathered information: {gathered_info}
Current proposed requirements: {proposed_requirements}

Guidelines:
- Be conversational, helpful, and professional
- Ask focused questions to understand requirements
- When the user wants to modify the proposal, understand their intent and apply changes
- Support adding pages, features, integrations, changing tech stack, etc.
- When user says "yes", "build it", "go ahead", etc., indicate readiness to build
- Keep responses concise but informative

Respond naturally to the user's message. If they want to modify the proposal, acknowledge their request and describe what changes you'll make."""

    def __init__(self, workspace: str | None = None) -> None:
        """
        Initialize KappaChat.

        Args:
            workspace: Optional workspace directory for project output.
        """
        self.state = ConversationState()
        self.workspace = workspace

        # Callbacks for UI updates
        self.on_phase_change: Callable[[ConversationPhase], None] | None = None
        self.on_progress_update: Callable[[dict[str, Any]], None] | None = None

        # Lazy-loaded components
        self._kappa: Any | None = None
        self._parser: Any | None = None
        self._claude_client: Any | None = None

    @property
    def claude_client(self) -> Any:
        """Get or create Claude API client."""
        if self._claude_client is None:
            try:
                from anthropic import AsyncAnthropic

                from src.core.config import get_settings

                settings = get_settings()
                self._claude_client = AsyncAnthropic(
                    api_key=settings.anthropic_api_key.get_secret_value()
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Claude client: {e}")
                self._claude_client = None
        return self._claude_client

    @property
    def kappa(self) -> Any:
        """Get or create Kappa orchestrator instance."""
        if self._kappa is None:
            from src.core.orchestrator import Kappa

            self._kappa = Kappa(workspace=self.workspace) if self.workspace else Kappa()
        return self._kappa

    @property
    def parser(self) -> Any:
        """Get or create requirements parser instance."""
        if self._parser is None:
            from src.decomposition.parser import RequirementsParser

            self._parser = RequirementsParser()
        return self._parser

    async def _call_claude(self, user_message: str, context: str = "") -> str:
        """Call Claude API for intelligent conversation.

        Args:
            user_message: The user's message.
            context: Additional context to include.

        Returns:
            Claude's response.
        """
        if not self.claude_client:
            return self._fallback_response(user_message)

        try:
            # Build conversation history for context
            messages = []

            # Add recent conversation history (last 10 messages)
            for msg in self.state.messages[-10:]:
                messages.append({
                    "role": "user" if msg.role == "user" else "assistant",
                    "content": msg.content
                })

            # Add current message
            messages.append({"role": "user", "content": user_message})

            # Build system prompt with current state
            system = self.SYSTEM_PROMPT.format(
                phase=self.state.phase.value,
                gathered_info=str(self.state.gathered_info),
                proposed_requirements=(
                    str(self.state.proposed_requirements.__dict__)
                    if self.state.proposed_requirements
                    else "None"
                ),
            )

            if context:
                system += f"\n\nAdditional context:\n{context}"

            response = await self.claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                system=system,
                messages=messages,
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            return self._fallback_response(user_message)

    def _fallback_response(self, user_message: str) -> str:  # noqa: ARG002
        """Fallback response when Claude API is unavailable."""
        return (
            "I apologize, but I'm having trouble processing your request right now. "
            "Please make sure your ANTHROPIC_API_KEY is set correctly.\n\n"
            "You can try:\n"
            "- Checking your API key: `echo $ANTHROPIC_API_KEY`\n"
            "- Setting it: `export ANTHROPIC_API_KEY=sk-ant-...`"
        )

    async def process_message(self, user_input: str) -> str:
        """
        Process user message and return response.

        This is the main entry point for chat interaction. Routes
        to appropriate handler based on current conversation phase.

        Args:
            user_input: The user's message text.

        Returns:
            The assistant's response text.
        """
        # Add user message to history
        self._add_message("user", user_input)

        # Route based on current phase
        handler_map = {
            ConversationPhase.GREETING: self._handle_greeting,
            ConversationPhase.DISCOVERY: self._handle_discovery,
            ConversationPhase.CLARIFICATION: self._handle_clarification,
            ConversationPhase.PROPOSAL: self._handle_proposal_feedback,
            ConversationPhase.REFINEMENT: self._handle_refinement,
            ConversationPhase.CONFIRMATION: self._handle_confirmation,
            ConversationPhase.EXECUTION: self._handle_execution_query,
            ConversationPhase.COMPLETION: self._handle_completion,
        }

        handler = handler_map.get(self.state.phase, self._handle_completion)
        response = await handler(user_input)

        # Add assistant response to history
        self._add_message("assistant", response)

        return response

    # =========================================================================
    # PHASE HANDLERS
    # =========================================================================

    async def _handle_greeting(self, user_input: str) -> str:
        """Handle initial greeting and project description."""
        # Extract initial project info
        info = await self._extract_project_info(user_input)
        self.state.gathered_info.update(info)

        # Move to discovery phase
        self._set_phase(ConversationPhase.DISCOVERY)

        # Generate clarifying questions
        questions = self._generate_discovery_questions(info)

        project_type = info.get("project_type", "web project")

        return f"""Great! I'd love to help you build that.

Based on what you've shared, I understand you want to build a **{project_type}**.

Let me ask a few questions to better understand your vision:

{questions}

Feel free to answer any or all of these!"""

    async def _handle_discovery(self, user_input: str) -> str:
        """Gather more information about the project."""
        # Extract additional info from response
        new_info = await self._extract_project_info(user_input)
        self.state.gathered_info.update(new_info)

        # Check if we have enough information
        completeness = self._check_info_completeness()

        if completeness >= 0.7:
            # Move to proposal
            self._set_phase(ConversationPhase.PROPOSAL)
            return await self._generate_proposal()
        else:
            # Ask more clarifying questions
            self._set_phase(ConversationPhase.CLARIFICATION)
            missing = self._get_missing_info()

            return f"""Thanks for those details! I just need a bit more information:

{missing}

This will help me create the perfect project structure for you."""

    async def _handle_clarification(self, user_input: str) -> str:
        """Handle responses to clarifying questions."""
        # Extract info
        new_info = await self._extract_project_info(user_input)
        self.state.gathered_info.update(new_info)

        # Move to proposal
        self._set_phase(ConversationPhase.PROPOSAL)
        return await self._generate_proposal()

    async def _generate_proposal(self) -> str:
        """Generate and present the project proposal."""
        # Convert gathered info to requirements
        requirements_text = self._info_to_requirements_text()

        # Try to parse into structured requirements
        try:
            self.state.proposed_requirements = await self.parser.parse(requirements_text)
        except Exception as e:
            logger.warning(f"Failed to parse requirements: {e}")
            # Use gathered info directly
            self.state.proposed_requirements = self._create_requirements_from_info()

        req = self.state.proposed_requirements
        self.state.project_name = req.name

        proposal = f"""# Project Proposal: {req.name}

## Overview
{req.description}

## Technical Stack
- **Framework:** {req.tech_stack.get('framework', 'Next.js 14')}
- **Language:** {req.tech_stack.get('language', 'TypeScript')}
- **Styling:** {req.tech_stack.get('styling', 'Tailwind CSS')}
- **CMS:** {req.tech_stack.get('cms', 'None')}

## Pages
{self._format_list(req.pages)}

## Features
{self._format_list(req.features)}

## Integrations
{self._format_list(req.integrations)}

---

**Does this look good?**

You can:
- Say **"yes"** or **"build it"** to start development
- Tell me what to **add, remove, or change**
- Ask me to **explain** any part in more detail"""

        self._set_phase(ConversationPhase.CONFIRMATION)
        return proposal

    async def _handle_proposal_feedback(self, user_input: str) -> str:
        """Handle feedback on the proposal."""
        return await self._handle_refinement(user_input)

    async def _handle_refinement(self, user_input: str) -> str:
        """Handle user refinements to the proposal using Claude."""
        lower_input = user_input.lower()

        # Check for clear approval signals
        approval_phrases = [
            "yes", "build it", "build", "proceed", "start building",
            "go ahead", "looks good", "perfect", "great", "let's go",
            "ok build", "okay build", "confirm", "approved", "ship it"
        ]

        # Only trigger build on clear approval, not just "ok" or "yes" in a sentence
        words = lower_input.split()
        is_approval = (
            any(phrase in lower_input for phrase in approval_phrases)
            and len(words) <= 5  # Short confirmations only
            and not any(w in lower_input for w in ["add", "remove", "change", "modify", "but"])
        )

        if is_approval:
            return await self._start_execution()

        # Use Claude to understand and process the modification request
        modifications = await self._extract_modifications_with_claude(user_input)

        if modifications:
            # Show updated proposal
            updated_proposal = await self._generate_proposal()
            return f"""Got it! I've updated the proposal:

**Changes made:**
{self._format_list(modifications)}

{updated_proposal}"""
        else:
            # Let Claude handle the conversation naturally
            context = f"""The user is asking about or trying to modify the project proposal.
Current proposal:
- Name: {self.state.proposed_requirements.name if self.state.proposed_requirements else 'Unknown'}
- Pages: {self.state.proposed_requirements.pages if self.state.proposed_requirements else []}
- Features: {self.state.proposed_requirements.features if self.state.proposed_requirements else []}
- Tech Stack: {self.state.proposed_requirements.tech_stack if self.state.proposed_requirements else {}}

Help the user modify the proposal or clarify what they want."""

            return await self._call_claude(user_input, context)

    async def _handle_confirmation(self, user_input: str) -> str:
        """Handle final confirmation before building using Claude for understanding."""
        lower_input = user_input.lower()
        words = lower_input.split()

        # Clear approval signals (short responses)
        approval_words = ["yes", "build", "proceed", "start", "go", "confirm", "ok", "okay", "ship"]
        is_approval = (
            any(word in lower_input for word in approval_words)
            and len(words) <= 5
            and not any(w in lower_input for w in ["add", "remove", "change", "modify", "but", "wait"])
        )

        if is_approval:
            return await self._start_execution()

        # Check for modification intent
        modification_words = ["no", "wait", "change", "modify", "add", "remove", "update", "instead"]
        if any(word in lower_input for word in modification_words):
            self._set_phase(ConversationPhase.REFINEMENT)
            # Process the modification directly
            return await self._handle_refinement(user_input)
        else:
            return """Just to confirm - would you like me to start building the project?

Say **"yes"** to begin, or tell me what you'd like to change."""

    async def _start_execution(self) -> str:
        """Start the project execution."""
        self._set_phase(ConversationPhase.EXECUTION)
        self.state.is_building = True

        response = """**Starting development!**

I'm now building your project. You can watch the progress in the dashboard.

**What's happening:**
1. Analyzing requirements and creating tasks
2. Organizing tasks into execution waves
3. Spawning parallel Claude Code sessions
4. Building components simultaneously
5. Merging outputs and resolving conflicts
6. Validating the final build

I'll let you know when it's ready!

---

*You can ask me questions while I'm building, like:*
- "What's the current progress?"
- "How many tasks are left?"
- "Any issues so far?"
"""

        # Start execution in background
        asyncio.create_task(self._execute_project())

        return response

    async def _execute_project(self) -> None:
        """Execute the project build (runs in background)."""
        try:
            # Convert requirements to text format for Kappa
            requirements_text = self._requirements_to_text()

            project_name = (
                self.state.proposed_requirements.name
                if self.state.proposed_requirements
                else self.state.project_name or "kappa-project"
            )

            # Execute through Kappa
            result = await self.kappa.execute(
                requirements=requirements_text,
                project_name=project_name,
            )

            self.state.project_id = result.get("project_id")
            self.state.build_result = result
            self.state.is_building = False
            self.state.build_complete = True

            if result["status"] == "completed":
                self._set_phase(ConversationPhase.COMPLETION)
                self._add_message(
                    "system",
                    f"Project completed successfully! Workspace: {result.get('workspace_path')}",
                )
            else:
                self._add_message(
                    "system",
                    f"Project completed with status: {result['status']}",
                )

            # Trigger callback if set
            if self.on_progress_update:
                self.on_progress_update(result)

        except Exception as e:
            logger.error(f"Execution failed: {e}")
            self.state.is_building = False
            self.state.build_complete = True
            self.state.build_result = {"status": "failed", "error": str(e)}
            self._add_message("system", f"Build failed: {str(e)}")

    async def _handle_execution_query(self, user_input: str) -> str:
        """Handle questions during execution."""
        lower_input = user_input.lower()

        # Check if build is complete
        if self.state.build_complete:
            self._set_phase(ConversationPhase.COMPLETION)
            if self.state.build_result and self.state.build_result.get("status") == "completed":
                return f"""Your project is complete!

**Workspace:** {self.state.build_result.get('workspace_path', 'N/A')}

You can:
- Say **"deploy"** to deploy to Vercel
- Say **"new project"** to start something else
- Ask me anything about the generated code"""
            else:
                error = (
                    self.state.build_result.get("error", "Unknown error")
                    if self.state.build_result
                    else "Unknown error"
                )
                return f"""Unfortunately, the build encountered an issue:

**Error:** {error}

Would you like to:
- Try building again?
- Start a new project?"""

        if "progress" in lower_input or "status" in lower_input:
            return self._get_progress_summary()

        elif any(word in lower_input for word in ["issue", "error", "problem"]):
            return self._get_issues_summary()

        elif "task" in lower_input:
            return self._get_tasks_summary()

        else:
            return """I'm currently building your project!

You can ask:
- **"What's the progress?"** - See overall completion
- **"Any issues?"** - Check for problems
- **"How many tasks left?"** - See remaining work

The dashboard also shows real-time updates."""

    async def _handle_completion(self, user_input: str) -> str:
        """Handle conversation after completion."""
        lower_input = user_input.lower()

        if "deploy" in lower_input:
            return await self._deploy_project()

        elif any(word in lower_input for word in ["new", "another", "start over", "reset"]):
            # Reset for new project
            self.state = ConversationState()
            self._kappa = None  # Reset orchestrator
            return """Ready to start a new project!

What would you like to build next?"""

        else:
            workspace = (
                self.state.build_result.get("workspace_path", "/tmp/kappa-workspace")
                if self.state.build_result
                else self.state.gathered_info.get("workspace", "/tmp/kappa-workspace")
            )

            return f"""Your project is complete!

**What's next?**
- Say **"deploy"** to deploy to Vercel
- Say **"new project"** to start something else
- Ask me anything about the generated code

The project is ready at: {workspace}"""

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _add_message(self, role: str, content: str) -> None:
        """Add message to conversation history."""
        self.state.messages.append(
            Message(
                role=role,
                content=content,
                timestamp=datetime.now().isoformat(),
            )
        )

    def _set_phase(self, phase: ConversationPhase) -> None:
        """Set conversation phase and trigger callback."""
        self.state.phase = phase
        logger.info(f"Phase changed to: {phase.value}")

        if self.on_phase_change:
            self.on_phase_change(phase)

    async def _extract_project_info(self, text: str) -> dict[str, Any]:
        """
        Extract project information from user text.

        Simple keyword-based extraction. In production, this could
        use NLP/LLM for better extraction.
        """
        info: dict[str, Any] = {}
        lower_text = text.lower()

        # Detect project type
        if any(w in lower_text for w in ["website", "site", "web", "page", "landing"]):
            info["project_type"] = "website"
        elif any(w in lower_text for w in ["api", "backend", "server", "rest", "graphql"]):
            info["project_type"] = "api"
        elif any(w in lower_text for w in ["app", "application", "mobile"]):
            info["project_type"] = "application"
        elif any(w in lower_text for w in ["dashboard", "admin", "panel"]):
            info["project_type"] = "dashboard"
        elif any(w in lower_text for w in ["cli", "command", "terminal", "tool"]):
            info["project_type"] = "cli_tool"

        # Extract name if mentioned
        for phrase in ["called ", "named ", "for ", " - "]:
            if phrase in lower_text:
                idx = lower_text.find(phrase) + len(phrase)
                remaining = text[idx:].strip()
                # Take first 3 words as potential name
                words = remaining.split()[:3]
                potential_name = " ".join(words).strip(".,!?")
                if potential_name and len(potential_name) > 2:
                    info["project_name"] = potential_name.replace(" ", "-").lower()
                break

        # Detect tech preferences
        if "next" in lower_text or "nextjs" in lower_text:
            info["framework"] = "Next.js 14"
        if "react" in lower_text and "next" not in lower_text:
            info["framework"] = "React"
        if "vue" in lower_text:
            info["framework"] = "Vue.js"
        if "angular" in lower_text:
            info["framework"] = "Angular"
        if "express" in lower_text:
            info["framework"] = "Express.js"
        if "fastapi" in lower_text:
            info["framework"] = "FastAPI"

        if "typescript" in lower_text or " ts " in lower_text:
            info["language"] = "TypeScript"
        if "javascript" in lower_text or " js " in lower_text:
            info["language"] = "JavaScript"
        if "python" in lower_text:
            info["language"] = "Python"

        if "tailwind" in lower_text:
            info["styling"] = "Tailwind CSS"
        if "bootstrap" in lower_text:
            info["styling"] = "Bootstrap"
        if "sass" in lower_text or "scss" in lower_text:
            info["styling"] = "SASS/SCSS"

        if "sanity" in lower_text:
            info["cms"] = "Sanity.io"
        if "contentful" in lower_text:
            info["cms"] = "Contentful"
        if "strapi" in lower_text:
            info["cms"] = "Strapi"

        # Detect pages mentioned
        pages = []
        page_keywords = [
            ("home", "Home"),
            ("about", "About"),
            ("contact", "Contact"),
            ("portfolio", "Portfolio"),
            ("service", "Services"),
            ("blog", "Blog"),
            ("product", "Products"),
            ("pricing", "Pricing"),
            ("faq", "FAQ"),
            ("team", "Team"),
            ("gallery", "Gallery"),
        ]
        for keyword, page_name in page_keywords:
            if keyword in lower_text:
                pages.append(page_name)
        if pages:
            info["pages"] = pages

        # Detect features mentioned
        features = []
        feature_keywords = [
            ("authentication", "User authentication"),
            ("login", "Login/Signup"),
            ("signup", "Login/Signup"),
            ("contact form", "Contact form"),
            ("form", "Form handling"),
            ("filter", "Filtering"),
            ("search", "Search functionality"),
            ("animation", "Animations"),
            ("dark mode", "Dark mode"),
            ("responsive", "Responsive design"),
            ("seo", "SEO optimization"),
            ("payment", "Payment integration"),
            ("stripe", "Stripe payments"),
            ("email", "Email notifications"),
            ("notification", "Notifications"),
        ]
        for keyword, feature_name in feature_keywords:
            if keyword in lower_text and feature_name not in features:
                features.append(feature_name)
        if features:
            info["features"] = features

        # Extract design description
        design_words = []
        if "minimal" in lower_text:
            design_words.append("minimal")
        if "modern" in lower_text:
            design_words.append("modern")
        if "clean" in lower_text:
            design_words.append("clean")
        if "tropical" in lower_text:
            design_words.append("tropical")
        if "professional" in lower_text:
            design_words.append("professional")
        if "elegant" in lower_text:
            design_words.append("elegant")
        if design_words:
            info["design_style"] = ", ".join(design_words)

        return info

    def _generate_discovery_questions(self, info: dict[str, Any]) -> str:
        """Generate questions based on missing info."""
        questions = []
        counter = 1

        if "project_name" not in info:
            questions.append(f"{counter}. What would you like to name this project?")
            counter += 1

        if "pages" not in info and info.get("project_type") in [
            "website",
            "application",
            "dashboard",
        ]:
            questions.append(
                f"{counter}. What pages do you need? (e.g., Home, About, Contact, Portfolio)"
            )
            counter += 1

        if "styling" not in info:
            questions.append(
                f"{counter}. Any design preferences? (colors, style, reference websites)"
            )
            counter += 1

        if "features" not in info:
            questions.append(
                f"{counter}. What special features do you need? (contact form, CMS, blog, etc.)"
            )
            counter += 1

        return "\n".join(questions) if questions else "Tell me more about your vision!"

    def _check_info_completeness(self) -> float:
        """Check how complete the gathered info is (0-1)."""
        required = ["project_type"]
        optional = [
            "project_name",
            "pages",
            "framework",
            "styling",
            "cms",
            "features",
            "design_style",
        ]

        score = 0.0
        total = len(required) + len(optional) * 0.5

        for key in required:
            if key in self.state.gathered_info:
                score += 1

        for key in optional:
            if key in self.state.gathered_info:
                score += 0.5

        return score / total if total > 0 else 0

    def _get_missing_info(self) -> str:
        """Get questions for missing information."""
        missing = []
        info = self.state.gathered_info

        if "project_name" not in info:
            missing.append("- What should we call this project?")
        if (
            ("pages" not in info or not info.get("pages"))
            and info.get("project_type") in ["website", "application", "dashboard", None]
        ):
            missing.append("- What pages do you need?")
        if "features" not in info:
            missing.append("- Any special features? (forms, CMS, authentication)")

        return "\n".join(missing) if missing else "I think I have everything I need!"

    def _format_list(self, items: list[str] | None) -> str:
        """Format list as bullet points."""
        if not items:
            return "- None specified"
        return "\n".join(f"- {item}" for item in items)

    def _info_to_requirements_text(self) -> str:
        """Convert gathered info to requirements text."""
        info = self.state.gathered_info

        pages_text = (
            "\n".join(f"- {p}" for p in info.get("pages", ["Home"]))
            if info.get("pages")
            else "- Home"
        )
        features_text = (
            "\n".join(f"- {f}" for f in info.get("features", []))
            if info.get("features")
            else "- Basic functionality"
        )

        return f"""
Project: {info.get('project_name', 'untitled-project')}
Type: {info.get('project_type', 'website')}
Framework: {info.get('framework', 'Next.js 14')}
Language: {info.get('language', 'TypeScript')}
Styling: {info.get('styling', 'Tailwind CSS')}
CMS: {info.get('cms', 'None')}

Pages:
{pages_text}

Features:
{features_text}

Design:
{info.get('design_style', 'Modern, clean design')}
"""

    def _create_requirements_from_info(self) -> Any:
        """Create ProjectRequirements from gathered info."""
        from src.decomposition.models import ProjectRequirements, ProjectType

        info = self.state.gathered_info

        # Map project type string to enum
        type_map = {
            "website": ProjectType.WEBSITE,
            "api": ProjectType.API,
            "application": ProjectType.WEBSITE,
            "dashboard": ProjectType.DASHBOARD,
            "cli_tool": ProjectType.CLI_TOOL,
        }
        project_type = type_map.get(info.get("project_type", "website"), ProjectType.WEBSITE)

        return ProjectRequirements(
            name=info.get("project_name", "untitled-project"),
            description=f"Project built with Kappa OS - {info.get('design_style', 'Modern design')}",
            project_type=project_type,
            tech_stack={
                "framework": info.get("framework", "Next.js 14"),
                "language": info.get("language", "TypeScript"),
                "styling": info.get("styling", "Tailwind CSS"),
                "cms": info.get("cms", "None"),
            },
            pages=info.get("pages", ["Home"]),
            features=info.get("features", []),
            integrations=info.get("integrations", []),
        )

    async def _extract_modifications_with_claude(self, text: str) -> list[str]:
        """Use Claude to understand and extract modifications from user request."""
        import json

        if not self.claude_client or not self.state.proposed_requirements:
            # Fallback to basic extraction
            return await self._extract_modifications_basic(text)

        try:
            req = self.state.proposed_requirements
            prompt = f"""Analyze this user request and extract the modifications they want to make to the project proposal.

Current proposal:
- Name: {req.name}
- Pages: {req.pages}
- Features: {req.features}
- Tech Stack: {req.tech_stack}
- Integrations: {req.integrations}

User request: "{text}"

Return a JSON object with these fields:
{{
    "understood": true/false,  // whether you understood the request
    "modifications": [
        {{
            "action": "add" | "remove" | "change",
            "type": "page" | "feature" | "tech_stack" | "integration",
            "item": "the item to add/remove",
            "value": "new value for changes (optional)"
        }}
    ],
    "summary": ["list of human-readable changes made"]
}}

Examples:
- "add our work page" -> {{"understood": true, "modifications": [{{"action": "add", "type": "page", "item": "Our Work"}}], "summary": ["Added page: Our Work"]}}
- "add services and portfolio pages" -> two modifications for each page
- "use Sanity CMS" -> {{"action": "change", "type": "tech_stack", "item": "cms", "value": "Sanity"}}

Return ONLY valid JSON."""

            response = await self.claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
            )

            response_text = response.content[0].text.strip()

            # Handle markdown code blocks
            if response_text.startswith("```"):
                import re
                response_text = re.sub(r"```(?:json)?\n?", "", response_text)
                response_text = response_text.rstrip("`").strip()

            parsed = json.loads(response_text)

            if not parsed.get("understood", False):
                return []

            # Apply the modifications
            modifications = parsed.get("modifications", [])
            for mod in modifications:
                action = mod.get("action")
                mod_type = mod.get("type")
                item = mod.get("item")
                value = mod.get("value")

                if action == "add":
                    if mod_type == "page" and item not in req.pages:
                        req.pages.append(item)
                    elif mod_type == "feature" and item not in req.features:
                        req.features.append(item)
                    elif mod_type == "integration" and item not in req.integrations:
                        req.integrations.append(item)
                elif action == "remove":
                    if mod_type == "page" and item in req.pages:
                        req.pages.remove(item)
                    elif mod_type == "feature" and item in req.features:
                        req.features.remove(item)
                    elif mod_type == "integration" and item in req.integrations:
                        req.integrations.remove(item)
                elif action == "change" and mod_type == "tech_stack" and item and value:
                    req.tech_stack[item] = value

            return parsed.get("summary", [])

        except Exception as e:
            logger.warning(f"Claude modification extraction failed: {e}")
            return await self._extract_modifications_basic(text)

    async def _extract_modifications_basic(self, text: str) -> list[str]:
        """Basic keyword-based modification extraction (fallback)."""
        modifications = []
        lower = text.lower()
        req = self.state.proposed_requirements

        if not req:
            return []

        # Detect "add page X" patterns
        import re
        add_page_match = re.findall(r"add\s+(?:page[s]?\s+)?([a-zA-Z\s,]+?)(?:\s+page)?(?:$|,|\.|and)", lower)
        for match in add_page_match:
            pages = [p.strip().title() for p in re.split(r"[,\s]+and\s+|,\s*", match) if p.strip()]
            for page in pages:
                if page and page not in req.pages and len(page) > 1:
                    req.pages.append(page)
                    modifications.append(f"Added page: {page}")

        # Detect "add feature X" patterns
        add_feature_match = re.findall(r"add\s+(?:feature[s]?\s+)?([a-zA-Z\s,]+?)(?:\s+feature)?(?:$|,|\.|and)", lower)
        for match in add_feature_match:
            features = [f.strip().title() for f in re.split(r"[,\s]+and\s+|,\s*", match) if f.strip()]
            for feature in features:
                if feature and feature not in req.features and len(feature) > 1:
                    req.features.append(feature)
                    modifications.append(f"Added feature: {feature}")

        # Detect CMS changes
        cms_patterns = ["sanity", "contentful", "strapi", "wordpress", "prismic"]
        for cms in cms_patterns:
            if cms in lower and ("use" in lower or "cms" in lower or "add" in lower):
                cms_name = cms.title()
                req.tech_stack["cms"] = cms_name
                modifications.append(f"Set CMS to: {cms_name}")
                break

        return modifications

    async def _extract_modifications(self, text: str) -> list[str]:
        """Extract modifications - delegates to Claude-powered version."""
        return await self._extract_modifications_with_claude(text)

    def _apply_modifications(self, modifications: list[str]) -> None:
        """Apply modifications to requirements."""
        # Already applied in _extract_modifications
        pass

    def _requirements_to_text(self) -> str:
        """Convert requirements object to text for Kappa."""
        req = self.state.proposed_requirements
        if not req:
            return self._info_to_requirements_text()

        pages_text = "\n".join(f"- {p}" for p in req.pages) if req.pages else "- Home"
        features_text = (
            "\n".join(f"- {f}" for f in req.features) if req.features else "- Basic functionality"
        )
        integrations_text = (
            "\n".join(f"- {i}" for i in req.integrations) if req.integrations else "- None"
        )

        return f"""
# {req.name}

## Description
{req.description}

## Technical Stack
- Framework: {req.tech_stack.get('framework', 'Next.js 14')}
- Language: {req.tech_stack.get('language', 'TypeScript')}
- Styling: {req.tech_stack.get('styling', 'Tailwind CSS')}
- CMS: {req.tech_stack.get('cms', 'None')}

## Pages
{pages_text}

## Features
{features_text}

## Integrations
{integrations_text}
"""

    def _get_progress_summary(self) -> str:
        """Get current progress summary."""
        # In production, query from database using project_id
        return """**Current Progress:**

Building... Please wait.

The dashboard shows real-time updates."""

    def _get_issues_summary(self) -> str:
        """Get issues summary."""
        return """**Issues Status:**

No critical issues detected.

All systems running smoothly."""

    def _get_tasks_summary(self) -> str:
        """Get tasks summary."""
        return """**Task Summary:**

Tasks are being executed in parallel.
Check the dashboard for detailed progress."""

    async def _deploy_project(self) -> str:
        """Deploy project to Vercel."""
        workspace = (
            self.state.build_result.get("workspace_path", "/tmp/kappa-workspace")
            if self.state.build_result
            else "/tmp/kappa-workspace"
        )

        return f"""**Deployment Instructions**

To deploy your project to Vercel:

1. Navigate to the project directory:
   cd {workspace}

2. Install Vercel CLI if not installed:
   npm i -g vercel

3. Deploy:
   vercel --prod

Alternatively, connect your GitHub repository to Vercel for automatic deployments.

Would you like to start a new project?"""

    def get_conversation_history(self) -> list[dict[str, Any]]:
        """Get conversation history as list of dicts."""
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "metadata": msg.metadata,
            }
            for msg in self.state.messages
        ]

    def reset(self) -> None:
        """Reset the conversation state."""
        self.state = ConversationState()
        self._kappa = None


# =============================================================================
# CLI INTEGRATION
# =============================================================================


def _read_file_content(file_path: str, max_lines: int = 500) -> str:
    """Read file content with line limit."""
    from pathlib import Path

    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        return f"[File not found: {file_path}]"
    if not path.is_file():
        return f"[Not a file: {file_path}]"

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()
        if len(lines) > max_lines:
            content = "\n".join(lines[:max_lines])
            content += f"\n... [truncated, {len(lines) - max_lines} more lines]"
        return content
    except Exception as e:
        return f"[Error reading file: {e}]"


def _list_directory(dir_path: str, max_depth: int = 2) -> str:
    """List directory contents with depth limit."""
    from pathlib import Path

    path = Path(dir_path).expanduser().resolve()
    if not path.exists():
        return f"[Directory not found: {dir_path}]"
    if not path.is_dir():
        return f"[Not a directory: {dir_path}]"

    try:
        result = []
        result.append(f"Directory: {path}")
        result.append("=" * 50)

        def walk_dir(p: Path, prefix: str = "", depth: int = 0) -> None:
            if depth > max_depth:
                return
            try:
                items = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
                for i, item in enumerate(items[:50]):  # Limit items per directory
                    is_last = i == len(items) - 1 or i == 49
                    connector = "└── " if is_last else "├── "
                    if item.is_dir():
                        result.append(f"{prefix}{connector}{item.name}/")
                        if depth < max_depth:
                            extension = "    " if is_last else "│   "
                            walk_dir(item, prefix + extension, depth + 1)
                    else:
                        size = item.stat().st_size
                        size_str = f"{size:,} bytes" if size < 1024 else f"{size/1024:.1f} KB"
                        result.append(f"{prefix}{connector}{item.name} ({size_str})")
                if len(items) > 50:
                    result.append(f"{prefix}... and {len(items) - 50} more items")
            except PermissionError:
                result.append(f"{prefix}[Permission denied]")

        walk_dir(path)
        return "\n".join(result)
    except Exception as e:
        return f"[Error listing directory: {e}]"


def _extract_file_references(text: str) -> list[tuple[str, str]]:
    """Extract file and folder references from text.

    Detects patterns like:
    - @file:path/to/file.py
    - @folder:path/to/dir
    - @./relative/path
    - @/absolute/path
    - @~/home/path
    """
    import re
    from pathlib import Path

    references = []

    # Pattern for explicit @file: and @folder: references
    file_pattern = r'@file:([^\s]+)'
    folder_pattern = r'@folder:([^\s]+)'

    # Pattern for path references starting with @
    path_pattern = r'@([./~][^\s]*)'

    # Find @file: references
    for match in re.finditer(file_pattern, text):
        path = match.group(1)
        references.append(("file", path))

    # Find @folder: references
    for match in re.finditer(folder_pattern, text):
        path = match.group(1)
        references.append(("folder", path))

    # Find @path references
    for match in re.finditer(path_pattern, text):
        path = match.group(1)
        # Skip if already matched by file: or folder: pattern
        full_match = match.group(0)
        if full_match.startswith("@file:") or full_match.startswith("@folder:"):
            continue

        expanded_path = Path(path).expanduser()
        if expanded_path.exists():
            if expanded_path.is_dir():
                references.append(("folder", path))
            else:
                references.append(("file", path))
        else:
            # Try relative to current directory
            cwd_path = Path.cwd() / path
            if cwd_path.exists():
                if cwd_path.is_dir():
                    references.append(("folder", str(cwd_path)))
                else:
                    references.append(("file", str(cwd_path)))

    return references


def _process_file_references(text: str) -> str:
    """Process text and inject file/folder contents."""
    references = _extract_file_references(text)

    if not references:
        return text

    # Build context from references
    context_parts = []
    for ref_type, ref_path in references:
        if ref_type == "file":
            content = _read_file_content(ref_path)
            context_parts.append(f"\n--- File: {ref_path} ---\n```\n{content}\n```\n")
        elif ref_type == "folder":
            content = _list_directory(ref_path)
            context_parts.append(f"\n--- Directory: {ref_path} ---\n```\n{content}\n```\n")

    if context_parts:
        # Clean the original text of reference markers for cleaner processing
        import re
        cleaned_text = re.sub(r'@file:[^\s]+', '', text)
        cleaned_text = re.sub(r'@folder:[^\s]+', '', cleaned_text)
        cleaned_text = re.sub(r'@([./~][^\s]*)', '', cleaned_text)
        cleaned_text = cleaned_text.strip()

        # Combine original intent with file context
        context_section = "\n".join(context_parts)
        return f"{cleaned_text}\n\n**Referenced Context:**{context_section}"

    return text


def _has_pending_input(timeout: float = 0.05) -> bool:
    """Check if there's pending input (indicates paste operation)."""
    import select
    import sys

    try:
        # Use select to check if stdin has data available
        readable, _, _ = select.select([sys.stdin], [], [], timeout)
        return bool(readable)
    except (ValueError, OSError):
        # select doesn't work on Windows or some edge cases
        return False


def _collect_pasted_content(first_line: str, timeout: float = 0.1) -> str:
    """Collect all pasted content by detecting rapid consecutive inputs."""
    import sys

    lines = [first_line]

    # Keep reading while there's pending input (paste detection)
    while _has_pending_input(timeout):
        try:
            line = sys.stdin.readline()
            if line:
                lines.append(line.rstrip('\n'))
            else:
                break
        except Exception:
            break

    return "\n".join(lines)


def _get_multiline_input(console: Console) -> str:
    """Get multi-line input from user with paste support.

    Supports:
    - Single line input: Type and press Enter
    - Multi-line paste: Just paste - auto-detected and collected
    - Explicit multi-line: Start with ``` and end with ```
    - Two empty lines to submit in explicit mode

    Paste detection works by checking if more input arrives immediately
    after the first line (within 100ms), indicating a paste operation.
    """
    import sys

    console.print("[bold green]You[/bold green] [dim](paste multi-line or use ``` mode)[/dim]")
    console.print("[bold green]>[/bold green] ", end="")
    sys.stdout.flush()

    try:
        first_line = input()
    except EOFError:
        return ""

    # Check for explicit multi-line mode with ```
    if first_line.strip() == "```" or first_line.strip().startswith("```"):
        console.print("[dim]Multi-line mode. End with ``` or two empty lines.[/dim]")
        lines = []
        if first_line.strip() != "```":
            # Has content after ```
            lines.append(first_line[first_line.find("```") + 3:])

        empty_count = 0
        while True:
            try:
                console.print("[bold green].[/bold green] ", end="")
                sys.stdout.flush()
                line = input()

                # End on closing ```
                if line.strip() == "```":
                    break

                # End on two consecutive empty lines
                if not line.strip():
                    empty_count += 1
                    if empty_count >= 2:
                        # Remove the trailing empty line we added
                        if lines and not lines[-1].strip():
                            lines.pop()
                        break
                else:
                    empty_count = 0

                lines.append(line)
            except EOFError:
                break

        return "\n".join(lines)

    # Check for pasted content (auto-detect multi-line paste)
    if _has_pending_input(0.1):
        # More input is immediately available - this is a paste
        console.print("[dim]Detecting pasted content...[/dim]")
        full_content = _collect_pasted_content(first_line, 0.15)
        line_count = full_content.count('\n') + 1
        console.print(f"[dim]Collected {line_count} lines[/dim]")
        return full_content

    # Single line mode - process escape sequences
    result = first_line.replace("\\n", "\n")
    return result


async def start_chat_cli() -> None:
    """Start interactive chat in terminal using Rich.

    Features:
    - Multi-line input: Start with ``` for multi-line mode, end with ``` or Ctrl+D
    - File references: Use @file:path/to/file or @./relative/path
    - Folder references: Use @folder:path/to/dir or @./dir/
    - Escape sequences: Use \\n for newlines in single-line mode

    Examples:
        You> Build a REST API @file:./requirements.txt
        You> ```
        . Build a CLI tool that:
        . - Reads config from @./config.yaml
        . - Outputs to ./dist/
        . ```
    """
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel

    console = Console()
    chat = KappaChat()

    # Print welcome banner
    console.print()
    console.print(Panel.fit(
        "[bold cyan]KAPPA OS[/bold cyan]\n"
        "[dim]Autonomous Development Operating System[/dim]",
        border_style="cyan"
    ))
    console.print()
    console.print("Tell me what you'd like to build!\n")
    console.print("[dim]Input modes:[/dim]")
    console.print("[dim]  • Single line: Type and press Enter[/dim]")
    console.print("[dim]  • Paste: Just paste multi-line text (auto-detected)[/dim]")
    console.print("[dim]  • Manual multi-line: Start with ``` and end with ``` or two empty lines[/dim]")
    console.print("[dim]  • File refs: @./path or @file:./path[/dim]")
    console.print("[dim]  • Folder refs: @./dir/ or @folder:./dir[/dim]")
    console.print("[dim]  • Exit: type 'exit' or 'quit'[/dim]")
    console.print()

    while True:
        try:
            user_input = _get_multiline_input(console)

            if not user_input or not user_input.strip():
                continue

            if user_input.lower().strip() in ["exit", "quit", "bye", "q"]:
                console.print("\n[dim]Goodbye! Happy building![/dim]\n")
                break

            # Process file/folder references
            processed_input = _process_file_references(user_input)

            # Show what references were found (if any)
            refs = _extract_file_references(user_input)
            if refs:
                console.print(f"[dim]📎 Attached {len(refs)} reference(s)[/dim]")

            response = await chat.process_message(processed_input)
            console.print()
            console.print("[bold cyan]Kappa[/bold cyan]")
            console.print(Markdown(response))
            console.print()

        except KeyboardInterrupt:
            console.print("\n\n[dim]Session interrupted. Goodbye![/dim]\n")
            break
        except Exception as e:
            logger.error(f"Chat error: {e}")
            console.print(f"\n[red]Error: {e}[/red]\n")
