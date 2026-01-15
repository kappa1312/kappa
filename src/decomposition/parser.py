"""Requirements parser - extracts structured requirements from natural language.

This module provides two parsing approaches:
1. RequirementsParser: Uses Claude API for intelligent parsing
2. SpecificationParser: Uses keyword-based heuristics (fallback)
"""

import json
import re
from typing import Any

from loguru import logger

from src.decomposition.models import (
    ParsedRequirement,
    ProjectRequirements,
    ProjectType,
    TaskCategory,
)

# =============================================================================
# REQUIREMENTS PARSER (Claude API)
# =============================================================================


class RequirementsParser:
    """
    Parse natural language requirements into structured ProjectRequirements.

    Uses Claude API for intelligent parsing with fallback to keyword-based
    parsing if API is unavailable.

    Example:
        >>> parser = RequirementsParser()
        >>> reqs = await parser.parse(
        ...     "Build a REST API with user authentication using Express.js"
        ... )
        >>> reqs.project_type
        ProjectType.API
    """

    # Prompt template for Claude API
    PARSE_PROMPT = """Analyze this project specification and extract structured requirements.

SPECIFICATION:
{specification}

Return a JSON object with these fields:
{{
    "name": "project name (infer from spec or generate a reasonable name)",
    "description": "brief description of the project",
    "project_type": "one of: website, api, dashboard, cli_tool, library, mobile_app",
    "tech_stack": {{
        "framework": "main framework (e.g., express, fastapi, react)",
        "language": "programming language (e.g., typescript, python)",
        "database": "database if mentioned (e.g., postgresql, mongodb)",
        "additional": "any other technologies mentioned"
    }},
    "features": ["list of features to implement"],
    "pages": ["list of pages/routes for web projects"],
    "integrations": ["list of third-party integrations (Stripe, OAuth, etc)"],
    "constraints": ["any constraints or requirements mentioned"],
    "timeline": "timeline if mentioned, null otherwise",
    "priority": "normal, high, or low"
}}

Be thorough in extracting features. If a feature is mentioned implicitly, include it.
Return ONLY valid JSON, no additional text."""

    def __init__(self, use_api: bool = True) -> None:
        """Initialize the requirements parser.

        Args:
            use_api: Whether to use Claude API. If False, uses keyword parsing.
        """
        self.use_api = use_api
        self._client: Any = None
        self._fallback_parser = SpecificationParser()

    async def parse(self, requirements_text: str) -> ProjectRequirements:
        """
        Parse natural language into structured ProjectRequirements.

        Args:
            requirements_text: Natural language project specification.

        Returns:
            ProjectRequirements with structured data.

        Example:
            >>> parser = RequirementsParser()
            >>> reqs = await parser.parse(
            ...     "Build a blog with posts, comments, and user auth"
            ... )
            >>> reqs.features
            ['posts', 'comments', 'user authentication']
        """
        logger.info("Parsing requirements from specification")
        logger.debug(f"Specification: {requirements_text[:200]}...")

        if self.use_api:
            try:
                return await self._parse_with_api(requirements_text)
            except Exception as e:
                logger.warning(f"Claude API parsing failed: {e}, using fallback")

        # Fallback to keyword-based parsing
        return self._parse_with_keywords(requirements_text)

    async def _parse_with_api(self, text: str) -> ProjectRequirements:
        """Parse using Claude API.

        Args:
            text: Specification text.

        Returns:
            Parsed ProjectRequirements.
        """
        from anthropic import AsyncAnthropic

        if self._client is None:
            from src.core.config import get_settings

            settings = get_settings()
            self._client = AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())

        prompt = self.PARSE_PROMPT.format(specification=text)

        logger.debug("Calling Claude API for requirements parsing")

        response = await self._client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract JSON from response
        response_text = response.content[0].text.strip()

        # Handle potential markdown code blocks
        if response_text.startswith("```"):
            response_text = re.sub(r"```(?:json)?\n?", "", response_text)
            response_text = response_text.rstrip("`").strip()

        parsed = json.loads(response_text)

        # Convert to ProjectRequirements
        return self._dict_to_requirements(parsed, text)

    def _parse_with_keywords(self, text: str) -> ProjectRequirements:
        """Parse using keyword-based heuristics.

        Args:
            text: Specification text.

        Returns:
            Parsed ProjectRequirements.
        """
        logger.debug("Using keyword-based parsing")

        # Determine project type
        project_type = self._determine_project_type(text)

        # Extract tech stack
        tech_stack = self._extract_tech_stack(text)

        # Extract features
        features = self._extract_features(text)

        # Extract pages (for web projects)
        pages = self._extract_pages(text)

        # Extract integrations
        integrations = self._extract_integrations(text)

        # Extract constraints
        constraints = self._extract_constraints(text)

        # Generate project name
        name = self._generate_project_name(text)

        return ProjectRequirements(
            name=name,
            description=text[:500] if len(text) > 500 else text,
            project_type=project_type,
            tech_stack=tech_stack,
            features=features,
            pages=pages,
            integrations=integrations,
            constraints=constraints,
        )

    def _dict_to_requirements(
        self,
        data: dict[str, Any],
        original_text: str,
    ) -> ProjectRequirements:
        """Convert parsed dict to ProjectRequirements.

        Args:
            data: Parsed data from Claude API.
            original_text: Original specification text.

        Returns:
            ProjectRequirements object.
        """
        # Map project type string to enum
        project_type_str = data.get("project_type", "api").lower()
        project_type_map = {
            "website": ProjectType.WEBSITE,
            "api": ProjectType.API,
            "dashboard": ProjectType.DASHBOARD,
            "cli_tool": ProjectType.CLI_TOOL,
            "cli": ProjectType.CLI_TOOL,
            "library": ProjectType.LIBRARY,
            "mobile_app": ProjectType.MOBILE_APP,
            "mobile": ProjectType.MOBILE_APP,
        }
        project_type = project_type_map.get(project_type_str, ProjectType.API)

        # Extract tech stack
        tech_stack_raw = data.get("tech_stack", {})
        tech_stack = {
            k: v for k, v in tech_stack_raw.items() if v and v != "null" and k != "additional"
        }
        if tech_stack_raw.get("additional"):
            tech_stack["additional"] = tech_stack_raw["additional"]

        return ProjectRequirements(
            name=data.get("name", self._generate_project_name(original_text)),
            description=data.get("description", original_text[:500]),
            project_type=project_type,
            tech_stack=tech_stack,
            features=data.get("features", []),
            pages=data.get("pages", []),
            integrations=data.get("integrations", []),
            constraints=data.get("constraints", []),
            timeline=data.get("timeline"),
            priority=data.get("priority", "normal"),
        )

    def _determine_project_type(self, text: str) -> ProjectType:
        """Determine project type from text.

        Args:
            text: Specification text.

        Returns:
            Inferred ProjectType.
        """
        text_lower = text.lower()

        # Check for specific keywords
        if any(kw in text_lower for kw in ["website", "web app", "frontend", "landing page"]):
            return ProjectType.WEBSITE

        if any(kw in text_lower for kw in ["dashboard", "admin panel", "analytics"]):
            return ProjectType.DASHBOARD

        if any(kw in text_lower for kw in ["cli", "command line", "terminal tool", "script"]):
            return ProjectType.CLI_TOOL

        if any(kw in text_lower for kw in ["library", "package", "module", "sdk"]):
            return ProjectType.LIBRARY

        if any(kw in text_lower for kw in ["mobile", "ios", "android", "react native", "flutter"]):
            return ProjectType.MOBILE_APP

        if any(kw in text_lower for kw in ["api", "rest", "graphql", "endpoint", "backend"]):
            return ProjectType.API

        # Default to API
        return ProjectType.API

    def _extract_tech_stack(self, text: str) -> dict[str, str]:
        """Extract technology stack from text.

        Args:
            text: Specification text.

        Returns:
            Dict of technology choices.
        """
        text_lower = text.lower()
        tech_stack: dict[str, str] = {}

        # Frameworks
        framework_map = {
            "express": "express",
            "fastapi": "fastapi",
            "django": "django",
            "flask": "flask",
            "nest": "nestjs",
            "next": "nextjs",
            "react": "react",
            "vue": "vue",
            "angular": "angular",
            "svelte": "svelte",
            "rails": "rails",
            "spring": "spring",
        }
        for keyword, framework in framework_map.items():
            if keyword in text_lower:
                tech_stack["framework"] = framework
                break

        # Languages
        language_map = {
            "typescript": "typescript",
            "javascript": "javascript",
            "python": "python",
            "ruby": "ruby",
            "java": "java",
            "go": "go",
            "rust": "rust",
            "c#": "csharp",
        }
        for keyword, language in language_map.items():
            if keyword in text_lower:
                tech_stack["language"] = language
                break

        # Databases
        db_map = {
            "postgresql": "postgresql",
            "postgres": "postgresql",
            "mysql": "mysql",
            "mongodb": "mongodb",
            "mongo": "mongodb",
            "redis": "redis",
            "sqlite": "sqlite",
            "dynamodb": "dynamodb",
        }
        for keyword, db in db_map.items():
            if keyword in text_lower:
                tech_stack["database"] = db
                break

        return tech_stack

    def _extract_features(self, text: str) -> list[str]:
        """Extract feature list from text.

        Args:
            text: Specification text.

        Returns:
            List of features.
        """
        features: list[str] = []

        # Common feature patterns
        feature_patterns = [
            r"(?:with|include|has|support)\s+([^,.]+)",
            r"(?:feature|functionality):\s*([^,.]+)",
            r"-\s*([^,.\n]+)",  # Bullet points
        ]

        for pattern in feature_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                feature = match.strip()
                if len(feature) > 3 and feature not in features:
                    features.append(feature)

        # Extract from parsed requirements
        parsed = self._fallback_parser.parse(text)
        for req in parsed:
            if req.text not in features and len(req.text) > 5:
                features.append(req.text)

        return features[:20]  # Limit to 20 features

    def _extract_pages(self, text: str) -> list[str]:
        """Extract pages/routes from text.

        Args:
            text: Specification text.

        Returns:
            List of page names.
        """
        pages: list[str] = []
        text_lower = text.lower()

        # Common page patterns
        page_patterns = [
            r"(?:page|screen|view|route)\s*:\s*([^,.]+)",
            r"(?:home|about|contact|login|register|dashboard|profile|settings)\s*page",
        ]

        for pattern in page_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                page = match.strip()
                if len(page) > 2 and page not in pages:
                    pages.append(page)

        # Check for common pages mentioned
        common_pages = [
            "home",
            "about",
            "contact",
            "login",
            "register",
            "signup",
            "dashboard",
            "profile",
            "settings",
            "admin",
            "landing",
        ]
        for page in common_pages:
            if page in text_lower and page not in pages:
                pages.append(page)

        return pages

    def _extract_integrations(self, text: str) -> list[str]:
        """Extract third-party integrations from text.

        Args:
            text: Specification text.

        Returns:
            List of integrations.
        """
        integrations: list[str] = []
        text_lower = text.lower()

        # Common integrations
        integration_keywords = [
            "stripe",
            "paypal",
            "oauth",
            "google auth",
            "github auth",
            "sendgrid",
            "twilio",
            "aws",
            "firebase",
            "supabase",
            "cloudflare",
            "vercel",
            "netlify",
            "heroku",
            "slack",
            "discord",
            "telegram",
            "whatsapp",
            "google analytics",
            "mixpanel",
            "segment",
            "openai",
            "anthropic",
            "claude",
        ]

        for integration in integration_keywords:
            if integration in text_lower:
                integrations.append(integration)

        return integrations

    def _extract_constraints(self, text: str) -> list[str]:
        """Extract constraints from text.

        Args:
            text: Specification text.

        Returns:
            List of constraints.
        """
        constraints: list[str] = []

        # Constraint patterns
        constraint_patterns = [
            r"(?:must|should|need to|require)\s+([^,.]+)",
            r"(?:constraint|requirement|limit):\s*([^,.]+)",
            r"(?:no|without|avoid)\s+([^,.]+)",
        ]

        for pattern in constraint_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                constraint = match.strip()
                if len(constraint) > 5 and constraint not in constraints:
                    constraints.append(constraint)

        return constraints[:10]  # Limit to 10 constraints

    def _generate_project_name(self, text: str) -> str:
        """Generate a project name from specification.

        Args:
            text: Specification text.

        Returns:
            Generated project name.
        """
        # Try to extract explicit name
        name_match = re.search(
            r"(?:called|named|create|build)\s+(?:a\s+)?[\"']?(\w+(?:-\w+)?)[\"']?",
            text,
            re.IGNORECASE,
        )
        if name_match:
            return name_match.group(1).lower().replace(" ", "-")

        # Extract key nouns
        words = re.findall(r"\b[a-z]{3,}\b", text.lower())
        common_words = {
            "the",
            "and",
            "with",
            "for",
            "that",
            "this",
            "from",
            "will",
            "have",
            "has",
            "can",
            "are",
            "was",
            "were",
            "been",
            "being",
            "build",
            "create",
            "make",
            "use",
            "using",
            "need",
            "want",
        }
        meaningful = [w for w in words if w not in common_words][:3]

        if meaningful:
            return "-".join(meaningful)

        return "kappa-project"


# =============================================================================
# SPECIFICATION PARSER (Keyword-based - Legacy/Fallback)
# =============================================================================


class SpecificationParser:
    """
    Parse natural language specifications into structured requirements.

    Uses pattern matching and heuristics to extract actionable
    requirements from free-form text. This is the fallback parser
    when Claude API is unavailable.

    Example:
        >>> parser = SpecificationParser()
        >>> reqs = parser.parse("Build a REST API with user auth and product catalog")
        >>> len(reqs)
        3
    """

    # Keywords that indicate different task categories
    CATEGORY_KEYWORDS: dict[TaskCategory, list[str]] = {
        TaskCategory.SETUP: [
            "setup",
            "install",
            "configure",
            "initialize",
            "bootstrap",
            "create project",
            "scaffold",
        ],
        TaskCategory.INFRASTRUCTURE: [
            "database",
            "postgresql",
            "redis",
            "docker",
            "kubernetes",
            "infrastructure",
            "deployment",
            "ci/cd",
            "pipeline",
        ],
        TaskCategory.DATA_MODEL: [
            "model",
            "schema",
            "entity",
            "table",
            "migration",
            "orm",
            "sqlalchemy",
            "pydantic",
        ],
        TaskCategory.BUSINESS_LOGIC: [
            "service",
            "logic",
            "process",
            "calculate",
            "validate",
            "transform",
            "business",
            "domain",
        ],
        TaskCategory.API: [
            "api",
            "endpoint",
            "rest",
            "graphql",
            "route",
            "controller",
            "request",
            "response",
            "http",
        ],
        TaskCategory.UI: [
            "ui",
            "frontend",
            "component",
            "page",
            "view",
            "template",
            "form",
            "button",
            "interface",
            "cli",
        ],
        TaskCategory.TESTING: [
            "test",
            "spec",
            "pytest",
            "unittest",
            "coverage",
            "integration test",
            "e2e",
            "mock",
        ],
        TaskCategory.DOCUMENTATION: [
            "documentation",
            "readme",
            "docstring",
            "api docs",
            "swagger",
            "openapi",
        ],
        TaskCategory.DEPLOYMENT: [
            "deploy",
            "release",
            "publish",
            "production",
            "staging",
            "environment",
        ],
        TaskCategory.INTEGRATION: [
            "integrate",
            "integration",
            "third-party",
            "external",
            "webhook",
            "oauth",
            "stripe",
            "payment",
        ],
        TaskCategory.TYPES: [
            "type",
            "interface",
            "typescript",
            "typing",
            "schema",
        ],
    }

    # Patterns for extracting entities
    ENTITY_PATTERNS = [
        r"(?:create|add|implement|build)\s+(?:a\s+)?(\w+(?:\s+\w+)?)\s+(?:model|entity|table)",
        r"(\w+)\s+(?:crud|api|endpoint|service)",
        r"(?:user|product|order|item|customer|account)\s*(?:s)?",
        r"(?:authentication|authorization|auth)\s*(?:entication)?",
    ]

    def __init__(self) -> None:
        """Initialize the specification parser."""
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.ENTITY_PATTERNS]

    def parse(self, specification: str) -> list[ParsedRequirement]:
        """
        Parse a specification into structured requirements.

        Args:
            specification: Natural language project specification.

        Returns:
            List of ParsedRequirement objects.

        Example:
            >>> parser = SpecificationParser()
            >>> reqs = parser.parse(
            ...     "Build an e-commerce API with user auth, "
            ...     "product catalog, and shopping cart"
            ... )
            >>> [r.category for r in reqs]
            [TaskCategory.API, TaskCategory.BUSINESS_LOGIC, ...]
        """
        logger.debug(f"Parsing specification: {specification[:100]}...")

        requirements: list[ParsedRequirement] = []

        # Split into sentences/clauses
        clauses = self._split_into_clauses(specification)

        for clause in clauses:
            if not clause.strip():
                continue

            # Determine category
            category = self._infer_category(clause)

            # Extract entities
            entities = self._extract_entities(clause)

            # Determine priority
            priority = self._infer_priority(clause)

            req = ParsedRequirement(
                text=clause.strip(),
                category=category,
                entities=entities,
                priority=priority,
            )
            requirements.append(req)

        logger.info(f"Parsed {len(requirements)} requirements from specification")
        return requirements

    def _split_into_clauses(self, text: str) -> list[str]:
        """Split text into logical clauses.

        Args:
            text: Input text.

        Returns:
            List of clause strings.
        """
        # Split on common delimiters
        delimiters = r"[.,;]\s+|\s+and\s+|\s+with\s+|\s+-\s+"
        clauses = re.split(delimiters, text, flags=re.IGNORECASE)

        # Also split on newlines for list-style specs
        expanded = []
        for clause in clauses:
            expanded.extend(clause.split("\n"))

        return [c.strip() for c in expanded if c.strip()]

    def _infer_category(self, clause: str) -> TaskCategory:
        """Infer task category from clause text.

        Args:
            clause: Text clause to analyze.

        Returns:
            Most likely TaskCategory.
        """
        clause_lower = clause.lower()

        # Score each category based on keyword matches
        scores: dict[TaskCategory, int] = {}

        for category, keywords in self.CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in clause_lower)
            if score > 0:
                scores[category] = score

        if scores:
            return max(scores, key=scores.get)  # type: ignore

        # Default to business logic
        return TaskCategory.BUSINESS_LOGIC

    def _extract_entities(self, clause: str) -> list[str]:
        """Extract named entities from clause.

        Args:
            clause: Text clause to analyze.

        Returns:
            List of extracted entity names.
        """
        entities: set[str] = set()

        for pattern in self._compiled_patterns:
            matches = pattern.findall(clause)
            for match in matches:
                if isinstance(match, tuple):
                    entities.update(m for m in match if m)
                else:
                    entities.add(match)

        # Normalize entities
        normalized = []
        for entity in entities:
            entity = entity.strip().lower()
            if len(entity) > 2 and entity not in ("the", "and", "with"):
                normalized.append(entity)

        return normalized

    def _infer_priority(self, clause: str) -> int:
        """Infer priority level from clause.

        Args:
            clause: Text clause to analyze.

        Returns:
            Priority level (1-5, 1=highest).
        """
        clause_lower = clause.lower()

        # High priority keywords
        if any(kw in clause_lower for kw in ["critical", "must", "required", "essential"]):
            return 1

        # Medium-high priority
        if any(kw in clause_lower for kw in ["important", "should", "need"]):
            return 2

        # Medium priority (default)
        if any(kw in clause_lower for kw in ["want", "would like", "feature"]):
            return 3

        # Lower priority
        if any(kw in clause_lower for kw in ["nice to have", "optional", "could"]):
            return 4

        # Lowest priority
        if any(kw in clause_lower for kw in ["maybe", "future", "later"]):
            return 5

        return 3  # Default to medium


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


async def parse_requirements(spec: str, use_api: bool = True) -> ProjectRequirements:
    """Convenience function to parse requirements.

    Args:
        spec: Natural language specification.
        use_api: Whether to use Claude API.

    Returns:
        ProjectRequirements object.
    """
    parser = RequirementsParser(use_api=use_api)
    return await parser.parse(spec)


def parse_specification(spec: str) -> list[ParsedRequirement]:
    """Convenience function to parse a specification.

    Args:
        spec: Natural language specification.

    Returns:
        List of ParsedRequirement objects.
    """
    parser = SpecificationParser()
    return parser.parse(spec)
