"""Specification parser - extracts requirements from natural language."""

import re
from typing import Any

from loguru import logger

from src.decomposition.models import ParsedRequirement, TaskCategory


class SpecificationParser:
    """
    Parse natural language specifications into structured requirements.

    Uses pattern matching and heuristics to extract actionable
    requirements from free-form text.

    Example:
        >>> parser = SpecificationParser()
        >>> reqs = parser.parse("Build a REST API with user auth and product catalog")
        >>> len(reqs)
        3
    """

    # Keywords that indicate different task categories
    CATEGORY_KEYWORDS: dict[TaskCategory, list[str]] = {
        TaskCategory.SETUP: [
            "setup", "install", "configure", "initialize", "bootstrap",
            "create project", "scaffold",
        ],
        TaskCategory.INFRASTRUCTURE: [
            "database", "postgresql", "redis", "docker", "kubernetes",
            "infrastructure", "deployment", "ci/cd", "pipeline",
        ],
        TaskCategory.DATA_MODEL: [
            "model", "schema", "entity", "table", "migration",
            "orm", "sqlalchemy", "pydantic",
        ],
        TaskCategory.BUSINESS_LOGIC: [
            "service", "logic", "process", "calculate", "validate",
            "transform", "business", "domain",
        ],
        TaskCategory.API: [
            "api", "endpoint", "rest", "graphql", "route", "controller",
            "request", "response", "http",
        ],
        TaskCategory.UI: [
            "ui", "frontend", "component", "page", "view", "template",
            "form", "button", "interface", "cli",
        ],
        TaskCategory.TESTING: [
            "test", "spec", "pytest", "unittest", "coverage",
            "integration test", "e2e", "mock",
        ],
        TaskCategory.DOCUMENTATION: [
            "documentation", "readme", "docstring", "api docs",
            "swagger", "openapi",
        ],
        TaskCategory.DEPLOYMENT: [
            "deploy", "release", "publish", "production",
            "staging", "environment",
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
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.ENTITY_PATTERNS
        ]

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


def parse_specification(spec: str) -> list[ParsedRequirement]:
    """Convenience function to parse a specification.

    Args:
        spec: Natural language specification.

    Returns:
        List of ParsedRequirement objects.
    """
    parser = SpecificationParser()
    return parser.parse(spec)
