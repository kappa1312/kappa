"""CLAUDE.md generator - creates project-specific development guides."""

from pathlib import Path
from typing import Any

from loguru import logger


class ClaudeMDGenerator:
    """
    Generate CLAUDE.md files for projects.

    Creates comprehensive development guides that help Claude
    understand project context, conventions, and requirements.

    Example:
        >>> generator = ClaudeMDGenerator()
        >>> content = await generator.generate(
        ...     project_name="MyAPI",
        ...     tech_stack=["fastapi", "postgresql"],
        ...     description="REST API for e-commerce"
        ... )
    """

    def __init__(self) -> None:
        """Initialize the generator."""
        self._templates: dict[str, str] = {}

    async def generate(
        self,
        project_name: str,
        description: str,
        tech_stack: list[str] | None = None,
        conventions: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        """
        Generate CLAUDE.md content.

        Args:
            project_name: Name of the project.
            description: Project description.
            tech_stack: List of technologies used.
            conventions: Code conventions and style guide.
            context: Additional context (decisions, discoveries).

        Returns:
            Generated CLAUDE.md content.
        """
        logger.info(f"Generating CLAUDE.md for {project_name}")

        sections = [
            self._generate_header(project_name, description),
            self._generate_tech_stack(tech_stack or []),
            self._generate_conventions(conventions or {}),
            self._generate_structure(),
            self._generate_context(context or {}),
            self._generate_workflow(),
        ]

        return "\n\n".join(filter(None, sections))

    def _generate_header(self, name: str, description: str) -> str:
        """Generate the header section."""
        return f"""# {name} Development Guide

## Overview

{description}

## Purpose

This document provides Claude with the context needed to contribute effectively to this project. It outlines the architecture, conventions, and guidelines that should be followed."""

    def _generate_tech_stack(self, stack: list[str]) -> str:
        """Generate the tech stack section."""
        if not stack:
            return ""

        items = "\n".join(f"- {tech}" for tech in stack)

        return f"""## Technology Stack

{items}"""

    def _generate_conventions(self, conventions: dict[str, Any]) -> str:
        """Generate the code conventions section."""
        lines = ["## Code Conventions"]

        # Default conventions
        defaults = {
            "formatting": "Black (line length 100)",
            "linting": "Ruff",
            "type_checking": "Mypy strict mode",
            "docstrings": "Google style",
            "testing": "Pytest with 80%+ coverage",
        }

        # Merge with provided conventions
        all_conventions = {**defaults, **conventions}

        lines.append("")
        for key, value in all_conventions.items():
            key_title = key.replace("_", " ").title()
            lines.append(f"- **{key_title}**: {value}")

        return "\n".join(lines)

    def _generate_structure(self) -> str:
        """Generate the project structure section."""
        return """## Project Structure

```
src/
├── models/      # Data models and schemas
├── services/    # Business logic layer
├── api/         # API endpoints
├── utils/       # Utility functions
└── tests/       # Test files
```

### Key Principles

1. **Separation of Concerns**: Keep models, services, and API layers distinct
2. **Dependency Injection**: Use DI for testability
3. **Async First**: Use async/await for all I/O operations
4. **Type Safety**: All functions must have type hints"""

    def _generate_context(self, context: dict[str, Any]) -> str:
        """Generate the context section."""
        if not context:
            return ""

        lines = ["## Project Context"]

        decisions = context.get("decisions", [])
        if decisions:
            lines.append("")
            lines.append("### Key Decisions")
            for dec in decisions[:5]:
                if isinstance(dec, dict):
                    lines.append(
                        f"- **{dec.get('category', 'General')}**: {dec.get('decision', '')}"
                    )
                else:
                    lines.append(f"- {dec}")

        discoveries = context.get("discoveries", [])
        if discoveries:
            lines.append("")
            lines.append("### Discoveries")
            for disc in discoveries[:5]:
                lines.append(f"- {disc}")

        return "\n".join(lines)

    def _generate_workflow(self) -> str:
        """Generate the workflow section."""
        return """## Development Workflow

### Making Changes

1. Read existing code before modifying
2. Follow established patterns in the codebase
3. Add appropriate error handling
4. Include type hints for all new code
5. Write or update tests as needed

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific tests
pytest tests/test_specific.py -v
```

### Code Quality

```bash
# Format
black src tests

# Lint
ruff check src tests

# Type check
mypy src
```"""

    async def save(
        self,
        content: str,
        project_path: str | Path,
    ) -> Path:
        """
        Save CLAUDE.md to the project directory.

        Args:
            content: Generated content.
            project_path: Project directory path.

        Returns:
            Path to saved file.
        """
        path = Path(project_path) / "CLAUDE.md"

        path.write_text(content, encoding="utf-8")

        logger.info(f"Saved CLAUDE.md to {path}")
        return path


async def generate_claude_md(
    project_name: str,
    project_path: str,
    description: str = "",
    **kwargs: Any,
) -> str:
    """
    Convenience function to generate CLAUDE.md.

    Args:
        project_name: Name of the project.
        project_path: Path to project directory.
        description: Project description.
        **kwargs: Additional generation options.

    Returns:
        Generated CLAUDE.md content.
    """
    generator = ClaudeMDGenerator()
    content = await generator.generate(
        project_name=project_name,
        description=description or f"{project_name} - A Kappa-generated project",
        **kwargs,
    )

    await generator.save(content, project_path)
    return content
