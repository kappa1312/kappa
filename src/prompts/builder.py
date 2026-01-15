"""Dynamic prompt builder for Kappa operations."""

from typing import Any

from loguru import logger

from src.prompts.templates import (
    DECOMPOSITION_PROMPT,
    SYSTEM_PROMPT,
    TASK_PROMPT,
    PromptTemplate,
)


class PromptBuilder:
    """
    Build dynamic prompts from templates and context.

    Combines templates with runtime context to create
    effective prompts for Claude sessions.

    Example:
        >>> builder = PromptBuilder()
        >>> prompt = builder.build_task_prompt(
        ...     task_name="Create user model",
        ...     task_description="Define User SQLAlchemy model",
        ...     file_targets=["src/models/user.py"],
        ... )
    """

    def __init__(self) -> None:
        """Initialize the prompt builder."""
        self._templates: dict[str, PromptTemplate] = {
            "system": SYSTEM_PROMPT,
            "task": TASK_PROMPT,
            "decomposition": DECOMPOSITION_PROMPT,
        }

    def build_system_prompt(
        self,
        project_name: str,
        context: str = "",
    ) -> str:
        """
        Build a system prompt for a session.

        Args:
            project_name: Name of the project.
            context: Additional context to include.

        Returns:
            Formatted system prompt.
        """
        template = self._templates["system"]
        return template.format(
            project_name=project_name,
            context=context or "No additional context provided.",
        )

    def build_task_prompt(
        self,
        task_name: str,
        task_description: str,
        file_targets: list[str] | None = None,
        dependencies: list[str] | None = None,
        requirements: str = "",
    ) -> str:
        """
        Build a task execution prompt.

        Args:
            task_name: Name of the task.
            task_description: Detailed description.
            file_targets: List of target files.
            dependencies: List of dependent tasks.
            requirements: Additional requirements.

        Returns:
            Formatted task prompt.
        """
        template = self._templates["task"]

        # Format file targets
        if file_targets:
            targets_str = "\n".join(f"- {f}" for f in file_targets)
        else:
            targets_str = "- No specific files targeted"

        # Format dependencies
        if dependencies:
            deps_str = ", ".join(dependencies)
        else:
            deps_str = "None"

        return template.format(
            task_name=task_name,
            task_description=task_description,
            file_targets=targets_str,
            dependencies=deps_str,
            requirements=requirements or "Follow standard best practices.",
        )

    def build_decomposition_prompt(
        self,
        specification: str,
        constraints: str = "",
    ) -> str:
        """
        Build a decomposition prompt.

        Args:
            specification: Project specification to decompose.
            constraints: Any constraints to consider.

        Returns:
            Formatted decomposition prompt.
        """
        template = self._templates["decomposition"]
        return template.format(
            specification=specification,
            constraints=constraints or "No specific constraints.",
        )

    def build_custom_prompt(
        self,
        template_name: str,
        **kwargs: Any,
    ) -> str:
        """
        Build a prompt from a named template.

        Args:
            template_name: Name of the template.
            **kwargs: Template variables.

        Returns:
            Formatted prompt.

        Raises:
            KeyError: If template not found.
        """
        if template_name not in self._templates:
            raise KeyError(f"Template '{template_name}' not found")

        template = self._templates[template_name]
        return template.format(**kwargs)

    def register_template(self, template: PromptTemplate) -> None:
        """
        Register a custom template.

        Args:
            template: PromptTemplate to register.
        """
        self._templates[template.name] = template
        logger.debug(f"Registered template: {template.name}")

    def list_templates(self) -> list[str]:
        """
        List available template names.

        Returns:
            List of template names.
        """
        return list(self._templates.keys())

    def build_context_aware_prompt(
        self,
        base_prompt: str,
        decisions: list[dict[str, Any]] | None = None,
        discoveries: list[str] | None = None,
        file_contents: dict[str, str] | None = None,
    ) -> str:
        """
        Enhance a prompt with project context.

        Args:
            base_prompt: The base prompt to enhance.
            decisions: List of project decisions.
            discoveries: List of discoveries.
            file_contents: Dict of file path -> content.

        Returns:
            Context-enhanced prompt.
        """
        sections = [base_prompt]

        if decisions:
            sections.append("\n## Relevant Decisions")
            for dec in decisions[:5]:  # Limit to 5
                category = dec.get("category", "general")
                decision = dec.get("decision", "")
                sections.append(f"- **{category}**: {decision}")

        if discoveries:
            sections.append("\n## Project Discoveries")
            for disc in discoveries[:5]:
                sections.append(f"- {disc}")

        if file_contents:
            sections.append("\n## Relevant Files")
            for path, content in list(file_contents.items())[:3]:
                # Truncate long files
                truncated = content[:1000] + "..." if len(content) > 1000 else content
                sections.append(f"\n### {path}\n```\n{truncated}\n```")

        return "\n".join(sections)


# Singleton instance
_builder: PromptBuilder | None = None


def get_prompt_builder() -> PromptBuilder:
    """Get the shared PromptBuilder instance.

    Returns:
        PromptBuilder singleton.
    """
    global _builder
    if _builder is None:
        _builder = PromptBuilder()
    return _builder
