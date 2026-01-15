"""
Dynamic prompt builder for Kappa operations.

This module provides comprehensive prompt building capabilities for
parallel task execution, including context sharing between sessions,
project-aware prompts, and validation prompts.
"""

from typing import Any
from pathlib import Path

from loguru import logger

from src.prompts.templates import (
    DECOMPOSITION_PROMPT,
    SYSTEM_PROMPT,
    TASK_PROMPT,
    PARALLEL_TASK_PROMPT,
    VALIDATION_PROMPT,
    CONTEXT_INJECTION_PROMPT,
    PromptTemplate,
)
from src.decomposition.models import (
    TaskSpec,
    ProjectRequirements,
    ProjectType,
    TaskCategory,
)


# =============================================================================
# PROMPT CONTEXT
# =============================================================================


class PromptContext:
    """
    Container for context information used in prompt building.

    Collects all contextual information needed to build effective prompts
    for parallel task execution.

    Example:
        >>> ctx = PromptContext(
        ...     project_name="my-api",
        ...     workspace="/path/to/project",
        ...     tech_stack={"framework": "express", "language": "typescript"},
        ... )
        >>> ctx.add_type_definition("User", "interface User { id: string; name: string; }")
    """

    def __init__(
        self,
        project_name: str = "",
        workspace: str = ".",
        tech_stack: dict[str, str] | None = None,
        project_type: ProjectType | None = None,
    ) -> None:
        """
        Initialize prompt context.

        Args:
            project_name: Name of the project.
            workspace: Working directory path.
            tech_stack: Technology stack dict.
            project_type: Type of project.
        """
        self.project_name = project_name
        self.workspace = workspace
        self.tech_stack = tech_stack or {}
        self.project_type = project_type

        # Shared context
        self.type_definitions: dict[str, str] = {}
        self.imports: dict[str, list[str]] = {}
        self.file_contents: dict[str, str] = {}
        self.decisions: list[dict[str, Any]] = []
        self.discoveries: list[str] = []
        self.conventions: list[str] = []

        # Wave-specific context
        self.wave_outputs: dict[int, dict[str, Any]] = {}
        self.completed_task_outputs: dict[str, dict[str, Any]] = {}

    def add_type_definition(self, name: str, definition: str) -> None:
        """Add a type/interface definition for sharing."""
        self.type_definitions[name] = definition
        logger.debug(f"Added type definition: {name}")

    def add_import(self, module: str, items: list[str]) -> None:
        """Add import statements for sharing."""
        if module not in self.imports:
            self.imports[module] = []
        self.imports[module].extend(items)
        logger.debug(f"Added imports from {module}: {items}")

    def add_file_content(self, path: str, content: str) -> None:
        """Add file content for context."""
        self.file_contents[path] = content

    def add_decision(
        self,
        category: str,
        decision: str,
        rationale: str = "",
    ) -> None:
        """Add a project decision."""
        self.decisions.append({
            "category": category,
            "decision": decision,
            "rationale": rationale,
        })

    def add_discovery(self, discovery: str) -> None:
        """Add a project discovery."""
        self.discoveries.append(discovery)

    def add_convention(self, convention: str) -> None:
        """Add a coding convention."""
        self.conventions.append(convention)

    def add_task_output(self, task_id: str, output: dict[str, Any]) -> None:
        """Record output from a completed task."""
        self.completed_task_outputs[task_id] = output

    def add_wave_output(self, wave_number: int, outputs: dict[str, Any]) -> None:
        """Record outputs from a completed wave."""
        self.wave_outputs[wave_number] = outputs

    def get_types_for_task(self, task: TaskSpec) -> dict[str, str]:
        """Get type definitions relevant for a task."""
        # Return all types if task needs models
        if task.category in (TaskCategory.DATA_MODEL, TaskCategory.API, TaskCategory.BUSINESS_LOGIC):
            return self.type_definitions
        return {}

    def get_imports_for_task(self, task: TaskSpec) -> dict[str, list[str]]:
        """Get imports relevant for a task."""
        relevant_imports: dict[str, list[str]] = {}

        # Get imports from files this task depends on
        for dep_id in task.requires_context_from:
            if dep_id in self.completed_task_outputs:
                output = self.completed_task_outputs[dep_id]
                if "exports" in output:
                    for module, items in output["exports"].items():
                        if module not in relevant_imports:
                            relevant_imports[module] = []
                        relevant_imports[module].extend(items)

        return relevant_imports

    def to_dict(self) -> dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "project_name": self.project_name,
            "workspace": self.workspace,
            "tech_stack": self.tech_stack,
            "project_type": self.project_type.value if self.project_type else None,
            "type_definitions": self.type_definitions,
            "imports": self.imports,
            "decisions": self.decisions,
            "discoveries": self.discoveries,
            "conventions": self.conventions,
        }


# =============================================================================
# PROMPT BUILDER
# =============================================================================


class PromptBuilder:
    """
    Build dynamic prompts from templates and context.

    Combines templates with runtime context to create effective prompts
    for parallel Claude session execution.

    Example:
        >>> builder = PromptBuilder()
        >>> ctx = PromptContext(project_name="my-api")
        >>> prompt = builder.build_parallel_task_prompt(task, ctx)
    """

    def __init__(self) -> None:
        """Initialize the prompt builder."""
        self._templates: dict[str, PromptTemplate] = {
            "system": SYSTEM_PROMPT,
            "task": TASK_PROMPT,
            "decomposition": DECOMPOSITION_PROMPT,
            "parallel_task": PARALLEL_TASK_PROMPT,
            "validation": VALIDATION_PROMPT,
            "context_injection": CONTEXT_INJECTION_PROMPT,
        }
        self._max_file_content_length = 2000
        self._max_decisions = 10
        self._max_discoveries = 10

    # -------------------------------------------------------------------------
    # CORE BUILD METHODS
    # -------------------------------------------------------------------------

    def build(
        self,
        task: TaskSpec,
        context: PromptContext,
        include_validation: bool = True,
    ) -> str:
        """
        Build a complete prompt for task execution.

        This is the main entry point for building prompts for parallel
        task execution.

        Args:
            task: Task specification.
            context: Prompt context with shared information.
            include_validation: Whether to include validation section.

        Returns:
            Complete formatted prompt string.
        """
        sections = []

        # Header
        sections.append(self._build_header(task, context))

        # Task section
        sections.append(self._build_task_section(task))

        # Context section (types, imports, etc.)
        context_section = self._build_context_section(task, context)
        if context_section:
            sections.append(context_section)

        # Dependencies section
        deps_section = self._build_dependencies_section(task, context)
        if deps_section:
            sections.append(deps_section)

        # Constraints section
        constraints_section = self._build_constraints_section(task, context)
        if constraints_section:
            sections.append(constraints_section)

        # Validation section
        if include_validation and task.validation_commands:
            sections.append(self._build_validation_section(task))

        # Instructions section
        sections.append(self._build_instructions_section(task, context))

        return "\n\n".join(sections)

    def _build_header(self, task: TaskSpec, context: PromptContext) -> str:
        """Build the header section."""
        lines = [
            f"# Task: {task.title}",
            f"**Project**: {context.project_name or 'Unknown'}",
            f"**Category**: {task.category.value}",
            f"**Priority**: {task.priority}",
        ]

        if task.wave_number is not None:
            lines.append(f"**Wave**: {task.wave_number}")

        if context.tech_stack:
            tech = ", ".join(f"{k}: {v}" for k, v in context.tech_stack.items())
            lines.append(f"**Tech Stack**: {tech}")

        return "\n".join(lines)

    def _build_task_section(self, task: TaskSpec) -> str:
        """Build the task description section."""
        lines = [
            "## Description",
            task.description,
        ]

        if task.files_to_create:
            lines.extend([
                "",
                "## Files to Create",
                *[f"- `{f}`" for f in task.files_to_create],
            ])

        if task.files_to_modify:
            lines.extend([
                "",
                "## Files to Modify",
                *[f"- `{f}`" for f in task.files_to_modify],
            ])

        return "\n".join(lines)

    def _build_context_section(
        self,
        task: TaskSpec,
        context: PromptContext,
    ) -> str:
        """Build context section with types, imports, etc."""
        sections = []

        # Type definitions
        types = context.get_types_for_task(task)
        if types:
            type_lines = ["## Available Types"]
            for name, definition in types.items():
                type_lines.append(f"\n### {name}")
                type_lines.append(f"```typescript\n{definition}\n```")
            sections.append("\n".join(type_lines))

        # Imports
        imports = context.get_imports_for_task(task)
        if imports:
            import_lines = ["## Available Imports"]
            for module, items in imports.items():
                import_lines.append(f"- From `{module}`: {', '.join(items)}")
            sections.append("\n".join(import_lines))

        # Relevant file contents
        relevant_files = self._get_relevant_files(task, context)
        if relevant_files:
            file_lines = ["## Relevant Files"]
            for path, content in relevant_files.items():
                truncated = self._truncate_content(content)
                file_lines.append(f"\n### {path}")
                file_lines.append(f"```\n{truncated}\n```")
            sections.append("\n".join(file_lines))

        return "\n\n".join(sections) if sections else ""

    def _build_dependencies_section(
        self,
        task: TaskSpec,
        context: PromptContext,
    ) -> str:
        """Build dependencies and context from other tasks."""
        if not task.dependencies and not task.requires_context_from:
            return ""

        lines = ["## Dependencies"]

        if task.dependencies:
            lines.append(f"\nThis task depends on: {', '.join(task.dependencies)}")

        # Add outputs from dependent tasks
        for dep_id in task.requires_context_from:
            if dep_id in context.completed_task_outputs:
                output = context.completed_task_outputs[dep_id]
                lines.append(f"\n### From task `{dep_id}`")
                if "summary" in output:
                    lines.append(output["summary"])
                if "files_created" in output:
                    lines.append("Files created: " + ", ".join(output["files_created"]))

        return "\n".join(lines)

    def _build_constraints_section(
        self,
        task: TaskSpec,
        context: PromptContext,
    ) -> str:
        """Build constraints from context."""
        lines = []

        # Project decisions
        if context.decisions:
            lines.append("## Project Decisions")
            for dec in context.decisions[:self._max_decisions]:
                lines.append(f"- **{dec['category']}**: {dec['decision']}")
            lines.append("")

        # Conventions
        if context.conventions:
            lines.append("## Coding Conventions")
            for conv in context.conventions[:10]:
                lines.append(f"- {conv}")
            lines.append("")

        # Discoveries
        if context.discoveries:
            lines.append("## Project Discoveries")
            for disc in context.discoveries[:self._max_discoveries]:
                lines.append(f"- {disc}")

        return "\n".join(lines) if lines else ""

    def _build_validation_section(self, task: TaskSpec) -> str:
        """Build validation commands section."""
        lines = [
            "## Validation",
            "After completing, run these commands to validate:",
        ]
        for cmd in task.validation_commands:
            lines.append(f"- `{cmd}`")
        return "\n".join(lines)

    def _build_instructions_section(
        self,
        task: TaskSpec,
        context: PromptContext,
    ) -> str:
        """Build the instructions section."""
        lines = [
            "## Instructions",
            "1. Read any existing relevant files first",
            "2. Implement the required functionality",
            "3. Follow project code conventions",
            "4. Include proper error handling",
            "5. Add type hints to all functions",
        ]

        # Category-specific instructions
        if task.category == TaskCategory.API:
            lines.extend([
                "6. Include input validation",
                "7. Add appropriate HTTP status codes",
                "8. Document the endpoint",
            ])
        elif task.category == TaskCategory.DATA_MODEL:
            lines.extend([
                "6. Include field validators",
                "7. Add model documentation",
                "8. Include serialization methods",
            ])
        elif task.category == TaskCategory.UI:
            lines.extend([
                "6. Ensure accessibility (ARIA attributes)",
                "7. Add loading/error states",
                "8. Make component responsive",
            ])
        elif task.category == TaskCategory.TESTING:
            lines.extend([
                "6. Test edge cases",
                "7. Add integration tests where needed",
                "8. Aim for high coverage",
            ])

        lines.extend([
            "",
            "Execute this task completely. Create production-quality code.",
        ])

        return "\n".join(lines)

    def _get_relevant_files(
        self,
        task: TaskSpec,
        context: PromptContext,
    ) -> dict[str, str]:
        """Get files relevant to the task."""
        relevant: dict[str, str] = {}

        # Files this task will modify
        for path in task.files_to_modify:
            if path in context.file_contents:
                relevant[path] = context.file_contents[path]

        return relevant

    def _truncate_content(self, content: str) -> str:
        """Truncate content to max length."""
        if len(content) <= self._max_file_content_length:
            return content
        return content[:self._max_file_content_length] + "\n... (truncated)"

    # -------------------------------------------------------------------------
    # SPECIALIZED BUILD METHODS
    # -------------------------------------------------------------------------

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
        Build a task execution prompt (legacy interface).

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

    def build_parallel_task_prompt(
        self,
        task: TaskSpec,
        context: PromptContext,
        wave_number: int,
        parallel_tasks: list[str] | None = None,
    ) -> str:
        """
        Build a prompt for parallel task execution.

        This builds a comprehensive prompt that includes awareness of
        parallel execution and context from previous waves.

        Args:
            task: Task specification.
            context: Prompt context.
            wave_number: Current wave number.
            parallel_tasks: Other tasks executing in parallel.

        Returns:
            Formatted prompt for parallel execution.
        """
        # Build the main prompt
        main_prompt = self.build(task, context)

        # Add parallel execution awareness
        parallel_section = self._build_parallel_awareness_section(
            task, wave_number, parallel_tasks or []
        )

        if parallel_section:
            return f"{main_prompt}\n\n{parallel_section}"
        return main_prompt

    def _build_parallel_awareness_section(
        self,
        task: TaskSpec,
        wave_number: int,
        parallel_tasks: list[str],
    ) -> str:
        """Build section about parallel execution awareness."""
        if not parallel_tasks:
            return ""

        lines = [
            "## Parallel Execution Notice",
            f"Wave {wave_number}: This task is executing in parallel with:",
        ]
        for other_task in parallel_tasks[:5]:
            if other_task != task.id:
                lines.append(f"- {other_task}")

        lines.extend([
            "",
            "**Important**: Do not modify files that may be touched by parallel tasks.",
            "Focus only on your assigned files:",
        ])
        for f in task.files_to_create + task.files_to_modify:
            lines.append(f"- `{f}`")

        return "\n".join(lines)

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

    def build_validation_prompt(
        self,
        workspace: str,
        tasks_completed: list[str],
        files_created: list[str],
        files_modified: list[str],
    ) -> str:
        """
        Build a validation prompt for project output.

        Args:
            workspace: Project workspace path.
            tasks_completed: List of completed task IDs.
            files_created: List of created file paths.
            files_modified: List of modified file paths.

        Returns:
            Formatted validation prompt.
        """
        template = self._templates["validation"]
        return template.format(
            workspace=workspace,
            task_count=len(tasks_completed),
            files_created="\n".join(f"- {f}" for f in files_created) or "None",
            files_modified="\n".join(f"- {f}" for f in files_modified) or "None",
        )

    def build_context_injection_prompt(
        self,
        task: TaskSpec,
        context: PromptContext,
    ) -> str:
        """
        Build a context injection prompt for mid-task context updates.

        Used when a task needs additional context from recently
        completed parallel tasks.

        Args:
            task: Current task.
            context: Updated context.

        Returns:
            Context injection prompt.
        """
        lines = [
            "## Context Update",
            "The following context has been discovered from parallel tasks:",
        ]

        # Add new type definitions
        types = context.get_types_for_task(task)
        if types:
            lines.append("\n### New Types Available")
            for name in types:
                lines.append(f"- `{name}`")

        # Add new imports
        imports = context.get_imports_for_task(task)
        if imports:
            lines.append("\n### New Imports Available")
            for module, items in imports.items():
                lines.append(f"- From `{module}`: {', '.join(items)}")

        lines.append("\nIncorporate this context as needed.")

        return "\n".join(lines)

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
            for dec in decisions[:5]:
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
                truncated = self._truncate_content(content)
                sections.append(f"\n### {path}\n```\n{truncated}\n```")

        return "\n".join(sections)

    # -------------------------------------------------------------------------
    # TEMPLATE MANAGEMENT
    # -------------------------------------------------------------------------

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

    def get_template(self, name: str) -> PromptTemplate | None:
        """
        Get a template by name.

        Args:
            name: Template name.

        Returns:
            PromptTemplate if found.
        """
        return self._templates.get(name)


# =============================================================================
# FACTORY AND SINGLETON
# =============================================================================


_builder: PromptBuilder | None = None


def get_prompt_builder() -> PromptBuilder:
    """
    Get the shared PromptBuilder instance.

    Returns:
        PromptBuilder singleton.
    """
    global _builder
    if _builder is None:
        _builder = PromptBuilder()
    return _builder


def create_prompt_context(
    project_name: str,
    workspace: str,
    requirements: ProjectRequirements | None = None,
) -> PromptContext:
    """
    Create a PromptContext from project requirements.

    Args:
        project_name: Project name.
        workspace: Workspace path.
        requirements: Optional ProjectRequirements.

    Returns:
        Configured PromptContext.
    """
    ctx = PromptContext(
        project_name=project_name,
        workspace=workspace,
    )

    if requirements:
        ctx.tech_stack = requirements.tech_stack
        ctx.project_type = requirements.project_type

        # Add tech-stack based conventions
        if requirements.uses_typescript():
            ctx.add_convention("Use TypeScript strict mode")
            ctx.add_convention("Export types from dedicated files")
            ctx.add_convention("Use interfaces for data shapes")

        if requirements.uses_database():
            ctx.add_convention("Use migrations for schema changes")
            ctx.add_convention("Include proper indexes")

    return ctx
