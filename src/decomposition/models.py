"""Pydantic models for task decomposition.

This module defines all the data structures used in Kappa's task
decomposition system, including project types, requirements, task
specifications, and dependency graphs.
"""

from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


# =============================================================================
# ENUMS
# =============================================================================


class ProjectType(str, Enum):
    """Type of project being built."""

    WEBSITE = "website"
    API = "api"
    DASHBOARD = "dashboard"
    CLI_TOOL = "cli_tool"
    LIBRARY = "library"
    MOBILE_APP = "mobile_app"


class SessionType(str, Enum):
    """Type of Claude session to use for task execution."""

    TERMINAL = "terminal"
    WEB = "web"
    NATIVE = "native"
    EXTENSION = "extension"


class Complexity(str, Enum):
    """Task complexity level."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TaskCategory(str, Enum):
    """Category of development task."""

    SETUP = "setup"
    INFRASTRUCTURE = "infrastructure"
    DATA_MODEL = "data_model"
    BUSINESS_LOGIC = "business_logic"
    API = "api"
    UI = "ui"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    DEPLOYMENT = "deployment"
    INTEGRATION = "integration"
    TYPES = "types"


class TaskStatus(str, Enum):
    """Execution status of a task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# =============================================================================
# PROJECT REQUIREMENTS
# =============================================================================


class ProjectRequirements(BaseModel):
    """Structured project requirements parsed from natural language.

    This model represents the fully parsed and structured requirements
    that drive task generation. It captures the project type, technology
    choices, features, and constraints.

    Example:
        >>> reqs = ProjectRequirements(
        ...     name="my-api",
        ...     description="REST API for user management",
        ...     project_type=ProjectType.API,
        ...     tech_stack={"framework": "express", "language": "typescript"},
        ...     features=["user registration", "authentication"],
        ... )
    """

    model_config = ConfigDict(frozen=False)

    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Project name",
    )
    description: str = Field(
        ...,
        description="Brief project description",
    )
    project_type: ProjectType = Field(
        default=ProjectType.API,
        description="Type of project",
    )

    tech_stack: dict[str, str] = Field(
        default_factory=dict,
        description="Technology choices (framework, language, database, etc.)",
    )

    features: list[str] = Field(
        default_factory=list,
        description="List of features to implement",
    )

    pages: list[str] = Field(
        default_factory=list,
        description="Pages/routes for websites/apps",
    )

    integrations: list[str] = Field(
        default_factory=list,
        description="Third-party integrations (APIs, services)",
    )

    constraints: list[str] = Field(
        default_factory=list,
        description="Development constraints or requirements",
    )

    timeline: str | None = Field(
        default=None,
        description="Project timeline if specified",
    )

    priority: str = Field(
        default="normal",
        description="Overall priority level",
    )

    @field_validator("tech_stack")
    @classmethod
    def validate_tech_stack(cls, v: dict[str, str]) -> dict[str, str]:
        """Ensure tech stack has required keys if provided."""
        # Allow empty tech_stack, but if provided, ensure reasonable defaults
        if v and "language" not in v:
            # Infer language from framework if possible
            framework = v.get("framework", "").lower()
            if framework in ("express", "nest", "next", "react", "vue", "angular"):
                v["language"] = "typescript"
            elif framework in ("django", "flask", "fastapi"):
                v["language"] = "python"
            elif framework in ("rails", "sinatra"):
                v["language"] = "ruby"
            elif framework in ("spring", "quarkus"):
                v["language"] = "java"
        return v

    def uses_typescript(self) -> bool:
        """Check if project uses TypeScript."""
        return self.tech_stack.get("language", "").lower() in (
            "typescript",
            "ts",
        )

    def uses_database(self) -> bool:
        """Check if project uses a database."""
        return "database" in self.tech_stack or any(
            kw in str(self.features).lower()
            for kw in ("database", "sql", "mongo", "postgres", "mysql", "redis")
        )


# =============================================================================
# TASK SPECIFICATIONS
# =============================================================================


class TaskSpec(BaseModel):
    """Atomic, executable task specification.

    TaskSpec represents a single unit of work that can be executed
    by a Claude session. It includes execution constraints, dependencies,
    and validation commands.

    Example:
        >>> task = TaskSpec(
        ...     title="Create User Model",
        ...     description="Define User SQLAlchemy model with fields...",
        ...     session_type=SessionType.TERMINAL,
        ...     dependencies=["setup-task"],
        ...     files_to_create=["src/models/user.py"],
        ... )
    """

    model_config = ConfigDict(frozen=False)

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique task identifier",
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Task title",
    )
    description: str = Field(
        ...,
        description="Detailed task description",
    )
    session_type: SessionType = Field(
        default=SessionType.TERMINAL,
        description="Type of session to use",
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="Task IDs this task depends on",
    )
    files_to_create: list[str] = Field(
        default_factory=list,
        description="Files this task will create",
    )
    files_to_modify: list[str] = Field(
        default_factory=list,
        description="Files this task will modify",
    )
    priority: int = Field(
        default=5,
        ge=0,
        le=10,
        description="Priority (0=highest, 10=lowest)",
    )
    estimated_duration_minutes: int = Field(
        default=30,
        gt=0,
        description="Estimated execution time in minutes",
    )
    wave_number: int | None = Field(
        default=None,
        description="Assigned execution wave",
    )

    # Execution constraints
    requires_context_from: list[str] = Field(
        default_factory=list,
        description="Task IDs whose context this task needs",
    )

    validation_commands: list[str] = Field(
        default_factory=list,
        description="Commands to run for validation",
    )

    # Additional metadata
    category: TaskCategory = Field(
        default=TaskCategory.BUSINESS_LOGIC,
        description="Task category",
    )
    complexity: Complexity = Field(
        default=Complexity.MEDIUM,
        description="Task complexity",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorization",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    def is_ready(self, completed_tasks: set[str]) -> bool:
        """Check if all dependencies are satisfied.

        Args:
            completed_tasks: Set of completed task IDs.

        Returns:
            True if all dependencies are in completed_tasks.
        """
        return all(dep in completed_tasks for dep in self.dependencies)

    def to_prompt(self) -> str:
        """Generate a prompt for Claude to execute this task.

        Returns:
            Formatted prompt string.
        """
        lines = [
            f"# Task: {self.title}",
            "",
            "## Description",
            self.description,
            "",
        ]

        if self.files_to_create:
            lines.extend([
                "## Files to Create",
                *[f"- {f}" for f in self.files_to_create],
                "",
            ])

        if self.files_to_modify:
            lines.extend([
                "## Files to Modify",
                *[f"- {f}" for f in self.files_to_modify],
                "",
            ])

        if self.validation_commands:
            lines.extend([
                "## Validation",
                "Run these commands to verify:",
                *[f"- `{cmd}`" for cmd in self.validation_commands],
                "",
            ])

        if self.tags:
            lines.extend([
                f"## Tags: {', '.join(self.tags)}",
                "",
            ])

        lines.extend([
            "## Requirements",
            "- Follow the project's code style and conventions",
            "- Include appropriate error handling",
            "- Add type hints for all functions",
            "- Write clean, maintainable code",
            "",
        ])

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()


# =============================================================================
# DEPENDENCY GRAPH
# =============================================================================


class DependencyGraph(BaseModel):
    """Graph representation of task dependencies.

    This model maintains the complete dependency structure, including
    nodes (tasks), edges (dependencies), and computed execution waves.

    Example:
        >>> graph = DependencyGraph()
        >>> graph.add_task(task1)
        >>> graph.add_task(task2)
        >>> graph.is_ready("task2-id", {"task1-id"})
        True
    """

    model_config = ConfigDict(frozen=False)

    nodes: dict[str, TaskSpec] = Field(
        default_factory=dict,
        description="Task ID -> TaskSpec mapping",
    )
    edges: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Task ID -> list of dependency IDs",
    )
    waves: list[list[str]] = Field(
        default_factory=list,
        description="Execution waves (list of task ID lists)",
    )

    def add_task(self, task: TaskSpec) -> None:
        """Add a task to the graph.

        Args:
            task: TaskSpec to add.
        """
        self.nodes[task.id] = task
        self.edges[task.id] = task.dependencies.copy()

    def get_task(self, task_id: str) -> TaskSpec | None:
        """Get a task by ID.

        Args:
            task_id: Task identifier.

        Returns:
            TaskSpec if found, None otherwise.
        """
        return self.nodes.get(task_id)

    def get_dependents(self, task_id: str) -> list[str]:
        """Get tasks that depend on this task.

        Args:
            task_id: Task identifier.

        Returns:
            List of task IDs that depend on this task.
        """
        return [
            tid for tid, deps in self.edges.items()
            if task_id in deps
        ]

    def is_ready(self, task_id: str, completed: set[str]) -> bool:
        """Check if a task's dependencies are satisfied.

        Args:
            task_id: Task to check.
            completed: Set of completed task IDs.

        Returns:
            True if all dependencies are completed.
        """
        deps = self.edges.get(task_id, [])
        return all(dep in completed for dep in deps)

    def get_all_tasks(self) -> list[TaskSpec]:
        """Get all tasks in the graph.

        Returns:
            List of all TaskSpec objects.
        """
        return list(self.nodes.values())

    def get_wave_tasks(self, wave_number: int) -> list[TaskSpec]:
        """Get tasks in a specific wave.

        Args:
            wave_number: Wave index.

        Returns:
            List of TaskSpec objects in that wave.
        """
        if wave_number >= len(self.waves):
            return []
        return [
            self.nodes[tid] for tid in self.waves[wave_number]
            if tid in self.nodes
        ]

    @property
    def total_waves(self) -> int:
        """Get total number of waves."""
        return len(self.waves)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "nodes": {k: v.model_dump() for k, v in self.nodes.items()},
            "edges": self.edges,
            "waves": self.waves,
            "total_waves": self.total_waves,
        }


# =============================================================================
# LEGACY COMPATIBILITY MODELS
# =============================================================================


class TaskDependency(BaseModel):
    """Represents a dependency relationship between tasks.

    Note: This is maintained for backward compatibility.
    New code should use TaskSpec.dependencies directly.
    """

    model_config = ConfigDict(frozen=True)

    task_id: str = Field(description="ID of the dependent task")
    dependency_id: str = Field(description="ID of the task this depends on")
    dependency_type: str = Field(
        default="required",
        description="Type: required, optional, or soft",
    )


class Task(BaseModel):
    """A single executable task in the decomposition.

    Note: This is maintained for backward compatibility.
    New code should use TaskSpec instead.
    """

    model_config = ConfigDict(frozen=False)

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(min_length=1, max_length=255)
    description: str = Field(min_length=1)
    category: TaskCategory = Field(default=TaskCategory.BUSINESS_LOGIC)
    complexity: Complexity = Field(default=Complexity.MEDIUM)
    dependencies: list[str] = Field(default_factory=list)
    wave: int = Field(ge=0, default=0)
    estimated_tokens: int = Field(ge=0, default=1000)
    file_targets: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def is_ready(self, completed_tasks: set[str]) -> bool:
        """Check if all dependencies are satisfied."""
        return all(dep in completed_tasks for dep in self.dependencies)

    def to_prompt(self) -> str:
        """Generate a prompt for Claude to execute this task."""
        lines = [
            f"# Task: {self.name}",
            "",
            "## Description",
            self.description,
            "",
        ]

        if self.file_targets:
            lines.extend([
                "## Target Files",
                *[f"- {f}" for f in self.file_targets],
                "",
            ])

        if self.tags:
            lines.extend([
                f"## Tags: {', '.join(self.tags)}",
                "",
            ])

        lines.extend([
            "## Requirements",
            "- Follow the project's code style and conventions",
            "- Include appropriate error handling",
            "- Add type hints for all functions",
            "- Write clean, maintainable code",
            "",
        ])

        return "\n".join(lines)

    def to_task_spec(self) -> TaskSpec:
        """Convert to TaskSpec for new code."""
        return TaskSpec(
            id=self.id,
            title=self.name,
            description=self.description,
            category=self.category,
            complexity=self.complexity,
            dependencies=self.dependencies,
            wave_number=self.wave,
            files_to_create=self.file_targets,
            tags=self.tags,
            metadata=self.metadata,
        )


class ParsedRequirement(BaseModel):
    """A requirement extracted from the specification."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(description="Original requirement text")
    category: TaskCategory = Field(description="Inferred category")
    entities: list[str] = Field(
        default_factory=list,
        description="Extracted entities (models, endpoints, etc.)",
    )
    priority: int = Field(
        ge=1,
        le=5,
        default=3,
        description="Priority level (1=highest)",
    )


class DecompositionResult(BaseModel):
    """Result of decomposing a specification."""

    model_config = ConfigDict(frozen=True)

    tasks: list[Task] = Field(description="Generated tasks")
    dependency_graph: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Task ID -> dependency IDs",
    )
    waves: list[list[str]] = Field(
        default_factory=list,
        description="Wave index -> task IDs",
    )
    total_estimated_tokens: int = Field(
        default=0,
        description="Total estimated token usage",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Any warnings from decomposition",
    )

    # New fields for Session 2
    project_requirements: ProjectRequirements | None = Field(
        default=None,
        description="Parsed project requirements",
    )
    task_specs: list[TaskSpec] = Field(
        default_factory=list,
        description="New-style task specifications",
    )


# =============================================================================
# CONFLICT DETECTION
# =============================================================================


class FileConflict(BaseModel):
    """Represents a file write conflict between tasks."""

    model_config = ConfigDict(frozen=True)

    conflict_type: str = Field(
        default="file_write",
        description="Type of conflict",
    )
    file_path: str = Field(
        description="Path to conflicting file",
    )
    task_ids: list[str] = Field(
        description="IDs of conflicting tasks",
    )
    wave_number: int = Field(
        description="Wave where conflict occurs",
    )
    description: str = Field(
        default="",
        description="Human-readable description",
    )
