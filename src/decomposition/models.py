"""Pydantic models for task decomposition."""

from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


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


class TaskDependency(BaseModel):
    """Represents a dependency relationship between tasks."""

    model_config = ConfigDict(frozen=True)

    task_id: str = Field(description="ID of the dependent task")
    dependency_id: str = Field(description="ID of the task this depends on")
    dependency_type: str = Field(
        default="required",
        description="Type: required, optional, or soft",
    )


class Task(BaseModel):
    """A single executable task in the decomposition."""

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
            f"# Task: {self.name}",
            "",
            f"## Description",
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
