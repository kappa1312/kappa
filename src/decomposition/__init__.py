"""Task decomposition - parsing specs into executable tasks.

This module provides the complete task decomposition pipeline:
- Requirements parsing (natural language -> structured requirements)
- Task generation (requirements -> atomic tasks)
- Dependency resolution (tasks -> execution waves)
- Parallel execution (waves -> results)
- Validation (results -> verified outputs)
"""

from src.decomposition.dependency_resolver import DependencyResolver
from src.decomposition.executor import (
    ParallelExecutor,
    RetryExecutor,
    SequentialExecutor,
    TaskExecutionResult,
    WaveExecutionResult,
    create_executor,
)
from src.decomposition.models import (
    Complexity,
    DependencyGraph,
    FileConflict,
    ProjectRequirements,
    ProjectType,
    SessionType,
    Task,
    TaskCategory,
    TaskDependency,
    TaskSpec,
)
from src.decomposition.parser import RequirementsParser, SpecificationParser
from src.decomposition.task_generator import TaskGenerator
from src.decomposition.validator import (
    FileValidator,
    ProjectValidationResult,
    ProjectValidator,
    ValidationResult,
    create_validator,
)

__all__ = [
    # Models - Legacy
    "Complexity",
    "Task",
    "TaskCategory",
    "TaskDependency",
    # Models - New
    "ProjectType",
    "SessionType",
    "ProjectRequirements",
    "TaskSpec",
    "DependencyGraph",
    "FileConflict",
    # Parser
    "SpecificationParser",
    "RequirementsParser",
    # Task Generation
    "TaskGenerator",
    # Dependency Resolution
    "DependencyResolver",
    # Execution
    "ParallelExecutor",
    "SequentialExecutor",
    "RetryExecutor",
    "TaskExecutionResult",
    "WaveExecutionResult",
    "create_executor",
    # Validation
    "ProjectValidator",
    "FileValidator",
    "ValidationResult",
    "ProjectValidationResult",
    "create_validator",
]
