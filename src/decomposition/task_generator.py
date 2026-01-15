"""Task generator - creates executable tasks from parsed requirements."""

from typing import Any

from loguru import logger

from src.decomposition.models import (
    Complexity,
    ParsedRequirement,
    Task,
    TaskCategory,
)
from src.decomposition.parser import SpecificationParser


class TaskGenerator:
    """
    Generate executable tasks from specifications.

    Takes a natural language specification and produces a list
    of well-defined tasks ready for parallel execution.

    Example:
        >>> generator = TaskGenerator()
        >>> tasks = await generator.generate("Build a REST API with auth")
        >>> len(tasks) > 0
        True
    """

    # Default task templates by category
    TASK_TEMPLATES: dict[TaskCategory, list[dict[str, Any]]] = {
        TaskCategory.SETUP: [
            {
                "name": "Initialize project structure",
                "description": "Create directory structure and configuration files",
                "complexity": Complexity.LOW,
                "file_targets": ["pyproject.toml", "README.md", ".gitignore"],
            },
        ],
        TaskCategory.DATA_MODEL: [
            {
                "name": "Create {entity} model",
                "description": "Define SQLAlchemy model for {entity} with all fields and relationships",
                "complexity": Complexity.MEDIUM,
                "file_targets": ["src/models/{entity}.py"],
            },
        ],
        TaskCategory.API: [
            {
                "name": "Implement {entity} API endpoints",
                "description": "Create CRUD API endpoints for {entity}",
                "complexity": Complexity.MEDIUM,
                "file_targets": ["src/api/{entity}.py"],
            },
        ],
        TaskCategory.BUSINESS_LOGIC: [
            {
                "name": "Implement {entity} service",
                "description": "Create service layer with business logic for {entity}",
                "complexity": Complexity.MEDIUM,
                "file_targets": ["src/services/{entity}.py"],
            },
        ],
        TaskCategory.TESTING: [
            {
                "name": "Write tests for {entity}",
                "description": "Create unit and integration tests for {entity}",
                "complexity": Complexity.MEDIUM,
                "file_targets": ["tests/test_{entity}.py"],
            },
        ],
        TaskCategory.DOCUMENTATION: [
            {
                "name": "Document {entity}",
                "description": "Write documentation for {entity} API and usage",
                "complexity": Complexity.LOW,
                "file_targets": ["docs/{entity}.md"],
            },
        ],
    }

    def __init__(self) -> None:
        """Initialize the task generator."""
        self.parser = SpecificationParser()

    async def generate(
        self,
        specification: str,
        context: dict[str, Any] | None = None,
    ) -> list[Task]:
        """
        Generate tasks from a specification.

        Args:
            specification: Natural language project specification.
            context: Optional context with existing project info.

        Returns:
            List of Task objects ready for execution.

        Example:
            >>> generator = TaskGenerator()
            >>> tasks = await generator.generate(
            ...     "Build a blog API with posts and comments"
            ... )
            >>> task_names = [t.name for t in tasks]
            >>> "Create post model" in task_names or len(tasks) > 0
            True
        """
        logger.info("Generating tasks from specification")

        # Parse specification into requirements
        requirements = self.parser.parse(specification)

        # Generate tasks for each requirement
        tasks: list[Task] = []

        # Always add setup task first
        setup_task = Task(
            name="Initialize project structure",
            description="Create project directory structure, configuration files, and base dependencies",
            category=TaskCategory.SETUP,
            complexity=Complexity.LOW,
            wave=0,
            file_targets=["pyproject.toml", "README.md", ".gitignore", "src/__init__.py"],
            tags=["setup", "infrastructure"],
        )
        tasks.append(setup_task)

        # Track entities for dependency management
        entities: set[str] = set()
        model_tasks: dict[str, str] = {}  # entity -> task_id

        # First pass: Create model/schema tasks
        for req in requirements:
            if req.category in (TaskCategory.DATA_MODEL, TaskCategory.INFRASTRUCTURE):
                for entity in req.entities:
                    if entity not in entities:
                        entities.add(entity)
                        task = self._create_model_task(entity, setup_task.id)
                        tasks.append(task)
                        model_tasks[entity] = task.id

        # If no explicit entities, extract from requirement text
        if not entities:
            entities = self._extract_implicit_entities(requirements)
            for entity in entities:
                task = self._create_model_task(entity, setup_task.id)
                tasks.append(task)
                model_tasks[entity] = task.id

        # Second pass: Create service/logic tasks
        for entity in entities:
            deps = [setup_task.id]
            if entity in model_tasks:
                deps.append(model_tasks[entity])

            task = self._create_service_task(entity, deps)
            tasks.append(task)

        # Third pass: Create API tasks
        api_tasks: dict[str, str] = {}
        for req in requirements:
            if req.category == TaskCategory.API:
                for entity in entities:
                    deps = [setup_task.id]
                    if entity in model_tasks:
                        deps.append(model_tasks[entity])

                    task = self._create_api_task(entity, deps)
                    tasks.append(task)
                    api_tasks[entity] = task.id
                break  # Only create API tasks once
        else:
            # Default: create API tasks for all entities if API-related keywords in spec
            if any(kw in specification.lower() for kw in ["api", "rest", "endpoint", "http"]):
                for entity in entities:
                    deps = [setup_task.id]
                    if entity in model_tasks:
                        deps.append(model_tasks[entity])
                    task = self._create_api_task(entity, deps)
                    tasks.append(task)
                    api_tasks[entity] = task.id

        # Fourth pass: Create test tasks
        for entity in entities:
            deps = [setup_task.id]
            if entity in model_tasks:
                deps.append(model_tasks[entity])
            if entity in api_tasks:
                deps.append(api_tasks[entity])

            task = self._create_test_task(entity, deps)
            tasks.append(task)

        logger.info(f"Generated {len(tasks)} tasks for {len(entities)} entities")
        return tasks

    def _create_model_task(self, entity: str, setup_task_id: str) -> Task:
        """Create a data model task for an entity."""
        return Task(
            name=f"Create {entity} model",
            description=f"Define the data model/schema for {entity} including all fields, types, and relationships",
            category=TaskCategory.DATA_MODEL,
            complexity=Complexity.MEDIUM,
            dependencies=[setup_task_id],
            wave=1,
            file_targets=[f"src/models/{entity.lower()}.py"],
            tags=["model", "schema", entity.lower()],
        )

    def _create_service_task(self, entity: str, deps: list[str]) -> Task:
        """Create a service layer task for an entity."""
        return Task(
            name=f"Implement {entity} service",
            description=f"Create business logic and service layer for {entity} including CRUD operations and validation",
            category=TaskCategory.BUSINESS_LOGIC,
            complexity=Complexity.MEDIUM,
            dependencies=deps,
            wave=2,
            file_targets=[f"src/services/{entity.lower()}.py"],
            tags=["service", "business-logic", entity.lower()],
        )

    def _create_api_task(self, entity: str, deps: list[str]) -> Task:
        """Create an API endpoint task for an entity."""
        return Task(
            name=f"Create {entity} API endpoints",
            description=f"Implement REST API endpoints for {entity} including GET, POST, PUT, DELETE operations",
            category=TaskCategory.API,
            complexity=Complexity.MEDIUM,
            dependencies=deps,
            wave=2,
            file_targets=[f"src/api/{entity.lower()}.py"],
            tags=["api", "endpoint", entity.lower()],
        )

    def _create_test_task(self, entity: str, deps: list[str]) -> Task:
        """Create a test task for an entity."""
        return Task(
            name=f"Write {entity} tests",
            description=f"Create unit and integration tests for {entity} model, service, and API",
            category=TaskCategory.TESTING,
            complexity=Complexity.MEDIUM,
            dependencies=deps,
            wave=3,
            file_targets=[f"tests/test_{entity.lower()}.py"],
            tags=["test", "pytest", entity.lower()],
        )

    def _extract_implicit_entities(
        self,
        requirements: list[ParsedRequirement],
    ) -> set[str]:
        """Extract entities implied by the requirements."""
        entities: set[str] = set()

        # Common entity names to look for
        common_entities = {
            "user", "account", "profile", "auth",
            "product", "item", "catalog",
            "order", "cart", "checkout",
            "post", "article", "blog", "comment",
            "message", "notification",
            "file", "document", "upload",
            "payment", "invoice", "subscription",
            "settings", "config", "preference",
        }

        for req in requirements:
            text_lower = req.text.lower()
            for entity in common_entities:
                if entity in text_lower:
                    entities.add(entity)

        # Default entities if none found
        if not entities:
            entities.add("item")

        return entities


async def generate_tasks(spec: str) -> list[Task]:
    """Convenience function to generate tasks.

    Args:
        spec: Natural language specification.

    Returns:
        List of Task objects.
    """
    generator = TaskGenerator()
    return await generator.generate(spec)
