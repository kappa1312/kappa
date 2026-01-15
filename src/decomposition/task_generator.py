"""Task generator - creates executable tasks from parsed requirements.

This module generates atomic, executable TaskSpec objects from
ProjectRequirements. It supports project-type aware generation
with appropriate task structures for websites, APIs, dashboards,
CLI tools, and libraries.
"""

from typing import Any
from uuid import uuid4

from loguru import logger

from src.decomposition.models import (
    Complexity,
    ParsedRequirement,
    ProjectRequirements,
    ProjectType,
    SessionType,
    Task,
    TaskCategory,
    TaskSpec,
)


class TaskGenerator:
    """
    Generate executable tasks from project requirements.

    Takes ProjectRequirements and produces a list of well-defined
    TaskSpec objects ready for parallel execution. Generation is
    project-type aware with different task structures for:
    - Website: foundation → types → components → pages → tests
    - API: foundation → models → services → routes → tests
    - Dashboard: foundation → data layer → components → pages → tests
    - CLI Tool: foundation → commands → utils → tests
    - Library: foundation → core → utils → tests → docs

    Example:
        >>> generator = TaskGenerator()
        >>> reqs = ProjectRequirements(
        ...     name="my-api",
        ...     description="REST API",
        ...     project_type=ProjectType.API
        ... )
        >>> tasks = await generator.generate(reqs)
        >>> len(tasks) > 0
        True
    """

    # Wave configuration by project type
    WAVE_CONFIGS: dict[ProjectType, list[str]] = {
        ProjectType.API: [
            "foundation",  # Wave 0: Setup, config, base structure
            "types_models",  # Wave 1: Types, schemas, database models
            "services",  # Wave 2: Business logic, services
            "routes",  # Wave 3: API routes, middleware
            "integration",  # Wave 4: Third-party integrations
            "testing",  # Wave 5: Tests
        ],
        ProjectType.WEBSITE: [
            "foundation",  # Wave 0: Setup, config
            "types_styles",  # Wave 1: Types, global styles, theme
            "components",  # Wave 2: UI components
            "layouts",  # Wave 3: Layout components
            "pages",  # Wave 4: Page components
            "integration",  # Wave 5: API integration
            "testing",  # Wave 6: Tests
        ],
        ProjectType.DASHBOARD: [
            "foundation",  # Wave 0: Setup, config
            "types_models",  # Wave 1: Types, data models
            "data_layer",  # Wave 2: Data fetching, state management
            "components",  # Wave 3: UI components, charts
            "pages",  # Wave 4: Dashboard pages
            "testing",  # Wave 5: Tests
        ],
        ProjectType.CLI_TOOL: [
            "foundation",  # Wave 0: Setup, config, CLI framework
            "types",  # Wave 1: Types, schemas
            "commands",  # Wave 2: CLI commands
            "utils",  # Wave 3: Utility functions
            "testing",  # Wave 4: Tests
        ],
        ProjectType.LIBRARY: [
            "foundation",  # Wave 0: Setup, build config
            "types",  # Wave 1: Type definitions
            "core",  # Wave 2: Core functionality
            "utils",  # Wave 3: Utilities, helpers
            "testing",  # Wave 4: Tests
            "documentation",  # Wave 5: API docs
        ],
        ProjectType.MOBILE_APP: [
            "foundation",  # Wave 0: Setup, config
            "types_models",  # Wave 1: Types, data models
            "services",  # Wave 2: API services, storage
            "components",  # Wave 3: UI components
            "screens",  # Wave 4: Screens/pages
            "navigation",  # Wave 5: Navigation setup
            "testing",  # Wave 6: Tests
        ],
    }

    def __init__(self) -> None:
        """Initialize the task generator."""
        self._task_counter = 0

    def _generate_task_id(self, prefix: str = "task") -> str:
        """Generate a unique task ID."""
        self._task_counter += 1
        return f"{prefix}-{self._task_counter}-{uuid4().hex[:8]}"

    async def generate(
        self,
        requirements: ProjectRequirements | str,
        context: dict[str, Any] | None = None,
    ) -> list[TaskSpec]:
        """
        Generate tasks from project requirements.

        Args:
            requirements: ProjectRequirements object or specification string.
            context: Optional context with existing project info.

        Returns:
            List of TaskSpec objects ready for execution.

        Example:
            >>> generator = TaskGenerator()
            >>> reqs = ProjectRequirements(
            ...     name="blog-api",
            ...     description="Blog REST API",
            ...     project_type=ProjectType.API,
            ...     features=["posts", "comments", "auth"]
            ... )
            >>> tasks = await generator.generate(reqs)
            >>> any(t.category == TaskCategory.SETUP for t in tasks)
            True
        """
        # Handle string input for backward compatibility
        if isinstance(requirements, str):
            from src.decomposition.parser import RequirementsParser

            parser = RequirementsParser()
            requirements = await parser.parse(requirements)

        logger.info(
            f"Generating tasks for {requirements.project_type.value} project: "
            f"{requirements.name}"
        )

        self._task_counter = 0
        tasks: list[TaskSpec] = []

        # Generate based on project type
        if requirements.project_type == ProjectType.API:
            tasks = await self._generate_api_tasks(requirements)
        elif requirements.project_type == ProjectType.WEBSITE:
            tasks = await self._generate_website_tasks(requirements)
        elif requirements.project_type == ProjectType.DASHBOARD:
            tasks = await self._generate_dashboard_tasks(requirements)
        elif requirements.project_type == ProjectType.CLI_TOOL:
            tasks = await self._generate_cli_tasks(requirements)
        elif requirements.project_type == ProjectType.LIBRARY:
            tasks = await self._generate_library_tasks(requirements)
        elif requirements.project_type == ProjectType.MOBILE_APP:
            tasks = await self._generate_mobile_tasks(requirements)
        else:
            # Fallback to generic tasks
            tasks = await self._generate_generic_tasks(requirements)

        logger.info(
            f"Generated {len(tasks)} tasks across "
            f"{max(t.wave_number or 0 for t in tasks) + 1} waves"
        )
        return tasks

    # =========================================================================
    # API PROJECT GENERATION
    # =========================================================================

    async def _generate_api_tasks(
        self,
        requirements: ProjectRequirements,
    ) -> list[TaskSpec]:
        """Generate tasks for API projects."""
        tasks: list[TaskSpec] = []
        task_ids: dict[str, str] = {}

        # Wave 0: Foundation
        foundation_tasks = self._generate_foundation_tasks(requirements, wave=0)
        tasks.extend(foundation_tasks)
        task_ids["setup"] = foundation_tasks[0].id

        # Wave 1: Types and Models
        type_tasks = self._generate_type_tasks(requirements, wave=1, deps=[task_ids["setup"]])
        tasks.extend(type_tasks)
        if type_tasks:
            task_ids["types"] = type_tasks[0].id

        model_tasks = self._generate_model_tasks(
            requirements,
            wave=1,
            deps=[task_ids["setup"]] + ([task_ids["types"]] if "types" in task_ids else []),
        )
        tasks.extend(model_tasks)
        for t in model_tasks:
            entity = t.title.replace("Create ", "").replace(" model", "").lower()
            task_ids[f"model_{entity}"] = t.id

        # Wave 2: Services
        service_deps = [task_ids["setup"]]
        service_deps.extend([tid for key, tid in task_ids.items() if key.startswith("model_")])
        service_tasks = self._generate_service_tasks(requirements, wave=2, deps=service_deps)
        tasks.extend(service_tasks)
        for t in service_tasks:
            entity = t.title.replace("Implement ", "").replace(" service", "").lower()
            task_ids[f"service_{entity}"] = t.id

        # Wave 3: Routes/Endpoints
        route_deps = [task_ids["setup"]]
        route_deps.extend([tid for key, tid in task_ids.items() if key.startswith("service_")])
        route_tasks = self._generate_api_route_tasks(requirements, wave=3, deps=route_deps)
        tasks.extend(route_tasks)
        for t in route_tasks:
            task_ids[f"route_{t.id}"] = t.id

        # Wave 4: Integrations
        if requirements.integrations:
            integration_deps = [task_ids["setup"]]
            integration_tasks = self._generate_integration_tasks(
                requirements, wave=4, deps=integration_deps
            )
            tasks.extend(integration_tasks)

        # Wave 5: Testing
        test_deps = [tid for key, tid in task_ids.items() if key.startswith("route_")]
        if not test_deps:
            test_deps = [task_ids["setup"]]
        test_tasks = self._generate_test_tasks(requirements, wave=5, deps=test_deps)
        tasks.extend(test_tasks)

        return tasks

    # =========================================================================
    # WEBSITE PROJECT GENERATION
    # =========================================================================

    async def _generate_website_tasks(
        self,
        requirements: ProjectRequirements,
    ) -> list[TaskSpec]:
        """Generate tasks for website projects."""
        tasks: list[TaskSpec] = []
        task_ids: dict[str, str] = {}

        # Wave 0: Foundation
        foundation_tasks = self._generate_website_foundation_tasks(requirements, wave=0)
        tasks.extend(foundation_tasks)
        task_ids["setup"] = foundation_tasks[0].id

        # Wave 1: Types and Styles
        type_tasks = self._generate_type_tasks(requirements, wave=1, deps=[task_ids["setup"]])
        tasks.extend(type_tasks)
        if type_tasks:
            task_ids["types"] = type_tasks[0].id

        style_tasks = self._generate_style_tasks(requirements, wave=1, deps=[task_ids["setup"]])
        tasks.extend(style_tasks)

        # Wave 2: Components
        component_deps = [task_ids["setup"]]
        if "types" in task_ids:
            component_deps.append(task_ids["types"])
        component_tasks = self._generate_component_tasks(requirements, wave=2, deps=component_deps)
        tasks.extend(component_tasks)
        for t in component_tasks:
            task_ids[f"component_{t.id}"] = t.id

        # Wave 3: Layouts
        layout_deps = [tid for key, tid in task_ids.items() if key.startswith("component_")]
        if not layout_deps:
            layout_deps = [task_ids["setup"]]
        layout_tasks = self._generate_layout_tasks(requirements, wave=3, deps=layout_deps)
        tasks.extend(layout_tasks)

        # Wave 4: Pages
        page_deps = [task_ids["setup"]]
        page_deps.extend([tid for key, tid in task_ids.items() if key.startswith("component_")])
        page_tasks = self._generate_page_tasks(requirements, wave=4, deps=page_deps)
        tasks.extend(page_tasks)

        # Wave 5: Integrations
        if requirements.integrations:
            integration_tasks = self._generate_integration_tasks(
                requirements, wave=5, deps=[task_ids["setup"]]
            )
            tasks.extend(integration_tasks)

        # Wave 6: Testing
        test_deps = [task_ids["setup"]]
        test_tasks = self._generate_test_tasks(requirements, wave=6, deps=test_deps)
        tasks.extend(test_tasks)

        return tasks

    # =========================================================================
    # DASHBOARD PROJECT GENERATION
    # =========================================================================

    async def _generate_dashboard_tasks(
        self,
        requirements: ProjectRequirements,
    ) -> list[TaskSpec]:
        """Generate tasks for dashboard projects."""
        tasks: list[TaskSpec] = []
        task_ids: dict[str, str] = {}

        # Wave 0: Foundation
        foundation_tasks = self._generate_website_foundation_tasks(requirements, wave=0)
        tasks.extend(foundation_tasks)
        task_ids["setup"] = foundation_tasks[0].id

        # Wave 1: Types and Models
        type_tasks = self._generate_type_tasks(requirements, wave=1, deps=[task_ids["setup"]])
        tasks.extend(type_tasks)

        # Wave 2: Data Layer
        data_tasks = self._generate_data_layer_tasks(requirements, wave=2, deps=[task_ids["setup"]])
        tasks.extend(data_tasks)

        # Wave 3: Components (including charts)
        component_tasks = self._generate_dashboard_component_tasks(
            requirements, wave=3, deps=[task_ids["setup"]]
        )
        tasks.extend(component_tasks)

        # Wave 4: Pages
        page_tasks = self._generate_page_tasks(requirements, wave=4, deps=[task_ids["setup"]])
        tasks.extend(page_tasks)

        # Wave 5: Testing
        test_tasks = self._generate_test_tasks(requirements, wave=5, deps=[task_ids["setup"]])
        tasks.extend(test_tasks)

        return tasks

    # =========================================================================
    # CLI PROJECT GENERATION
    # =========================================================================

    async def _generate_cli_tasks(
        self,
        requirements: ProjectRequirements,
    ) -> list[TaskSpec]:
        """Generate tasks for CLI tool projects."""
        tasks: list[TaskSpec] = []
        task_ids: dict[str, str] = {}

        # Wave 0: Foundation
        foundation_tasks = self._generate_cli_foundation_tasks(requirements, wave=0)
        tasks.extend(foundation_tasks)
        task_ids["setup"] = foundation_tasks[0].id

        # Wave 1: Types
        type_tasks = self._generate_type_tasks(requirements, wave=1, deps=[task_ids["setup"]])
        tasks.extend(type_tasks)

        # Wave 2: Commands
        command_tasks = self._generate_command_tasks(requirements, wave=2, deps=[task_ids["setup"]])
        tasks.extend(command_tasks)

        # Wave 3: Utils
        util_tasks = self._generate_util_tasks(requirements, wave=3, deps=[task_ids["setup"]])
        tasks.extend(util_tasks)

        # Wave 4: Testing
        test_tasks = self._generate_test_tasks(requirements, wave=4, deps=[task_ids["setup"]])
        tasks.extend(test_tasks)

        return tasks

    # =========================================================================
    # LIBRARY PROJECT GENERATION
    # =========================================================================

    async def _generate_library_tasks(
        self,
        requirements: ProjectRequirements,
    ) -> list[TaskSpec]:
        """Generate tasks for library projects."""
        tasks: list[TaskSpec] = []
        task_ids: dict[str, str] = {}

        # Wave 0: Foundation
        foundation_tasks = self._generate_library_foundation_tasks(requirements, wave=0)
        tasks.extend(foundation_tasks)
        task_ids["setup"] = foundation_tasks[0].id

        # Wave 1: Types
        type_tasks = self._generate_type_tasks(requirements, wave=1, deps=[task_ids["setup"]])
        tasks.extend(type_tasks)

        # Wave 2: Core
        core_tasks = self._generate_core_tasks(requirements, wave=2, deps=[task_ids["setup"]])
        tasks.extend(core_tasks)

        # Wave 3: Utils
        util_tasks = self._generate_util_tasks(requirements, wave=3, deps=[task_ids["setup"]])
        tasks.extend(util_tasks)

        # Wave 4: Testing
        test_tasks = self._generate_test_tasks(requirements, wave=4, deps=[task_ids["setup"]])
        tasks.extend(test_tasks)

        # Wave 5: Documentation
        doc_tasks = self._generate_doc_tasks(requirements, wave=5, deps=[task_ids["setup"]])
        tasks.extend(doc_tasks)

        return tasks

    # =========================================================================
    # MOBILE APP PROJECT GENERATION
    # =========================================================================

    async def _generate_mobile_tasks(
        self,
        requirements: ProjectRequirements,
    ) -> list[TaskSpec]:
        """Generate tasks for mobile app projects."""
        tasks: list[TaskSpec] = []
        task_ids: dict[str, str] = {}

        # Wave 0: Foundation
        foundation_tasks = self._generate_mobile_foundation_tasks(requirements, wave=0)
        tasks.extend(foundation_tasks)
        task_ids["setup"] = foundation_tasks[0].id

        # Wave 1: Types and Models
        type_tasks = self._generate_type_tasks(requirements, wave=1, deps=[task_ids["setup"]])
        tasks.extend(type_tasks)

        # Wave 2: Services
        service_tasks = self._generate_mobile_service_tasks(
            requirements, wave=2, deps=[task_ids["setup"]]
        )
        tasks.extend(service_tasks)

        # Wave 3: Components
        component_tasks = self._generate_mobile_component_tasks(
            requirements, wave=3, deps=[task_ids["setup"]]
        )
        tasks.extend(component_tasks)

        # Wave 4: Screens
        screen_tasks = self._generate_screen_tasks(requirements, wave=4, deps=[task_ids["setup"]])
        tasks.extend(screen_tasks)

        # Wave 5: Navigation
        nav_task = TaskSpec(
            id=self._generate_task_id("nav"),
            title="Setup navigation",
            description="Configure app navigation with routes and transitions",
            session_type=SessionType.TERMINAL,
            category=TaskCategory.INFRASTRUCTURE,
            complexity=Complexity.MEDIUM,
            dependencies=[task_ids["setup"]],
            wave_number=5,
            files_to_create=["src/navigation/index.ts", "src/navigation/types.ts"],
            validation_commands=["npm run type-check"],
        )
        tasks.append(nav_task)

        # Wave 6: Testing
        test_tasks = self._generate_test_tasks(requirements, wave=6, deps=[task_ids["setup"]])
        tasks.extend(test_tasks)

        return tasks

    # =========================================================================
    # GENERIC PROJECT GENERATION
    # =========================================================================

    async def _generate_generic_tasks(
        self,
        requirements: ProjectRequirements,
    ) -> list[TaskSpec]:
        """Generate generic tasks when project type is unknown."""
        tasks: list[TaskSpec] = []
        task_ids: dict[str, str] = {}

        # Wave 0: Foundation
        foundation_tasks = self._generate_foundation_tasks(requirements, wave=0)
        tasks.extend(foundation_tasks)
        task_ids["setup"] = foundation_tasks[0].id

        # Wave 1: Types
        type_tasks = self._generate_type_tasks(requirements, wave=1, deps=[task_ids["setup"]])
        tasks.extend(type_tasks)

        # Wave 2: Core features
        for i, feature in enumerate(requirements.features[:5]):  # Limit to 5 features
            task = TaskSpec(
                id=self._generate_task_id("feature"),
                title=f"Implement {feature}",
                description=f"Implement the {feature} feature as specified in requirements",
                session_type=SessionType.TERMINAL,
                category=TaskCategory.BUSINESS_LOGIC,
                complexity=Complexity.MEDIUM,
                dependencies=[task_ids["setup"]],
                wave_number=2,
                files_to_create=[f"src/features/{self._slugify(feature)}.py"],
                validation_commands=["python -m pytest tests/ -v"],
            )
            tasks.append(task)

        # Wave 3: Testing
        test_tasks = self._generate_test_tasks(requirements, wave=3, deps=[task_ids["setup"]])
        tasks.extend(test_tasks)

        return tasks

    # =========================================================================
    # TASK GENERATION HELPERS
    # =========================================================================

    def _generate_foundation_tasks(
        self,
        requirements: ProjectRequirements,
        wave: int,
    ) -> list[TaskSpec]:
        """Generate foundation/setup tasks."""
        language = requirements.tech_stack.get("language", "python").lower()
        framework = requirements.tech_stack.get("framework", "").lower()

        # Determine config files based on language
        if language in ("typescript", "javascript", "ts", "js"):
            config_files = ["package.json", "tsconfig.json", ".eslintrc.json", ".prettierrc"]
            validation = ["npm install", "npm run type-check"]
        elif language == "python":
            config_files = ["pyproject.toml", "setup.py", ".flake8", "pytest.ini"]
            validation = ["poetry install", "poetry run pytest --collect-only"]
        else:
            config_files = ["README.md", ".gitignore"]
            validation = []

        setup_task = TaskSpec(
            id=self._generate_task_id("setup"),
            title="Initialize project structure",
            description=(
                f"Set up {requirements.name} project with {framework or language} stack. "
                f"Create directory structure, configuration files, and install dependencies."
            ),
            session_type=SessionType.TERMINAL,
            category=TaskCategory.SETUP,
            complexity=Complexity.LOW,
            dependencies=[],
            wave_number=wave,
            files_to_create=config_files + ["src/__init__.py", "README.md", ".gitignore"],
            validation_commands=validation,
            tags=["setup", "infrastructure", language],
        )

        tasks = [setup_task]

        # Add database setup if needed
        if requirements.uses_database():
            db_type = requirements.tech_stack.get("database", "postgresql")
            db_task = TaskSpec(
                id=self._generate_task_id("db"),
                title="Configure database",
                description=f"Set up {db_type} database connection, models base, and migrations",
                session_type=SessionType.TERMINAL,
                category=TaskCategory.INFRASTRUCTURE,
                complexity=Complexity.MEDIUM,
                dependencies=[setup_task.id],
                wave_number=wave,
                files_to_create=[
                    "src/database/connection.py",
                    "src/database/models.py",
                    "alembic.ini",
                ],
                validation_commands=["alembic check"],
                tags=["database", db_type],
            )
            tasks.append(db_task)

        return tasks

    def _generate_website_foundation_tasks(
        self,
        requirements: ProjectRequirements,
        wave: int,
    ) -> list[TaskSpec]:
        """Generate foundation tasks for website projects."""
        framework = requirements.tech_stack.get("framework", "react").lower()

        setup_task = TaskSpec(
            id=self._generate_task_id("setup"),
            title=f"Initialize {framework} project",
            description=(
                f"Set up {requirements.name} with {framework}. "
                "Configure build tools, linting, and development environment."
            ),
            session_type=SessionType.TERMINAL,
            category=TaskCategory.SETUP,
            complexity=Complexity.LOW,
            dependencies=[],
            wave_number=wave,
            files_to_create=[
                "package.json",
                "tsconfig.json",
                "vite.config.ts" if framework != "next" else "next.config.js",
                ".eslintrc.json",
                "src/main.tsx",
                "src/App.tsx",
            ],
            validation_commands=["npm install", "npm run build"],
            tags=["setup", framework, "frontend"],
        )

        return [setup_task]

    def _generate_cli_foundation_tasks(
        self,
        requirements: ProjectRequirements,
        wave: int,
    ) -> list[TaskSpec]:
        """Generate foundation tasks for CLI projects."""
        language = requirements.tech_stack.get("language", "python").lower()

        if language == "python":
            files = [
                "pyproject.toml",
                "src/cli/__init__.py",
                "src/cli/main.py",
                "src/cli/commands/__init__.py",
            ]
            validation = ["poetry install", "poetry run python -m src.cli --help"]
        else:
            files = ["package.json", "src/cli/index.ts", "src/cli/commands/index.ts"]
            validation = ["npm install", "npm run build"]

        setup_task = TaskSpec(
            id=self._generate_task_id("setup"),
            title="Initialize CLI project",
            description=f"Set up {requirements.name} CLI tool with argument parsing and command structure",
            session_type=SessionType.TERMINAL,
            category=TaskCategory.SETUP,
            complexity=Complexity.LOW,
            dependencies=[],
            wave_number=wave,
            files_to_create=files,
            validation_commands=validation,
            tags=["setup", "cli", language],
        )

        return [setup_task]

    def _generate_library_foundation_tasks(
        self,
        requirements: ProjectRequirements,
        wave: int,
    ) -> list[TaskSpec]:
        """Generate foundation tasks for library projects."""
        language = requirements.tech_stack.get("language", "python").lower()

        if language == "python":
            files = ["pyproject.toml", "src/__init__.py", "src/core.py", "py.typed"]
            validation = ["poetry install", "poetry build"]
        else:
            files = ["package.json", "tsconfig.json", "src/index.ts", "rollup.config.js"]
            validation = ["npm install", "npm run build"]

        setup_task = TaskSpec(
            id=self._generate_task_id("setup"),
            title="Initialize library project",
            description=f"Set up {requirements.name} library with build configuration and exports",
            session_type=SessionType.TERMINAL,
            category=TaskCategory.SETUP,
            complexity=Complexity.LOW,
            dependencies=[],
            wave_number=wave,
            files_to_create=files,
            validation_commands=validation,
            tags=["setup", "library", language],
        )

        return [setup_task]

    def _generate_mobile_foundation_tasks(
        self,
        requirements: ProjectRequirements,
        wave: int,
    ) -> list[TaskSpec]:
        """Generate foundation tasks for mobile app projects."""
        framework = requirements.tech_stack.get("framework", "react-native").lower()

        setup_task = TaskSpec(
            id=self._generate_task_id("setup"),
            title=f"Initialize {framework} project",
            description=f"Set up {requirements.name} mobile app with {framework}",
            session_type=SessionType.TERMINAL,
            category=TaskCategory.SETUP,
            complexity=Complexity.MEDIUM,
            dependencies=[],
            wave_number=wave,
            files_to_create=[
                "package.json",
                "app.json",
                "tsconfig.json",
                "src/App.tsx",
            ],
            validation_commands=["npm install", "npm run type-check"],
            tags=["setup", "mobile", framework],
        )

        return [setup_task]

    def _generate_type_tasks(
        self,
        requirements: ProjectRequirements,
        wave: int,
        deps: list[str],
    ) -> list[TaskSpec]:
        """Generate type definition tasks."""
        if not requirements.uses_typescript():
            return []

        task = TaskSpec(
            id=self._generate_task_id("types"),
            title="Create type definitions",
            description="Define TypeScript types, interfaces, and enums for the project",
            session_type=SessionType.TERMINAL,
            category=TaskCategory.TYPES,
            complexity=Complexity.LOW,
            dependencies=deps,
            wave_number=wave,
            files_to_create=["src/types/index.ts", "src/types/models.ts", "src/types/api.ts"],
            validation_commands=["npm run type-check"],
            tags=["types", "typescript"],
        )

        return [task]

    def _generate_model_tasks(
        self,
        requirements: ProjectRequirements,
        wave: int,
        deps: list[str],
    ) -> list[TaskSpec]:
        """Generate data model tasks."""
        tasks: list[TaskSpec] = []
        entities = self._extract_entities(requirements)

        for entity in entities:
            task = TaskSpec(
                id=self._generate_task_id("model"),
                title=f"Create {entity} model",
                description=f"Define data model/schema for {entity} with fields, types, and relationships",
                session_type=SessionType.TERMINAL,
                category=TaskCategory.DATA_MODEL,
                complexity=Complexity.MEDIUM,
                dependencies=deps,
                wave_number=wave,
                files_to_create=[f"src/models/{entity.lower()}.py"],
                validation_commands=["python -m pytest tests/models/ -v"],
                tags=["model", entity.lower()],
            )
            tasks.append(task)

        return tasks

    def _generate_service_tasks(
        self,
        requirements: ProjectRequirements,
        wave: int,
        deps: list[str],
    ) -> list[TaskSpec]:
        """Generate service layer tasks."""
        tasks: list[TaskSpec] = []
        entities = self._extract_entities(requirements)

        for entity in entities:
            task = TaskSpec(
                id=self._generate_task_id("service"),
                title=f"Implement {entity} service",
                description=f"Create service layer for {entity} with business logic and CRUD operations",
                session_type=SessionType.TERMINAL,
                category=TaskCategory.BUSINESS_LOGIC,
                complexity=Complexity.MEDIUM,
                dependencies=deps,
                wave_number=wave,
                files_to_create=[f"src/services/{entity.lower()}_service.py"],
                validation_commands=["python -m pytest tests/services/ -v"],
                tags=["service", entity.lower()],
            )
            tasks.append(task)

        return tasks

    def _generate_api_route_tasks(
        self,
        requirements: ProjectRequirements,
        wave: int,
        deps: list[str],
    ) -> list[TaskSpec]:
        """Generate API route tasks."""
        tasks: list[TaskSpec] = []
        entities = self._extract_entities(requirements)
        framework = requirements.tech_stack.get("framework", "fastapi").lower()

        for entity in entities:
            if framework in ("fastapi", "flask", "django"):
                file_path = f"src/api/routes/{entity.lower()}.py"
            else:
                file_path = f"src/routes/{entity.lower()}.ts"

            task = TaskSpec(
                id=self._generate_task_id("route"),
                title=f"Create {entity} API routes",
                description=f"Implement REST endpoints for {entity}: GET, POST, PUT, DELETE",
                session_type=SessionType.TERMINAL,
                category=TaskCategory.API,
                complexity=Complexity.MEDIUM,
                dependencies=deps,
                wave_number=wave,
                files_to_create=[file_path],
                validation_commands=["python -m pytest tests/api/ -v"],
                tags=["api", "routes", entity.lower()],
            )
            tasks.append(task)

        return tasks

    def _generate_component_tasks(
        self,
        requirements: ProjectRequirements,
        wave: int,
        deps: list[str],
    ) -> list[TaskSpec]:
        """Generate UI component tasks."""
        tasks: list[TaskSpec] = []

        # Common components based on features
        components = self._extract_components(requirements)

        for component in components:
            task = TaskSpec(
                id=self._generate_task_id("component"),
                title=f"Create {component} component",
                description=f"Build reusable {component} UI component with props and styling",
                session_type=SessionType.TERMINAL,
                category=TaskCategory.UI,
                complexity=Complexity.MEDIUM,
                dependencies=deps,
                wave_number=wave,
                files_to_create=[
                    f"src/components/{component}/{component}.tsx",
                    f"src/components/{component}/index.ts",
                ],
                validation_commands=["npm run build", "npm run test"],
                tags=["component", "ui", component.lower()],
            )
            tasks.append(task)

        return tasks

    def _generate_dashboard_component_tasks(
        self,
        requirements: ProjectRequirements,
        wave: int,
        deps: list[str],
    ) -> list[TaskSpec]:
        """Generate dashboard-specific component tasks."""
        tasks: list[TaskSpec] = []

        # Dashboard typically needs charts and data visualization
        dashboard_components = ["Chart", "DataTable", "StatCard", "FilterPanel"]

        for component in dashboard_components:
            task = TaskSpec(
                id=self._generate_task_id("component"),
                title=f"Create {component} component",
                description=f"Build {component} dashboard component with data visualization",
                session_type=SessionType.TERMINAL,
                category=TaskCategory.UI,
                complexity=Complexity.MEDIUM,
                dependencies=deps,
                wave_number=wave,
                files_to_create=[
                    f"src/components/{component}/{component}.tsx",
                    f"src/components/{component}/index.ts",
                ],
                validation_commands=["npm run build"],
                tags=["component", "dashboard", component.lower()],
            )
            tasks.append(task)

        return tasks

    def _generate_mobile_component_tasks(
        self,
        requirements: ProjectRequirements,
        wave: int,
        deps: list[str],
    ) -> list[TaskSpec]:
        """Generate mobile app component tasks."""
        tasks: list[TaskSpec] = []

        components = self._extract_components(requirements)
        if not components:
            components = ["Button", "Card", "Input", "Header"]

        for component in components:
            task = TaskSpec(
                id=self._generate_task_id("component"),
                title=f"Create {component} component",
                description=f"Build {component} mobile UI component",
                session_type=SessionType.TERMINAL,
                category=TaskCategory.UI,
                complexity=Complexity.MEDIUM,
                dependencies=deps,
                wave_number=wave,
                files_to_create=[f"src/components/{component}.tsx"],
                validation_commands=["npm run type-check"],
                tags=["component", "mobile", component.lower()],
            )
            tasks.append(task)

        return tasks

    def _generate_layout_tasks(
        self,
        requirements: ProjectRequirements,
        wave: int,
        deps: list[str],
    ) -> list[TaskSpec]:
        """Generate layout component tasks."""
        task = TaskSpec(
            id=self._generate_task_id("layout"),
            title="Create layout components",
            description="Build main layout, header, footer, and sidebar components",
            session_type=SessionType.TERMINAL,
            category=TaskCategory.UI,
            complexity=Complexity.MEDIUM,
            dependencies=deps,
            wave_number=wave,
            files_to_create=[
                "src/layouts/MainLayout.tsx",
                "src/components/Header/Header.tsx",
                "src/components/Footer/Footer.tsx",
            ],
            validation_commands=["npm run build"],
            tags=["layout", "ui"],
        )

        return [task]

    def _generate_page_tasks(
        self,
        requirements: ProjectRequirements,
        wave: int,
        deps: list[str],
    ) -> list[TaskSpec]:
        """Generate page tasks."""
        tasks: list[TaskSpec] = []
        pages = requirements.pages or ["Home", "About"]

        for page in pages:
            page_name = self._capitalize_page(page)
            task = TaskSpec(
                id=self._generate_task_id("page"),
                title=f"Create {page_name} page",
                description=f"Build {page_name} page with routing and components",
                session_type=SessionType.TERMINAL,
                category=TaskCategory.UI,
                complexity=Complexity.MEDIUM,
                dependencies=deps,
                wave_number=wave,
                files_to_create=[f"src/pages/{page_name}/{page_name}.tsx"],
                validation_commands=["npm run build"],
                tags=["page", page_name.lower()],
            )
            tasks.append(task)

        return tasks

    def _generate_screen_tasks(
        self,
        requirements: ProjectRequirements,
        wave: int,
        deps: list[str],
    ) -> list[TaskSpec]:
        """Generate mobile screen tasks."""
        tasks: list[TaskSpec] = []
        screens = requirements.pages or ["Home", "Profile", "Settings"]

        for screen in screens:
            screen_name = self._capitalize_page(screen)
            task = TaskSpec(
                id=self._generate_task_id("screen"),
                title=f"Create {screen_name} screen",
                description=f"Build {screen_name} screen for mobile app",
                session_type=SessionType.TERMINAL,
                category=TaskCategory.UI,
                complexity=Complexity.MEDIUM,
                dependencies=deps,
                wave_number=wave,
                files_to_create=[f"src/screens/{screen_name}Screen.tsx"],
                validation_commands=["npm run type-check"],
                tags=["screen", "mobile", screen_name.lower()],
            )
            tasks.append(task)

        return tasks

    def _generate_style_tasks(
        self,
        requirements: ProjectRequirements,
        wave: int,
        deps: list[str],
    ) -> list[TaskSpec]:
        """Generate styling/theme tasks."""
        task = TaskSpec(
            id=self._generate_task_id("styles"),
            title="Setup global styles and theme",
            description="Configure global styles, CSS variables, and theme system",
            session_type=SessionType.TERMINAL,
            category=TaskCategory.UI,
            complexity=Complexity.LOW,
            dependencies=deps,
            wave_number=wave,
            files_to_create=[
                "src/styles/globals.css",
                "src/styles/variables.css",
                "src/theme/index.ts",
            ],
            validation_commands=["npm run build"],
            tags=["styles", "theme"],
        )

        return [task]

    def _generate_data_layer_tasks(
        self,
        requirements: ProjectRequirements,
        wave: int,
        deps: list[str],
    ) -> list[TaskSpec]:
        """Generate data layer tasks for dashboards."""
        tasks = [
            TaskSpec(
                id=self._generate_task_id("data"),
                title="Setup data fetching layer",
                description="Configure API client, data fetching hooks, and caching",
                session_type=SessionType.TERMINAL,
                category=TaskCategory.INFRASTRUCTURE,
                complexity=Complexity.MEDIUM,
                dependencies=deps,
                wave_number=wave,
                files_to_create=[
                    "src/api/client.ts",
                    "src/hooks/useQuery.ts",
                    "src/store/index.ts",
                ],
                validation_commands=["npm run type-check"],
                tags=["data", "api"],
            ),
        ]

        return tasks

    def _generate_command_tasks(
        self,
        requirements: ProjectRequirements,
        wave: int,
        deps: list[str],
    ) -> list[TaskSpec]:
        """Generate CLI command tasks."""
        tasks: list[TaskSpec] = []
        features = requirements.features or ["run", "config", "version"]

        for feature in features[:5]:  # Limit commands
            command_name = self._slugify(feature)
            task = TaskSpec(
                id=self._generate_task_id("cmd"),
                title=f"Implement '{command_name}' command",
                description=f"Create CLI command for {feature}",
                session_type=SessionType.TERMINAL,
                category=TaskCategory.BUSINESS_LOGIC,
                complexity=Complexity.MEDIUM,
                dependencies=deps,
                wave_number=wave,
                files_to_create=[f"src/cli/commands/{command_name}.py"],
                validation_commands=[f"poetry run python -m src.cli {command_name} --help"],
                tags=["cli", "command", command_name],
            )
            tasks.append(task)

        return tasks

    def _generate_util_tasks(
        self,
        requirements: ProjectRequirements,
        wave: int,
        deps: list[str],
    ) -> list[TaskSpec]:
        """Generate utility function tasks."""
        task = TaskSpec(
            id=self._generate_task_id("utils"),
            title="Create utility functions",
            description="Implement common utility functions and helpers",
            session_type=SessionType.TERMINAL,
            category=TaskCategory.BUSINESS_LOGIC,
            complexity=Complexity.LOW,
            dependencies=deps,
            wave_number=wave,
            files_to_create=["src/utils/helpers.py", "src/utils/validators.py"],
            validation_commands=["python -m pytest tests/utils/ -v"],
            tags=["utils", "helpers"],
        )

        return [task]

    def _generate_core_tasks(
        self,
        requirements: ProjectRequirements,
        wave: int,
        deps: list[str],
    ) -> list[TaskSpec]:
        """Generate core library functionality tasks."""
        tasks: list[TaskSpec] = []

        for feature in requirements.features[:3]:  # Core features
            task = TaskSpec(
                id=self._generate_task_id("core"),
                title=f"Implement {feature}",
                description=f"Build core {feature} functionality for the library",
                session_type=SessionType.TERMINAL,
                category=TaskCategory.BUSINESS_LOGIC,
                complexity=Complexity.HIGH,
                dependencies=deps,
                wave_number=wave,
                files_to_create=[f"src/{self._slugify(feature)}.py"],
                validation_commands=["python -m pytest -v"],
                tags=["core", self._slugify(feature)],
            )
            tasks.append(task)

        return tasks

    def _generate_mobile_service_tasks(
        self,
        requirements: ProjectRequirements,
        wave: int,
        deps: list[str],
    ) -> list[TaskSpec]:
        """Generate mobile service tasks."""
        tasks = [
            TaskSpec(
                id=self._generate_task_id("api"),
                title="Setup API service",
                description="Configure API client for backend communication",
                session_type=SessionType.TERMINAL,
                category=TaskCategory.INFRASTRUCTURE,
                complexity=Complexity.MEDIUM,
                dependencies=deps,
                wave_number=wave,
                files_to_create=["src/services/api.ts", "src/services/auth.ts"],
                validation_commands=["npm run type-check"],
                tags=["service", "api"],
            ),
            TaskSpec(
                id=self._generate_task_id("storage"),
                title="Setup local storage",
                description="Configure local storage for offline data",
                session_type=SessionType.TERMINAL,
                category=TaskCategory.INFRASTRUCTURE,
                complexity=Complexity.LOW,
                dependencies=deps,
                wave_number=wave,
                files_to_create=["src/services/storage.ts"],
                validation_commands=["npm run type-check"],
                tags=["service", "storage"],
            ),
        ]

        return tasks

    def _generate_integration_tasks(
        self,
        requirements: ProjectRequirements,
        wave: int,
        deps: list[str],
    ) -> list[TaskSpec]:
        """Generate third-party integration tasks."""
        tasks: list[TaskSpec] = []

        for integration in requirements.integrations:
            task = TaskSpec(
                id=self._generate_task_id("integration"),
                title=f"Integrate {integration}",
                description=f"Set up {integration} integration with configuration and client",
                session_type=SessionType.TERMINAL,
                category=TaskCategory.INTEGRATION,
                complexity=Complexity.MEDIUM,
                dependencies=deps,
                wave_number=wave,
                files_to_create=[f"src/integrations/{self._slugify(integration)}.py"],
                validation_commands=["python -m pytest tests/integrations/ -v"],
                tags=["integration", self._slugify(integration)],
            )
            tasks.append(task)

        return tasks

    def _generate_test_tasks(
        self,
        requirements: ProjectRequirements,
        wave: int,
        deps: list[str],
    ) -> list[TaskSpec]:
        """Generate testing tasks."""
        language = requirements.tech_stack.get("language", "python").lower()

        if language in ("typescript", "javascript", "ts", "js"):
            files = ["tests/setup.ts", "tests/unit/example.test.ts"]
            validation = ["npm run test"]
        else:
            files = ["tests/conftest.py", "tests/unit/test_example.py"]
            validation = ["python -m pytest -v"]

        task = TaskSpec(
            id=self._generate_task_id("test"),
            title="Create test suite",
            description="Set up testing framework and write initial tests",
            session_type=SessionType.TERMINAL,
            category=TaskCategory.TESTING,
            complexity=Complexity.MEDIUM,
            dependencies=deps,
            wave_number=wave,
            files_to_create=files,
            validation_commands=validation,
            tags=["testing", "pytest" if language == "python" else "jest"],
        )

        return [task]

    def _generate_doc_tasks(
        self,
        requirements: ProjectRequirements,
        wave: int,
        deps: list[str],
    ) -> list[TaskSpec]:
        """Generate documentation tasks."""
        task = TaskSpec(
            id=self._generate_task_id("docs"),
            title="Create documentation",
            description="Write API documentation and usage examples",
            session_type=SessionType.TERMINAL,
            category=TaskCategory.DOCUMENTATION,
            complexity=Complexity.LOW,
            dependencies=deps,
            wave_number=wave,
            files_to_create=["docs/README.md", "docs/API.md", "docs/examples.md"],
            validation_commands=[],
            tags=["documentation"],
        )

        return [task]

    # =========================================================================
    # EXTRACTION HELPERS
    # =========================================================================

    def _extract_entities(self, requirements: ProjectRequirements) -> list[str]:
        """Extract entity names from requirements."""
        entities: set[str] = set()

        # Common entity patterns to look for
        common_entities = {
            "user": ["user", "account", "profile", "member"],
            "product": ["product", "item", "goods"],
            "order": ["order", "purchase", "transaction"],
            "post": ["post", "article", "blog", "content"],
            "comment": ["comment", "review", "feedback"],
            "category": ["category", "tag", "label"],
            "message": ["message", "chat", "notification"],
            "file": ["file", "document", "upload", "media"],
            "payment": ["payment", "invoice", "billing"],
            "settings": ["settings", "config", "preference"],
        }

        text = requirements.description.lower() + " " + " ".join(requirements.features).lower()

        for entity, keywords in common_entities.items():
            if any(kw in text for kw in keywords):
                entities.add(entity.capitalize())

        # Default if nothing found
        if not entities:
            entities.add("Item")

        return sorted(entities)

    def _extract_components(self, requirements: ProjectRequirements) -> list[str]:
        """Extract UI component names from requirements."""
        components: set[str] = set()

        # Map features to components
        feature_component_map = {
            "auth": ["LoginForm", "SignupForm", "AuthProvider"],
            "login": ["LoginForm"],
            "signup": ["SignupForm"],
            "search": ["SearchBar", "SearchResults"],
            "filter": ["FilterPanel"],
            "form": ["Form", "FormField"],
            "table": ["DataTable"],
            "list": ["ListView"],
            "card": ["Card"],
            "modal": ["Modal"],
            "nav": ["Navbar", "Sidebar"],
            "menu": ["Menu", "Dropdown"],
        }

        text = " ".join(requirements.features).lower()

        for keyword, comps in feature_component_map.items():
            if keyword in text:
                components.update(comps)

        # Default components
        if not components:
            components = {"Button", "Card", "Input"}

        return sorted(components)

    def _capitalize_page(self, page: str) -> str:
        """Capitalize page name properly."""
        # Handle routes like '/about' or 'about-us'
        page = page.strip("/").replace("-", " ").replace("_", " ")
        return "".join(word.capitalize() for word in page.split())

    def _slugify(self, text: str) -> str:
        """Convert text to slug format."""
        import re

        slug = text.lower().strip()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"[-\s]+", "_", slug)
        return slug


# =============================================================================
# LEGACY SUPPORT
# =============================================================================


class LegacyTaskGenerator:
    """
    Legacy task generator for backward compatibility.

    This class maintains the original string-based interface for existing code.

    Example:
        >>> generator = LegacyTaskGenerator()
        >>> tasks = await generator.generate("Build a REST API")
        >>> len(tasks) > 0
        True
    """

    def __init__(self) -> None:
        """Initialize the legacy generator."""
        from src.decomposition.parser import SpecificationParser

        self.parser = SpecificationParser()

    async def generate(
        self,
        specification: str,
        context: dict[str, Any] | None = None,
    ) -> list[Task]:
        """
        Generate legacy Task objects from specification.

        Args:
            specification: Natural language project specification.
            context: Optional context.

        Returns:
            List of legacy Task objects.
        """
        logger.info("Using legacy task generation")

        # Parse specification into requirements
        requirements = self.parser.parse(specification)

        tasks: list[Task] = []

        # Setup task
        setup_task = Task(
            name="Initialize project structure",
            description="Create project structure and configuration",
            category=TaskCategory.SETUP,
            complexity=Complexity.LOW,
            wave=0,
            file_targets=["pyproject.toml", "README.md"],
            tags=["setup"],
        )
        tasks.append(setup_task)

        # Extract entities
        entities = self._extract_entities(requirements)

        # Model tasks
        model_ids: dict[str, str] = {}
        for entity in entities:
            task = Task(
                name=f"Create {entity} model",
                description=f"Define data model for {entity}",
                category=TaskCategory.DATA_MODEL,
                complexity=Complexity.MEDIUM,
                dependencies=[setup_task.id],
                wave=1,
                file_targets=[f"src/models/{entity.lower()}.py"],
                tags=["model", entity.lower()],
            )
            tasks.append(task)
            model_ids[entity] = task.id

        # API tasks
        for entity in entities:
            deps = [setup_task.id]
            if entity in model_ids:
                deps.append(model_ids[entity])

            task = Task(
                name=f"Create {entity} API",
                description=f"Implement API endpoints for {entity}",
                category=TaskCategory.API,
                complexity=Complexity.MEDIUM,
                dependencies=deps,
                wave=2,
                file_targets=[f"src/api/{entity.lower()}.py"],
                tags=["api", entity.lower()],
            )
            tasks.append(task)

        # Test task
        test_task = Task(
            name="Create test suite",
            description="Write unit and integration tests",
            category=TaskCategory.TESTING,
            complexity=Complexity.MEDIUM,
            dependencies=[setup_task.id],
            wave=3,
            file_targets=["tests/conftest.py"],
            tags=["testing"],
        )
        tasks.append(test_task)

        logger.info(f"Generated {len(tasks)} legacy tasks")
        return tasks

    def _extract_entities(
        self,
        requirements: list[ParsedRequirement],
    ) -> set[str]:
        """Extract entity names from parsed requirements."""
        entities: set[str] = set()

        common_entities = {
            "user",
            "product",
            "order",
            "post",
            "comment",
            "item",
        }

        for req in requirements:
            text_lower = req.text.lower()
            for entity in common_entities:
                if entity in text_lower:
                    entities.add(entity.capitalize())

        if not entities:
            entities.add("Item")

        return entities


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


async def generate_tasks(
    requirements: ProjectRequirements | str,
) -> list[TaskSpec] | list[Task]:
    """
    Convenience function to generate tasks.

    Automatically detects input type and uses appropriate generator.

    Args:
        requirements: ProjectRequirements object or specification string.

    Returns:
        List of TaskSpec (for ProjectRequirements) or Task (for string).

    Example:
        >>> from src.decomposition.models import ProjectRequirements, ProjectType
        >>> reqs = ProjectRequirements(
        ...     name="test",
        ...     description="Test API",
        ...     project_type=ProjectType.API
        ... )
        >>> tasks = await generate_tasks(reqs)
    """
    if isinstance(requirements, str):
        # Legacy string input
        generator = LegacyTaskGenerator()
        return await generator.generate(requirements)
    else:
        # New ProjectRequirements input
        generator = TaskGenerator()
        return await generator.generate(requirements)
