"""
Conflict Detection System for Kappa OS

Detects potential conflicts before parallel execution:
- File write conflicts (multiple tasks writing same file)
- Import collisions (circular dependencies, duplicate imports)
- Naming conflicts (duplicate component/function names)
- Type mismatches (incompatible type definitions)
- Dependency violations (task needs output from incomplete task)
- Resource contention (same external resource access)
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger

from src.decomposition.models import DependencyGraph, TaskSpec


class ConflictType(str, Enum):
    """Types of conflicts that can occur during parallel execution."""

    FILE_WRITE = "file_write"  # Multiple tasks write same file
    IMPORT_COLLISION = "import_collision"  # Circular or duplicate imports
    NAMING_CONFLICT = "naming_conflict"  # Duplicate names in same scope
    TYPE_MISMATCH = "type_mismatch"  # Incompatible type definitions
    DEPENDENCY_VIOLATION = "dependency_violation"  # Missing dependency
    RESOURCE_CONTENTION = "resource_contention"  # Same external resource


class ConflictSeverity(str, Enum):
    """Severity levels for detected conflicts."""

    CRITICAL = "critical"  # Must be resolved before execution
    HIGH = "high"  # Should be resolved, may cause failures
    MEDIUM = "medium"  # May cause issues, can often auto-resolve
    LOW = "low"  # Minor issues, informational


@dataclass
class Conflict:
    """Represents a detected conflict."""

    id: str
    conflict_type: ConflictType
    severity: ConflictSeverity
    description: str
    affected_tasks: list[str]
    affected_files: list[str] = field(default_factory=list)
    suggested_resolution: str | None = None
    auto_resolvable: bool = False
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert conflict to dictionary."""
        return {
            "id": self.id,
            "type": self.conflict_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "affected_tasks": self.affected_tasks,
            "affected_files": self.affected_files,
            "suggested_resolution": self.suggested_resolution,
            "auto_resolvable": self.auto_resolvable,
            "metadata": self.metadata,
        }


@dataclass
class ConflictReport:
    """Complete conflict analysis report."""

    total_conflicts: int = 0
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    conflicts: list[Conflict] = field(default_factory=list)
    can_proceed: bool = True
    warnings: list[str] = field(default_factory=list)

    def add_conflict(self, conflict: Conflict):
        """Add a conflict to the report."""
        self.conflicts.append(conflict)
        self.total_conflicts += 1

        if conflict.severity == ConflictSeverity.CRITICAL:
            self.critical_count += 1
            self.can_proceed = False
        elif conflict.severity == ConflictSeverity.HIGH:
            self.high_count += 1
        elif conflict.severity == ConflictSeverity.MEDIUM:
            self.medium_count += 1
        else:
            self.low_count += 1

    def get_by_type(self, conflict_type: ConflictType) -> list[Conflict]:
        """Get conflicts by type."""
        return [c for c in self.conflicts if c.conflict_type == conflict_type]

    def get_auto_resolvable(self) -> list[Conflict]:
        """Get auto-resolvable conflicts."""
        return [c for c in self.conflicts if c.auto_resolvable]

    def get_by_severity(self, severity: ConflictSeverity) -> list[Conflict]:
        """Get conflicts by severity."""
        return [c for c in self.conflicts if c.severity == severity]

    def to_dict(self) -> dict:
        """Convert report to dictionary."""
        return {
            "total_conflicts": self.total_conflicts,
            "critical_count": self.critical_count,
            "high_count": self.high_count,
            "medium_count": self.medium_count,
            "low_count": self.low_count,
            "can_proceed": self.can_proceed,
            "warnings": self.warnings,
            "conflicts": [c.to_dict() for c in self.conflicts],
        }


class ConflictDetector:
    """
    Detects conflicts before parallel task execution.

    Usage:
        detector = ConflictDetector()
        report = detector.analyze(tasks, dependency_graph)

        if not report.can_proceed:
            # Handle critical conflicts
            pass
    """

    def __init__(self):
        self._conflict_counter = 0

    def analyze(
        self,
        tasks: list[TaskSpec],
        dependency_graph: DependencyGraph | None = None,
    ) -> ConflictReport:
        """
        Perform comprehensive conflict analysis on task batch.

        Args:
            tasks: List of tasks to analyze
            dependency_graph: Optional dependency graph for deeper analysis

        Returns:
            ConflictReport with all detected conflicts
        """
        logger.info(f"Analyzing {len(tasks)} tasks for conflicts")
        report = ConflictReport()

        # Run all detection methods
        self._detect_file_write_conflicts(tasks, report)
        self._detect_import_collisions(tasks, report)
        self._detect_naming_conflicts(tasks, report)
        self._detect_type_mismatches(tasks, report)

        if dependency_graph:
            self._detect_dependency_violations(tasks, dependency_graph, report)

        self._detect_resource_contention(tasks, report)

        logger.info(
            f"Conflict analysis complete: {report.total_conflicts} conflicts "
            f"({report.critical_count} critical, {report.high_count} high, "
            f"{report.medium_count} medium, {report.low_count} low)"
        )

        return report

    def analyze_wave(
        self,
        wave_tasks: list[TaskSpec],
        completed_tasks: set[str],
        context: dict,
    ) -> ConflictReport:
        """
        Analyze a specific execution wave for conflicts.

        More focused analysis for tasks about to execute in parallel.
        """
        logger.info(f"Analyzing wave with {len(wave_tasks)} tasks")
        report = ConflictReport()

        # Wave-specific analysis
        self._detect_file_write_conflicts(wave_tasks, report)
        self._detect_shared_state_conflicts(wave_tasks, context, report)
        self._detect_output_overlap(wave_tasks, report)

        return report

    def _generate_conflict_id(self) -> str:
        """Generate unique conflict identifier."""
        self._conflict_counter += 1
        return f"CONFLICT-{self._conflict_counter:04d}"

    def _detect_file_write_conflicts(
        self,
        tasks: list[TaskSpec],
        report: ConflictReport,
    ):
        """
        Detect when multiple tasks write to the same file.

        This is CRITICAL for parallel execution - two tasks writing
        the same file will corrupt output.
        """
        logger.debug("Checking for file write conflicts")

        # Build file -> tasks mapping
        file_writers: dict[str, list[str]] = {}

        for task in tasks:
            for filepath in task.files_to_create:
                normalized = self._normalize_path(filepath)
                if normalized not in file_writers:
                    file_writers[normalized] = []
                file_writers[normalized].append(task.id)

            for filepath in task.files_to_modify:
                normalized = self._normalize_path(filepath)
                if normalized not in file_writers:
                    file_writers[normalized] = []
                file_writers[normalized].append(task.id)

        # Find conflicts (multiple writers)
        for filepath, task_ids in file_writers.items():
            if len(task_ids) > 1:
                # Check if tasks are in same wave (parallel conflict)
                # vs sequential (acceptable)
                waves = self._get_task_waves(task_ids, tasks)

                if len(set(waves)) < len(waves):
                    # Same wave = parallel conflict
                    severity = ConflictSeverity.CRITICAL
                    auto_resolvable = False
                else:
                    # Different waves = sequential, might be intentional
                    severity = ConflictSeverity.MEDIUM
                    auto_resolvable = True

                conflict = Conflict(
                    id=self._generate_conflict_id(),
                    conflict_type=ConflictType.FILE_WRITE,
                    severity=severity,
                    description=f"Multiple tasks write to '{filepath}'",
                    affected_tasks=task_ids,
                    affected_files=[filepath],
                    suggested_resolution="Merge file contents or serialize task execution",
                    auto_resolvable=auto_resolvable,
                    metadata={"waves": waves},
                )
                report.add_conflict(conflict)

    def _detect_import_collisions(
        self,
        tasks: list[TaskSpec],
        report: ConflictReport,
    ):
        """
        Detect import-related conflicts:
        - Circular imports between generated files
        - Duplicate import statements
        - Conflicting re-exports
        """
        logger.debug("Checking for import collisions")

        # Build import graph from task descriptions and file patterns
        imports: dict[str, set[str]] = {}  # file -> imports
        exports: dict[str, set[str]] = {}  # file -> exports

        for task in tasks:
            task_imports, task_exports = self._extract_import_patterns(task)

            for filepath in task.files_to_create:
                normalized = self._normalize_path(filepath)
                imports[normalized] = task_imports
                exports[normalized] = task_exports

        # Detect circular imports
        checked_pairs: set[tuple[str, str]] = set()
        for filepath, file_imports in imports.items():
            for imported in file_imports:
                if imported in imports:
                    # Check if imported file imports this file
                    pair = tuple(sorted([filepath, imported]))
                    if pair not in checked_pairs and filepath in imports.get(imported, set()):
                        checked_pairs.add(pair)
                        conflict = Conflict(
                            id=self._generate_conflict_id(),
                            conflict_type=ConflictType.IMPORT_COLLISION,
                            severity=ConflictSeverity.HIGH,
                            description=f"Circular import between '{filepath}' and '{imported}'",
                            affected_tasks=self._find_tasks_for_files([filepath, imported], tasks),
                            affected_files=[filepath, imported],
                            suggested_resolution="Restructure imports or use lazy loading",
                            auto_resolvable=False,
                        )
                        report.add_conflict(conflict)

    def _detect_naming_conflicts(
        self,
        tasks: list[TaskSpec],
        report: ConflictReport,
    ):
        """
        Detect naming conflicts:
        - Duplicate component names
        - Duplicate function names in same module
        - Conflicting type definitions
        """
        logger.debug("Checking for naming conflicts")

        # Extract names from task descriptions
        component_names: dict[str, list[str]] = {}  # name -> task_ids
        function_names: dict[str, list[str]] = {}
        type_names: dict[str, list[str]] = {}

        for task in tasks:
            names = self._extract_names_from_task(task)

            for name in names.get("components", []):
                if name not in component_names:
                    component_names[name] = []
                component_names[name].append(task.id)

            for name in names.get("functions", []):
                if name not in function_names:
                    function_names[name] = []
                function_names[name].append(task.id)

            for name in names.get("types", []):
                if name not in type_names:
                    type_names[name] = []
                type_names[name].append(task.id)

        # Check for duplicates
        for name, task_ids in component_names.items():
            if len(task_ids) > 1:
                conflict = Conflict(
                    id=self._generate_conflict_id(),
                    conflict_type=ConflictType.NAMING_CONFLICT,
                    severity=ConflictSeverity.HIGH,
                    description=f"Duplicate component name '{name}'",
                    affected_tasks=task_ids,
                    suggested_resolution=f"Rename one component (e.g., '{name}V2' or add prefix)",
                    auto_resolvable=True,
                    metadata={"name_type": "component", "name": name},
                )
                report.add_conflict(conflict)

        for name, task_ids in type_names.items():
            if len(task_ids) > 1:
                conflict = Conflict(
                    id=self._generate_conflict_id(),
                    conflict_type=ConflictType.NAMING_CONFLICT,
                    severity=ConflictSeverity.MEDIUM,
                    description=f"Duplicate type name '{name}'",
                    affected_tasks=task_ids,
                    suggested_resolution="Consolidate type definitions",
                    auto_resolvable=True,
                    metadata={"name_type": "type", "name": name},
                )
                report.add_conflict(conflict)

    def _detect_type_mismatches(
        self,
        tasks: list[TaskSpec],
        report: ConflictReport,
    ):
        """
        Detect type definition mismatches:
        - Same type defined differently in multiple tasks
        - Incompatible interface definitions
        """
        logger.debug("Checking for type mismatches")

        # This requires deeper analysis of task prompts/outputs
        # For now, flag tasks that both define types in the same domain

        type_domains: dict[str, list[str]] = {}  # domain -> task_ids

        for task in tasks:
            domains = self._extract_type_domains(task)
            for domain in domains:
                if domain not in type_domains:
                    type_domains[domain] = []
                type_domains[domain].append(task.id)

        for domain, task_ids in type_domains.items():
            if len(task_ids) > 1:
                conflict = Conflict(
                    id=self._generate_conflict_id(),
                    conflict_type=ConflictType.TYPE_MISMATCH,
                    severity=ConflictSeverity.MEDIUM,
                    description=f"Multiple tasks define types in '{domain}' domain",
                    affected_tasks=task_ids,
                    suggested_resolution="Consolidate type definitions in shared types file",
                    auto_resolvable=True,
                    metadata={"domain": domain},
                )
                report.add_conflict(conflict)

    def _detect_dependency_violations(
        self,
        tasks: list[TaskSpec],
        graph: DependencyGraph,
        report: ConflictReport,
    ):
        """
        Detect dependency violations:
        - Task scheduled before its dependencies complete
        - Missing required context
        """
        logger.debug("Checking for dependency violations")

        for task in tasks:
            for dep_id in task.dependencies:
                dep_task = graph.get_task(dep_id)

                if dep_task is None:
                    conflict = Conflict(
                        id=self._generate_conflict_id(),
                        conflict_type=ConflictType.DEPENDENCY_VIOLATION,
                        severity=ConflictSeverity.CRITICAL,
                        description=f"Task '{task.id}' depends on non-existent task '{dep_id}'",
                        affected_tasks=[task.id],
                        suggested_resolution="Fix dependency reference or remove dependency",
                        auto_resolvable=False,
                    )
                    report.add_conflict(conflict)

                elif task.wave_number is not None and dep_task.wave_number is not None:
                    if task.wave_number <= dep_task.wave_number:
                        conflict = Conflict(
                            id=self._generate_conflict_id(),
                            conflict_type=ConflictType.DEPENDENCY_VIOLATION,
                            severity=ConflictSeverity.CRITICAL,
                            description=(
                                f"Task '{task.id}' (wave {task.wave_number}) scheduled "
                                f"before dependency '{dep_id}' (wave {dep_task.wave_number})"
                            ),
                            affected_tasks=[task.id, dep_id],
                            suggested_resolution="Reorder waves or fix dependency graph",
                            auto_resolvable=False,
                        )
                        report.add_conflict(conflict)

    def _detect_resource_contention(
        self,
        tasks: list[TaskSpec],
        report: ConflictReport,
    ):
        """
        Detect resource contention:
        - Multiple tasks accessing same API endpoint
        - Database table locks
        - External service rate limits
        """
        logger.debug("Checking for resource contention")

        # Extract external resources from tasks
        resources: dict[str, list[str]] = {}

        for task in tasks:
            task_resources = self._extract_resources(task)
            for resource in task_resources:
                if resource not in resources:
                    resources[resource] = []
                resources[resource].append(task.id)

        for resource, task_ids in resources.items():
            if len(task_ids) > 1:
                conflict = Conflict(
                    id=self._generate_conflict_id(),
                    conflict_type=ConflictType.RESOURCE_CONTENTION,
                    severity=ConflictSeverity.LOW,
                    description=f"Multiple tasks access resource '{resource}'",
                    affected_tasks=task_ids,
                    suggested_resolution="Consider rate limiting or sequential access",
                    auto_resolvable=True,
                    metadata={"resource": resource},
                )
                report.add_conflict(conflict)

    def _detect_shared_state_conflicts(
        self,
        tasks: list[TaskSpec],
        context: dict,
        report: ConflictReport,
    ):
        """Detect conflicts with shared state/context."""
        logger.debug("Checking for shared state conflicts")

        # Check if multiple tasks modify same context keys
        context_modifiers: dict[str, list[str]] = {}

        for task in tasks:
            modified_keys = self._get_modified_context_keys(task)
            for key in modified_keys:
                if key not in context_modifiers:
                    context_modifiers[key] = []
                context_modifiers[key].append(task.id)

        for key, task_ids in context_modifiers.items():
            if len(task_ids) > 1:
                conflict = Conflict(
                    id=self._generate_conflict_id(),
                    conflict_type=ConflictType.RESOURCE_CONTENTION,
                    severity=ConflictSeverity.MEDIUM,
                    description=f"Multiple tasks modify context key '{key}'",
                    affected_tasks=task_ids,
                    suggested_resolution="Use task-specific context keys or merge strategy",
                    auto_resolvable=True,
                    metadata={"context_key": key},
                )
                report.add_conflict(conflict)

    def _detect_output_overlap(
        self,
        tasks: list[TaskSpec],
        report: ConflictReport,
    ):
        """Detect overlapping output directories or patterns."""
        logger.debug("Checking for output overlap")

        output_dirs: dict[str, list[str]] = {}

        for task in tasks:
            for filepath in task.files_to_create:
                dir_path = str(Path(filepath).parent)
                if dir_path not in output_dirs:
                    output_dirs[dir_path] = []
                output_dirs[dir_path].append(task.id)

        # Flag directories with many writers (potential index.ts conflicts)
        for dir_path, task_ids in output_dirs.items():
            if len(task_ids) > 3:  # Threshold for concern
                report.warnings.append(
                    f"Directory '{dir_path}' has {len(task_ids)} tasks writing to it - "
                    "watch for index file conflicts"
                )

    # ═══════════════════════════════════════════════════════════════
    # HELPER METHODS
    # ═══════════════════════════════════════════════════════════════

    def _normalize_path(self, filepath: str) -> str:
        """Normalize file path for comparison."""
        if not filepath:
            return ""
        # Use relative path normalization for comparison
        return str(Path(filepath).as_posix())

    def _get_task_waves(
        self,
        task_ids: list[str],
        tasks: list[TaskSpec],
    ) -> list[int]:
        """Get wave numbers for given task IDs."""
        task_map = {t.id: t for t in tasks}
        return [
            (
                task_map[tid].wave_number
                if tid in task_map and task_map[tid].wave_number is not None
                else -1
            )
            for tid in task_ids
        ]

    def _extract_import_patterns(
        self,
        task: TaskSpec,
    ) -> tuple[set[str], set[str]]:
        """Extract likely imports and exports from task description."""
        imports: set[str] = set()
        exports: set[str] = set()

        # Pattern matching for common import statements
        import_patterns = [
            r"import\s+.*\s+from\s+['\"]([^'\"]+)['\"]",
            r"require\(['\"]([^'\"]+)['\"]\)",
            r"from\s+(\S+)\s+import",
        ]

        text = f"{task.description} {task.title}"

        for pattern in import_patterns:
            matches = re.findall(pattern, text)
            imports.update(matches)

        # Export patterns
        export_patterns = [
            r"export\s+(?:default\s+)?(?:class|function|const|interface)\s+(\w+)",
        ]

        for pattern in export_patterns:
            matches = re.findall(pattern, text)
            exports.update(matches)

        return imports, exports

    def _extract_names_from_task(self, task: TaskSpec) -> dict[str, list[str]]:
        """Extract component, function, and type names from task."""
        names: dict[str, list[str]] = {
            "components": [],
            "functions": [],
            "types": [],
        }

        text = f"{task.description} {task.title}"

        # Component patterns (React-style)
        component_pattern = r"(?:component|Component)\s+(\w+)|(\w+)(?:Component|Page|Layout)"
        for match in re.findall(component_pattern, text):
            name = match[0] or match[1]
            if name and name[0].isupper():
                names["components"].append(name)

        # Function patterns
        function_pattern = r"(?:function|def|const)\s+(\w+)"
        for match in re.findall(function_pattern, text):
            if match and not match[0].isupper():
                names["functions"].append(match)

        # Type patterns
        type_pattern = r"(?:type|interface|Type|Interface)\s+(\w+)"
        for match in re.findall(type_pattern, text):
            if match:
                names["types"].append(match)

        return names

    def _extract_type_domains(self, task: TaskSpec) -> list[str]:
        """Extract type domains from task (e.g., 'user', 'product', 'auth')."""
        domains = []

        text = f"{task.description} {task.title}".lower()

        common_domains = [
            "user",
            "auth",
            "product",
            "order",
            "cart",
            "payment",
            "api",
            "database",
            "config",
            "utils",
            "common",
            "shared",
        ]

        for domain in common_domains:
            if domain in text:
                domains.append(domain)

        return domains

    def _extract_resources(self, task: TaskSpec) -> list[str]:
        """Extract external resources referenced by task."""
        resources = []

        text = f"{task.description} {task.title}"

        # API endpoints
        api_pattern = r"(?:api|endpoint|route)[:\s]+([^\s,]+)"
        resources.extend(re.findall(api_pattern, text, re.IGNORECASE))

        # Database tables
        table_pattern = r"(?:table|collection)[:\s]+(\w+)"
        resources.extend(re.findall(table_pattern, text, re.IGNORECASE))

        # External services
        service_pattern = r"(?:service|integration)[:\s]+(\w+)"
        resources.extend(re.findall(service_pattern, text, re.IGNORECASE))

        return list(set(resources))

    def _find_tasks_for_files(
        self,
        files: list[str],
        tasks: list[TaskSpec],
    ) -> list[str]:
        """Find task IDs that create/modify given files."""
        task_ids = []
        normalized_files = {self._normalize_path(f) for f in files}

        for task in tasks:
            task_files = {
                self._normalize_path(f) for f in task.files_to_create + task.files_to_modify
            }
            if task_files & normalized_files:
                task_ids.append(task.id)

        return task_ids

    def _get_modified_context_keys(self, task: TaskSpec) -> list[str]:
        """Extract context keys that task might modify."""
        keys = []

        text = f"{task.description} {task.title}"

        # Look for context update patterns
        patterns = [
            r"update[s]?\s+(?:context|state)[:\s]+(\w+)",
            r"set[s]?\s+(\w+)\s+(?:in|to)",
            r"context\.(\w+)\s*=",
        ]

        for pattern in patterns:
            keys.extend(re.findall(pattern, text, re.IGNORECASE))

        return list(set(keys))


# Convenience functions for backward compatibility
async def detect_conflicts(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convenience function to detect conflicts from task results.

    Args:
        results: Task results to analyze.

    Returns:
        List of detected conflicts as dictionaries.
    """
    # Convert task results to TaskSpec-like objects for analysis
    # This maintains backward compatibility with the old interface
    detector = ConflictDetector()

    # Build file -> sessions mapping (old behavior)
    from collections import defaultdict

    file_sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for result in results:
        session_id = result.get("session_id", "")
        files_modified = result.get("files_modified", [])

        for file_path in files_modified:
            file_sessions[file_path].append(
                {
                    "session_id": session_id,
                    "task_id": result.get("task_id", ""),
                    "output": result.get("output", ""),
                }
            )

    # Find conflicts
    conflicts: list[dict[str, Any]] = []

    for file_path, sessions in file_sessions.items():
        if len(sessions) > 1:
            conflicts.append(
                {
                    "file_path": file_path,
                    "session_a_id": sessions[0]["session_id"],
                    "session_b_id": (
                        sessions[1]["session_id"]
                        if len(sessions) > 1
                        else sessions[0]["session_id"]
                    ),
                    "conflict_type": (
                        "semantic"
                        if any(ext in file_path for ext in [".py", ".ts", ".js"])
                        else "merge"
                    ),
                    "description": f"File {file_path} modified by {len(sessions)} sessions",
                    "sessions": sessions,
                }
            )

    logger.info(f"Detected {len(conflicts)} conflicts")
    return conflicts
