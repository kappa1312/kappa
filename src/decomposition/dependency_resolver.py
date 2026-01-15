"""Dependency resolver - builds task graphs and assigns execution waves.

This module provides sophisticated dependency resolution for task graphs,
including cycle detection, topological sorting, wave assignment, and
file conflict detection.
"""

from collections import defaultdict, deque
from typing import TypeVar

from loguru import logger

from src.decomposition.models import (
    DependencyGraph,
    FileConflict,
    Task,
    TaskSpec,
)

# Type variable for task types
TaskType = TypeVar("TaskType", Task, TaskSpec)


class DependencyResolver:
    """
    Resolve task dependencies and organize into execution waves.

    Uses topological sorting to determine execution order and
    groups independent tasks into waves for parallel execution.
    Supports both legacy Task objects and new TaskSpec objects.

    Example:
        >>> resolver = DependencyResolver()
        >>> graph = resolver.resolve(tasks)
        >>> graph.total_waves
        3
        >>> graph.get_wave_tasks(0)
        [TaskSpec(...), TaskSpec(...)]
    """

    def __init__(self, tasks: list[TaskSpec] | None = None) -> None:
        """
        Initialize the resolver.

        Args:
            tasks: Optional list of TaskSpec objects to resolve.
        """
        self._graph = DependencyGraph()
        self._task_map: dict[str, TaskSpec] = {}
        self._conflicts: list[FileConflict] = []

        if tasks:
            for task in tasks:
                self._graph.add_task(task)
                self._task_map[task.id] = task

    def resolve(self, tasks: list[TaskSpec] | None = None) -> DependencyGraph:
        """
        Resolve dependencies and build complete dependency graph.

        Args:
            tasks: Optional list of tasks (uses constructor tasks if not provided).

        Returns:
            DependencyGraph with nodes, edges, and computed waves.

        Raises:
            ValueError: If circular dependencies are detected.

        Example:
            >>> resolver = DependencyResolver()
            >>> graph = resolver.resolve(tasks)
            >>> len(graph.waves)
            4
        """
        if tasks:
            self._graph = DependencyGraph()
            self._task_map = {}
            for task in tasks:
                self._graph.add_task(task)
                self._task_map[task.id] = task

        logger.info(f"Resolving dependencies for {len(self._task_map)} tasks")

        # Validate no cycles
        self._validate_no_cycles()

        # Calculate waves
        self._calculate_waves()

        # Detect file conflicts
        self._conflicts = self._detect_file_conflicts()

        if self._conflicts:
            logger.warning(f"Detected {len(self._conflicts)} potential file conflicts")

        logger.info(
            f"Resolved into {self._graph.total_waves} waves with "
            f"{len(self._conflicts)} conflicts"
        )

        return self._graph

    def get_graph(self) -> DependencyGraph:
        """Get the resolved dependency graph."""
        return self._graph

    def get_conflicts(self) -> list[FileConflict]:
        """Get detected file conflicts."""
        return self._conflicts

    # =========================================================================
    # CYCLE DETECTION
    # =========================================================================

    def _validate_no_cycles(self) -> None:
        """
        Validate that the dependency graph has no cycles.

        Raises:
            ValueError: If circular dependencies are detected.
        """
        cycles = self.detect_cycles(self._graph.edges)
        if cycles:
            cycle_str = " -> ".join(cycles[0])
            raise ValueError(f"Circular dependency detected: {cycle_str}")

    def detect_cycles(
        self,
        graph: dict[str, list[str]],
    ) -> list[list[str]] | None:
        """
        Detect cycles in the dependency graph using DFS.

        Args:
            graph: Dependency graph (task_id -> [dependency_ids]).

        Returns:
            List of cycle paths if found, None otherwise.

        Example:
            >>> cycles = resolver.detect_cycles({"a": ["b"], "b": ["a"]})
            >>> cycles[0]
            ["a", "b", "a"]
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        colors: dict[str, int] = {node: WHITE for node in graph}
        cycles: list[list[str]] = []

        def dfs(node: str, path: list[str]) -> bool:
            colors[node] = GRAY
            path.append(node)

            for neighbor in graph.get(node, []):
                if neighbor not in colors:
                    continue  # Skip invalid dependencies
                if colors[neighbor] == GRAY:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:] + [neighbor])
                    return True
                if colors[neighbor] == WHITE and dfs(neighbor, path):
                    return True

            path.pop()
            colors[node] = BLACK
            return False

        for node in graph:
            if colors[node] == WHITE:
                dfs(node, [])

        return cycles if cycles else None

    # =========================================================================
    # WAVE CALCULATION
    # =========================================================================

    def _calculate_waves(self) -> None:
        """Calculate execution waves using topological levels."""
        waves: list[list[str]] = []
        assigned: set[str] = set()
        task_ids = set(self._task_map.keys())

        logger.debug(f"Calculating waves for {len(task_ids)} tasks")

        # Keep assigning waves until all tasks are assigned
        iteration = 0
        max_iterations = len(task_ids) + 1

        while len(assigned) < len(task_ids) and iteration < max_iterations:
            iteration += 1
            wave: list[str] = []

            for task_id in task_ids:
                if task_id in assigned:
                    continue

                # Check if all dependencies are satisfied
                deps = self._graph.edges.get(task_id, [])
                valid_deps = [d for d in deps if d in task_ids]

                if all(d in assigned for d in valid_deps):
                    wave.append(task_id)

            if not wave:
                # Deadlock - remaining tasks have unsatisfiable dependencies
                remaining = task_ids - assigned
                logger.error(f"Cannot assign remaining tasks: {remaining}")
                # Force-add remaining tasks to a final wave
                wave = list(remaining)

            # Sort wave for deterministic ordering
            wave.sort()
            waves.append(wave)
            assigned.update(wave)

            # Update task wave numbers
            wave_num = len(waves) - 1
            for task_id in wave:
                if task_id in self._task_map:
                    self._task_map[task_id].wave_number = wave_num

        self._graph.waves = waves

        for i, wave in enumerate(waves):
            logger.debug(f"Wave {i}: {len(wave)} tasks")

    def assign_waves(
        self,
        tasks: list[Task | TaskSpec],
        graph: dict[str, list[str]],
    ) -> list[list[str]]:
        """
        Assign tasks to execution waves (legacy interface).

        Tasks in the same wave can execute in parallel.
        Tasks in later waves depend on earlier waves.

        Args:
            tasks: List of Task or TaskSpec objects.
            graph: Dependency graph.

        Returns:
            List of waves, where each wave is a list of task_ids.

        Example:
            >>> waves = resolver.assign_waves(tasks, graph)
            >>> waves[0]  # First wave: tasks with no dependencies
            ["setup", "init"]
        """
        logger.debug("Assigning tasks to execution waves")

        task_map = {self._get_task_id(t): t for t in tasks}
        waves: list[list[str]] = []
        assigned: set[str] = set()

        # Keep assigning waves until all tasks are assigned
        while len(assigned) < len(tasks):
            wave: list[str] = []

            for task in tasks:
                task_id = self._get_task_id(task)
                if task_id in assigned:
                    continue

                # Check if all dependencies are satisfied
                deps = graph.get(task_id, [])
                if all(d in assigned for d in deps):
                    wave.append(task_id)

            if not wave:
                # Deadlock - remaining tasks have unsatisfiable dependencies
                remaining = [
                    self._get_task_id(t) for t in tasks if self._get_task_id(t) not in assigned
                ]
                logger.error(f"Cannot assign remaining tasks: {remaining}")
                break

            wave.sort()  # Deterministic ordering
            waves.append(wave)
            assigned.update(wave)

            # Update task wave numbers
            for task_id in wave:
                task = task_map[task_id]
                if isinstance(task, TaskSpec):
                    task.wave_number = len(waves) - 1
                else:
                    task.wave = len(waves) - 1

        logger.info(f"Organized {len(tasks)} tasks into {len(waves)} waves")

        return waves

    # =========================================================================
    # CONFLICT DETECTION
    # =========================================================================

    def _detect_file_conflicts(self) -> list[FileConflict]:
        """
        Detect file write conflicts within execution waves.

        Tasks in the same wave might write to the same file,
        causing conflicts during parallel execution.

        Returns:
            List of FileConflict objects describing conflicts.
        """
        conflicts: list[FileConflict] = []

        for wave_num, wave_task_ids in enumerate(self._graph.waves):
            # Collect files written by each task in this wave
            file_writers: dict[str, list[str]] = defaultdict(list)

            for task_id in wave_task_ids:
                task = self._task_map.get(task_id)
                if not task:
                    continue

                # Check files_to_create
                for file_path in task.files_to_create:
                    file_writers[file_path].append(task_id)

                # Check files_to_modify
                for file_path in task.files_to_modify:
                    file_writers[file_path].append(task_id)

            # Find files with multiple writers
            for file_path, writers in file_writers.items():
                if len(writers) > 1:
                    conflict = FileConflict(
                        conflict_type="file_write",
                        file_path=file_path,
                        task_ids=writers,
                        wave_number=wave_num,
                        description=(
                            f"Multiple tasks in wave {wave_num} write to {file_path}: "
                            f"{', '.join(writers)}"
                        ),
                    )
                    conflicts.append(conflict)
                    logger.warning(
                        f"File conflict detected: {file_path} written by "
                        f"{len(writers)} tasks in wave {wave_num}"
                    )

        return conflicts

    def detect_conflicts(self) -> list[FileConflict]:
        """
        Detect file write conflicts (public interface).

        Returns:
            List of FileConflict objects.
        """
        return self._detect_file_conflicts()

    # =========================================================================
    # GRAPH BUILDING
    # =========================================================================

    def build_graph(self, tasks: list[Task | TaskSpec]) -> dict[str, list[str]]:
        """
        Build a dependency graph from tasks (legacy interface).

        Args:
            tasks: List of Task or TaskSpec objects with dependencies.

        Returns:
            Dictionary mapping task_id -> list of dependency task_ids.

        Example:
            >>> resolver = DependencyResolver()
            >>> graph = resolver.build_graph(tasks)
            >>> graph["b"]
            ["a"]
        """
        logger.debug(f"Building dependency graph for {len(tasks)} tasks")

        graph: dict[str, list[str]] = {}
        task_ids = {self._get_task_id(t) for t in tasks}

        for task in tasks:
            task_id = self._get_task_id(task)
            deps = self._get_task_deps(task)

            # Filter to only include dependencies that exist in our task set
            valid_deps = [d for d in deps if d in task_ids]

            if len(valid_deps) != len(deps):
                invalid = set(deps) - set(valid_deps)
                logger.warning(f"Task {task_id} has invalid dependencies: {invalid}")

            graph[task_id] = valid_deps

        return graph

    # =========================================================================
    # TOPOLOGICAL SORT
    # =========================================================================

    def topological_sort(
        self,
        tasks: list[Task | TaskSpec],
        graph: dict[str, list[str]],
    ) -> list[str]:
        """
        Perform topological sort on tasks.

        Args:
            tasks: List of Task or TaskSpec objects.
            graph: Dependency graph.

        Returns:
            List of task_ids in topological order.

        Raises:
            ValueError: If circular dependencies are detected.
        """
        # Check for cycles first
        cycles = self.detect_cycles(graph)
        if cycles:
            cycle_str = " -> ".join(cycles[0])
            raise ValueError(f"Circular dependency detected: {cycle_str}")

        # Calculate in-degree (number of dependencies)
        in_degree: dict[str, int] = {}
        for task in tasks:
            task_id = self._get_task_id(task)
            in_degree[task_id] = len(graph.get(task_id, []))

        # Start with tasks that have no dependencies
        queue: deque[str] = deque()
        for task_id, degree in in_degree.items():
            if degree == 0:
                queue.append(task_id)

        sorted_tasks: list[str] = []

        while queue:
            node = queue.popleft()
            sorted_tasks.append(node)

            # Find tasks that depend on this node and decrement their in-degree
            for task_id, deps in graph.items():
                if node in deps and task_id not in sorted_tasks:
                    in_degree[task_id] -= 1
                    if in_degree[task_id] == 0:
                        queue.append(task_id)

        if len(sorted_tasks) != len(tasks):
            missing = set(self._get_task_id(t) for t in tasks) - set(sorted_tasks)
            logger.warning(f"Some tasks not sorted: {missing}")

        return sorted_tasks

    def get_execution_order(self) -> list[str]:
        """
        Get flat list of task IDs in execution order.

        Returns:
            List of task IDs in wave-based order.
        """
        order: list[str] = []
        for wave in self._graph.waves:
            order.extend(sorted(wave))
        return order

    # =========================================================================
    # CRITICAL PATH
    # =========================================================================

    def get_critical_path(
        self,
        tasks: list[Task | TaskSpec],
        graph: dict[str, list[str]],
    ) -> list[str]:
        """
        Find the critical path through the task graph.

        The critical path is the longest chain of dependencies,
        which determines the minimum total execution time.

        Args:
            tasks: List of Task or TaskSpec objects.
            graph: Dependency graph.

        Returns:
            List of task_ids forming the critical path.
        """
        task_map = {self._get_task_id(t): t for t in tasks}

        # Calculate depth for each task
        depths: dict[str, int] = {}

        def get_depth(task_id: str) -> int:
            if task_id in depths:
                return depths[task_id]

            deps = graph.get(task_id, [])
            if not deps:
                depths[task_id] = 0
            else:
                valid_deps = [d for d in deps if d in task_map]
                if valid_deps:
                    depths[task_id] = 1 + max(get_depth(d) for d in valid_deps)
                else:
                    depths[task_id] = 0

            return depths[task_id]

        for task in tasks:
            get_depth(self._get_task_id(task))

        # Find task with maximum depth
        if not depths:
            return []

        max_task = max(depths, key=lambda k: depths[k])

        # Trace back the critical path
        path = [max_task]
        current = max_task

        while graph.get(current, []):
            deps = [d for d in graph[current] if d in depths]
            if not deps:
                break
            # Pick the dependency with highest depth
            next_task = max(deps, key=lambda d: depths.get(d, 0))
            path.append(next_task)
            current = next_task

        return list(reversed(path))

    def calculate_critical_path(self) -> list[str]:
        """
        Calculate critical path for the resolved graph.

        Returns:
            List of task IDs on the critical path.
        """
        tasks = list(self._task_map.values())
        return self.get_critical_path(tasks, self._graph.edges)

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_wave(self, wave_number: int) -> list[TaskSpec]:
        """
        Get tasks in a specific wave.

        Args:
            wave_number: Wave index (0-based).

        Returns:
            List of TaskSpec objects in that wave.
        """
        return self._graph.get_wave_tasks(wave_number)

    def get_dependents(self, task_id: str) -> list[str]:
        """
        Get tasks that depend on the given task.

        Args:
            task_id: Task identifier.

        Returns:
            List of task IDs that depend on this task.
        """
        return self._graph.get_dependents(task_id)

    def is_ready(self, task_id: str, completed: set[str]) -> bool:
        """
        Check if a task is ready to execute.

        Args:
            task_id: Task identifier.
            completed: Set of completed task IDs.

        Returns:
            True if all dependencies are satisfied.
        """
        return self._graph.is_ready(task_id, completed)

    @staticmethod
    def _get_task_id(task: Task | TaskSpec) -> str:
        """Get task ID from either Task or TaskSpec."""
        return task.id

    @staticmethod
    def _get_task_deps(task: Task | TaskSpec) -> list[str]:
        """Get task dependencies from either Task or TaskSpec."""
        return task.dependencies


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def resolve_dependencies(
    tasks: list[Task | TaskSpec],
) -> tuple[dict[str, list[str]], list[list[str]]]:
    """
    Convenience function to resolve dependencies.

    Args:
        tasks: List of Task or TaskSpec objects.

    Returns:
        Tuple of (dependency_graph, waves).

    Example:
        >>> graph, waves = resolve_dependencies(tasks)
        >>> len(waves)
        3
    """
    resolver = DependencyResolver()
    graph = resolver.build_graph(tasks)
    waves = resolver.assign_waves(tasks, graph)
    return graph, waves


def resolve_to_graph(tasks: list[TaskSpec]) -> DependencyGraph:
    """
    Resolve tasks and return DependencyGraph model.

    Args:
        tasks: List of TaskSpec objects.

    Returns:
        DependencyGraph with computed waves.

    Example:
        >>> graph = resolve_to_graph(tasks)
        >>> graph.total_waves
        4
    """
    resolver = DependencyResolver(tasks)
    return resolver.resolve()


def detect_conflicts(tasks: list[TaskSpec]) -> list[FileConflict]:
    """
    Detect file conflicts in task list.

    Args:
        tasks: List of TaskSpec objects.

    Returns:
        List of FileConflict objects.

    Example:
        >>> conflicts = detect_conflicts(tasks)
        >>> for c in conflicts:
        ...     print(f"{c.file_path}: {c.task_ids}")
    """
    resolver = DependencyResolver(tasks)
    resolver.resolve()
    return resolver.get_conflicts()
