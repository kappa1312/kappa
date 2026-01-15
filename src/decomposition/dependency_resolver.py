"""Dependency resolver - builds task graphs and assigns execution waves."""

from collections import defaultdict, deque

from loguru import logger

from src.decomposition.models import Task


class DependencyResolver:
    """
    Resolve task dependencies and organize into execution waves.

    Uses topological sorting to determine execution order and
    groups independent tasks into waves for parallel execution.

    Example:
        >>> resolver = DependencyResolver()
        >>> graph = resolver.build_graph(tasks)
        >>> waves = resolver.assign_waves(tasks, graph)
        >>> len(waves)  # Number of parallel execution waves
        3
    """

    def build_graph(self, tasks: list[Task]) -> dict[str, list[str]]:
        """
        Build a dependency graph from tasks.

        Args:
            tasks: List of Task objects with dependencies.

        Returns:
            Dictionary mapping task_id -> list of dependency task_ids.

        Example:
            >>> resolver = DependencyResolver()
            >>> tasks = [Task(id="a", ...), Task(id="b", dependencies=["a"], ...)]
            >>> graph = resolver.build_graph(tasks)
            >>> graph["b"]
            ["a"]
        """
        logger.debug(f"Building dependency graph for {len(tasks)} tasks")

        graph: dict[str, list[str]] = {}
        task_ids = {t.id for t in tasks}

        for task in tasks:
            # Filter to only include dependencies that exist in our task set
            valid_deps = [d for d in task.dependencies if d in task_ids]

            if len(valid_deps) != len(task.dependencies):
                invalid = set(task.dependencies) - set(valid_deps)
                logger.warning(
                    f"Task {task.id} has invalid dependencies: {invalid}"
                )

            graph[task.id] = valid_deps

        return graph

    def detect_cycles(
        self,
        graph: dict[str, list[str]],
    ) -> list[list[str]] | None:
        """
        Detect cycles in the dependency graph.

        Args:
            graph: Dependency graph (task_id -> [dependency_ids]).

        Returns:
            List of cycle paths if found, None otherwise.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        colors: dict[str, int] = {node: WHITE for node in graph}
        cycles: list[list[str]] = []

        def dfs(node: str, path: list[str]) -> bool:
            colors[node] = GRAY
            path.append(node)

            for neighbor in graph.get(node, []):
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

    def topological_sort(
        self,
        tasks: list[Task],
        graph: dict[str, list[str]],
    ) -> list[str]:
        """
        Perform topological sort on tasks.

        Args:
            tasks: List of Task objects.
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

        # Kahn's algorithm
        in_degree: dict[str, int] = {t.id: 0 for t in tasks}

        for deps in graph.values():
            for dep in deps:
                if dep in in_degree:
                    in_degree[dep] += 1

        # Actually we need reverse: tasks with no dependencies first
        # Recalculate: in_degree = number of tasks depending on this
        in_degree = {t.id: len(graph.get(t.id, [])) for t in tasks}

        queue: deque[str] = deque()
        for task_id, degree in in_degree.items():
            if degree == 0:
                queue.append(task_id)

        sorted_tasks: list[str] = []

        while queue:
            node = queue.popleft()
            sorted_tasks.append(node)

            # Find tasks that depend on this node
            for task_id, deps in graph.items():
                if node in deps:
                    in_degree[task_id] -= 1
                    if in_degree[task_id] == 0 and task_id not in sorted_tasks:
                        queue.append(task_id)

        if len(sorted_tasks) != len(tasks):
            missing = set(t.id for t in tasks) - set(sorted_tasks)
            logger.warning(f"Some tasks not sorted: {missing}")

        return sorted_tasks

    def assign_waves(
        self,
        tasks: list[Task],
        graph: dict[str, list[str]],
    ) -> list[list[str]]:
        """
        Assign tasks to execution waves.

        Tasks in the same wave can execute in parallel.
        Tasks in later waves depend on earlier waves.

        Args:
            tasks: List of Task objects.
            graph: Dependency graph.

        Returns:
            List of waves, where each wave is a list of task_ids.

        Example:
            >>> waves = resolver.assign_waves(tasks, graph)
            >>> waves[0]  # First wave: tasks with no dependencies
            ["setup", "init"]
            >>> waves[1]  # Second wave: tasks depending on wave 0
            ["model_user", "model_product"]
        """
        logger.debug("Assigning tasks to execution waves")

        task_map = {t.id: t for t in tasks}
        waves: list[list[str]] = []
        assigned: set[str] = set()

        # Keep assigning waves until all tasks are assigned
        while len(assigned) < len(tasks):
            wave: list[str] = []

            for task in tasks:
                if task.id in assigned:
                    continue

                # Check if all dependencies are satisfied
                deps = graph.get(task.id, [])
                if all(d in assigned for d in deps):
                    wave.append(task.id)

            if not wave:
                # Deadlock - remaining tasks have unsatisfiable dependencies
                remaining = [t.id for t in tasks if t.id not in assigned]
                logger.error(f"Cannot assign remaining tasks: {remaining}")
                break

            waves.append(wave)
            assigned.update(wave)

            # Update task wave numbers
            for task_id in wave:
                task_map[task_id].wave = len(waves) - 1

        logger.info(f"Organized {len(tasks)} tasks into {len(waves)} waves")

        for i, wave in enumerate(waves):
            logger.debug(f"Wave {i}: {wave}")

        return waves

    def get_critical_path(
        self,
        tasks: list[Task],
        graph: dict[str, list[str]],
    ) -> list[str]:
        """
        Find the critical path through the task graph.

        The critical path is the longest chain of dependencies,
        which determines the minimum total execution time.

        Args:
            tasks: List of Task objects.
            graph: Dependency graph.

        Returns:
            List of task_ids forming the critical path.
        """
        task_map = {t.id: t for t in tasks}

        # Calculate depth for each task
        depths: dict[str, int] = {}

        def get_depth(task_id: str) -> int:
            if task_id in depths:
                return depths[task_id]

            deps = graph.get(task_id, [])
            if not deps:
                depths[task_id] = 0
            else:
                depths[task_id] = 1 + max(get_depth(d) for d in deps)

            return depths[task_id]

        for task in tasks:
            get_depth(task.id)

        # Find task with maximum depth
        if not depths:
            return []

        max_task = max(depths, key=depths.get)  # type: ignore

        # Trace back the critical path
        path = [max_task]
        current = max_task

        while graph.get(current, []):
            deps = graph[current]
            # Pick the dependency with highest depth
            next_task = max(deps, key=lambda d: depths.get(d, 0))
            path.append(next_task)
            current = next_task

        return list(reversed(path))


def resolve_dependencies(tasks: list[Task]) -> tuple[dict[str, list[str]], list[list[str]]]:
    """Convenience function to resolve dependencies.

    Args:
        tasks: List of Task objects.

    Returns:
        Tuple of (dependency_graph, waves).
    """
    resolver = DependencyResolver()
    graph = resolver.build_graph(tasks)
    waves = resolver.assign_waves(tasks, graph)
    return graph, waves
