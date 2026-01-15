"""LangGraph builder for constructing the Kappa execution graph.

This module provides functions to build the complete Kappa orchestration
graph with all 7 nodes and comprehensive edge routing.
"""

from typing import Any

from langgraph.graph import END, START, StateGraph
from loguru import logger

from src.core.config import Settings
from src.graph.state import KappaState


def create_orchestration_graph() -> Any:
    """
    Build the complete Kappa orchestration graph.

    Creates a state machine with the following flow:
    1. parse_requirements - Parse natural language
    2. generate_tasks - Generate TaskSpecs
    3. resolve_dependencies - Build dependency graph
    4. execute_wave (loop) - Execute waves in parallel
    5. merge_outputs - Resolve conflicts
    6. validate - Validate outputs
    7. handle_error - Handle failures

    Returns:
        Compiled LangGraph application.

    Example:
        >>> graph = create_orchestration_graph()
        >>> result = await graph.ainvoke(initial_state)
    """
    from src.graph.edges import (
        route_after_dependency_resolution,
        route_after_merge,
        route_after_parsing,
        route_after_task_generation,
        route_after_validation,
        should_continue_execution,
    )
    from src.graph.nodes import (
        execute_wave_node,
        generate_tasks_node,
        handle_error_node,
        merge_outputs_node,
        parse_requirements_node,
        resolve_dependencies_node,
        validate_node,
    )

    logger.info("Building Kappa orchestration graph")

    # Create state graph
    graph = StateGraph(KappaState)

    # Add all 7 nodes
    graph.add_node("parse_requirements", parse_requirements_node)
    graph.add_node("generate_tasks", generate_tasks_node)
    graph.add_node("resolve_dependencies", resolve_dependencies_node)
    graph.add_node("execute_wave", execute_wave_node)
    graph.add_node("merge_outputs", merge_outputs_node)
    graph.add_node("validate", validate_node)
    graph.add_node("handle_error", handle_error_node)

    # Set entry point
    graph.set_entry_point("parse_requirements")

    # Add edges with conditional routing

    # parse_requirements -> (generate_tasks | handle_error)
    graph.add_conditional_edges(
        "parse_requirements",
        route_after_parsing,
        {
            "generate_tasks": "generate_tasks",
            "handle_error": "handle_error",
        },
    )

    # generate_tasks -> (resolve_dependencies | handle_error)
    graph.add_conditional_edges(
        "generate_tasks",
        route_after_task_generation,
        {
            "resolve_dependencies": "resolve_dependencies",
            "handle_error": "handle_error",
        },
    )

    # resolve_dependencies -> (execute_wave | handle_error)
    graph.add_conditional_edges(
        "resolve_dependencies",
        route_after_dependency_resolution,
        {
            "execute_wave": "execute_wave",
            "handle_error": "handle_error",
        },
    )

    # execute_wave -> (execute_wave | merge_outputs | handle_error)
    graph.add_conditional_edges(
        "execute_wave",
        should_continue_execution,
        {
            "execute_wave": "execute_wave",
            "merge_outputs": "merge_outputs",
            "handle_error": "handle_error",
        },
    )

    # merge_outputs -> (validate | handle_error)
    graph.add_conditional_edges(
        "merge_outputs",
        route_after_merge,
        {
            "validate": "validate",
            "handle_error": "handle_error",
        },
    )

    # validate -> (END | handle_error)
    graph.add_conditional_edges(
        "validate",
        route_after_validation,
        {
            "end": END,
            "handle_error": "handle_error",
        },
    )

    # handle_error -> END
    graph.add_edge("handle_error", END)

    # Compile graph
    app = graph.compile()

    logger.info("Kappa orchestration graph compiled successfully")
    return app


async def build_kappa_graph(settings: Settings | None = None) -> Any:
    """
    Build the Kappa LangGraph (async version).

    This is the primary function for building the graph with
    optional configuration settings.

    Args:
        settings: Optional Kappa configuration settings.

    Returns:
        Compiled LangGraph application.

    Example:
        >>> from src.core.config import get_settings
        >>> settings = get_settings()
        >>> graph = await build_kappa_graph(settings)
        >>> result = await graph.ainvoke(initial_state)
    """
    return create_orchestration_graph()


async def build_kappa_graph_with_persistence(settings: Settings) -> Any:
    """
    Build Kappa LangGraph with PostgreSQL persistence.

    Enables durable execution that can recover from failures
    and resume from checkpoints.

    Args:
        settings: Kappa configuration settings.

    Returns:
        Compiled LangGraph application with checkpointer.

    Example:
        >>> graph = await build_kappa_graph_with_persistence(settings)
        >>> result = await graph.ainvoke(state, config={"configurable": {"thread_id": "123"}})
    """
    from src.graph.edges import (
        route_after_dependency_resolution,
        route_after_merge,
        route_after_parsing,
        route_after_task_generation,
        route_after_validation,
        should_continue_execution,
    )
    from src.graph.nodes import (
        execute_wave_node,
        generate_tasks_node,
        handle_error_node,
        merge_outputs_node,
        parse_requirements_node,
        resolve_dependencies_node,
        validate_node,
    )

    logger.info("Building Kappa graph with PostgreSQL persistence")

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        # Create checkpointer
        checkpointer = AsyncPostgresSaver.from_conn_string(settings.database_url_async)
        await checkpointer.setup()

    except ImportError:
        logger.warning("PostgreSQL checkpointer not available, using in-memory")
        checkpointer = None

    # Build base graph
    graph = StateGraph(KappaState)

    # Add all nodes
    graph.add_node("parse_requirements", parse_requirements_node)
    graph.add_node("generate_tasks", generate_tasks_node)
    graph.add_node("resolve_dependencies", resolve_dependencies_node)
    graph.add_node("execute_wave", execute_wave_node)
    graph.add_node("merge_outputs", merge_outputs_node)
    graph.add_node("validate", validate_node)
    graph.add_node("handle_error", handle_error_node)

    # Set entry point
    graph.set_entry_point("parse_requirements")

    # Add all edges
    graph.add_conditional_edges(
        "parse_requirements",
        route_after_parsing,
        {"generate_tasks": "generate_tasks", "handle_error": "handle_error"},
    )

    graph.add_conditional_edges(
        "generate_tasks",
        route_after_task_generation,
        {"resolve_dependencies": "resolve_dependencies", "handle_error": "handle_error"},
    )

    graph.add_conditional_edges(
        "resolve_dependencies",
        route_after_dependency_resolution,
        {"execute_wave": "execute_wave", "handle_error": "handle_error"},
    )

    graph.add_conditional_edges(
        "execute_wave",
        should_continue_execution,
        {
            "execute_wave": "execute_wave",
            "merge_outputs": "merge_outputs",
            "handle_error": "handle_error",
        },
    )

    graph.add_conditional_edges(
        "merge_outputs",
        route_after_merge,
        {"validate": "validate", "handle_error": "handle_error"},
    )

    graph.add_conditional_edges(
        "validate",
        route_after_validation,
        {"end": END, "handle_error": "handle_error"},
    )

    graph.add_edge("handle_error", END)

    # Compile with checkpointer if available
    if checkpointer:
        app = graph.compile(checkpointer=checkpointer)
    else:
        app = graph.compile()

    logger.info("Kappa graph with persistence compiled successfully")
    return app


def build_legacy_graph() -> Any:
    """
    Build the legacy Kappa graph for backward compatibility.

    This graph uses the original 5-node structure:
    1. initialize
    2. decompose
    3. execute_wave (loop)
    4. resolve_conflicts
    5. finalize

    Returns:
        Compiled LangGraph application.
    """
    from src.graph.edges import (
        route_after_decomposition,
        route_after_resolution,
        route_after_wave,
    )
    from src.graph.nodes import (
        conflict_resolution_node,
        decomposition_node,
        finalize_node,
        initialize_node,
        wave_execution_node,
    )

    logger.info("Building legacy Kappa execution graph")

    # Create state graph
    graph = StateGraph(KappaState)

    # Add nodes
    graph.add_node("initialize", initialize_node)
    graph.add_node("decompose", decomposition_node)
    graph.add_node("execute_wave", wave_execution_node)
    graph.add_node("resolve_conflicts", conflict_resolution_node)
    graph.add_node("finalize", finalize_node)

    # Add edges
    # START -> initialize
    graph.add_edge(START, "initialize")

    # initialize -> decompose
    graph.add_edge("initialize", "decompose")

    # decompose -> (execute_wave | finalize)
    graph.add_conditional_edges(
        "decompose",
        route_after_decomposition,
        {
            "execute_wave": "execute_wave",
            "finalize": "finalize",
        },
    )

    # execute_wave -> (execute_wave | resolve_conflicts | finalize)
    graph.add_conditional_edges(
        "execute_wave",
        route_after_wave,
        {
            "execute_wave": "execute_wave",
            "resolve_conflicts": "resolve_conflicts",
            "finalize": "finalize",
        },
    )

    # resolve_conflicts -> finalize
    graph.add_conditional_edges(
        "resolve_conflicts",
        route_after_resolution,
        {
            "finalize": "finalize",
        },
    )

    # finalize -> END
    graph.add_edge("finalize", END)

    # Compile graph
    app = graph.compile()

    logger.info("Legacy Kappa graph compiled successfully")
    return app


def visualize_graph() -> str:
    """
    Generate a Mermaid diagram of the Kappa graph.

    Returns:
        Mermaid diagram string.
    """
    return """
```mermaid
graph TD
    START([Start]) --> parse[Parse Requirements]
    parse -->|success| generate[Generate Tasks]
    parse -->|error| error[Handle Error]
    generate -->|success| resolve[Resolve Dependencies]
    generate -->|error| error
    resolve -->|success| execute[Execute Wave]
    resolve -->|error| error
    execute -->|more waves| execute
    execute -->|done| merge[Merge Outputs]
    execute -->|critical failure| error
    merge -->|success| validate[Validate]
    merge -->|critical conflict| error
    validate -->|passed| END([End])
    validate -->|failed| error
    error --> END
```
"""


def visualize_legacy_graph() -> str:
    """
    Generate a Mermaid diagram of the legacy Kappa graph.

    Returns:
        Mermaid diagram string.
    """
    return """
```mermaid
graph TD
    START([Start]) --> init[Initialize]
    init --> decompose[Decompose Spec]
    decompose -->|has tasks| execute[Execute Wave]
    decompose -->|no tasks| finalize[Finalize]
    execute -->|more waves| execute
    execute -->|conflicts| resolve[Resolve Conflicts]
    execute -->|done| finalize
    resolve --> finalize
    finalize --> END([End])
```
"""


def get_graph_info() -> dict[str, Any]:
    """
    Get information about the Kappa graph structure.

    Returns:
        Dict with graph metadata.
    """
    return {
        "name": "Kappa Orchestration Graph",
        "version": "2.0",
        "nodes": [
            {
                "name": "parse_requirements",
                "description": "Parse natural language into ProjectRequirements",
                "inputs": ["requirements_text"],
                "outputs": ["requirements"],
            },
            {
                "name": "generate_tasks",
                "description": "Generate TaskSpecs from ProjectRequirements",
                "inputs": ["requirements"],
                "outputs": ["tasks"],
            },
            {
                "name": "resolve_dependencies",
                "description": "Build dependency graph and assign waves",
                "inputs": ["tasks"],
                "outputs": ["dependency_graph", "waves"],
            },
            {
                "name": "execute_wave",
                "description": "Execute all tasks in current wave in parallel",
                "inputs": ["tasks", "current_wave"],
                "outputs": ["completed_tasks", "failed_tasks", "created_files"],
            },
            {
                "name": "merge_outputs",
                "description": "Merge outputs and resolve conflicts",
                "inputs": ["task_results"],
                "outputs": ["conflicts"],
            },
            {
                "name": "validate",
                "description": "Validate all outputs",
                "inputs": ["workspace_path"],
                "outputs": ["validation_results"],
            },
            {
                "name": "handle_error",
                "description": "Handle execution errors",
                "inputs": ["error", "error_node"],
                "outputs": ["status"],
            },
        ],
        "edges": [
            ("START", "parse_requirements"),
            ("parse_requirements", "generate_tasks"),
            ("parse_requirements", "handle_error"),
            ("generate_tasks", "resolve_dependencies"),
            ("generate_tasks", "handle_error"),
            ("resolve_dependencies", "execute_wave"),
            ("resolve_dependencies", "handle_error"),
            ("execute_wave", "execute_wave"),
            ("execute_wave", "merge_outputs"),
            ("execute_wave", "handle_error"),
            ("merge_outputs", "validate"),
            ("merge_outputs", "handle_error"),
            ("validate", "END"),
            ("validate", "handle_error"),
            ("handle_error", "END"),
        ],
    }
