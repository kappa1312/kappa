"""LangGraph builder for constructing the Kappa execution graph."""

from typing import Any

from langgraph.graph import END, START, StateGraph
from loguru import logger

from src.core.config import Settings
from src.core.state import KappaState
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


async def build_kappa_graph(settings: Settings) -> Any:
    """
    Build the complete Kappa LangGraph.

    Creates a state machine with the following flow:
    1. Initialize -> Decompose
    2. Decompose -> Execute Waves (loop)
    3. Execute Waves -> Resolve Conflicts (if needed)
    4. Resolve Conflicts -> Finalize
    5. Finalize -> END

    Args:
        settings: Kappa configuration settings.

    Returns:
        Compiled LangGraph application.

    Example:
        >>> settings = get_settings()
        >>> graph = await build_kappa_graph(settings)
        >>> result = await graph.ainvoke(initial_state)
    """
    logger.info("Building Kappa execution graph")

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
    # Note: For production, add checkpointer for durability
    # checkpointer = await get_checkpointer(settings)
    # app = graph.compile(checkpointer=checkpointer)

    app = graph.compile()

    logger.info("Kappa graph compiled successfully")
    return app


async def build_kappa_graph_with_persistence(settings: Settings) -> Any:
    """
    Build Kappa LangGraph with PostgreSQL persistence.

    Enables durable execution that can recover from failures
    and resume from checkpoints.

    Args:
        settings: Kappa configuration settings.

    Returns:
        Compiled LangGraph application with checkpointer.
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    logger.info("Building Kappa graph with PostgreSQL persistence")

    # Create checkpointer
    checkpointer = AsyncPostgresSaver.from_conn_string(
        settings.database_url_async
    )
    await checkpointer.setup()

    # Build base graph
    graph = StateGraph(KappaState)

    # Add all nodes
    graph.add_node("initialize", initialize_node)
    graph.add_node("decompose", decomposition_node)
    graph.add_node("execute_wave", wave_execution_node)
    graph.add_node("resolve_conflicts", conflict_resolution_node)
    graph.add_node("finalize", finalize_node)

    # Add all edges
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "decompose")

    graph.add_conditional_edges(
        "decompose",
        route_after_decomposition,
        {"execute_wave": "execute_wave", "finalize": "finalize"},
    )

    graph.add_conditional_edges(
        "execute_wave",
        route_after_wave,
        {
            "execute_wave": "execute_wave",
            "resolve_conflicts": "resolve_conflicts",
            "finalize": "finalize",
        },
    )

    graph.add_conditional_edges(
        "resolve_conflicts",
        route_after_resolution,
        {"finalize": "finalize"},
    )

    graph.add_edge("finalize", END)

    # Compile with checkpointer
    app = graph.compile(checkpointer=checkpointer)

    logger.info("Kappa graph with persistence compiled successfully")
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
