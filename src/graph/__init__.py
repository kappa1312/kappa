"""LangGraph-based state machine for Kappa orchestration.

This module provides the complete LangGraph orchestration system for Kappa,
including state management, node implementations, edge routing, and graph
building utilities.
"""

from src.graph.builder import (
    build_kappa_graph,
    build_kappa_graph_with_persistence,
    build_legacy_graph,
    create_orchestration_graph,
    get_graph_info,
    visualize_graph,
    visualize_legacy_graph,
)
from src.graph.edges import (
    get_execution_progress,
    get_next_wave_tasks,
    get_ready_tasks,
    has_unresolved_conflicts,
    route_after_decomposition,
    route_after_dependency_resolution,
    route_after_merge,
    route_after_parsing,
    route_after_resolution,
    route_after_task_generation,
    route_after_validation,
    route_after_wave,
    should_abort_execution,
    should_continue_execution,
)
from src.graph.nodes import (
    conflict_resolution_node,
    decomposition_node,
    execute_wave_node,
    finalize_node,
    generate_tasks_node,
    handle_error_node,
    initialize_node,
    merge_outputs_node,
    parse_requirements_node,
    resolve_dependencies_node,
    validate_node,
    wave_execution_node,
)
from src.graph.state import (
    ExecutionStatus,
    KappaState,
    WaveStatus,
    calculate_progress,
    create_execution_log,
    create_initial_state,
    get_pending_task_ids,
    get_state_summary,
    get_task_by_id,
    get_wave_task_ids,
    is_task_ready,
    reconstruct_state_from_db,
    save_state_to_db,
    state_from_dict,
    state_to_dict,
)

__all__ = [
    # State
    "ExecutionStatus",
    "KappaState",
    "WaveStatus",
    "calculate_progress",
    "create_execution_log",
    "create_initial_state",
    "get_pending_task_ids",
    "get_state_summary",
    "get_task_by_id",
    "get_wave_task_ids",
    "is_task_ready",
    "reconstruct_state_from_db",
    "save_state_to_db",
    "state_from_dict",
    "state_to_dict",
    # Nodes (new)
    "parse_requirements_node",
    "generate_tasks_node",
    "resolve_dependencies_node",
    "execute_wave_node",
    "merge_outputs_node",
    "validate_node",
    "handle_error_node",
    # Nodes (legacy)
    "initialize_node",
    "decomposition_node",
    "wave_execution_node",
    "conflict_resolution_node",
    "finalize_node",
    # Edges (new)
    "route_after_parsing",
    "route_after_task_generation",
    "route_after_dependency_resolution",
    "should_continue_execution",
    "route_after_merge",
    "route_after_validation",
    # Edges (legacy)
    "route_after_decomposition",
    "route_after_wave",
    "route_after_resolution",
    # Edge helpers
    "should_abort_execution",
    "get_ready_tasks",
    "get_next_wave_tasks",
    "has_unresolved_conflicts",
    "get_execution_progress",
    # Builder
    "create_orchestration_graph",
    "build_kappa_graph",
    "build_kappa_graph_with_persistence",
    "build_legacy_graph",
    "visualize_graph",
    "visualize_legacy_graph",
    "get_graph_info",
]
