# LangGraph Patterns Skill

## Activation

This skill activates when working on:
- State machine design
- Graph node implementation
- Edge routing logic
- Checkpoint and persistence

## LangGraph 1.0 Fundamentals

### Core Concepts

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# Define state as TypedDict
class MyState(TypedDict):
    messages: list[str]
    current_step: str
    results: dict[str, Any]

# Create graph
graph = StateGraph(MyState)

# Add nodes (functions that transform state)
graph.add_node("parse", parse_node)
graph.add_node("execute", execute_node)
graph.add_node("validate", validate_node)

# Add edges (connections between nodes)
graph.add_edge(START, "parse")
graph.add_edge("parse", "execute")
graph.add_conditional_edges(
    "execute",
    route_after_execute,  # Routing function
    {"success": "validate", "retry": "execute", "fail": END}
)
graph.add_edge("validate", END)

# Compile with checkpointer for durability
checkpointer = AsyncPostgresSaver.from_conn_string(DATABASE_URL)
app = graph.compile(checkpointer=checkpointer)
```

### Node Pattern

```python
async def my_node(state: MyState) -> dict[str, Any]:
    """
    Nodes receive current state and return updates.

    Args:
        state: Current graph state (read-only view)

    Returns:
        Dictionary of state updates to merge
    """
    # Read from state
    current = state["current_step"]

    # Perform work
    result = await do_work(current)

    # Return updates (will be merged with state)
    return {
        "current_step": "next_step",
        "results": {**state["results"], current: result}
    }
```

### Conditional Edge Pattern

```python
def route_after_execute(state: MyState) -> str:
    """
    Routing functions determine next node.

    Args:
        state: Current graph state

    Returns:
        String key matching add_conditional_edges mapping
    """
    if state.get("error"):
        if state.get("retry_count", 0) < 3:
            return "retry"
        return "fail"
    return "success"
```

### Parallel Execution Pattern

```python
from langgraph.graph import StateGraph

# Fan-out: Run multiple nodes in parallel
graph.add_node("task_a", task_a_node)
graph.add_node("task_b", task_b_node)
graph.add_node("task_c", task_c_node)
graph.add_node("aggregate", aggregate_node)

# All three run in parallel after start
graph.add_edge(START, "task_a")
graph.add_edge(START, "task_b")
graph.add_edge(START, "task_c")

# Fan-in: All must complete before aggregate
graph.add_edge("task_a", "aggregate")
graph.add_edge("task_b", "aggregate")
graph.add_edge("task_c", "aggregate")
graph.add_edge("aggregate", END)
```

### Subgraph Pattern

```python
# Create a subgraph for reusable logic
subgraph = StateGraph(SubState)
subgraph.add_node("sub_step_1", sub_step_1)
subgraph.add_node("sub_step_2", sub_step_2)
subgraph.add_edge(START, "sub_step_1")
subgraph.add_edge("sub_step_1", "sub_step_2")
subgraph.add_edge("sub_step_2", END)
compiled_subgraph = subgraph.compile()

# Use subgraph as a node in parent graph
parent_graph = StateGraph(ParentState)
parent_graph.add_node("prepare", prepare_node)
parent_graph.add_node("subprocess", compiled_subgraph)
parent_graph.add_node("finalize", finalize_node)
```

### Human-in-the-Loop Pattern

```python
from langgraph.graph import StateGraph
from langgraph.types import interrupt

async def human_review_node(state: MyState) -> dict:
    """Node that pauses for human review."""

    # Prepare data for review
    review_data = prepare_for_review(state)

    # This will pause execution and return control
    human_response = interrupt({
        "type": "review_required",
        "data": review_data,
        "options": ["approve", "reject", "modify"]
    })

    # Execution resumes here after human responds
    if human_response["decision"] == "approve":
        return {"approved": True}
    elif human_response["decision"] == "reject":
        return {"approved": False, "reason": human_response.get("reason")}
    else:
        return {"modifications": human_response["changes"]}
```

### Persistence Pattern

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async def create_app():
    # Create checkpointer
    checkpointer = AsyncPostgresSaver.from_conn_string(
        "postgresql+asyncpg://user:pass@localhost/db"
    )
    await checkpointer.setup()  # Create tables if needed

    # Compile graph with checkpointer
    app = graph.compile(checkpointer=checkpointer)
    return app

# Run with thread_id for persistence
async def run_with_persistence():
    app = await create_app()

    config = {"configurable": {"thread_id": "my-session-123"}}

    # First run
    result = await app.ainvoke(initial_state, config)

    # Later: Resume from checkpoint
    result = await app.ainvoke(None, config)  # Continues from last state
```

### Error Handling Pattern

```python
async def safe_node(state: MyState) -> dict:
    """Node with error handling."""
    try:
        result = await risky_operation(state)
        return {"result": result, "error": None}
    except RecoverableError as e:
        return {
            "error": str(e),
            "retry_count": state.get("retry_count", 0) + 1
        }
    except FatalError as e:
        # Will propagate and stop graph
        raise
```

## Kappa-Specific Patterns

### Wave Execution

```python
async def wave_executor_node(state: KappaState) -> dict:
    """Execute all tasks in current wave in parallel."""
    current_wave = state["current_wave"]
    wave_tasks = state["waves"][current_wave]

    # Run all tasks in wave concurrently
    async with anyio.create_task_group() as tg:
        results = {}
        for task_id in wave_tasks:
            tg.start_soon(
                execute_task_and_store,
                task_id,
                state["tasks"],
                results
            )

    return {
        "completed_tasks": state["completed_tasks"] + list(results.keys()),
        "current_wave": current_wave + 1
    }
```

### Session Management

```python
async def spawn_session_node(state: KappaState) -> dict:
    """Spawn Claude sessions for wave tasks."""
    from claude_agent_sdk import ClaudeSDKClient

    sessions = {}
    for task_id in state["waves"][state["current_wave"]]:
        client = ClaudeSDKClient()
        sessions[task_id] = {
            "client": client,
            "started_at": datetime.utcnow().isoformat()
        }

    return {"active_sessions": sessions}
```

## Best Practices

1. **Keep nodes small and focused** - Each node should do one thing
2. **Use TypedDict for state** - Enables type checking and IDE support
3. **Always handle errors** - Return error state instead of raising
4. **Use checkpointing** - Enable durability for long-running graphs
5. **Test routing functions** - They control critical flow decisions
