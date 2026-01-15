# Kappa OS Architecture

## Overview

Kappa OS is an autonomous development operating system that orchestrates multiple parallel Claude Code sessions to build complete software projects significantly faster than traditional manual coding.

## Core Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          KAPPA ORCHESTRATOR                              │
│                                                                          │
│  ┌──────────────┐   ┌──────────────┐   ┌─────────────────────────────┐  │
│  │  CLI Layer   │───│   Core       │───│      LangGraph Engine       │  │
│  │  (Typer)     │   │   (Python)   │   │  (State Machine + Durable)  │  │
│  └──────────────┘   └──────────────┘   └─────────────────────────────┘  │
│                              │                                           │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    EXECUTION PIPELINE                             │   │
│  │                                                                   │   │
│  │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────────────┐   │   │
│  │  │ Parse   │──▶│Decompose│──▶│ Execute │──▶│ Resolve & Merge │   │   │
│  │  │  Spec   │   │  Tasks  │   │  Waves  │   │   Conflicts     │   │   │
│  │  └─────────┘   └─────────┘   └─────────┘   └─────────────────┘   │   │
│  │                                    │                              │   │
│  │                      ┌─────────────┴─────────────┐               │   │
│  │                      ▼             ▼             ▼               │   │
│  │                 ┌─────────┐   ┌─────────┐   ┌─────────┐         │   │
│  │                 │Session 1│   │Session 2│   │Session N│         │   │
│  │                 │ (Clone) │   │ (Clone) │   │ (Clone) │         │   │
│  │                 └─────────┘   └─────────┘   └─────────┘         │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    PERSISTENCE LAYER                              │   │
│  │                                                                   │   │
│  │  ┌─────────────────┐   ┌─────────────────┐   ┌────────────────┐  │   │
│  │  │   PostgreSQL    │   │  Context Store  │   │    Metrics     │  │   │
│  │  │   (asyncpg)     │   │   (Knowledge)   │   │   (Metrics)    │  │   │
│  │  └─────────────────┘   └─────────────────┘   └────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Core Module (`src/core/`)

The heart of Kappa OS, containing:

- **Orchestrator** (`orchestrator.py`): Main entry point that coordinates the entire execution pipeline
- **State** (`state.py`): TypedDict-based state management for LangGraph
- **Config** (`config.py`): Pydantic Settings-based configuration management

### 2. Graph Module (`src/graph/`)

LangGraph-based state machine implementation:

- **Nodes** (`nodes.py`): Individual pipeline stages (initialize, decompose, execute, resolve, finalize)
- **Edges** (`edges.py`): Routing logic between nodes
- **Builder** (`builder.py`): Graph construction and compilation

### 3. Decomposition Module (`src/decomposition/`)

Specification parsing and task generation:

- **Parser** (`parser.py`): NLP-based specification parsing
- **Task Generator** (`task_generator.py`): Creates executable tasks from requirements
- **Dependency Resolver** (`dependency_resolver.py`): Builds dependency graph and assigns waves

### 4. Sessions Module (`src/sessions/`)

Claude Agent SDK integration:

- **Base** (`base.py`): Abstract session interface
- **Terminal** (`terminal.py`): Terminal-based Claude sessions
- **Router** (`router.py`): Parallel session management and load balancing

### 5. Knowledge Module (`src/knowledge/`)

Persistence and context sharing:

- **Database** (`database.py`): Async PostgreSQL connection management
- **Models** (`models.py`): SQLAlchemy ORM models
- **Context Manager** (`context_manager.py`): Cross-session context sharing

### 6. Conflict Module (`src/conflict/`)

Conflict detection and resolution:

- **Detector** (`detector.py`): Identifies overlapping modifications
- **Resolver** (`resolver.py`): Applies resolution strategies
- **Strategies** (`strategies.py`): Pluggable resolution algorithms

## Data Flow

### 1. Specification → Tasks

```
User Specification
       │
       ▼
┌──────────────┐
│   Parser     │ ─── Extract requirements, entities, priorities
└──────────────┘
       │
       ▼
┌──────────────┐
│  Generator   │ ─── Create tasks with categories, complexity
└──────────────┘
       │
       ▼
┌──────────────┐
│  Resolver    │ ─── Build dependency graph, assign waves
└──────────────┘
       │
       ▼
Task List with Waves
```

### 2. Tasks → Results

```
Task Wave
    │
    ├──── Task A ──▶ Session 1 ──▶ Result A
    │
    ├──── Task B ──▶ Session 2 ──▶ Result B
    │
    └──── Task C ──▶ Session 3 ──▶ Result C
           │
           ▼
    Conflict Detection
           │
           ▼
    Resolution / Merge
           │
           ▼
    Final Output
```

## Key Design Decisions

### 1. LangGraph for Orchestration

**Why?**
- Durable state with automatic checkpointing
- Built-in recovery from failures
- Human-in-the-loop patterns
- Visualization and debugging support

### 2. asyncpg for PostgreSQL

**Why?**
- 5x faster than psycopg3 in benchmarks
- Native asyncio support
- Better connection pooling
- First-class PostgreSQL protocol support

### 3. Claude Agent SDK

**Why?**
- Official SDK with full Claude Code capabilities
- Terminal automation built-in
- File operations, bash execution
- Proper session management

### 4. Wave-Based Execution

**Why?**
- Maximizes parallelism within dependency constraints
- Clear execution phases for monitoring
- Natural checkpointing boundaries
- Easier conflict resolution

## State Management

### KappaState

```python
class KappaState(TypedDict, total=False):
    # Project identification
    project_id: str
    project_name: str

    # Input
    specification: str
    project_path: str

    # Decomposition
    tasks: list[dict]
    dependency_graph: dict[str, list[str]]
    waves: list[list[str]]

    # Execution
    current_wave: int
    active_sessions: dict[str, dict]
    completed_tasks: list[str]
    failed_tasks: list[str]

    # Results
    task_results: list[dict]
    conflicts: list[dict]
    final_output: str | None

    # Metadata
    status: str
    started_at: str
    completed_at: str | None
```

## Error Handling

### Retry Strategy

```
Session Failure
      │
      ▼
Attempt < Max Retries?
      │
  ┌───┴───┐
  │ Yes   │ No
  ▼       ▼
Retry    Mark Failed
with     └──▶ Skip Dependents
backoff       └──▶ Continue
```

### Conflict Resolution

```
File Modified by Multiple Sessions
            │
            ▼
┌───────────────────────┐
│   Detect Conflict     │
└───────────────────────┘
            │
            ▼
┌───────────────────────┐
│  Select Strategy      │
│  (Merge/Newer/Manual) │
└───────────────────────┘
            │
            ▼
┌───────────────────────┐
│  Apply Resolution     │
└───────────────────────┘
            │
        ┌───┴───┐
    Success   Failure
        │        │
        ▼        ▼
    Continue  Queue for
              Manual Review
```

## Scalability

### Horizontal Scaling

- Multiple Kappa instances can share the same PostgreSQL database
- Each instance manages its own session pool
- Context sharing enables coordination

### Vertical Scaling

- Adjust `KAPPA_MAX_PARALLEL_SESSIONS` based on available resources
- Connection pooling prevents database exhaustion
- Async I/O maximizes throughput

## Security Considerations

- API keys stored in environment variables, never in code
- Database credentials use SSL in production
- Session isolation prevents cross-project access
- Input validation on all external data
