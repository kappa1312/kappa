# Kappa OS

**Autonomous Development Operating System** - Build complete software projects 10x faster through parallel Claude Code orchestration.

## Overview

Kappa OS transforms software development by orchestrating multiple parallel Claude Code sessions to work on different parts of your project simultaneously. Instead of weeks of manual coding, complex projects can be completed in hours.

### Key Features

- **Parallel Execution**: Run multiple Claude Code sessions simultaneously
- **Intelligent Task Decomposition**: Automatically break down projects into parallelizable tasks
- **Cross-Session Context**: Shared knowledge base prevents redundant work
- **Conflict Resolution**: Intelligent merging when sessions modify overlapping code
- **Production Quality**: All generated code compiles, tests, and deploys

## Requirements

- Python 3.11+
- PostgreSQL 14+
- Node.js 18+ (for Claude Code CLI)
- Anthropic API key

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/your-org/kappa-os.git
cd kappa-os

# Install dependencies
poetry install

# Copy environment template
cp .env.example .env
# Edit .env with your credentials
```

### 2. Database Setup

```bash
# Create PostgreSQL database
createdb kappa_db

# Run schema setup
psql -d kappa_db -f scripts/setup_db.sql
```

### 3. Run Kappa

```bash
# Start Kappa CLI
poetry run kappa

# Or run a specific project
poetry run kappa run --project ./my-project --spec "Build a REST API with authentication"
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Kappa Orchestrator                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Parser    │──│ Decomposer  │──│ Dependency Resolver │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
        ┌───────────────────┐ ┌───────────────────┐
        │  Session Manager  │ │  Context Manager  │
        │                   │ │                   │
        │  ┌─────┐ ┌─────┐  │ │  ┌─────────────┐  │
        │  │ S1  │ │ S2  │  │ │  │ PostgreSQL  │  │
        │  └─────┘ └─────┘  │ │  │  Knowledge  │  │
        │  ┌─────┐ ┌─────┐  │ │  │    Base     │  │
        │  │ S3  │ │ S4  │  │ │  └─────────────┘  │
        │  └─────┘ └─────┘  │ │                   │
        └───────────────────┘ └───────────────────┘
                    │                   │
                    └─────────┬─────────┘
                              ▼
                    ┌───────────────────┐
                    │ Conflict Resolver │
                    └───────────────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │  Merged Output    │
                    └───────────────────┘
```

## Usage

### Basic Commands

```bash
# Initialize a new project
kappa init --name my-project

# Run with a specification
kappa run --spec "Build an e-commerce platform with user auth, product catalog, and checkout"

# Check status
kappa status

# View session logs
kappa logs --session <session-id>

# Resolve conflicts manually
kappa resolve --conflict <conflict-id>
```

### Configuration

Edit `CLAUDE.md` in your project root to customize:
- Code style preferences
- Testing requirements
- Architecture constraints
- Technology stack

## Development

### Running Tests

```bash
# All tests
poetry run pytest

# Unit tests only
poetry run pytest tests/unit

# With coverage
poetry run pytest --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
poetry run black src tests

# Lint
poetry run ruff check src tests

# Type check
poetry run mypy src
```

## Tech Stack

- **Python 3.11+** - Async/await for concurrency
- **LangGraph 1.0** - Durable state machine orchestration
- **Claude Agent SDK** - Terminal automation and file operations
- **PostgreSQL** - State persistence and context sharing
- **asyncpg** - High-performance async PostgreSQL driver
- **Typer** - Modern CLI framework
- **Pydantic** - Data validation and settings

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

- [Documentation](docs/)
- [Issues](https://github.com/your-org/kappa-os/issues)
- [Discussions](https://github.com/your-org/kappa-os/discussions)
