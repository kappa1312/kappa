# Kappa OS User Guide

## Introduction

Kappa OS transforms how you build software. Instead of manually coding features one by one, you describe what you want to build, and Kappa orchestrates multiple AI sessions to build it for you in parallel.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- PostgreSQL 14 or higher
- Node.js 18 or higher (for Claude Code CLI)
- Anthropic API key

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/kappa-os.git
cd kappa-os

# Install dependencies
poetry install

# Copy environment template
cp .env.example .env

# Edit .env with your credentials
# - Set ANTHROPIC_API_KEY
# - Set DATABASE_URL
```

### Database Setup

```bash
# Create database
createdb kappa_db

# Run schema setup
psql -d kappa_db -f scripts/setup_db.sql
```

### Verify Installation

```bash
# Run health check
poetry run kappa health
```

## Basic Usage

### Running Your First Project

```bash
# Simple specification
poetry run kappa run "Build a hello world Python CLI"

# With a project directory
poetry run kappa run "Build a REST API with user auth" --project ./my-api

# With a spec file
echo "Build a blog with posts and comments" > spec.txt
poetry run kappa run spec.txt --project ./blog
```

### Understanding the Output

Kappa will:
1. Parse your specification
2. Generate tasks
3. Organize them into waves
4. Execute each wave in parallel
5. Resolve any conflicts
6. Output a summary

### Previewing Tasks

Before running a full execution, you can preview what tasks will be generated:

```bash
poetry run kappa decompose "Build an e-commerce platform"
```

This shows the task breakdown without executing anything.

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Your Anthropic API key | Required |
| `DATABASE_URL` | PostgreSQL connection URL | Required |
| `KAPPA_LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | INFO |
| `KAPPA_MAX_PARALLEL_SESSIONS` | Max concurrent Claude sessions | 5 |
| `KAPPA_SESSION_TIMEOUT` | Session timeout in seconds | 3600 |
| `KAPPA_DEBUG` | Enable debug mode | false |

### Project Configuration

Create a `CLAUDE.md` file in your project root to customize behavior:

```markdown
# My Project

## Tech Stack
- Python 3.12
- FastAPI
- PostgreSQL
- Redis

## Code Style
- Use Black formatting
- Type hints required
- Docstrings in Google style

## Constraints
- No external API calls
- All data must be validated
```

## Advanced Usage

### Custom Working Directory

```bash
poetry run kappa run "Build a CLI tool" --project /path/to/project
```

### Adjusting Parallelism

```bash
# More parallel sessions (uses more API quota)
KAPPA_MAX_PARALLEL_SESSIONS=10 poetry run kappa run "Complex project"

# Fewer sessions (slower but cheaper)
KAPPA_MAX_PARALLEL_SESSIONS=2 poetry run kappa run "Simple project"
```

### Debug Mode

```bash
KAPPA_DEBUG=true poetry run kappa run "Debug this"
```

### Viewing Logs

```bash
# Recent logs
poetry run kappa logs --tail 100

# Follow logs
poetry run kappa logs --follow
```

## Workflow Best Practices

### 1. Start with Clear Specifications

**Good:**
```
Build a REST API for a todo application with:
- User authentication using JWT
- CRUD operations for todos
- PostgreSQL database with SQLAlchemy
- Pytest tests for all endpoints
```

**Less effective:**
```
Build a todo app
```

### 2. Be Specific About Tech Stack

If you have preferences, state them:
```
Build a blog using:
- FastAPI (not Flask)
- PostgreSQL with asyncpg
- Pydantic for validation
- Alembic for migrations
```

### 3. Define Constraints

Include what NOT to do:
```
Requirements:
- No external API calls
- No JavaScript dependencies
- Must work offline
- No Docker required for development
```

### 4. Iterate on Complex Projects

For large projects, consider building incrementally:

```bash
# Phase 1: Core
poetry run kappa run "Build user authentication module"

# Phase 2: Features
poetry run kappa run "Add product catalog to existing auth system"

# Phase 3: Integration
poetry run kappa run "Add shopping cart with Stripe checkout"
```

## Troubleshooting

### Common Issues

#### "API key not configured"

Make sure your `.env` file contains:
```
ANTHROPIC_API_KEY=sk-ant-api03-xxxxx
```

#### "Database connection failed"

1. Check PostgreSQL is running
2. Verify DATABASE_URL format:
   ```
   postgresql+asyncpg://user:password@localhost:5432/kappa_db
   ```
3. Ensure database exists: `createdb kappa_db`

#### "Session timeout"

Increase timeout for complex tasks:
```bash
KAPPA_SESSION_TIMEOUT=7200 poetry run kappa run "Complex task"
```

#### "Conflict resolution failed"

Some conflicts require manual review. Check the output for affected files and resolve manually.

### Getting Help

```bash
# CLI help
poetry run kappa --help
poetry run kappa run --help

# Health check
poetry run kappa health
```

## Examples

### Example 1: Simple CLI Tool

```bash
poetry run kappa run "Create a Python CLI tool that converts Markdown to HTML"
```

### Example 2: REST API

```bash
poetry run kappa run "
Build a REST API for a bookstore with:
- Book model (title, author, ISBN, price)
- CRUD endpoints
- SQLite database
- Basic input validation
"
```

### Example 3: Full-Stack Application

```bash
poetry run kappa run "
Build a blog platform with:
- User registration and authentication
- Create, edit, delete posts
- Comment system
- Categories and tags
- PostgreSQL database
- REST API with FastAPI
- Unit tests
"
```

## Next Steps

- Read the [Architecture Guide](architecture.md) to understand how Kappa works
- Check the [API Reference](api_reference.md) for programmatic usage
- Join the community for support and updates
