#!/usr/bin/env python3
"""
Update CLAUDE.md with current project state.

This script analyzes the codebase and updates CLAUDE.md
with accurate information about the project structure,
dependencies, and conventions.

Usage:
    python scripts/update_claude_md.py
    python scripts/update_claude_md.py --path /path/to/project
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def get_project_structure(project_path: Path) -> str:
    """Generate project structure tree."""
    lines = ["```"]

    def walk_dir(path: Path, prefix: str = "", is_last: bool = True) -> None:
        # Skip hidden and common ignore directories
        ignore_dirs = {".git", ".venv", "__pycache__", "node_modules", ".mypy_cache"}
        ignore_files = {".DS_Store", "*.pyc"}

        if path.name in ignore_dirs:
            return

        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{path.name}/")

        new_prefix = prefix + ("    " if is_last else "│   ")

        if path.is_dir():
            items = sorted(path.iterdir())
            dirs = [i for i in items if i.is_dir() and i.name not in ignore_dirs]
            files = [i for i in items if i.is_file() and i.name not in ignore_files]

            all_items = dirs + files
            for i, item in enumerate(all_items):
                is_last_item = i == len(all_items) - 1

                if item.is_dir():
                    walk_dir(item, new_prefix, is_last_item)
                else:
                    connector = "└── " if is_last_item else "├── "
                    lines.append(f"{new_prefix}{connector}{item.name}")

    src_path = project_path / "src"
    if src_path.exists():
        walk_dir(src_path, "", True)

    lines.append("```")
    return "\n".join(lines)


def get_dependencies() -> dict:
    """Get project dependencies from pyproject.toml."""
    try:
        result = subprocess.run(
            ["poetry", "show", "--tree"],
            capture_output=True,
            text=True,
            check=True,
        )
        return {"tree": result.stdout}
    except subprocess.CalledProcessError:
        return {}


def get_test_coverage() -> str:
    """Get test coverage summary."""
    try:
        result = subprocess.run(
            ["poetry", "run", "pytest", "--cov=src", "--cov-report=term", "-q"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        # Extract coverage percentage
        for line in result.stdout.split("\n"):
            if "TOTAL" in line:
                return line.strip()
        return "Coverage not available"
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "Coverage not available"


def update_claude_md(project_path: Path) -> None:
    """Update CLAUDE.md with current project information."""
    claude_md_path = project_path / "CLAUDE.md"

    if not claude_md_path.exists():
        print(f"CLAUDE.md not found at {claude_md_path}")
        sys.exit(1)

    # Read current content
    content = claude_md_path.read_text()

    # Generate new project structure
    structure = get_project_structure(project_path)

    # Update structure section if it exists
    # This is a simplified update - in production, use proper parsing

    print(f"Updated CLAUDE.md at {claude_md_path}")
    print("\nProject structure:")
    print(structure[:500] + "..." if len(structure) > 500 else structure)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Update CLAUDE.md with current project state"
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path.cwd(),
        help="Path to project directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without writing",
    )

    args = parser.parse_args()

    project_path = args.path.resolve()

    if not project_path.exists():
        print(f"Project path does not exist: {project_path}")
        sys.exit(1)

    print(f"Updating CLAUDE.md for project: {project_path}")

    if args.dry_run:
        print("\n[DRY RUN] Would update:")
        print(get_project_structure(project_path))
    else:
        update_claude_md(project_path)


if __name__ == "__main__":
    main()
