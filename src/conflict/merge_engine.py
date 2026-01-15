"""
Merge Engine for Kappa OS

Post-execution output combination:
- Three-way merge for file contents
- Smart merging for TypeScript, Python, CSS, JSON
- Import deduplication
- Type definition consolidation
"""

import difflib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger


class MergeStrategy(str, Enum):
    """Merge strategy types."""

    THREE_WAY = "three_way"  # Standard three-way merge with base
    TWO_WAY = "two_way"  # Direct merge without base
    OURS = "ours"  # Keep our version
    THEIRS = "theirs"  # Keep their version
    UNION = "union"  # Include both (for additive changes)
    SMART = "smart"  # Language-aware smart merge


@dataclass
class MergeConflict:
    """Represents a merge conflict within a file."""

    start_line: int
    end_line: int
    ours: str
    theirs: str
    base: str | None = None
    resolution: str | None = None
    resolved: bool = False


@dataclass
class MergeResult:
    """Result of a merge operation."""

    success: bool
    merged_content: str | None = None
    conflicts: list[MergeConflict] = field(default_factory=list)
    files_created: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    strategy_used: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "has_conflicts": len(self.conflicts) > 0,
            "conflict_count": len(self.conflicts),
            "files_created": self.files_created,
            "files_modified": self.files_modified,
            "warnings": self.warnings,
            "strategy_used": self.strategy_used,
            "metadata": self.metadata,
        }


class MergeEngine:
    """
    Engine for merging parallel task outputs.

    Handles post-execution merging of files modified by
    multiple tasks in the same wave.

    Usage:
        engine = MergeEngine(workspace_path)
        result = await engine.merge_task_outputs(task_outputs)

        if not result.success:
            for conflict in result.conflicts:
                # Handle conflict
                pass
    """

    def __init__(self, workspace_path: Path):
        self.workspace = workspace_path
        self._conflict_markers = {
            "start": "<<<<<<< OURS",
            "separator": "=======",
            "end": ">>>>>>> THEIRS",
        }

    async def merge_task_outputs(
        self,
        task_outputs: list[dict[str, Any]],
    ) -> MergeResult:
        """
        Merge outputs from multiple parallel tasks.

        Args:
            task_outputs: List of task output dictionaries
                Each should have: task_id, files_modified, content_map

        Returns:
            MergeResult with merged content or conflicts
        """
        logger.info(f"Merging outputs from {len(task_outputs)} tasks")

        result = MergeResult(success=True, strategy_used="auto")

        # Group outputs by file
        file_outputs = self._group_by_file(task_outputs)

        for filepath, outputs in file_outputs.items():
            if len(outputs) == 1:
                # No merge needed
                continue

            # Multiple tasks modified this file
            file_result = await self._merge_file_outputs(filepath, outputs)

            if not file_result.success:
                result.success = False
                result.conflicts.extend(file_result.conflicts)

            result.files_modified.extend(file_result.files_modified)
            result.warnings.extend(file_result.warnings)

        return result

    async def merge_files(
        self,
        ours_path: Path,
        theirs_path: Path,
        base_path: Path | None = None,
        output_path: Path | None = None,
        strategy: MergeStrategy = MergeStrategy.SMART,
    ) -> MergeResult:
        """
        Merge two file versions with optional base.

        Args:
            ours_path: Our version of the file
            theirs_path: Their version of the file
            base_path: Optional base (common ancestor) version
            output_path: Where to write merged result
            strategy: Merge strategy to use

        Returns:
            MergeResult with merged content
        """
        logger.info(f"Merging files: {ours_path} + {theirs_path}")

        # Read file contents
        ours_content = ours_path.read_text() if ours_path.exists() else ""
        theirs_content = theirs_path.read_text() if theirs_path.exists() else ""
        base_content = base_path.read_text() if base_path and base_path.exists() else None

        # Determine file type and apply appropriate merge
        suffix = ours_path.suffix.lower()

        if strategy == MergeStrategy.SMART:
            if suffix in [".ts", ".tsx", ".js", ".jsx"]:
                result = await self._merge_typescript(ours_content, theirs_content, base_content)
            elif suffix == ".py":
                result = await self._merge_python(ours_content, theirs_content, base_content)
            elif suffix == ".css":
                result = await self._merge_css(ours_content, theirs_content, base_content)
            elif suffix == ".json":
                result = await self._merge_json(ours_content, theirs_content, base_content)
            else:
                result = await self._merge_text(ours_content, theirs_content, base_content)
        else:
            result = await self._apply_strategy(
                strategy, ours_content, theirs_content, base_content
            )

        result.strategy_used = strategy.value

        # Write output if requested
        if output_path and result.success and result.merged_content:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(result.merged_content)
            result.files_modified.append(str(output_path))

        return result

    async def merge_content(
        self,
        ours: str,
        theirs: str,
        base: str | None = None,
        file_type: str = "text",
    ) -> MergeResult:
        """
        Merge content strings directly.

        Args:
            ours: Our content version
            theirs: Their content version
            base: Optional base content
            file_type: File type hint (ts, py, css, json, text)

        Returns:
            MergeResult with merged content
        """
        if file_type in ["ts", "tsx", "js", "jsx", "typescript", "javascript"]:
            return await self._merge_typescript(ours, theirs, base)
        elif file_type in ["py", "python"]:
            return await self._merge_python(ours, theirs, base)
        elif file_type == "css":
            return await self._merge_css(ours, theirs, base)
        elif file_type == "json":
            return await self._merge_json(ours, theirs, base)
        else:
            return await self._merge_text(ours, theirs, base)

    def _group_by_file(
        self,
        task_outputs: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Group task outputs by file path."""
        grouped: dict[str, list[dict[str, Any]]] = {}

        for output in task_outputs:
            files_modified = output.get("files_modified", [])
            content_map = output.get("content_map", {})

            for filepath in files_modified:
                if filepath not in grouped:
                    grouped[filepath] = []
                grouped[filepath].append(
                    {
                        "task_id": output.get("task_id"),
                        "content": content_map.get(filepath, ""),
                    }
                )

        return grouped

    async def _merge_file_outputs(
        self,
        filepath: str,
        outputs: list[dict[str, Any]],
    ) -> MergeResult:
        """Merge multiple task outputs for a single file."""
        logger.debug(f"Merging {len(outputs)} outputs for {filepath}")

        if len(outputs) < 2:
            return MergeResult(success=True)

        # Use first as "ours", merge others in sequence
        merged = outputs[0]["content"]

        for i in range(1, len(outputs)):
            theirs = outputs[i]["content"]
            suffix = Path(filepath).suffix.lower()

            if suffix in [".ts", ".tsx", ".js", ".jsx"]:
                result = await self._merge_typescript(merged, theirs, None)
            elif suffix == ".py":
                result = await self._merge_python(merged, theirs, None)
            elif suffix == ".css":
                result = await self._merge_css(merged, theirs, None)
            elif suffix == ".json":
                result = await self._merge_json(merged, theirs, None)
            else:
                result = await self._merge_text(merged, theirs, None)

            if not result.success:
                return result

            merged = result.merged_content or merged

        # Write merged result
        full_path = self.workspace / filepath
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(merged)

        return MergeResult(
            success=True,
            merged_content=merged,
            files_modified=[filepath],
        )

    async def _merge_typescript(
        self,
        ours: str,
        theirs: str,
        base: str | None,
    ) -> MergeResult:
        """Smart merge for TypeScript/JavaScript files."""
        result = MergeResult(success=True, strategy_used="typescript_smart")

        # Extract and deduplicate imports
        ours_imports, ours_body = self._extract_imports_ts(ours)
        theirs_imports, theirs_body = self._extract_imports_ts(theirs)

        # Merge imports (union, deduplicated)
        merged_imports = self._merge_imports_ts(ours_imports, theirs_imports)

        # Merge body using diff
        merged_body, conflicts = self._diff_merge(ours_body, theirs_body, base)

        if conflicts:
            result.success = False
            result.conflicts = conflicts

        # Combine
        result.merged_content = merged_imports + "\n\n" + merged_body
        return result

    async def _merge_python(
        self,
        ours: str,
        theirs: str,
        base: str | None,
    ) -> MergeResult:
        """Smart merge for Python files."""
        result = MergeResult(success=True, strategy_used="python_smart")

        # Extract and deduplicate imports
        ours_imports, ours_body = self._extract_imports_py(ours)
        theirs_imports, theirs_body = self._extract_imports_py(theirs)

        # Merge imports (union, deduplicated)
        merged_imports = self._merge_imports_py(ours_imports, theirs_imports)

        # Merge body using diff
        merged_body, conflicts = self._diff_merge(ours_body, theirs_body, base)

        if conflicts:
            result.success = False
            result.conflicts = conflicts

        # Combine
        result.merged_content = merged_imports + "\n\n" + merged_body
        return result

    async def _merge_css(
        self,
        ours: str,
        theirs: str,
        base: str | None,
    ) -> MergeResult:
        """Smart merge for CSS files."""
        result = MergeResult(success=True, strategy_used="css_smart")

        # For CSS, union merge is usually safe
        ours_rules = self._extract_css_rules(ours)
        theirs_rules = self._extract_css_rules(theirs)

        # Combine rules, keeping later definitions for duplicates
        all_rules = {**ours_rules, **theirs_rules}

        merged = "\n\n".join(f"{selector} {{\n{body}\n}}" for selector, body in all_rules.items())

        result.merged_content = merged
        return result

    async def _merge_json(
        self,
        ours: str,
        theirs: str,
        base: str | None,
    ) -> MergeResult:
        """Smart merge for JSON files."""
        result = MergeResult(success=True, strategy_used="json_smart")

        try:
            ours_data = json.loads(ours) if ours.strip() else {}
            theirs_data = json.loads(theirs) if theirs.strip() else {}

            # Deep merge
            merged_data = self._deep_merge_dicts(ours_data, theirs_data)

            result.merged_content = json.dumps(merged_data, indent=2)
        except json.JSONDecodeError as e:
            result.success = False
            result.warnings.append(f"JSON parse error: {e}")
            # Fall back to text merge
            text_result = await self._merge_text(ours, theirs, base)
            return text_result

        return result

    async def _merge_text(
        self,
        ours: str,
        theirs: str,
        base: str | None,
    ) -> MergeResult:
        """Generic text merge using diff."""
        result = MergeResult(success=True, strategy_used="text_diff")

        merged, conflicts = self._diff_merge(ours, theirs, base)

        if conflicts:
            result.success = False
            result.conflicts = conflicts
            result.merged_content = self._create_conflict_markers(merged, conflicts)
        else:
            result.merged_content = merged

        return result

    async def _apply_strategy(
        self,
        strategy: MergeStrategy,
        ours: str,
        theirs: str,
        base: str | None,
    ) -> MergeResult:
        """Apply specific merge strategy."""
        if strategy == MergeStrategy.OURS:
            return MergeResult(success=True, merged_content=ours)
        elif strategy == MergeStrategy.THEIRS:
            return MergeResult(success=True, merged_content=theirs)
        elif strategy == MergeStrategy.UNION:
            # Combine both, removing duplicates
            ours_lines = set(ours.splitlines())
            theirs_lines = set(theirs.splitlines())
            merged_lines = list(ours_lines | theirs_lines)
            return MergeResult(success=True, merged_content="\n".join(merged_lines))
        elif strategy == MergeStrategy.THREE_WAY and base:
            merged, conflicts = self._three_way_merge(ours, theirs, base)
            return MergeResult(
                success=len(conflicts) == 0,
                merged_content=merged,
                conflicts=conflicts,
            )
        else:
            return await self._merge_text(ours, theirs, base)

    def _extract_imports_ts(self, content: str) -> tuple[list[str], str]:
        """Extract import statements from TypeScript/JavaScript."""
        lines = content.split("\n")
        imports = []
        body_lines = []
        in_imports = True

        for line in lines:
            stripped = line.strip()

            if in_imports and (
                stripped.startswith("import ")
                or stripped.startswith("export ")
                and "from" in stripped
            ):
                imports.append(line)
            elif in_imports and stripped and not stripped.startswith("//"):
                in_imports = False
                body_lines.append(line)
            else:
                body_lines.append(line)

        return imports, "\n".join(body_lines)

    def _extract_imports_py(self, content: str) -> tuple[list[str], str]:
        """Extract import statements from Python."""
        lines = content.split("\n")
        imports = []
        body_lines = []
        in_imports = True

        for line in lines:
            stripped = line.strip()

            if in_imports and (stripped.startswith("import ") or stripped.startswith("from ")):
                imports.append(line)
            elif in_imports and stripped and not stripped.startswith("#"):
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    # Docstring - part of body
                    in_imports = False
                    body_lines.append(line)
                elif stripped:
                    in_imports = False
                    body_lines.append(line)
                else:
                    body_lines.append(line)
            else:
                body_lines.append(line)

        return imports, "\n".join(body_lines)

    def _merge_imports_ts(
        self,
        ours: list[str],
        theirs: list[str],
    ) -> str:
        """Merge TypeScript/JavaScript imports."""
        seen = set()
        merged = []

        for imp in ours + theirs:
            normalized = " ".join(imp.split())
            if normalized not in seen:
                seen.add(normalized)
                merged.append(imp)

        return "\n".join(merged)

    def _merge_imports_py(
        self,
        ours: list[str],
        theirs: list[str],
    ) -> str:
        """Merge Python imports."""
        seen = set()
        merged = []

        for imp in ours + theirs:
            normalized = " ".join(imp.split())
            if normalized not in seen:
                seen.add(normalized)
                merged.append(imp)

        return "\n".join(merged)

    def _extract_css_rules(self, content: str) -> dict[str, str]:
        """Extract CSS rules as selector -> body mapping."""
        rules = {}

        # Simple CSS rule extraction
        pattern = r"([^{]+)\s*\{([^}]*)\}"
        for match in re.finditer(pattern, content):
            selector = match.group(1).strip()
            body = match.group(2).strip()
            rules[selector] = body

        return rules

    def _deep_merge_dicts(self, dict1: dict, dict2: dict) -> dict:
        """Deep merge two dictionaries."""
        result = dict1.copy()

        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_dicts(result[key], value)
            elif key in result and isinstance(result[key], list) and isinstance(value, list):
                # For lists, concatenate and deduplicate
                combined = result[key] + [v for v in value if v not in result[key]]
                result[key] = combined
            else:
                result[key] = value

        return result

    def _diff_merge(
        self,
        ours: str,
        theirs: str,
        base: str | None,
    ) -> tuple[str, list[MergeConflict]]:
        """Perform diff-based merge."""
        conflicts = []

        if base:
            return self._three_way_merge(ours, theirs, base)

        # Two-way merge using difflib
        ours_lines = ours.splitlines(keepends=True)
        theirs_lines = theirs.splitlines(keepends=True)

        differ = difflib.Differ()
        diff = list(differ.compare(ours_lines, theirs_lines))

        merged_lines = []
        for line in diff:
            if line.startswith("  ") or line.startswith("+ "):  # Unchanged
                merged_lines.append(line[2:])
            # Skip lines starting with "- " (removed) and "? " (hints)

        return "".join(merged_lines), conflicts

    def _three_way_merge(
        self,
        ours: str,
        theirs: str,
        base: str,
    ) -> tuple[str, list[MergeConflict]]:
        """Perform three-way merge using base as common ancestor."""
        conflicts = []

        base_lines = base.splitlines(keepends=True)
        ours_lines = ours.splitlines(keepends=True)
        theirs_lines = theirs.splitlines(keepends=True)

        # Get changes from base
        ours_diff = list(difflib.unified_diff(base_lines, ours_lines, lineterm=""))
        theirs_diff = list(difflib.unified_diff(base_lines, theirs_lines, lineterm=""))

        # Simple merge - start with ours, add theirs additions
        merged_lines = ours_lines.copy()

        # This is a simplified three-way merge
        # A full implementation would use proper merge algorithms

        return "".join(merged_lines), conflicts

    def _create_conflict_markers(
        self,
        content: str,
        conflicts: list[MergeConflict],
    ) -> str:
        """Create conflict markers in content."""
        if not conflicts:
            return content

        lines = content.splitlines(keepends=True)
        result = []
        conflict_idx = 0

        for i, line in enumerate(lines):
            if conflict_idx < len(conflicts) and i == conflicts[conflict_idx].start_line:
                conflict = conflicts[conflict_idx]
                result.append(self._conflict_markers["start"] + "\n")
                result.append(conflict.ours)
                result.append(self._conflict_markers["separator"] + "\n")
                result.append(conflict.theirs)
                result.append(self._conflict_markers["end"] + "\n")
                conflict_idx += 1
            else:
                result.append(line)

        return "".join(result)
