"""
Unit tests for the MergeEngine module.

Tests the merge engine including:
- MergeStrategy enum
- MergeConflict and MergeResult dataclasses
- MergeEngine class with file-type specific merging
"""

import json

import pytest

from src.conflict.merge_engine import (
    MergeConflict,
    MergeEngine,
    MergeResult,
    MergeStrategy,
)

# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def merge_engine(tmp_path):
    """Create MergeEngine with temp workspace."""
    return MergeEngine(workspace_path=tmp_path)


@pytest.fixture
def ts_content_ours():
    """Sample TypeScript content - ours version."""
    return """import { useState } from 'react';
import { Button } from './Button';

export function App() {
  const [count, setCount] = useState(0);
  return <Button onClick={() => setCount(c => c + 1)}>{count}</Button>;
}
"""


@pytest.fixture
def ts_content_theirs():
    """Sample TypeScript content - theirs version."""
    return """import { useState, useEffect } from 'react';
import { Button } from './Button';

export function App() {
  const [count, setCount] = useState(0);
  useEffect(() => { console.log(count); }, [count]);
  return <Button onClick={() => setCount(c => c + 1)}>{count}</Button>;
}
"""


@pytest.fixture
def json_content_ours():
    """Sample JSON content - ours version."""
    return json.dumps({"name": "test", "version": "1.0.0", "dependencies": {"react": "^18.0.0"}})


@pytest.fixture
def json_content_theirs():
    """Sample JSON content - theirs version."""
    return json.dumps({"name": "test", "version": "1.0.0", "dependencies": {"lodash": "^4.0.0"}})


# =============================================================================
# TEST MergeStrategy ENUM
# =============================================================================


class TestMergeStrategy:
    """Tests for MergeStrategy enum."""

    def test_merge_strategy_values(self):
        """Test MergeStrategy enum values."""
        assert MergeStrategy.THREE_WAY.value == "three_way"
        assert MergeStrategy.TWO_WAY.value == "two_way"
        assert MergeStrategy.OURS.value == "ours"
        assert MergeStrategy.THEIRS.value == "theirs"
        assert MergeStrategy.UNION.value == "union"
        assert MergeStrategy.SMART.value == "smart"


# =============================================================================
# TEST MergeConflict DATACLASS
# =============================================================================


class TestMergeConflict:
    """Tests for MergeConflict dataclass."""

    def test_create_merge_conflict(self):
        """Test creating a MergeConflict."""
        conflict = MergeConflict(
            start_line=10,
            end_line=15,
            ours="our content",
            theirs="their content",
        )

        assert conflict.start_line == 10
        assert conflict.end_line == 15
        assert conflict.ours == "our content"
        assert conflict.theirs == "their content"
        assert conflict.resolved is False

    def test_merge_conflict_with_base(self):
        """Test MergeConflict with base content."""
        conflict = MergeConflict(
            start_line=5,
            end_line=8,
            ours="ours",
            theirs="theirs",
            base="base content",
        )

        assert conflict.base == "base content"


# =============================================================================
# TEST MergeResult DATACLASS
# =============================================================================


class TestMergeResult:
    """Tests for MergeResult dataclass."""

    def test_create_success_result(self):
        """Test creating a successful MergeResult."""
        result = MergeResult(
            success=True,
            merged_content="merged content",
            strategy_used="typescript_smart",
        )

        assert result.success is True
        assert result.merged_content == "merged content"
        assert len(result.conflicts) == 0

    def test_create_failed_result(self):
        """Test creating a failed MergeResult."""
        conflict = MergeConflict(start_line=1, end_line=5, ours="a", theirs="b")

        result = MergeResult(
            success=False,
            conflicts=[conflict],
        )

        assert result.success is False
        assert len(result.conflicts) == 1

    def test_result_to_dict(self):
        """Test MergeResult.to_dict() serialization."""
        result = MergeResult(
            success=True,
            merged_content="content",
            files_modified=["src/index.ts"],
            strategy_used="smart",
        )

        d = result.to_dict()

        assert d["success"] is True
        assert "has_conflicts" in d
        assert d["conflict_count"] == 0


# =============================================================================
# TEST MergeEngine CLASS
# =============================================================================


@pytest.mark.asyncio
class TestMergeEngine:
    """Tests for MergeEngine class."""

    async def test_engine_init(self, tmp_path):
        """Test MergeEngine initialization."""
        engine = MergeEngine(workspace_path=tmp_path)

        assert engine.workspace == tmp_path

    async def test_merge_content_text(self, merge_engine):
        """Test merging plain text content."""
        ours = "line 1\nline 2\nline 3"
        theirs = "line 1\nline 2\nline 4"

        result = await merge_engine.merge_content(ours, theirs, file_type="text")

        assert result.success is True
        assert result.merged_content is not None

    async def test_merge_content_typescript(self, merge_engine, ts_content_ours, ts_content_theirs):
        """Test merging TypeScript content."""
        result = await merge_engine.merge_content(
            ts_content_ours,
            ts_content_theirs,
            file_type="typescript",
        )

        assert result.success is True
        assert result.merged_content is not None
        # Should include imports from both
        assert "useState" in result.merged_content

    async def test_merge_content_json(self, merge_engine, json_content_ours, json_content_theirs):
        """Test merging JSON content."""
        result = await merge_engine.merge_content(
            json_content_ours,
            json_content_theirs,
            file_type="json",
        )

        assert result.success is True

        # Parse merged content
        merged = json.loads(result.merged_content)
        # Should have dependencies from both
        assert "react" in merged.get("dependencies", {})
        assert "lodash" in merged.get("dependencies", {})

    async def test_merge_files(self, tmp_path):
        """Test merging actual files."""
        engine = MergeEngine(workspace_path=tmp_path)

        # Create test files
        ours_file = tmp_path / "ours.txt"
        theirs_file = tmp_path / "theirs.txt"
        output_file = tmp_path / "merged.txt"

        ours_file.write_text("line 1\nline 2")
        theirs_file.write_text("line 1\nline 3")

        result = await engine.merge_files(
            ours_path=ours_file,
            theirs_path=theirs_file,
            output_path=output_file,
        )

        assert result.success is True
        assert output_file.exists()

    async def test_merge_task_outputs(self, tmp_path):
        """Test merging multiple task outputs."""
        engine = MergeEngine(workspace_path=tmp_path)

        task_outputs = [
            {
                "task_id": "task-1",
                "files_modified": ["src/index.ts"],
                "content_map": {"src/index.ts": "export const a = 1;"},
            },
            {
                "task_id": "task-2",
                "files_modified": ["src/utils.ts"],
                "content_map": {"src/utils.ts": "export const b = 2;"},
            },
        ]

        result = await engine.merge_task_outputs(task_outputs)

        assert result.success is True

    async def test_merge_strategy_ours(self, merge_engine):
        """Test OURS merge strategy."""
        ours = "our content"
        theirs = "their content"

        result = await merge_engine._apply_strategy(
            MergeStrategy.OURS,
            ours,
            theirs,
            None,
        )

        assert result.success is True
        assert result.merged_content == "our content"

    async def test_merge_strategy_theirs(self, merge_engine):
        """Test THEIRS merge strategy."""
        ours = "our content"
        theirs = "their content"

        result = await merge_engine._apply_strategy(
            MergeStrategy.THEIRS,
            ours,
            theirs,
            None,
        )

        assert result.success is True
        assert result.merged_content == "their content"


# =============================================================================
# TEST MERGE HELPERS
# =============================================================================


class TestMergeEngineHelpers:
    """Tests for MergeEngine helper methods."""

    @pytest.fixture
    def engine(self, tmp_path):
        return MergeEngine(workspace_path=tmp_path)

    def test_extract_imports_ts(self, engine):
        """Test TypeScript import extraction."""
        content = """import { useState } from 'react';
import lodash from 'lodash';

export function App() {}"""

        imports, body = engine._extract_imports_ts(content)

        assert len(imports) == 2
        assert "useState" in imports[0]

    def test_extract_imports_py(self, engine):
        """Test Python import extraction."""
        content = """import os
from pathlib import Path

def main():
    pass"""

        imports, body = engine._extract_imports_py(content)

        assert len(imports) == 2
        assert "import os" in imports

    def test_extract_css_rules(self, engine):
        """Test CSS rule extraction."""
        content = """.button {
  color: red;
}

.header {
  font-size: 16px;
}"""

        rules = engine._extract_css_rules(content)

        assert ".button" in rules
        assert ".header" in rules

    def test_deep_merge_dicts(self, engine):
        """Test deep dictionary merging."""
        dict1 = {"a": 1, "nested": {"x": 1}}
        dict2 = {"b": 2, "nested": {"y": 2}}

        result = engine._deep_merge_dicts(dict1, dict2)

        assert result["a"] == 1
        assert result["b"] == 2
        assert result["nested"]["x"] == 1
        assert result["nested"]["y"] == 2
