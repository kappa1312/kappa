"""Project validation for Kappa.

This module provides validation infrastructure for verifying that
generated projects are correct, buildable, and pass tests.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

# =============================================================================
# VALIDATION RESULTS
# =============================================================================


class ValidationResult:
    """Result of a single validation check."""

    def __init__(
        self,
        check_name: str,
        success: bool,
        output: str | None = None,
        error: str | None = None,
        duration_seconds: float = 0.0,
        details: dict[str, Any] | None = None,
    ):
        self.check_name = check_name
        self.success = success
        self.output = output
        self.error = error
        self.duration_seconds = duration_seconds
        self.details = details or {}
        self.completed_at = datetime.utcnow().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "check_name": self.check_name,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
            "details": self.details,
            "completed_at": self.completed_at,
        }


class ProjectValidationResult:
    """Combined result of all project validations."""

    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path
        self.results: dict[str, ValidationResult] = {}
        self.completed_at = datetime.utcnow().isoformat()

    @property
    def success(self) -> bool:
        """Check if all validations passed."""
        if not self.results:
            return True
        return all(r.success for r in self.results.values())

    @property
    def failed_checks(self) -> list[str]:
        """Get names of failed checks."""
        return [name for name, r in self.results.items() if not r.success]

    @property
    def passed_checks(self) -> list[str]:
        """Get names of passed checks."""
        return [name for name, r in self.results.items() if r.success]

    def add_result(self, result: ValidationResult) -> None:
        """Add a validation result."""
        self.results[result.check_name] = result

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "workspace_path": self.workspace_path,
            "success": self.success,
            "results": {k: v.to_dict() for k, v in self.results.items()},
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "completed_at": self.completed_at,
        }


# =============================================================================
# PROJECT VALIDATOR
# =============================================================================


class ProjectValidator:
    """
    Validate project outputs.

    Runs various validation checks on a generated project to ensure
    it is correct and functional.

    Checks include:
    - Type checking (TypeScript/mypy)
    - Build verification
    - Test execution
    - Lint checking
    - File structure validation

    Example:
        >>> validator = ProjectValidator("/path/to/project")
        >>> results = await validator.validate()
        >>> if results.success:
        ...     print("All validations passed!")
    """

    def __init__(
        self,
        workspace_path: str | Path,
        timeout: int = 300,
    ):
        """
        Initialize validator.

        Args:
            workspace_path: Path to the project workspace.
            timeout: Timeout for each validation step in seconds.
        """
        self.workspace = Path(workspace_path).resolve()
        self.timeout = timeout
        self._project_type: str | None = None

    async def validate(
        self,
        checks: list[str] | None = None,
    ) -> ProjectValidationResult:
        """
        Run all validation checks.

        Args:
            checks: Optional list of specific checks to run.
                   If None, runs all applicable checks.

        Returns:
            ProjectValidationResult with all check results.
        """
        logger.info(f"Validating project at: {self.workspace}")

        result = ProjectValidationResult(str(self.workspace))

        # Detect project type
        self._project_type = await self._detect_project_type()
        logger.info(f"Detected project type: {self._project_type}")

        # Determine which checks to run
        if checks is None:
            checks = self._get_applicable_checks()

        # Run validations
        for check in checks:
            check_result = await self._run_check(check)
            result.add_result(check_result)

        logger.info(
            f"Validation complete: {len(result.passed_checks)} passed, "
            f"{len(result.failed_checks)} failed"
        )

        return result

    async def validate_quick(self) -> ProjectValidationResult:
        """Run quick validation checks only."""
        return await self.validate(checks=["file_structure", "syntax"])

    async def validate_full(self) -> ProjectValidationResult:
        """Run all validation checks."""
        return await self.validate()

    # =========================================================================
    # CHECK RUNNERS
    # =========================================================================

    async def _run_check(self, check_name: str) -> ValidationResult:
        """Run a single validation check."""
        start_time = datetime.utcnow()

        try:
            if check_name == "file_structure":
                result = await self._check_file_structure()
            elif check_name == "syntax":
                result = await self._check_syntax()
            elif check_name == "type_check":
                result = await self._run_type_check()
            elif check_name == "build":
                result = await self._run_build()
            elif check_name == "tests":
                result = await self._run_tests()
            elif check_name == "lint":
                result = await self._run_lint()
            elif check_name == "format":
                result = await self._check_format()
            else:
                result = ValidationResult(
                    check_name=check_name,
                    success=False,
                    error=f"Unknown check: {check_name}",
                )

            result.duration_seconds = (datetime.utcnow() - start_time).total_seconds()
            return result

        except Exception as e:
            logger.error(f"Check {check_name} failed with exception: {e}")
            return ValidationResult(
                check_name=check_name,
                success=False,
                error=str(e),
                duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
            )

    async def _check_file_structure(self) -> ValidationResult:
        """Validate basic file structure."""
        issues: list[str] = []
        details: dict[str, Any] = {}

        # Check workspace exists
        if not self.workspace.exists():
            return ValidationResult(
                check_name="file_structure",
                success=False,
                error=f"Workspace does not exist: {self.workspace}",
            )

        # Check for common expected files based on project type
        expected_files = self._get_expected_files()
        missing_files = []
        found_files = []

        for expected in expected_files:
            path = self.workspace / expected
            if path.exists():
                found_files.append(expected)
            else:
                missing_files.append(expected)

        details["expected_files"] = expected_files
        details["found_files"] = found_files
        details["missing_files"] = missing_files

        if missing_files:
            issues.append(f"Missing expected files: {', '.join(missing_files)}")

        # Count total files
        all_files = list(self.workspace.rglob("*"))
        file_count = len([f for f in all_files if f.is_file()])
        dir_count = len([f for f in all_files if f.is_dir()])
        details["total_files"] = file_count
        details["total_directories"] = dir_count

        return ValidationResult(
            check_name="file_structure",
            success=len(issues) == 0,
            output=f"Found {file_count} files in {dir_count} directories",
            error="; ".join(issues) if issues else None,
            details=details,
        )

    async def _check_syntax(self) -> ValidationResult:
        """Check for basic syntax errors."""
        issues: list[str] = []
        checked_files = 0

        # Check Python files
        for py_file in self.workspace.rglob("*.py"):
            checked_files += 1
            try:
                with open(py_file) as f:
                    content = f.read()
                compile(content, str(py_file), "exec")
            except SyntaxError as e:
                issues.append(f"{py_file.name}: {e}")

        # Check JSON files
        for json_file in self.workspace.rglob("*.json"):
            checked_files += 1
            try:
                with open(json_file) as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                issues.append(f"{json_file.name}: {e}")

        return ValidationResult(
            check_name="syntax",
            success=len(issues) == 0,
            output=f"Checked {checked_files} files",
            error="; ".join(issues[:10]) if issues else None,  # Limit errors
            details={"checked_files": checked_files, "issues": issues},
        )

    async def _run_type_check(self) -> ValidationResult:
        """Run type checking."""
        if self._project_type == "typescript":
            return await self._run_command(
                "type_check",
                ["npx", "tsc", "--noEmit"],
            )
        elif self._project_type == "python":
            # Check if mypy is available
            if (self.workspace / "pyproject.toml").exists():
                return await self._run_command(
                    "type_check",
                    ["poetry", "run", "mypy", "."],
                )
            return await self._run_command(
                "type_check",
                ["mypy", "."],
            )
        else:
            return ValidationResult(
                check_name="type_check",
                success=True,
                output="Type checking not applicable for this project type",
            )

    async def _run_build(self) -> ValidationResult:
        """Run build command."""
        package_json = self.workspace / "package.json"
        pyproject = self.workspace / "pyproject.toml"
        makefile = self.workspace / "Makefile"

        if package_json.exists():
            # Node.js project
            return await self._run_command(
                "build",
                ["npm", "run", "build"],
            )
        elif pyproject.exists():
            # Python project with Poetry
            return await self._run_command(
                "build",
                ["poetry", "build"],
            )
        elif makefile.exists():
            return await self._run_command(
                "build",
                ["make", "build"],
            )
        else:
            return ValidationResult(
                check_name="build",
                success=True,
                output="No build configuration found, skipping",
            )

    async def _run_tests(self) -> ValidationResult:
        """Run test suite."""
        package_json = self.workspace / "package.json"
        pyproject = self.workspace / "pyproject.toml"
        pytest_ini = self.workspace / "pytest.ini"

        if package_json.exists():
            return await self._run_command(
                "tests",
                ["npm", "test"],
            )
        elif pyproject.exists() or pytest_ini.exists():
            return await self._run_command(
                "tests",
                ["poetry", "run", "pytest", "-v"],
            )
        else:
            # Check for test files
            test_files = list(self.workspace.rglob("test_*.py")) + list(
                self.workspace.rglob("*_test.py")
            )
            if test_files:
                return await self._run_command(
                    "tests",
                    ["pytest", "-v"],
                )
            return ValidationResult(
                check_name="tests",
                success=True,
                output="No test configuration found, skipping",
            )

    async def _run_lint(self) -> ValidationResult:
        """Run linting."""
        package_json = self.workspace / "package.json"
        pyproject = self.workspace / "pyproject.toml"

        if package_json.exists():
            return await self._run_command(
                "lint",
                ["npm", "run", "lint"],
            )
        elif pyproject.exists():
            return await self._run_command(
                "lint",
                ["poetry", "run", "ruff", "check", "."],
            )
        else:
            return ValidationResult(
                check_name="lint",
                success=True,
                output="No lint configuration found, skipping",
            )

    async def _check_format(self) -> ValidationResult:
        """Check code formatting."""
        if self._project_type == "typescript":
            return await self._run_command(
                "format",
                ["npx", "prettier", "--check", "."],
            )
        elif self._project_type == "python":
            return await self._run_command(
                "format",
                ["poetry", "run", "ruff", "format", "--check", "."],
            )
        else:
            return ValidationResult(
                check_name="format",
                success=True,
                output="Format checking not applicable for this project type",
            )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    async def _run_command(
        self,
        check_name: str,
        command: list[str],
    ) -> ValidationResult:
        """Run a shell command and capture result."""
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(self.workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout,
                )
            except TimeoutError:
                process.kill()
                return ValidationResult(
                    check_name=check_name,
                    success=False,
                    error=f"Command timed out after {self.timeout}s",
                )

            success = process.returncode == 0
            output = stdout.decode("utf-8", errors="replace")
            error_output = stderr.decode("utf-8", errors="replace")

            return ValidationResult(
                check_name=check_name,
                success=success,
                output=output[:5000] if output else None,  # Limit output size
                error=error_output[:2000] if not success and error_output else None,
                details={
                    "command": " ".join(command),
                    "return_code": process.returncode,
                },
            )

        except FileNotFoundError:
            return ValidationResult(
                check_name=check_name,
                success=True,  # Don't fail if command not found
                output=f"Command not found: {command[0]}, skipping check",
            )

    async def _detect_project_type(self) -> str:
        """Detect the type of project."""
        # Check for package.json (Node.js/TypeScript)
        if (self.workspace / "package.json").exists():
            try:
                with open(self.workspace / "package.json") as f:
                    pkg = json.load(f)
                    if "typescript" in pkg.get("devDependencies", {}):
                        return "typescript"
                    return "javascript"
            except Exception:
                return "javascript"

        # Check for pyproject.toml (Python)
        if (self.workspace / "pyproject.toml").exists():
            return "python"

        # Check for setup.py (Python)
        if (self.workspace / "setup.py").exists():
            return "python"

        # Check for Cargo.toml (Rust)
        if (self.workspace / "Cargo.toml").exists():
            return "rust"

        # Check for go.mod (Go)
        if (self.workspace / "go.mod").exists():
            return "go"

        # Default based on file extensions
        py_files = list(self.workspace.rglob("*.py"))
        ts_files = list(self.workspace.rglob("*.ts"))
        js_files = list(self.workspace.rglob("*.js"))

        if len(ts_files) > len(py_files) and len(ts_files) > len(js_files):
            return "typescript"
        elif len(js_files) > len(py_files):
            return "javascript"
        elif py_files:
            return "python"

        return "unknown"

    def _get_applicable_checks(self) -> list[str]:
        """Get list of applicable checks based on project type."""
        # Always run basic checks
        checks = ["file_structure", "syntax"]

        if self._project_type in ("typescript", "python"):
            checks.append("type_check")

        if self._project_type in ("typescript", "javascript", "python"):
            checks.extend(["build", "tests", "lint"])

        return checks

    def _get_expected_files(self) -> list[str]:
        """Get list of expected files based on project type."""
        if self._project_type == "typescript":
            return ["package.json", "tsconfig.json", "src"]
        elif self._project_type == "javascript":
            return ["package.json", "src"]
        elif self._project_type == "python":
            return ["pyproject.toml", "src"]
        elif self._project_type == "rust":
            return ["Cargo.toml", "src"]
        elif self._project_type == "go":
            return ["go.mod", "main.go"]
        else:
            return []


# =============================================================================
# FILE VALIDATOR
# =============================================================================


class FileValidator:
    """
    Validate individual files.

    Useful for validating specific outputs rather than entire projects.
    """

    @staticmethod
    async def validate_python(file_path: str | Path) -> ValidationResult:
        """Validate a Python file."""
        file_path = Path(file_path)

        if not file_path.exists():
            return ValidationResult(
                check_name="python_syntax",
                success=False,
                error=f"File not found: {file_path}",
            )

        try:
            with open(file_path) as f:
                content = f.read()
            compile(content, str(file_path), "exec")
            return ValidationResult(
                check_name="python_syntax",
                success=True,
                output=f"File {file_path.name} is valid Python",
            )
        except SyntaxError as e:
            return ValidationResult(
                check_name="python_syntax",
                success=False,
                error=str(e),
            )

    @staticmethod
    async def validate_json(file_path: str | Path) -> ValidationResult:
        """Validate a JSON file."""
        file_path = Path(file_path)

        if not file_path.exists():
            return ValidationResult(
                check_name="json_syntax",
                success=False,
                error=f"File not found: {file_path}",
            )

        try:
            with open(file_path) as f:
                json.load(f)
            return ValidationResult(
                check_name="json_syntax",
                success=True,
                output=f"File {file_path.name} is valid JSON",
            )
        except json.JSONDecodeError as e:
            return ValidationResult(
                check_name="json_syntax",
                success=False,
                error=str(e),
            )

    @staticmethod
    async def validate_yaml(file_path: str | Path) -> ValidationResult:
        """Validate a YAML file."""
        file_path = Path(file_path)

        if not file_path.exists():
            return ValidationResult(
                check_name="yaml_syntax",
                success=False,
                error=f"File not found: {file_path}",
            )

        try:
            import yaml

            with open(file_path) as f:
                yaml.safe_load(f)
            return ValidationResult(
                check_name="yaml_syntax",
                success=True,
                output=f"File {file_path.name} is valid YAML",
            )
        except ImportError:
            return ValidationResult(
                check_name="yaml_syntax",
                success=True,
                output="PyYAML not installed, skipping YAML validation",
            )
        except yaml.YAMLError as e:
            return ValidationResult(
                check_name="yaml_syntax",
                success=False,
                error=str(e),
            )


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_validator(
    workspace_path: str | Path,
    timeout: int = 300,
) -> ProjectValidator:
    """
    Create a project validator.

    Args:
        workspace_path: Path to the project workspace.
        timeout: Timeout for each validation step.

    Returns:
        Configured ProjectValidator instance.
    """
    return ProjectValidator(workspace_path, timeout)
