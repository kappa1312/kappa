"""
Terminal Session Manager for Kappa OS.

Spawns and manages Claude Code terminal sessions using subprocess.
This is the primary execution environment for 80%+ of tasks.
"""

import asyncio
import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from src.core.state import TaskResult

from .base import (
    BaseSession,
    BaseSessionManager,
    SessionConfig,
    SessionEvent,
    SessionEventType,
    SessionInfo,
    SessionResult,
    SessionStatus,
)

# =============================================================================
# TERMINAL SESSION MANAGER (NEW)
# =============================================================================


class TerminalSessionManager(BaseSessionManager):
    """
    Manages Claude Code terminal sessions.

    Uses subprocess to spawn `claude` CLI in non-interactive mode.
    Each session runs in its own process with isolated workspace.

    Example:
        >>> manager = TerminalSessionManager(max_concurrent=5)
        >>> session_id = await manager.create_session(
        ...     task_id="task-1",
        ...     prompt="Create a hello world script",
        ...     workspace="/tmp/project",
        ...     context={}
        ... )
        >>> result = await manager.wait_for_completion(session_id)
        >>> print(result.is_success())
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        default_config: SessionConfig | None = None,
        claude_path: str | None = None,
    ):
        """
        Initialize terminal session manager.

        Args:
            max_concurrent: Maximum concurrent sessions.
            default_config: Default session configuration.
            claude_path: Path to claude CLI (auto-detected if None).
        """
        super().__init__(max_concurrent, default_config)

        # Find claude CLI
        self.claude_path = claude_path or self._find_claude_path()
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._output_buffers: dict[str, dict[str, list[str]]] = {}

    def _find_claude_path(self) -> str:
        """Find the claude CLI executable."""
        # Check common locations
        locations = [
            "claude",  # In PATH
            "/usr/local/bin/claude",
            "/opt/homebrew/bin/claude",
            os.path.expanduser("~/.local/bin/claude"),
        ]

        for loc in locations:
            if shutil.which(loc):
                return loc

        # Default to "claude" and let it fail if not found
        return "claude"

    async def create_session(
        self,
        task_id: str,
        prompt: str,
        workspace: str,
        context: dict[str, Any],
        config: SessionConfig | None = None,
    ) -> str:
        """Spawn a new Claude Code terminal session."""

        session_config = config or self.default_config
        session_id = str(uuid.uuid4())

        logger.info(f"Creating terminal session {session_id} for task {task_id}")

        # Acquire semaphore (limits concurrent sessions)
        await self._semaphore.acquire()

        try:
            # Prepare workspace
            workspace_path = Path(workspace)
            workspace_path.mkdir(parents=True, exist_ok=True)

            # Write prompt to file for reference
            prompt_file = workspace_path / f".kappa_prompt_{session_id[:8]}.md"
            prompt_file.write_text(prompt)

            # Write context to file if provided
            context_file = None
            if context:
                context_file = workspace_path / f".kappa_context_{session_id[:8]}.json"
                context_file.write_text(json.dumps(context, indent=2, default=str))

            # Build environment
            env = os.environ.copy()
            env.update(session_config.environment)
            env["KAPPA_SESSION_ID"] = session_id
            env["KAPPA_TASK_ID"] = task_id

            # Build claude command arguments
            args = [
                self.claude_path,
                "--print",  # Non-interactive mode
                "--output-format",
                "text",
                "--max-turns",
                str(session_config.max_turns),
            ]

            # Add permission flags if configured
            if session_config.dangerously_skip_permissions:
                args.append("--dangerously-skip-permissions")

            # Add prompt
            args.extend(["-p", prompt])

            logger.debug(f"Spawning: {' '.join(args[:5])}...")

            # Spawn Claude Code process
            process = await asyncio.create_subprocess_exec(
                *args,
                cwd=str(workspace_path),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE,
            )

            # Track session
            async with self._lock:
                self._processes[session_id] = process
                self._output_buffers[session_id] = {"stdout": [], "stderr": []}
                self.active_sessions[session_id] = SessionInfo(
                    id=session_id,
                    task_id=task_id,
                    status=SessionStatus.RUNNING,
                    started_at=datetime.utcnow(),
                    pid=process.pid,
                )

            # Start output collection tasks
            asyncio.create_task(self._collect_output(session_id, "stdout", process.stdout))
            asyncio.create_task(self._collect_output(session_id, "stderr", process.stderr))

            # Emit started event
            await self._emit_event(
                SessionEvent(
                    event_type=SessionEventType.STARTED,
                    session_id=session_id,
                    task_id=task_id,
                    timestamp=datetime.utcnow(),
                    data={"workspace": workspace, "pid": process.pid},
                )
            )

            logger.info(f"Session {session_id} started with PID {process.pid}")
            return session_id

        except Exception as e:
            self._semaphore.release()
            logger.error(f"Failed to create session: {e}")
            raise

    async def _collect_output(
        self, session_id: str, stream_name: str, stream: asyncio.StreamReader
    ) -> None:
        """Collect output from a stream."""
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break

                decoded = line.decode("utf-8", errors="replace")

                async with self._lock:
                    if session_id in self._output_buffers:
                        self._output_buffers[session_id][stream_name].append(decoded)

                # Emit output event for real-time monitoring
                task_id = ""
                async with self._lock:
                    if session_id in self.active_sessions:
                        task_id = self.active_sessions[session_id].task_id or ""

                await self._emit_event(
                    SessionEvent(
                        event_type=SessionEventType.OUTPUT,
                        session_id=session_id,
                        task_id=task_id,
                        timestamp=datetime.utcnow(),
                        data={"stream": stream_name, "line": decoded.strip()},
                    )
                )

        except Exception as e:
            logger.debug(f"Output collection ended for {session_id}/{stream_name}: {e}")

    async def monitor_session(self, session_id: str) -> dict[str, Any]:
        """Get real-time status of a session."""

        async with self._lock:
            if session_id not in self.active_sessions:
                if session_id in self.completed_sessions:
                    result = self.completed_sessions[session_id]
                    return {
                        "status": result.status.value,
                        "completed": True,
                        "return_code": result.return_code,
                    }
                return {"error": "Session not found", "status": SessionStatus.FAILED.value}

            session = self.active_sessions[session_id]
            process = self._processes.get(session_id)

        if not process:
            return {"error": "Process not found", "status": SessionStatus.FAILED.value}

        # Check if process is still running
        if process.returncode is None:
            elapsed = (
                (datetime.utcnow() - session.started_at).total_seconds()
                if session.started_at
                else 0
            )

            # Try to get memory usage
            memory_mb = await self._get_process_memory(process.pid)

            return {
                "status": SessionStatus.RUNNING.value,
                "elapsed_seconds": elapsed,
                "pid": process.pid,
                "memory_mb": memory_mb,
                "output_lines": len(self._output_buffers.get(session_id, {}).get("stdout", [])),
            }
        else:
            return {
                "status": (
                    SessionStatus.COMPLETED.value
                    if process.returncode == 0
                    else SessionStatus.FAILED.value
                ),
                "return_code": process.returncode,
                "elapsed_seconds": session.duration_seconds,
            }

    async def _get_process_memory(self, pid: int | None) -> float | None:
        """Get memory usage of a process in MB."""
        if pid is None:
            return None
        try:
            result = await asyncio.create_subprocess_exec(
                "ps",
                "-o",
                "rss=",
                "-p",
                str(pid),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await result.communicate()
            if stdout:
                kb = int(stdout.decode().strip())
                return kb / 1024
        except Exception:
            pass
        return None

    async def get_output(self, session_id: str) -> SessionResult:
        """Get complete session output."""

        async with self._lock:
            if session_id in self.completed_sessions:
                return self.completed_sessions[session_id]

            if session_id not in self.active_sessions:
                return SessionResult(
                    session_id=session_id,
                    task_id="unknown",
                    status=SessionStatus.FAILED,
                    started_at=datetime.utcnow(),
                    error_message="Session not found",
                )

            session = self.active_sessions[session_id]
            buffers = self._output_buffers.get(session_id, {"stdout": [], "stderr": []})

        return SessionResult(
            session_id=session_id,
            task_id=session.task_id or "",
            status=session.status,
            started_at=session.started_at or datetime.utcnow(),
            stdout="".join(buffers["stdout"]),
            stderr="".join(buffers["stderr"]),
        )

    async def send_input(self, session_id: str, input_text: str) -> bool:
        """Send input to a running session."""

        async with self._lock:
            process = self._processes.get(session_id)

        if not process or process.returncode is not None:
            return False

        try:
            if process.stdin:
                process.stdin.write(input_text.encode() + b"\n")
                await process.stdin.drain()
            return True
        except Exception as e:
            logger.error(f"Failed to send input to session {session_id}: {e}")
            return False

    async def kill_session(self, session_id: str, force: bool = False) -> bool:
        """Terminate a running session."""

        async with self._lock:
            process = self._processes.get(session_id)
            session = self.active_sessions.get(session_id)

        if not process:
            return False

        try:
            if force:
                process.kill()
            else:
                process.terminate()

            # Wait a bit for process to terminate
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except TimeoutError:
                process.kill()
                await process.wait()

            # Update status
            async with self._lock:
                if session:
                    session.status = SessionStatus.CANCELLED
                    session.completed_at = datetime.utcnow()

                # Create result
                buffers = self._output_buffers.get(session_id, {"stdout": [], "stderr": []})
                result = SessionResult(
                    session_id=session_id,
                    task_id=session.task_id if session else "",
                    status=SessionStatus.CANCELLED,
                    started_at=session.started_at if session else datetime.utcnow(),
                    completed_at=datetime.utcnow(),
                    stdout="".join(buffers["stdout"]),
                    stderr="".join(buffers["stderr"]),
                    return_code=-15 if not force else -9,
                )
                self.completed_sessions[session_id] = result

                # Cleanup
                self.active_sessions.pop(session_id, None)
                self._processes.pop(session_id, None)
                self._output_buffers.pop(session_id, None)

            self._semaphore.release()

            logger.info(f"Session {session_id} killed")
            return True

        except Exception as e:
            logger.error(f"Failed to kill session {session_id}: {e}")
            return False

    async def wait_for_completion(
        self, session_id: str, timeout: int | None = None, poll_interval: float = 1.0
    ) -> SessionResult:
        """Wait for session to complete."""

        async with self._lock:
            if session_id not in self.active_sessions:
                if session_id in self.completed_sessions:
                    return self.completed_sessions[session_id]
                return SessionResult(
                    session_id=session_id,
                    task_id="unknown",
                    status=SessionStatus.FAILED,
                    started_at=datetime.utcnow(),
                    error_message="Session not found",
                )

            session = self.active_sessions[session_id]
            process = self._processes.get(session_id)

        if not process:
            return SessionResult(
                session_id=session_id,
                task_id=session.task_id or "",
                status=SessionStatus.FAILED,
                started_at=session.started_at or datetime.utcnow(),
                error_message="Process not found",
            )

        effective_timeout = timeout or self.default_config.timeout_seconds

        try:
            # Wait for process to complete
            await asyncio.wait_for(process.wait(), timeout=effective_timeout)

            # Collect final output
            async with self._lock:
                buffers = self._output_buffers.get(session_id, {"stdout": [], "stderr": []})

                completed_at = datetime.utcnow()
                started_at = session.started_at or datetime.utcnow()
                duration = (completed_at - started_at).total_seconds()

                # Determine status
                if process.returncode == 0:
                    status = SessionStatus.COMPLETED
                else:
                    status = SessionStatus.FAILED

                # Parse created files from output
                files_created = self._parse_created_files(buffers["stdout"])
                files_modified = self._parse_modified_files(buffers["stdout"])

                result = SessionResult(
                    session_id=session_id,
                    task_id=session.task_id or "",
                    status=status,
                    started_at=started_at,
                    completed_at=completed_at,
                    duration_seconds=duration,
                    stdout="".join(buffers["stdout"]),
                    stderr="".join(buffers["stderr"]),
                    return_code=process.returncode,
                    files_created=files_created,
                    files_modified=files_modified,
                )

                # Store result
                self.completed_sessions[session_id] = result

                # Update session info
                session.status = status
                session.completed_at = completed_at

                # Cleanup
                self.active_sessions.pop(session_id, None)
                self._processes.pop(session_id, None)
                self._output_buffers.pop(session_id, None)

            self._semaphore.release()

            # Emit completed event
            await self._emit_event(
                SessionEvent(
                    event_type=SessionEventType.COMPLETED,
                    session_id=session_id,
                    task_id=session.task_id or "",
                    timestamp=completed_at,
                    data={
                        "status": status.value,
                        "duration": duration,
                        "return_code": process.returncode,
                    },
                )
            )

            return result

        except TimeoutError:
            logger.warning(f"Session {session_id} timed out after {effective_timeout}s")

            # Kill the process
            await self.kill_session(session_id, force=True)

            async with self._lock:
                buffers = self._output_buffers.get(session_id, {"stdout": [], "stderr": []})

            result = SessionResult(
                session_id=session_id,
                task_id=session.task_id or "",
                status=SessionStatus.TIMEOUT,
                started_at=session.started_at or datetime.utcnow(),
                completed_at=datetime.utcnow(),
                stdout="".join(buffers["stdout"]),
                stderr="".join(buffers["stderr"]),
                error_message=f"Session timed out after {effective_timeout} seconds",
            )

            self.completed_sessions[session_id] = result

            await self._emit_event(
                SessionEvent(
                    event_type=SessionEventType.TIMEOUT,
                    session_id=session_id,
                    task_id=session.task_id or "",
                    timestamp=datetime.utcnow(),
                    data={"timeout_seconds": effective_timeout},
                )
            )

            return result

    def _parse_created_files(self, stdout_lines: list[str]) -> list[str]:
        """Parse created files from Claude Code output."""
        files = []
        for line in stdout_lines:
            line = line.strip()
            # Look for file creation patterns in output
            if "Created" in line or "Wrote" in line or "wrote" in line:
                # Try to extract file path
                for pattern in ["Created: ", "Created ", "Wrote: ", "Wrote ", "wrote "]:
                    if pattern in line:
                        parts = line.split(pattern)
                        if len(parts) > 1:
                            file_path = parts[1].strip().strip("`").strip("'").strip('"')
                            if (
                                file_path
                                and "/" in file_path
                                or file_path.endswith(
                                    (
                                        ".py",
                                        ".ts",
                                        ".js",
                                        ".tsx",
                                        ".jsx",
                                        ".json",
                                        ".md",
                                        ".yml",
                                        ".yaml",
                                    )
                                )
                            ):
                                files.append(file_path)
                                break
            # Also check for Write tool usage patterns
            if '"tool":"Write"' in line or '"tool": "Write"' in line:
                try:
                    data = json.loads(line)
                    if "file_path" in data.get("input", {}):
                        files.append(data["input"]["file_path"])
                except (json.JSONDecodeError, KeyError):
                    pass
        return list(set(files))

    def _parse_modified_files(self, stdout_lines: list[str]) -> list[str]:
        """Parse modified files from Claude Code output."""
        files = []
        for line in stdout_lines:
            line = line.strip()
            # Look for edit patterns
            if "Modified" in line or "Edited" in line or "Updated" in line:
                for pattern in [
                    "Modified: ",
                    "Edited: ",
                    "Updated: ",
                    "Modified ",
                    "Edited ",
                    "Updated ",
                ]:
                    if pattern in line:
                        parts = line.split(pattern)
                        if len(parts) > 1:
                            file_path = parts[1].strip().strip("`").strip("'").strip('"')
                            if file_path:
                                files.append(file_path)
                                break
            # Also check for Edit tool usage patterns
            if '"tool":"Edit"' in line or '"tool": "Edit"' in line:
                try:
                    data = json.loads(line)
                    if "file_path" in data.get("input", {}):
                        files.append(data["input"]["file_path"])
                except (json.JSONDecodeError, KeyError):
                    pass
        return list(set(files))


# =============================================================================
# LEGACY TERMINAL SESSION (kept for backward compatibility)
# =============================================================================


class TerminalSession(BaseSession):
    """
    Terminal-based Claude session using Claude Agent SDK.

    Executes tasks by interfacing with Claude through the
    official Agent SDK or subprocess.

    Example:
        >>> session = TerminalSession(
        ...     config=SessionConfig(working_directory="./project")
        ... )
        >>> await session.start()
        >>> result = await session.execute("Create a hello world script")
        >>> await session.close()
    """

    def __init__(
        self,
        config: SessionConfig | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> None:
        """Initialize terminal session."""
        super().__init__(config, session_id)
        self.task_id = task_id
        self._client: Any = None
        self._files_modified: list[str] = []
        self._token_usage: dict[str, int] = {"input": 0, "output": 0}

    async def start(self) -> None:
        """Start the terminal session."""
        await super().start()

        logger.info(f"Starting terminal session {self.id}")

        try:
            # Try to import claude_agent_sdk first
            from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

            options = ClaudeAgentOptions(
                system_prompt=self.config.system_prompt,
                max_turns=self.config.max_turns,
                cwd=self.config.working_directory,
            )

            self._client = ClaudeSDKClient(options=options)
            logger.debug(f"Session {self.id} client initialized")

        except ImportError:
            logger.warning("claude_agent_sdk not available, using mock client")
            self._client = MockClaudeClient()

        except Exception as e:
            logger.error(f"Failed to initialize session {self.id}: {e}")
            self.update_status(SessionStatus.FAILED, str(e))
            raise

    async def execute(self, prompt: str) -> TaskResult:
        """Execute a prompt/task in the terminal session."""
        if not self._client:
            await self.start()

        logger.info(f"Session {self.id} executing: {prompt[:50]}...")
        self.record_message("user", prompt)

        start_time = datetime.utcnow()

        try:
            # Execute with timeout
            import anyio

            with anyio.fail_after(self.config.timeout_seconds):
                response = await self._execute_with_client(prompt)

            duration = (datetime.utcnow() - start_time).total_seconds()

            self.record_message("assistant", response)

            result = TaskResult(
                task_id=self.task_id or "",
                session_id=self.id,
                success=True,
                output=response,
                files_modified=self._files_modified.copy(),
                duration_seconds=duration,
                token_usage=self._token_usage.copy(),
            )

            logger.info(f"Session {self.id} completed in {duration:.1f}s")
            return result

        except TimeoutError:
            logger.error(f"Session {self.id} timed out")
            self.update_status(SessionStatus.TIMEOUT, "Execution timed out")

            return TaskResult(
                task_id=self.task_id or "",
                session_id=self.id,
                success=False,
                error="Session timed out",
                duration_seconds=self.config.timeout_seconds,
            )

        except Exception as e:
            logger.error(f"Session {self.id} failed: {e}")
            self.update_status(SessionStatus.FAILED, str(e))

            return TaskResult(
                task_id=self.task_id or "",
                session_id=self.id,
                success=False,
                error=str(e),
            )

    async def _execute_with_client(self, prompt: str) -> str:
        """Execute prompt with the Claude client."""
        if hasattr(self._client, "query"):
            # Using claude_agent_sdk
            messages = []
            async for message in self._client.query(prompt):
                messages.append(str(message))

                # Track file modifications
                if hasattr(message, "tool_use"):
                    self._track_file_modifications(message)

                # Track token usage
                if hasattr(message, "usage"):
                    self._token_usage["input"] += message.usage.get("input_tokens", 0)
                    self._token_usage["output"] += message.usage.get("output_tokens", 0)

            return "\n".join(messages)

        elif hasattr(self._client, "send"):
            response = await self._client.send(prompt)
            return str(response)

        else:
            return await self._client.execute(prompt)

    def _track_file_modifications(self, message: Any) -> None:
        """Track file modifications from tool usage."""
        if not hasattr(message, "tool_use"):
            return

        tool_use = message.tool_use
        tool_name = getattr(tool_use, "name", "")

        if tool_name in ("Write", "Edit"):
            file_path = getattr(tool_use, "input", {}).get("file_path", "")
            if file_path and file_path not in self._files_modified:
                self._files_modified.append(file_path)

    async def close(self) -> None:
        """Close the terminal session."""
        logger.info(f"Closing session {self.id}")

        if self._client and hasattr(self._client, "close"):
            await self._client.close()

        await self.stop()

    async def __aenter__(self) -> "TerminalSession":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()


# =============================================================================
# MOCK CLIENT
# =============================================================================


class MockClaudeClient:
    """Mock Claude client for testing without API access."""

    async def execute(self, prompt: str) -> str:
        """Mock execution."""
        import asyncio

        await asyncio.sleep(0.1)
        return f"Mock execution completed for: {prompt[:50]}..."

    async def close(self) -> None:
        """Mock close."""
        pass
