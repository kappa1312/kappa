"""Terminal session - Claude Agent SDK integration."""

from datetime import datetime
from pathlib import Path
from typing import Any

import anyio
from loguru import logger

from src.core.state import SessionStatus, TaskResult
from src.sessions.base import BaseSession, SessionConfig


class TerminalSession(BaseSession):
    """
    Terminal-based Claude session using Claude Agent SDK.

    Executes tasks by interfacing with Claude through the
    official Agent SDK, which provides terminal automation
    capabilities.

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
        """Initialize terminal session.

        Args:
            config: Session configuration.
            session_id: Optional session ID.
            task_id: Optional associated task ID.
        """
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
            from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

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
        """
        Execute a prompt/task in the terminal session.

        Args:
            prompt: The task prompt to execute.

        Returns:
            TaskResult with execution outcome.
        """
        if not self._client:
            await self.start()

        logger.info(f"Session {self.id} executing: {prompt[:50]}...")
        self.record_message("user", prompt)

        start_time = datetime.utcnow()

        try:
            # Execute with timeout
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
        """Execute prompt with the Claude client.

        Args:
            prompt: Task prompt.

        Returns:
            Response text.
        """
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
            # Using ClaudeSDKClient
            response = await self._client.send(prompt)
            return str(response)

        else:
            # Mock client
            return await self._client.execute(prompt)

    def _track_file_modifications(self, message: Any) -> None:
        """Track file modifications from tool usage.

        Args:
            message: Message with potential tool usage.
        """
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


class MockClaudeClient:
    """Mock Claude client for testing without API access."""

    async def execute(self, prompt: str) -> str:
        """Mock execution.

        Args:
            prompt: Task prompt.

        Returns:
            Mock response.
        """
        # Simulate some work
        await anyio.sleep(0.1)

        return f"Mock execution completed for: {prompt[:50]}..."

    async def close(self) -> None:
        """Mock close."""
        pass
