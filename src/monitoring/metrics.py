"""Metrics collection and reporting."""

from datetime import datetime
from typing import Any

from loguru import logger


class MetricsCollector:
    """
    Collect and aggregate performance metrics.

    Tracks execution time, token usage, success rates,
    and other key performance indicators.

    Example:
        >>> collector = MetricsCollector()
        >>> collector.record_task_completion("task-1", 45.2, True)
        >>> metrics = collector.get_summary()
    """

    def __init__(self) -> None:
        """Initialize the metrics collector."""
        self._task_metrics: list[dict[str, Any]] = []
        self._session_metrics: list[dict[str, Any]] = []
        self._wave_metrics: list[dict[str, Any]] = []
        self._start_time: datetime | None = None

    def start_collection(self) -> None:
        """Start metrics collection."""
        self._start_time = datetime.utcnow()
        logger.info("Metrics collection started")

    def record_task_completion(
        self,
        task_id: str,
        duration_seconds: float,
        success: bool,
        token_usage: dict[str, int] | None = None,
    ) -> None:
        """
        Record task completion metrics.

        Args:
            task_id: Task ID.
            duration_seconds: Execution duration.
            success: Whether task succeeded.
            token_usage: Token usage statistics.
        """
        self._task_metrics.append({
            "task_id": task_id,
            "duration_seconds": duration_seconds,
            "success": success,
            "token_usage": token_usage or {},
            "timestamp": datetime.utcnow().isoformat(),
        })

    def record_session_metrics(
        self,
        session_id: str,
        duration_seconds: float,
        success: bool,
        files_modified: int = 0,
        token_usage: dict[str, int] | None = None,
    ) -> None:
        """
        Record session metrics.

        Args:
            session_id: Session ID.
            duration_seconds: Session duration.
            success: Whether session succeeded.
            files_modified: Number of files modified.
            token_usage: Token usage statistics.
        """
        self._session_metrics.append({
            "session_id": session_id,
            "duration_seconds": duration_seconds,
            "success": success,
            "files_modified": files_modified,
            "token_usage": token_usage or {},
            "timestamp": datetime.utcnow().isoformat(),
        })

    def record_wave_completion(
        self,
        wave_index: int,
        task_count: int,
        success_count: int,
        duration_seconds: float,
    ) -> None:
        """
        Record wave completion metrics.

        Args:
            wave_index: Wave index.
            task_count: Total tasks in wave.
            success_count: Successful tasks.
            duration_seconds: Wave duration.
        """
        self._wave_metrics.append({
            "wave_index": wave_index,
            "task_count": task_count,
            "success_count": success_count,
            "success_rate": success_count / task_count if task_count > 0 else 0,
            "duration_seconds": duration_seconds,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def get_summary(self) -> dict[str, Any]:
        """
        Get metrics summary.

        Returns:
            Summary dictionary with aggregated metrics.
        """
        total_duration = 0.0
        if self._start_time:
            total_duration = (datetime.utcnow() - self._start_time).total_seconds()

        # Task metrics
        task_count = len(self._task_metrics)
        task_success = sum(1 for t in self._task_metrics if t["success"])
        task_duration = sum(t["duration_seconds"] for t in self._task_metrics)

        # Session metrics
        session_count = len(self._session_metrics)
        session_success = sum(1 for s in self._session_metrics if s["success"])

        # Token usage
        total_input_tokens = sum(
            t.get("token_usage", {}).get("input", 0)
            for t in self._task_metrics
        )
        total_output_tokens = sum(
            t.get("token_usage", {}).get("output", 0)
            for t in self._task_metrics
        )

        return {
            "total_duration_seconds": total_duration,
            "tasks": {
                "total": task_count,
                "successful": task_success,
                "failed": task_count - task_success,
                "success_rate": task_success / task_count if task_count > 0 else 0,
                "total_duration_seconds": task_duration,
                "avg_duration_seconds": task_duration / task_count if task_count > 0 else 0,
            },
            "sessions": {
                "total": session_count,
                "successful": session_success,
                "success_rate": session_success / session_count if session_count > 0 else 0,
            },
            "waves": {
                "total": len(self._wave_metrics),
                "details": self._wave_metrics,
            },
            "tokens": {
                "input": total_input_tokens,
                "output": total_output_tokens,
                "total": total_input_tokens + total_output_tokens,
            },
        }

    def export(self) -> dict[str, Any]:
        """
        Export all collected metrics.

        Returns:
            Complete metrics data.
        """
        return {
            "summary": self.get_summary(),
            "tasks": self._task_metrics,
            "sessions": self._session_metrics,
            "waves": self._wave_metrics,
            "collection_started": self._start_time.isoformat() if self._start_time else None,
            "exported_at": datetime.utcnow().isoformat(),
        }

    def reset(self) -> None:
        """Reset all collected metrics."""
        self._task_metrics = []
        self._session_metrics = []
        self._wave_metrics = []
        self._start_time = None
        logger.info("Metrics reset")
