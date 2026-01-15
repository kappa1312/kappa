"""
Metrics Collection and Reporting for Kappa OS

Provides comprehensive metrics collection and reporting:
- MetricsCollector: Collect and aggregate metrics
- MetricType: Types of metrics (counter, gauge, histogram)
- MetricReporter: Export metrics to various formats
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from loguru import logger


class MetricType(str, Enum):
    """Types of metrics that can be collected."""

    COUNTER = "counter"  # Monotonically increasing value
    GAUGE = "gauge"  # Point-in-time value
    HISTOGRAM = "histogram"  # Distribution of values
    SUMMARY = "summary"  # Statistical summary


@dataclass
class MetricValue:
    """Single metric value with metadata."""

    name: str
    type: MetricType
    value: float
    labels: dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    description: str = ""


class MetricReporter:
    """
    Export metrics to various formats.

    Supports:
    - JSON export
    - Prometheus format
    - CSV format
    - Custom formatters

    Usage:
        reporter = MetricReporter()
        json_output = reporter.to_json(metrics)
        prometheus_output = reporter.to_prometheus(metrics)
    """

    def __init__(self):
        self._formatters: dict[str, Callable] = {}
        self._register_default_formatters()

    def _register_default_formatters(self):
        """Register default output formatters."""
        self._formatters["json"] = self._format_json
        self._formatters["prometheus"] = self._format_prometheus
        self._formatters["csv"] = self._format_csv

    def to_json(self, collector: "MetricsCollector") -> str:
        """Export metrics to JSON format."""
        return self._formatters["json"](collector)

    def to_prometheus(self, collector: "MetricsCollector") -> str:
        """Export metrics to Prometheus format."""
        return self._formatters["prometheus"](collector)

    def to_csv(self, collector: "MetricsCollector") -> str:
        """Export metrics to CSV format."""
        return self._formatters["csv"](collector)

    def export(
        self,
        collector: "MetricsCollector",
        format: str = "json",
    ) -> str:
        """Export metrics in specified format."""
        formatter = self._formatters.get(format)
        if not formatter:
            raise ValueError(f"Unknown format: {format}")
        return formatter(collector)

    def register_formatter(
        self,
        name: str,
        formatter: Callable[["MetricsCollector"], str],
    ):
        """Register a custom formatter."""
        self._formatters[name] = formatter

    def _format_json(self, collector: "MetricsCollector") -> str:
        """Format metrics as JSON."""
        data = collector.export()
        return json.dumps(data, indent=2, default=str)

    def _format_prometheus(self, collector: "MetricsCollector") -> str:
        """Format metrics in Prometheus exposition format."""
        lines = []
        summary = collector.get_summary()

        # Task metrics
        lines.append("# HELP kappa_tasks_total Total number of tasks")
        lines.append("# TYPE kappa_tasks_total counter")
        lines.append(f"kappa_tasks_total {summary['tasks']['total']}")

        lines.append("# HELP kappa_tasks_successful Successful tasks")
        lines.append("# TYPE kappa_tasks_successful counter")
        lines.append(f"kappa_tasks_successful {summary['tasks']['successful']}")

        lines.append("# HELP kappa_tasks_failed Failed tasks")
        lines.append("# TYPE kappa_tasks_failed counter")
        lines.append(f"kappa_tasks_failed {summary['tasks']['failed']}")

        lines.append("# HELP kappa_task_duration_seconds Total task duration")
        lines.append("# TYPE kappa_task_duration_seconds gauge")
        lines.append(f"kappa_task_duration_seconds {summary['tasks']['total_duration_seconds']}")

        # Session metrics
        lines.append("# HELP kappa_sessions_total Total sessions")
        lines.append("# TYPE kappa_sessions_total counter")
        lines.append(f"kappa_sessions_total {summary['sessions']['total']}")

        # Token metrics
        lines.append("# HELP kappa_tokens_input Total input tokens")
        lines.append("# TYPE kappa_tokens_input counter")
        lines.append(f"kappa_tokens_input {summary['tokens']['input']}")

        lines.append("# HELP kappa_tokens_output Total output tokens")
        lines.append("# TYPE kappa_tokens_output counter")
        lines.append(f"kappa_tokens_output {summary['tokens']['output']}")

        # Wave metrics
        for wave in summary["waves"]["details"]:
            wave_idx = wave.get("wave_index", 0)
            lines.append(f"# HELP kappa_wave_{wave_idx}_tasks Wave {wave_idx} tasks")
            lines.append(f"# TYPE kappa_wave_{wave_idx}_tasks gauge")
            lines.append(f"kappa_wave_{wave_idx}_tasks {wave.get('task_count', 0)}")
            lines.append(f"kappa_wave_{wave_idx}_success_rate {wave.get('success_rate', 0)}")

        return "\n".join(lines)

    def _format_csv(self, collector: "MetricsCollector") -> str:
        """Format metrics as CSV."""
        lines = ["metric,type,value,timestamp"]
        summary = collector.get_summary()

        # Flatten metrics to CSV
        lines.append(f"tasks_total,counter,{summary['tasks']['total']},{time.time()}")
        lines.append(f"tasks_successful,counter,{summary['tasks']['successful']},{time.time()}")
        lines.append(f"tasks_failed,counter,{summary['tasks']['failed']},{time.time()}")
        lines.append(
            f"task_duration_avg,gauge,{summary['tasks']['avg_duration_seconds']},{time.time()}"
        )
        lines.append(f"sessions_total,counter,{summary['sessions']['total']},{time.time()}")
        lines.append(f"tokens_input,counter,{summary['tokens']['input']},{time.time()}")
        lines.append(f"tokens_output,counter,{summary['tokens']['output']},{time.time()}")

        return "\n".join(lines)


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
        self._custom_metrics: list[MetricValue] = []
        self._start_time: datetime | None = None
        self._counters: dict[str, float] = {}
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = {}

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
        self._task_metrics.append(
            {
                "task_id": task_id,
                "duration_seconds": duration_seconds,
                "success": success,
                "token_usage": token_usage or {},
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        # Update counters
        self.increment_counter("tasks_total")
        if success:
            self.increment_counter("tasks_successful")
        else:
            self.increment_counter("tasks_failed")

        # Record duration in histogram
        self.record_histogram("task_duration", duration_seconds)

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
        self._session_metrics.append(
            {
                "session_id": session_id,
                "duration_seconds": duration_seconds,
                "success": success,
                "files_modified": files_modified,
                "token_usage": token_usage or {},
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        # Update counters
        self.increment_counter("sessions_total")
        if success:
            self.increment_counter("sessions_successful")
        self.increment_counter("files_modified", files_modified)

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
        self._wave_metrics.append(
            {
                "wave_index": wave_index,
                "task_count": task_count,
                "success_count": success_count,
                "success_rate": success_count / task_count if task_count > 0 else 0,
                "duration_seconds": duration_seconds,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        # Update gauges
        self.set_gauge(f"wave_{wave_index}_tasks", task_count)
        self.set_gauge(
            f"wave_{wave_index}_success_rate", success_count / task_count if task_count > 0 else 0
        )

    def record_conflict_metrics(
        self,
        conflict_count: int,
        resolved_count: int,
        critical_count: int = 0,
    ) -> None:
        """
        Record conflict resolution metrics.

        Args:
            conflict_count: Total conflicts detected.
            resolved_count: Conflicts successfully resolved.
            critical_count: Critical conflicts.
        """
        self.increment_counter("conflicts_total", conflict_count)
        self.increment_counter("conflicts_resolved", resolved_count)
        self.increment_counter("conflicts_critical", critical_count)

    # Generic metric methods
    def increment_counter(self, name: str, value: float = 1.0):
        """Increment a counter metric."""
        if name not in self._counters:
            self._counters[name] = 0
        self._counters[name] += value

    def set_gauge(self, name: str, value: float):
        """Set a gauge metric."""
        self._gauges[name] = value

    def record_histogram(self, name: str, value: float):
        """Record a value in a histogram."""
        if name not in self._histograms:
            self._histograms[name] = []
        self._histograms[name].append(value)

    def record_custom(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        labels: dict[str, str] | None = None,
        description: str = "",
    ):
        """Record a custom metric."""
        self._custom_metrics.append(
            MetricValue(
                name=name,
                type=metric_type,
                value=value,
                labels=labels or {},
                description=description,
            )
        )

    def get_counter(self, name: str) -> float:
        """Get current counter value."""
        return self._counters.get(name, 0)

    def get_gauge(self, name: str) -> float:
        """Get current gauge value."""
        return self._gauges.get(name, 0)

    def get_histogram_stats(self, name: str) -> dict[str, float]:
        """Get histogram statistics."""
        values = self._histograms.get(name, [])
        if not values:
            return {"count": 0}

        sorted_values = sorted(values)
        count = len(values)

        return {
            "count": count,
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / count,
            "p50": sorted_values[count // 2],
            "p90": sorted_values[int(count * 0.9)] if count > 10 else max(values),
            "p99": sorted_values[int(count * 0.99)] if count > 100 else max(values),
        }

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
            t.get("token_usage", {}).get("input", 0) for t in self._task_metrics
        )
        total_output_tokens = sum(
            t.get("token_usage", {}).get("output", 0) for t in self._task_metrics
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
            "counters": self._counters.copy(),
            "gauges": self._gauges.copy(),
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
            "custom": [
                {
                    "name": m.name,
                    "type": m.type.value,
                    "value": m.value,
                    "labels": m.labels,
                    "timestamp": m.timestamp,
                    "description": m.description,
                }
                for m in self._custom_metrics
            ],
            "histograms": {name: self.get_histogram_stats(name) for name in self._histograms},
            "collection_started": self._start_time.isoformat() if self._start_time else None,
            "exported_at": datetime.utcnow().isoformat(),
        }

    def reset(self) -> None:
        """Reset all collected metrics."""
        self._task_metrics = []
        self._session_metrics = []
        self._wave_metrics = []
        self._custom_metrics = []
        self._counters = {}
        self._gauges = {}
        self._histograms = {}
        self._start_time = None
        logger.info("Metrics reset")
