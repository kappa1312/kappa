"""Health check utilities for Kappa system."""

from datetime import datetime
from typing import Any

from loguru import logger


class HealthChecker:
    """
    System health checker.

    Verifies that all required components are operational
    and within healthy parameters.

    Example:
        >>> checker = HealthChecker()
        >>> health = await checker.check_all()
        >>> health["status"]
        'healthy'
    """

    def __init__(self) -> None:
        """Initialize the health checker."""
        self._checks: dict[str, Any] = {}

    async def check_database(self) -> dict[str, Any]:
        """
        Check database connectivity.

        Returns:
            Health check result.
        """
        try:
            from src.knowledge.database import health_check

            healthy = await health_check()

            return {
                "name": "database",
                "status": "healthy" if healthy else "unhealthy",
                "message": "Connected" if healthy else "Connection failed",
            }

        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "name": "database",
                "status": "unhealthy",
                "message": str(e),
            }

    async def check_claude_sdk(self) -> dict[str, Any]:
        """
        Check Claude Agent SDK availability.

        Returns:
            Health check result.
        """
        try:
            from claude_agent_sdk import query  # noqa: F401

            return {
                "name": "claude_sdk",
                "status": "healthy",
                "message": "SDK available",
            }

        except ImportError:
            return {
                "name": "claude_sdk",
                "status": "degraded",
                "message": "SDK not installed, using mock",
            }

        except Exception as e:
            return {
                "name": "claude_sdk",
                "status": "unhealthy",
                "message": str(e),
            }

    async def check_api_key(self) -> dict[str, Any]:
        """
        Check API key configuration.

        Returns:
            Health check result.
        """
        try:
            from src.core.config import get_settings

            settings = get_settings()
            api_key = settings.anthropic_api_key.get_secret_value()

            if api_key and api_key.startswith("sk-ant-"):
                return {
                    "name": "api_key",
                    "status": "healthy",
                    "message": "API key configured",
                }
            else:
                return {
                    "name": "api_key",
                    "status": "degraded",
                    "message": "API key format unexpected",
                }

        except Exception as e:
            return {
                "name": "api_key",
                "status": "unhealthy",
                "message": str(e),
            }

    async def check_disk_space(self) -> dict[str, Any]:
        """
        Check available disk space.

        Returns:
            Health check result.
        """
        import shutil

        try:
            total, used, free = shutil.disk_usage("/")
            free_gb = free / (1024**3)
            used_percent = (used / total) * 100

            if free_gb < 1:
                status = "unhealthy"
                message = f"Low disk space: {free_gb:.1f}GB free"
            elif free_gb < 5:
                status = "degraded"
                message = f"Disk space warning: {free_gb:.1f}GB free"
            else:
                status = "healthy"
                message = f"{free_gb:.1f}GB free ({used_percent:.1f}% used)"

            return {
                "name": "disk_space",
                "status": status,
                "message": message,
                "details": {
                    "free_gb": round(free_gb, 2),
                    "used_percent": round(used_percent, 1),
                },
            }

        except Exception as e:
            return {
                "name": "disk_space",
                "status": "unknown",
                "message": str(e),
            }

    async def check_all(self) -> dict[str, Any]:
        """
        Run all health checks.

        Returns:
            Aggregate health status.
        """
        logger.info("Running health checks")

        checks = [
            await self.check_database(),
            await self.check_claude_sdk(),
            await self.check_api_key(),
            await self.check_disk_space(),
        ]

        # Determine overall status
        statuses = [c["status"] for c in checks]

        if "unhealthy" in statuses:
            overall = "unhealthy"
        elif "degraded" in statuses:
            overall = "degraded"
        else:
            overall = "healthy"

        return {
            "status": overall,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": checks,
        }


async def check_health() -> dict[str, Any]:
    """
    Convenience function to run health checks.

    Returns:
        Health check results.
    """
    checker = HealthChecker()
    return await checker.check_all()
