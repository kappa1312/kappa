"""Configuration management using Pydantic Settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )

    # Anthropic API
    anthropic_api_key: SecretStr = Field(
        ...,
        description="Anthropic API key for Claude access",
    )

    # Database
    database_url: PostgresDsn = Field(
        ...,
        description="PostgreSQL connection URL",
    )

    # Kappa Configuration
    kappa_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )
    kappa_max_parallel_sessions: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of parallel Claude sessions",
    )
    kappa_session_timeout: int = Field(
        default=3600,
        ge=60,
        description="Session timeout in seconds",
    )
    kappa_working_dir: str = Field(
        default="./workspace",
        description="Working directory for Kappa operations",
    )
    kappa_debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )
    kappa_hot_reload_skills: bool = Field(
        default=True,
        description="Enable hot reload for skills",
    )
    kappa_metrics_interval: int = Field(
        default=60,
        ge=10,
        description="Metrics collection interval in seconds",
    )

    # Claude Agent SDK
    claude_code_exit_after_stop_delay: int | None = Field(
        default=None,
        description="Exit SDK mode after idle duration (ms)",
    )
    claude_code_use_bedrock: bool = Field(
        default=False,
        description="Use Amazon Bedrock as provider",
    )
    claude_code_use_vertex: bool = Field(
        default=False,
        description="Use Google Vertex AI as provider",
    )

    # LangSmith (optional)
    langchain_tracing_v2: bool = Field(
        default=False,
        description="Enable LangSmith tracing",
    )
    langchain_api_key: SecretStr | None = Field(
        default=None,
        description="LangSmith API key",
    )
    langchain_project: str | None = Field(
        default=None,
        description="LangSmith project name",
    )

    @property
    def database_url_async(self) -> str:
        """Get async database URL (with asyncpg driver)."""
        url = str(self.database_url)
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return url

    @property
    def database_url_sync(self) -> str:
        """Get sync database URL (with psycopg2 driver)."""
        url = str(self.database_url)
        if "+asyncpg" in url:
            return url.replace("+asyncpg", "", 1)
        return url


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Settings instance loaded from environment.

    Example:
        >>> settings = get_settings()
        >>> settings.kappa_max_parallel_sessions
        5
    """
    return Settings()  # type: ignore[call-arg]


def clear_settings_cache() -> None:
    """Clear the settings cache (useful for testing)."""
    get_settings.cache_clear()
