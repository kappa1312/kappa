"""Conflict detection and resolution for parallel session outputs."""

from src.conflict.detector import ConflictDetector
from src.conflict.resolver import ConflictResolver
from src.conflict.strategies import (
    MergeStrategy,
    NewerWinsStrategy,
    ResolutionStrategy,
    SemanticMergeStrategy,
)

__all__ = [
    "ConflictDetector",
    "ConflictResolver",
    "MergeStrategy",
    "NewerWinsStrategy",
    "ResolutionStrategy",
    "SemanticMergeStrategy",
]
