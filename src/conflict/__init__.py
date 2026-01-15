"""
Conflict Detection and Resolution Module for Kappa OS

Provides comprehensive conflict handling for parallel session outputs:
- ConflictDetector: Pre-execution conflict analysis
- ConflictResolver: Strategy-based conflict resolution
- MergeEngine: Post-execution output combination
"""

from src.conflict.detector import (
    Conflict,
    ConflictDetector,
    ConflictReport,
    ConflictSeverity,
    ConflictType,
    detect_conflicts,
)
from src.conflict.merge_engine import (
    MergeConflict,
    MergeEngine,
    MergeResult,
)
from src.conflict.merge_engine import (
    MergeStrategy as MergeStrategyEnum,
)
from src.conflict.resolver import (
    ConflictResolver,
    ResolutionPlan,
    resolve_conflict,
)
from src.conflict.strategies import (
    CompositeStrategy,
    DependencyViolationStrategy,
    FileWriteStrategy,
    ImportCollisionStrategy,
    ManualReviewStrategy,
    MergeStrategy,
    NamingConflictStrategy,
    NewerWinsStrategy,
    ResolutionResult,
    ResolutionStatus,
    ResolutionStrategy,
    ResourceContentionStrategy,
    SemanticMergeStrategy,
    TypeMismatchStrategy,
)

__all__ = [
    # Detector
    "Conflict",
    "ConflictDetector",
    "ConflictReport",
    "ConflictSeverity",
    "ConflictType",
    "detect_conflicts",
    # Resolver
    "ConflictResolver",
    "ResolutionPlan",
    "resolve_conflict",
    # Strategies
    "CompositeStrategy",
    "DependencyViolationStrategy",
    "FileWriteStrategy",
    "ImportCollisionStrategy",
    "ManualReviewStrategy",
    "MergeStrategy",
    "NamingConflictStrategy",
    "NewerWinsStrategy",
    "ResolutionResult",
    "ResolutionStatus",
    "ResolutionStrategy",
    "ResourceContentionStrategy",
    "SemanticMergeStrategy",
    "TypeMismatchStrategy",
    # Merge Engine
    "MergeConflict",
    "MergeEngine",
    "MergeResult",
    "MergeStrategyEnum",
]
