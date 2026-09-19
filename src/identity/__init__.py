"""Identity resolution (Phase 2)."""
from src.identity.graph import IdentityGraph
from src.identity.matcher import IdentityRecord, MatchMethod, MatchResult, match
from src.identity.resolver import IdentityResolver, Resolution

__all__ = ["IdentityGraph", "IdentityRecord", "MatchMethod", "MatchResult", "match", "IdentityResolver", "Resolution"]
