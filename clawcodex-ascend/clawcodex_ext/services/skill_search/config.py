from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clawcodex_ext.feature_gate import FeatureRegistry


@dataclass(frozen=True)
class SkillSearchConfig:
    """Configuration for skill search.

    Use :meth:`from_feature_gate` to create a config that reads
    ``enabled`` from the :class:`~clawcodex_ext.feature_gate.FeatureRegistry`
    (``SKILL_SEARCH_TFIDF`` flag).
    """

    enabled: bool = False
    top_k: int = 8
    min_score: float = 0.05
    index_path: Path = Path("~/.clawcodex/skill_search/index.json")
    refresh_interval_seconds: int = 300
    save_cooldown_seconds: int = 5
    source_weights: dict[str, float] = field(
        default_factory=lambda: {
            "project": 1.3,
            "local": 1.1,
            "template": 1.0,
            "mcp": 0.9,
        }
    )

    @classmethod
    def from_feature_gate(
        cls,
        *,
        registry: FeatureRegistry | None = None,
        **overrides: object,
    ) -> SkillSearchConfig:
        """Create a config whose ``enabled`` is read from the feature gate.

        Args:
            registry: Optional :class:`FeatureRegistry` instance.  When
                ``None``, uses the process-global singleton.
            **overrides: Additional keyword arguments to override config
                fields (e.g. ``top_k=10``, ``index_path=...``).

        Returns:
            A :class:`SkillSearchConfig` instance.
        """
        from clawcodex_ext.feature_gate import get_registry  # pylint: disable=no-name-in-module

        reg = registry or get_registry()
        enabled = reg.is_enabled("SKILL_SEARCH_TFIDF")
        return cls(enabled=enabled, **overrides)  # type: ignore[arg-type]
