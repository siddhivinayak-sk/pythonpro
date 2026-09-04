"""Runtime types for the LLM registry."""

from __future__ import annotations

from dataclasses import dataclass

from ..config.models import Capability, Provider


@dataclass(frozen=True)
class ResolvedModel:
    """A concrete model reachable through a specific connection.

    ``ref`` is the stable ``(connection_id, model_name)`` identifier used across the platform.
    """

    connection_id: str
    provider: Provider
    model_name: str
    display_name: str
    capabilities: tuple[Capability, ...]
    dimensions: int | None = None
    context_window: int | None = None

    @property
    def ref(self) -> tuple[str, str]:
        return (self.connection_id, self.model_name)

    def has(self, capability: Capability) -> bool:
        return capability in self.capabilities
