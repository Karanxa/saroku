"""
saroku.policy.dsl — Declarative policy definitions for the execution engine.

A Policy binds behavioral properties (PolicyProperty) to classifier ids
(saroku.classifiers.registry.ClassifierRegistry ids, e.g. "llm:gpt-4o-mini",
"rule:sycophancy") and describes, per mode ("balanced", "thorough", ...),
an ordered list of ExecutionLayer cascades that ExecutionEngine
(saroku.execution.engine) walks to produce a verdict.

Layer strategy:
    "cascade"     — try each classifier in the layer in order, stop at the
                    first result whose confidence meets the property's
                    confidence_threshold.
    "speculative" — run every classifier in the layer concurrently, use
                    whichever confident result lands first, cancel the rest.

Example::

    from saroku.policy import Policy

    policy = Policy.from_yaml("src/saroku/policy/policies/default.yml")
    prop = policy.get_property("sycophancy")
    layers = policy.get_layers("balanced")
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import yaml

VALID_STRATEGIES = ("cascade", "speculative")


@dataclass
class PolicyProperty:
    """A single behavioral property and how it should be classified."""
    name: str
    classifier: str
    description: str = ""
    fallback: Optional[str] = None
    context_requirements: list[str] = field(default_factory=list)
    confidence_threshold: float = 0.8

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PolicyProperty":
        return cls(
            name=data["name"],
            classifier=data["classifier"],
            description=data.get("description", ""),
            fallback=data.get("fallback"),
            context_requirements=list(data.get("context_requirements", [])),
            confidence_threshold=float(data.get("confidence_threshold", 0.8)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionLayer:
    """One cascade step: a set of classifiers, a strategy, and a timeout."""
    name: str
    classifiers: list[str] = field(default_factory=list)
    timeout_ms: int = 5000
    strategy: str = "cascade"

    def __post_init__(self) -> None:
        if self.strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy '{self.strategy}' for layer '{self.name}'. "
                f"Expected one of {VALID_STRATEGIES}."
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionLayer":
        return cls(
            name=data["name"],
            classifiers=list(data.get("classifiers", [])),
            timeout_ms=int(data.get("timeout_ms", 5000)),
            strategy=data.get("strategy", "cascade"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PolicyMetadata:
    """Minimal bookkeeping for a policy file."""
    version: str = "1"
    author: str = ""
    description: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PolicyMetadata":
        return cls(
            version=str(data.get("version", "1")),
            author=data.get("author", ""),
            description=data.get("description", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Policy:
    """A full policy: properties + per-mode execution cascades."""
    version: str
    policy_id: str
    properties: list[PolicyProperty] = field(default_factory=list)
    execution: dict[str, list[ExecutionLayer]] = field(default_factory=dict)
    metadata: PolicyMetadata = field(default_factory=PolicyMetadata)

    def get_property(self, name: str) -> PolicyProperty:
        for prop in self.properties:
            if prop.name == name:
                return prop
        raise KeyError(f"Property '{name}' not defined in policy '{self.policy_id}'.")

    def get_layers(self, mode: str) -> list[ExecutionLayer]:
        if mode not in self.execution:
            raise KeyError(
                f"Mode '{mode}' not defined in policy '{self.policy_id}'. "
                f"Available modes: {list(self.execution)}."
            )
        return self.execution[mode]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Policy":
        properties = [PolicyProperty.from_dict(p) for p in data.get("properties", [])]
        execution = {
            mode: [ExecutionLayer.from_dict(layer) for layer in layers]
            for mode, layers in data.get("execution", {}).items()
        }
        metadata = PolicyMetadata.from_dict(data.get("metadata", {}))
        return cls(
            version=str(data.get("version", "1")),
            policy_id=data["policy_id"],
            properties=properties,
            execution=execution,
            metadata=metadata,
        )

    @classmethod
    def from_yaml(cls, path: str) -> "Policy":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "policy_id": self.policy_id,
            "properties": [p.to_dict() for p in self.properties],
            "execution": {
                mode: [layer.to_dict() for layer in layers]
                for mode, layers in self.execution.items()
            },
            "metadata": self.metadata.to_dict(),
        }

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)
