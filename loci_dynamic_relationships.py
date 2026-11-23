"""
loci_dynamic_relationships.py — Relationships as dynamic QSymPrev states.

Masai Mara model
────────────────
In the Masai Mara, predator-prey relationships are not static (lion HUNTS zebra
forever) — they are dynamic temporal states with valid intervals, shifting with
season, territory, and pack dynamics.

LOCI models relationships the same way.  A relationship is NOT a static edge
in a knowledge graph.  It is a sequence of append-only QSymPrev nodes, each
carrying a VersionedInterval [valid_from, valid_until) encoding the relationship's
strength, valence, and type at that point in time.

Heat decay
──────────
Relationship heat decays at the same rate as episodic memory, but with a longer
half-life (default 72 h vs ~24 h for single memories).
  H(t) = H(0) · exp(−ln2 · t / T½)

LOCI Heat Law tiers (θ_HOT = 0.7250):
  HOT   (H ≥ 0.7250) → active context — included in every prompt
  MILD  (H ≥ 0.350)  → background awareness — available on demand
  COLD  (H <  0.350) → archived — queryable, never injected automatically

Append-only semantics
─────────────────────
When a relationship changes (strength, type, valence), the current node's
interval is closed (valid_until = now) and a new node is appended.
Nothing is ever deleted; the history is fully queryable.

Usage
─────
    from loci_dynamic_relationships import RelationshipStore, RelType

    store = RelationshipStore()

    # Assert the Masai Mara dynamic: initial trust relationship
    store.assert_relationship(
        subject="user",
        object="masai_mara_guide",
        rel_type=RelType.TRUST,
        strength=0.85,
        valence=0.9,
        evidence="shared camp, guided wildebeest migration route",
    )

    # Relationship evolves — assert new state (closes previous version)
    store.assert_relationship(
        subject="user",
        object="masai_mara_guide",
        rel_type=RelType.TRUST,
        strength=0.60,
        valence=0.75,
        evidence="disagreement on route, trust partially eroded",
    )

    # Decay over 48 h
    store.decay_all(elapsed_hours=48)

    hot = store.get_hot_relationships()
    history = store.history("user", "masai_mara_guide", RelType.TRUST)
    facts = store.to_qsymprev_facts()   # inject into QSymPrev stream
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


from loci_adaptive_theta import current_theta as _theta_hot, THETA_ANCHOR as THETA_HOT
# THETA_HOT alias kept for display; routing code should call _theta_hot() for the live value
THETA_MILD  = 0.35
ALPHA_HOPE  = 0.050

# Relationship heat half-life: 72 h (slower decay than episodic memory at ~24 h)
RELATIONSHIP_HALFLIFE_HOURS = 72.0


class RelType(str, Enum):
    TRUST        = "TRUST"
    PROXIMITY    = "PROXIMITY"
    SHARED_GOAL  = "SHARED_GOAL"
    CONFLICT     = "CONFLICT"
    DEPENDENCY   = "DEPENDENCY"
    AFFINITY     = "AFFINITY"
    ADVERSARIAL  = "ADVERSARIAL"


class RelState(str, Enum):
    HOT    = "HOT"      # H ≥ θ_HOT — active context
    MILD   = "MILD"     # θ_MILD ≤ H < θ_HOT — background awareness
    COLD   = "COLD"     # H < θ_MILD — archived
    ENDED  = "ENDED"    # interval explicitly closed (retracted)


@dataclass
class VersionedInterval:
    """Half-open interval [valid_from, valid_until) for one relationship version."""
    valid_from:  float
    valid_until: Optional[float] = None   # None = still active

    @property
    def is_active(self) -> bool:
        return self.valid_until is None or self.valid_until > time.time()

    def duration_hours(self) -> float:
        end = self.valid_until or time.time()
        return (end - self.valid_from) / 3600.0


@dataclass
class RelationshipNode:
    """
    One versioned state of a relationship between subject and object.

    Append-only: nodes are never mutated after creation.  When a relationship
    changes, the current node's interval is closed and a new node is appended.

    Maps to QSymPrev via to_qsymprev_dict():
      fact_key  : "rel:<subject>:<rel_type>:<object>"
      value     : {strength, valence}
      interval  : VersionedInterval
    """
    node_id:    str             = field(default_factory=lambda: str(uuid.uuid4())[:8])
    subject:    str             = ""
    object:     str             = ""
    rel_type:   RelType         = RelType.TRUST
    strength:   float           = 0.5    # [0, 1]
    valence:    float           = 0.5    # [0, 1] — 0=negative, 0.5=neutral, 1=positive
    heat:       float           = THETA_HOT
    interval:   VersionedInterval = field(
        default_factory=lambda: VersionedInterval(time.time())
    )
    source_stream: str          = "QSymPrev"
    evidence:      str          = ""

    @property
    def state(self) -> RelState:
        if not self.interval.is_active:
            return RelState.ENDED
        if self.heat >= THETA_HOT:
            return RelState.HOT
        if self.heat >= THETA_MILD:
            return RelState.MILD
        return RelState.COLD

    def decay(self, elapsed_hours: float) -> None:
        """Exponential heat decay: H(t) = H(0) · exp(−ln2 · t / T½)"""
        factor    = math.exp(-math.log(2) * elapsed_hours / RELATIONSHIP_HALFLIFE_HOURS)
        self.heat = max(ALPHA_HOPE, self.heat * factor)

    def to_qsymprev_dict(self) -> dict:
        return {
            "fact_key":         f"rel:{self.subject}:{self.rel_type.value}:{self.object}",
            "value":            {"strength": self.strength, "valence": self.valence},
            "valid_from":       self.interval.valid_from,
            "valid_until":      self.interval.valid_until,
            "is_deterministic": False,
            "heat":             self.heat,
            "source_stream":    self.source_stream,
            "recompute_rate":   None,
            "evidence":         self.evidence,
        }


class RelationshipStore:
    """
    Append-only store of dynamic relationship states (Masai Mara model).

    Relationships are never deleted — only superseded and decayed.
    The full version history is always queryable for all (subject, object) pairs.
    """

    def __init__(self) -> None:
        self._nodes:          list[RelationshipNode] = []
        self._last_decay_ts:  float                  = time.time()

    def assert_relationship(
        self,
        subject:       str,
        object:        str,
        rel_type:      RelType | str = RelType.TRUST,
        strength:      float         = 0.7,
        valence:       float         = 0.7,
        evidence:      str           = "",
        source_stream: str           = "QSymPrev",
    ) -> RelationshipNode:
        """
        Assert a new (or evolved) relationship state.

        If an active node with the same (subject, object, rel_type) exists,
        its interval is closed first; then the new node is appended.
        """
        if isinstance(rel_type, str):
            rel_type = RelType(rel_type)

        # Close the previously-active version
        for node in self._nodes:
            if (node.subject == subject
                    and node.object   == object
                    and node.rel_type == rel_type
                    and node.interval.is_active):
                node.interval.valid_until = time.time()

        new_node = RelationshipNode(
            subject=subject,
            object=object,
            rel_type=rel_type,
            strength=strength,
            valence=valence,
            heat=THETA_HOT,
            interval=VersionedInterval(time.time()),
            evidence=evidence,
            source_stream=source_stream,
        )
        self._nodes.append(new_node)
        return new_node

    def retract_relationship(
        self,
        subject:  str,
        object:   str,
        rel_type: RelType | str = RelType.TRUST,
        reason:   str           = "",
    ) -> None:
        """Close the active node's interval (relationship ended)."""
        if isinstance(rel_type, str):
            rel_type = RelType(rel_type)
        for node in self._nodes:
            if (node.subject == subject
                    and node.object   == object
                    and node.rel_type == rel_type
                    and node.interval.is_active):
                node.interval.valid_until = time.time()
                if reason:
                    node.evidence = f"[RETRACTED: {reason}] " + node.evidence

    def decay_all(self, elapsed_hours: Optional[float] = None) -> None:
        """
        Apply heat decay to all active nodes.
        When elapsed_hours is None, computes from last call time automatically.
        """
        now = time.time()
        if elapsed_hours is None:
            elapsed_hours = (now - self._last_decay_ts) / 3600.0
        self._last_decay_ts = now
        for node in self._nodes:
            if node.interval.is_active:
                node.decay(elapsed_hours)

    def get_active(
        self,
        subject:  Optional[str]   = None,
        object:   Optional[str]   = None,
        min_heat: float           = 0.0,
    ) -> list[RelationshipNode]:
        return [
            n for n in self._nodes
            if n.interval.is_active
            and n.heat     >= min_heat
            and (subject is None or n.subject == subject)
            and (object  is None or n.object  == object)
        ]

    def get_hot_relationships(self) -> list[RelationshipNode]:
        return self.get_active(min_heat=THETA_HOT)

    def get_mild_relationships(self) -> list[RelationshipNode]:
        return [n for n in self.get_active() if THETA_MILD <= n.heat < THETA_HOT]

    def history(
        self,
        subject:  str,
        object:   str,
        rel_type: Optional[RelType] = None,
    ) -> list[RelationshipNode]:
        """Full version history for (subject, object[, rel_type])."""
        return [
            n for n in self._nodes
            if n.subject == subject
            and n.object  == object
            and (rel_type is None or n.rel_type == rel_type)
        ]

    def to_qsymprev_facts(self) -> list[dict]:
        """Export all HOT active nodes as QSymPrev fact dicts for stream injection."""
        return [n.to_qsymprev_dict() for n in self.get_hot_relationships()]

    def summary(self) -> str:
        active = self.get_active()
        hot    = sum(1 for n in active if n.state == RelState.HOT)
        mild   = sum(1 for n in active if n.state == RelState.MILD)
        return (
            f"RelationshipStore: {len(active)} active "
            f"({hot} HOT, {mild} MILD, {len(self._nodes)} total nodes)"
        )
