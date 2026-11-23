"""
model_fusion/context_fusion.py — Heat-weighted multi-agent context fusion.

Extends arc/core/cache_layer.py's LociItem to handle context blocks from
all five IDE agents (Arc · Claude · Codex · Copilot · local-metal).

How it works
────────────
Each agent contributes AgentContext blocks.  Every block carries:
  • text   — the context content
  • heat   — LOCI heat score from the Arc memory engine
  • agent  — which model produced it ("arc", "claude", "codex", "copilot", "local")
  • kind   — "memory" | "code" | "tool" | "message" | "completion"

Fusion algorithm:
  1. Score each block → heat tier (HOT / MILD / COLD).
  2. Drop COLD blocks entirely (noise, not worth the tokens).
  3. Sort remaining by heat descending.
  4. Compute heat-weighted attention across MILD+HOT blocks:
       score_i = softmax(heat_i / τ)
       where τ=0.3 (temperature — sharpens around HOT blocks)
  5. Truncate to max_tokens using the weighted order.
  6. Annotate each block with the correct cache_control TTL:
       HOT  → extended (1 hr)   injected first in context window
       MILD → ephemeral (5 min) injected after HOT blocks
  7. Wrap in <loci_context> so PromptCacheLayer's BP3 breakpoint fires.

Output is a single string ready to insert as the system/assistant context
block, plus a FusionSummary for the IDE status panel.

Usage
─────
    from model_fusion.context_fusion import ContextFuser, AgentContext

    fuser = ContextFuser(max_tokens=4096)

    blocks = [
        AgentContext(agent="arc",    text="...", heat=0.91, kind="memory"),
        AgentContext(agent="claude", text="...", heat=0.60, kind="message"),
        AgentContext(agent="codex",  text="...", heat=0.78, kind="code"),
        AgentContext(agent="copilot",text="...", heat=0.42, kind="completion"),
        AgentContext(agent="local",  text="...", heat=0.22, kind="tool"),  # COLD → dropped
    ]
    result = fuser.fuse(blocks)
    # result.context_text → inject into prompt
    # result.cache_blocks → pass to PromptCacheLayer.annotate()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from model_fusion.heat_tiers import (
    HeatTier, classify,
    TIER_HOT_BOUNDARY, TIER_MILD_BOUNDARY,
)


# ── Agent context block ───────────────────────────────────────────────────────

@dataclass
class AgentContext:
    """
    One context contribution from a single agent.

    agent  : "arc" | "claude" | "codex" | "copilot" | "local" | ...
    text   : the context text to inject
    heat   : LOCI heat score for this block (0.0–1.0)
    kind   : "memory" | "code" | "tool" | "message" | "completion"
    tokens : estimated token count (chars/4 if not provided)
    """
    agent:  str
    text:   str
    heat:   float
    kind:   str           = "message"
    tokens: Optional[int] = None

    def __post_init__(self) -> None:
        if self.tokens is None:
            self.tokens = max(1, len(self.text) // 4)

    @property
    def tier(self) -> HeatTier:
        return classify(self.heat)

    @property
    def cache_ttl(self) -> str:
        """Cache TTL string for PromptCacheLayer annotation."""
        if self.tier in (HeatTier.SOVEREIGN, HeatTier.HOT):
            return "extended"
        if self.tier == HeatTier.MILD:
            return "ephemeral"
        return "none"


# ── Fusion result ─────────────────────────────────────────────────────────────

@dataclass
class CacheBlock:
    """One annotated block ready for PromptCacheLayer.annotate()."""
    text:  str
    agent: str
    heat:  float
    ttl:   str               # "extended" | "ephemeral" | "none"
    cache_control: Optional[dict] = None


@dataclass
class FusionResult:
    """
    Output of ContextFuser.fuse().

    context_text   : full <loci_context>...</loci_context> string to inject
    cache_blocks   : ordered list of CacheBlocks (HOT first) for the cache layer
    hot_blocks     : count of HOT-tier blocks included
    mild_blocks    : count of MILD-tier blocks included
    dropped_blocks : count of COLD blocks dropped
    total_tokens   : estimated token count of context_text
    fusion_weights : {agent: attention_weight} for the IDE panel
    """
    context_text:   str
    cache_blocks:   list[CacheBlock]
    hot_blocks:     int
    mild_blocks:    int
    dropped_blocks: int
    total_tokens:   int
    fusion_weights: dict[str, float] = field(default_factory=dict)


# ── Attention softmax over heat ───────────────────────────────────────────────

def _heat_attention(blocks: list[AgentContext], temperature: float = 0.3) -> list[float]:
    """
    Compute softmax(heat_i / τ) over active (non-COLD) blocks.
    Returns a weight per block; they sum to 1.0.
    """
    scores = [b.heat / temperature for b in blocks]
    m = max(scores, default=0.0)
    exp_ = [math.exp(s - m) for s in scores]
    total = sum(exp_) or 1.0
    return [e / total for e in exp_]


# ── ContextFuser ──────────────────────────────────────────────────────────────

class ContextFuser:
    """
    Heat-weighted context merger for multi-agent IDE prompts.

    max_tokens     : hard cap on the fused context window
    temperature    : softmax sharpness — lower = more weight on HOT blocks
    prefer_agents  : if set, only include these agents (e.g. ["arc", "claude"])
    """

    def __init__(
        self,
        max_tokens:     int   = 4096,
        temperature:    float = 0.3,
        prefer_agents:  Optional[list[str]] = None,
    ) -> None:
        self.max_tokens    = max_tokens
        self.temperature   = temperature
        self.prefer_agents = prefer_agents

    def fuse(self, blocks: list[AgentContext]) -> FusionResult:
        """
        Merge agent context blocks into a single prompt-ready context string.
        """
        # 0. Filter by preferred agents if specified
        if self.prefer_agents:
            filtered = [b for b in blocks if b.agent in self.prefer_agents]
            dropped_pref = len(blocks) - len(filtered)
            blocks = filtered
        else:
            dropped_pref = 0

        # 1. Tier split: drop COLD, keep HOT+MILD
        hot_blocks  = [b for b in blocks if b.tier in (HeatTier.HOT, HeatTier.SOVEREIGN)]
        mild_blocks = [b for b in blocks if b.tier == HeatTier.MILD]
        cold_blocks = [b for b in blocks if b.tier == HeatTier.COLD]

        active = hot_blocks + mild_blocks

        if not active:
            # Everything was COLD — emit minimal context
            return FusionResult(
                context_text   = "",
                cache_blocks   = [],
                hot_blocks     = 0,
                mild_blocks    = 0,
                dropped_blocks = len(cold_blocks) + dropped_pref,
                total_tokens   = 0,
            )

        # 2. Heat-weighted attention scores
        weights = _heat_attention(active, temperature=self.temperature)

        # 3. Sort by weight descending (HOT blocks naturally sort first)
        ranked = sorted(zip(weights, active), key=lambda x: x[0], reverse=True)

        # 4. Build ordered context, respecting token budget
        ordered_blocks: list[tuple[float, AgentContext]] = []
        budget_used = 0
        for w, b in ranked:
            if budget_used + (b.tokens or 0) > self.max_tokens:
                break
            ordered_blocks.append((w, b))
            budget_used += b.tokens or 0

        # 5. Compose context text
        lines: list[str] = ["<loci_context>"]
        cache_blocks: list[CacheBlock] = []
        fw: dict[str, float] = {}

        for w, b in ordered_blocks:
            header = f"[{b.agent.upper()} · {b.kind} · heat={b.heat:.2f} · w={w:.3f}]"
            lines.append(header)
            lines.append(b.text.strip())
            lines.append("")
            cc = {"type": "ephemeral"} if b.cache_ttl in ("ephemeral", "extended") else None
            cache_blocks.append(CacheBlock(
                text=b.text, agent=b.agent, heat=b.heat, ttl=b.cache_ttl, cache_control=cc
            ))
            fw[b.agent] = round(fw.get(b.agent, 0.0) + w, 4)

        lines.append("</loci_context>")
        context_text = "\n".join(lines)

        return FusionResult(
            context_text   = context_text,
            cache_blocks   = cache_blocks,
            hot_blocks     = len([b for _, b in ordered_blocks if b.tier != HeatTier.MILD]),
            mild_blocks    = len([b for _, b in ordered_blocks if b.tier == HeatTier.MILD]),
            dropped_blocks = len(cold_blocks) + dropped_pref,
            total_tokens   = budget_used,
            fusion_weights = fw,
        )

    def summary(self, result: FusionResult) -> str:
        """One-line summary for the IDE status panel."""
        return (
            f"Context fusion: HOT={result.hot_blocks} MILD={result.mild_blocks} "
            f"DROPPED={result.dropped_blocks} tokens≈{result.total_tokens} "
            f"agents={list(result.fusion_weights.keys())}"
        )
